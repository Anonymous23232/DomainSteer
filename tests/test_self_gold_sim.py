"""Tests for dense+BM25 hybrid similarity — no GPU, no API, no downloads."""

import json
from pathlib import Path

import numpy as np
import pytest

from extras.self_gold_sim import (
    BM25Index, DEFAULT_EMBED_MODEL, alphas_for_record, as_gpt_gold_records,
    as_short_sys_records, assign_key, assignment_rows, assignment_summary,
    attach_short_sys, average_heatmaps, bm25_similarity, candidate_text,
    collect_texts, cosine, domain_plot_dir, domain_slug_from_plot_path,
    fit_encoders, flatten_layer_records, hash_embeddings, heatmap_matrix,
    hybrid_similarity, iter_scored_plot_json_paths, nested_steered,
    plot_json_identity, resolve_embed_route, score_records, summarize,
    summarize_by_layer, tokenize,
)


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Materiality, in Accounting!") == [
        "materiality", "in", "accounting",
    ]


def test_cosine_identical_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, v) == pytest.approx(1.0)
    assert cosine(v, np.zeros(3)) == 0.0


def test_bm25_identical_is_one():
    corpus = [
        tokenize("A stack is a genotype carrying two traits."),
        tokenize("A stack is a last-in first-out structure."),
    ]
    index = BM25Index.fit(corpus)
    toks = corpus[0]
    assert bm25_similarity(toks, toks, index) == pytest.approx(1.0)
    assert bm25_similarity(toks, tokenize("unrelated quantum foam"), index) < 0.2


def test_hybrid_is_one_for_identical_text():
    text = "Materiality is the threshold for financial misstatement."
    records = [{
        "gold": text, "baseline": text,
        "steered": {"0.1000": text},
        "kind": "direct", "index": 1, "alphas": [0.0, 0.1],
    }]
    embeddings = hash_embeddings([text])
    enc = fit_encoders(records, embeddings=embeddings)
    metrics = hybrid_similarity(text, text, enc)
    assert metrics["dense"] == pytest.approx(1.0)
    assert metrics["bm25"] == pytest.approx(1.0)
    assert metrics["hybrid"] == pytest.approx(1.0)


def test_hybrid_drops_when_texts_diverge():
    gold = "A stack is a genotype carrying two or more combined traits."
    other = "A stack is a last-in first-out data structure in computer science."
    records = [{"gold": gold, "baseline": other, "steered": {}, "kind": "direct"}]
    embeddings = hash_embeddings([gold, other])
    enc = fit_encoders(records, embeddings=embeddings)
    same = hybrid_similarity(gold, gold, enc)
    diff = hybrid_similarity(gold, other, enc)
    assert same["hybrid"] > diff["hybrid"]
    assert same["dense"] > diff["dense"]
    assert same["bm25"] > diff["bm25"]


def test_score_records_include_baseline_as_alpha_zero():
    records = [{
        "index": 1, "term": "stack", "kind": "direct",
        "question": "What is a stack?",
        "gold": "A stack combines two genetic events.",
        "baseline": "A stack is a LIFO structure.",
        "steered": {"0.1000": "A stack stores elements in computer memory."},
        "alphas": [0.0, 0.1],
    }]
    texts = [
        records[0]["gold"], records[0]["baseline"],
        records[0]["steered"]["0.1000"],
    ]
    enc = fit_encoders(records, embeddings=hash_embeddings(texts))
    scored = score_records(records, enc)
    alphas = sorted({row["alpha"] for row in scored})
    assert alphas == [0.0, 0.1]
    summary = summarize(scored)
    kinds = {row["kind"] for row in summary}
    pairs = {row["pair"] for row in scored}
    assert "all" in kinds and "direct" in kinds
    assert "gold_vs_answer" in pairs
    assert "steered_vs_baseline" in pairs
    assert "dense" in scored[0] and "bm25" in scored[0]


def test_candidate_text_reads_baseline_and_steered():
    row = {"baseline": "base", "steered": {"0.1600": "steer"}}
    assert candidate_text(row, 0.0) == "base"
    assert candidate_text(row, 0.16) == "steer"
    assert 0.0 in alphas_for_record(row)
    assert 0.16 in alphas_for_record(row)


def test_plot_rows_keeps_direct_and_tricky_apart():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "extras" / "scripts" / "plot_self_gold_similarity.py"
    spec = importlib.util.spec_from_file_location("plot_self_gold_similarity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    summary = [
        {"kind": "direct", "pair": "gold_vs_answer", "alpha": 0.1, "hybrid_mean": 0.5},
        {"kind": "tricky", "pair": "gold_vs_answer", "alpha": 0.1, "hybrid_mean": 0.2},
        {"kind": "direct", "pair": "steered_vs_baseline", "alpha": 0.1, "hybrid_mean": 0.8},
    ]
    rows = mod._rows(summary, "direct", "gold_vs_answer")
    assert len(rows) == 1
    assert rows[0]["hybrid_mean"] == 0.5
    assert not mod._rows(summary, "tricky", "steered_vs_baseline")


def test_flatten_layer_records_does_not_treat_layers_as_alphas():
    gold = "A stack combines two genetic events."
    baseline = "A stack is a LIFO structure."
    nested = {
        "index": 1, "kind": "direct", "chosen_layer": 12,
        "gold": gold, "baseline": baseline,
        "steered": {
            "10": {"0.1000": "layer ten answer about genotypes."},
            "12": {"0.1000": gold},
        },
        "alphas": [0.0, 0.1],
    }
    assert nested_steered(nested["steered"]) is True
    views = flatten_layer_records([nested])
    assert {v["layer"] for v in views} == {10, 12}
    assert alphas_for_record(views[0]) == [0.0, 0.1]
    texts = collect_texts([nested])
    assert gold in texts
    assert "layer ten answer about genotypes." in texts


def _load_domain_question_plot():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "extras" / "scripts" / "plot_domain_question_similarity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "plot_domain_question_similarity", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pair_questions_tracks_gpt_and_contextual_lifts():
    mod = _load_domain_question_plot()
    scored = []
    for alpha, gpt, ctx in ((0.0, 0.40, 0.50), (0.20, 0.70, 0.80)):
        scored.append({
            "index": 1, "kind": "direct", "term": "event",
            "question": "What is an event?", "alpha": alpha,
            "pair": "gold_vs_answer", "dense": gpt, "bm25": 0.1,
            "hybrid": gpt,
        })
        scored.append({
            "index": 1, "kind": "direct", "term": "event",
            "question": "What is an event?", "alpha": alpha,
            "pair": "contextual_vs_answer", "dense": ctx, "bm25": 0.1,
            "hybrid": ctx,
        })
    questions = mod.pair_questions(scored)
    assert len(questions) == 1
    rows = mod.question_rows(questions, 0.20)
    assert rows[0]["gpt_win"] is True
    assert rows[0]["ctx_win"] is True
    assert rows[0]["gpt_lift_dense"] == pytest.approx(0.30)
    stats = mod.kind_stats(rows, "all")
    assert stats["gpt_wins"] == 1
    assert stats["gpt_winrate"] == pytest.approx(1.0)


def test_as_gpt_gold_swaps_llama_gold_to_contextual_baseline():
    records = [{
        "index": 1, "kind": "direct",
        "gpt_answer": "GPT: arrest is a cell-cycle halt.",
        "gold": "Llama: arrest stops the cell cycle.",
        "baseline": "Police detain a suspect.",
        "steered": {"0.2000": "Arrest is G1 cell-cycle arrest."},
    }]
    remapped = as_gpt_gold_records(records)
    assert len(remapped) == 1
    assert remapped[0]["gold"] == "GPT: arrest is a cell-cycle halt."
    assert remapped[0]["baseline"] == "Police detain a suspect."
    assert remapped[0]["contextual"] == "Llama: arrest stops the cell cycle."
    embeddings = hash_embeddings(collect_texts(remapped))
    enc = fit_encoders(remapped, embeddings=embeddings)
    scored = score_records(remapped, enc)
    zero = next(r for r in scored if r["alpha"] == 0.0
                and r["pair"] == "gold_vs_answer")
    ctx = next(r for r in scored if r["pair"] == "baseline_vs_contextual")
    steered = next(r for r in scored if abs(r["alpha"] - 0.2) < 1e-9
                   and r["pair"] == "gold_vs_answer")
    ctx_curve = next(r for r in scored if abs(r["alpha"] - 0.2) < 1e-9
                     and r["pair"] == "contextual_vs_answer")
    assert zero["dense"] == hybrid_similarity(
        remapped[0]["gold"], remapped[0]["baseline"], enc,
    )["dense"]
    assert ctx["dense"] == hybrid_similarity(
        remapped[0]["baseline"], remapped[0]["contextual"], enc,
    )["dense"]
    assert steered["pair"] == "gold_vs_answer"
    assert ctx_curve["dense"] == hybrid_similarity(
        remapped[0]["contextual"], remapped[0]["steered"]["0.2000"], enc,
    )["dense"]
    empty = as_gpt_gold_records([{"gold": "x", "gpt_answer": ""}])
    assert empty == []


def test_as_short_sys_records_uses_gpt_gold_and_short_sys_contextual():
    gpt = "GPT: event is a DNA insertion."
    short = "Short: event is a GM insertion."
    baseline = "Everyday concert."
    steered_text = "An event is a transgenic insertion."
    records = [{
        "index": 1, "kind": "direct",
        "gpt_answer": gpt,
        "gold": "Llama coaching gold.",
        "short_sys": short,
        "baseline": baseline,
        "steered": {"0.2000": steered_text},
    }]
    remapped = as_short_sys_records(records)
    assert len(remapped) == 1
    assert remapped[0]["gold"] == gpt
    assert remapped[0]["contextual"] == short
    assert remapped[0]["baseline"] == baseline
    embeddings = hash_embeddings(collect_texts(remapped))
    enc = fit_encoders(remapped, embeddings=embeddings)
    scored = score_records(remapped, enc)
    zero = next(r for r in scored if r["alpha"] == 0.0
                and r["pair"] == "gold_vs_answer")
    gpt_short = next(r for r in scored if r["pair"] == "gold_vs_contextual")
    steered = next(r for r in scored if abs(r["alpha"] - 0.2) < 1e-9
                   and r["pair"] == "gold_vs_answer")
    short_steered = next(r for r in scored if abs(r["alpha"] - 0.2) < 1e-9
                         and r["pair"] == "contextual_vs_answer")
    assert zero["dense"] == hybrid_similarity(gpt, baseline, enc)["dense"]
    assert gpt_short["dense"] == hybrid_similarity(gpt, short, enc)["dense"]
    assert steered["dense"] == hybrid_similarity(gpt, steered_text, enc)["dense"]
    assert short_steered["dense"] == hybrid_similarity(
        short, steered_text, enc,
    )["dense"]
    empty = as_short_sys_records([{"gpt_answer": "x", "short_sys": ""}])
    assert empty == []


def test_attach_short_sys_fills_from_gold_json(tmp_path: Path):
    (tmp_path / "gold.json").write_text(json.dumps({
        "items": [{
            "question": "What is an event?",
            "short_sys": "Event means a GM insertion.",
        }],
    }), encoding="utf-8")
    rows = attach_short_sys([{
        "question": "What is an event?",
        "gpt_answer": "A unique insertion.",
        "baseline": "A concert.",
    }], tmp_path)
    assert rows[0]["short_sys"] == "Event means a GM insertion."
    kept = attach_short_sys([{
        "question": "What is an event?",
        "short_sys": "Already set.",
    }], tmp_path)
    assert kept[0]["short_sys"] == "Already set."


def test_layer_heatmap_matrix_from_nested_steered():
    gold = "Materiality is the threshold for financial misstatement."
    close = "Materiality is the cutoff for a financial misstatement."
    far = "A stack is a last-in first-out structure."
    records = [{
        "index": 1, "kind": "direct", "chosen_layer": 12,
        "gold": gold, "baseline": far,
        "steered": {
            "10": {"0.0500": far, "0.1500": close},
            "12": {"0.0500": close, "0.1500": gold},
        },
        "alphas": [0.0, 0.05, 0.15],
    }]
    texts = collect_texts(records)
    enc = fit_encoders(records, embeddings=hash_embeddings(texts))
    scored = score_records(records, enc)
    layers = {row["layer"] for row in scored}
    assert layers == {10, 12}
    summary = summarize_by_layer(scored)
    layers, alphas, grid = heatmap_matrix(summary, "direct")
    assert layers == [10, 12]
    assert 0.0 in alphas and 0.05 in alphas and 0.15 in alphas
    gold_vs = [
        r for r in summary
        if r["kind"] == "direct" and r["pair"] == "gold_vs_answer"
    ]
    best = max(gold_vs, key=lambda r: r["hybrid_mean"])
    assert best["layer"] == 12
    assert best["alpha"] == 0.15
    other = heatmap_matrix(summary, "direct")
    averaged = average_heatmaps([other, other])
    assert averaged[0] == layers
    assert averaged[2].shape == grid.shape


def test_plot_layer_heatmap_script_builds_matrix():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "extras" / "scripts" / "plot_self_gold_layer_heatmap.py"
    )
    spec = importlib.util.spec_from_file_location(
        "plot_self_gold_layer_heatmap", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    records = [{"chosen_layer": 16, "layer": 12}]
    assert mod._chosen_layer(records, Path("missing")) == 16


def _load_metrics_plot():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "extras" / "scripts" / "plot_self_gold_metrics.py"
    )
    spec = importlib.util.spec_from_file_location(
        "plot_self_gold_metrics", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_peak_lift_picks_best_alpha_not_largest():
    mod = _load_metrics_plot()
    summary = [
        {"kind": "direct", "pair": "gold_vs_answer", "alpha": 0.0,
         "dense_mean": 0.60, "bm25_mean": 0.20, "n": 7},
        {"kind": "direct", "pair": "gold_vs_answer", "alpha": 0.15,
         "dense_mean": 0.72, "bm25_mean": 0.29, "n": 7},
        {"kind": "direct", "pair": "gold_vs_answer", "alpha": 0.30,
         "dense_mean": 0.68, "bm25_mean": 0.18, "n": 7},
        {"kind": "tricky", "pair": "gold_vs_answer", "alpha": 0.15,
         "dense_mean": 0.99, "bm25_mean": 0.99, "n": 6},
    ]
    dense = mod.peak_lift(summary, "direct", "dense_mean")
    assert dense["peak_alpha"] == 0.15
    assert dense["lift"] == pytest.approx(0.12)
    bm25 = mod.peak_lift(summary, "direct", "bm25_mean")
    assert bm25["peak_alpha"] == 0.15
    assert bm25["lift"] == pytest.approx(0.09)


def test_peak_lift_zero_when_alpha_zero_is_best():
    mod = _load_metrics_plot()
    summary = [
        {"kind": "tricky", "pair": "gold_vs_answer", "alpha": 0.0,
         "dense_mean": 0.50, "n": 6},
        {"kind": "tricky", "pair": "gold_vs_answer", "alpha": 0.30,
         "dense_mean": 0.40, "n": 6},
    ]
    peak = mod.peak_lift(summary, "tricky", "dense_mean")
    assert peak["peak_alpha"] == 0.0
    assert peak["lift"] == pytest.approx(0.0)


def test_load_plot_jsons_skips_overview_and_filters_ids(tmp_path):
    mod = _load_metrics_plot()
    model_dir = tmp_path / "llama"
    model_dir.mkdir()
    payload = {
        "model": "llama", "cluster_id": "3101",
        "cluster": "Biochemistry",
        "summary": [{"kind": "direct", "pair": "gold_vs_answer",
                     "alpha": 0.0, "dense_mean": 0.5}],
    }
    (model_dir / "3101-biochemistry.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    (model_dir / "overview-direct.json").write_text(
        json.dumps({"summary": []}), encoding="utf-8",
    )
    other = dict(payload)
    other["cluster_id"] = "4012"
    (model_dir / "4012-fluids.json").write_text(
        json.dumps(other), encoding="utf-8",
    )
    loaded = mod.load_plot_jsons(tmp_path, ids=["3101"])
    assert list(loaded) == ["llama"]
    assert [d["cluster_id"] for d in loaded["llama"]] == ["3101"]


def test_item_win_rate_and_hardness_groups():
    mod = _load_metrics_plot()
    items = []
    for index, (kind, base, steered) in enumerate((
        ("direct", 0.40, 0.70),
        ("direct", 0.80, 0.80),
        ("direct", 0.55, 0.68),
        ("tricky", 0.30, 0.50),
    ), start=1):
        q = f"What is t{index}?"
        for alpha, dense, pair in (
            (0.0, base, "gold_vs_answer"),
            (0.0, 1.0, "steered_vs_baseline"),
            (0.2, steered, "gold_vs_answer"),
            (0.2, 0.6, "steered_vs_baseline"),
        ):
            items.append({
                "index": index, "kind": kind, "question": q, "term": "t",
                "alpha": alpha, "pair": pair, "dense": dense, "bm25": 0.1,
            })
    paired = mod.pair_scored_items(items)
    win = mod.win_rate(paired, "direct", 0.2, margin=0.01)
    assert win["n"] == 3
    assert win["wins"] == 2
    groups = mod.hardness_groups(paired, "direct", 0.2)
    assert [g["label"] for g in groups] == ["hard", "mid", "easy"]
    assert groups[0]["mean_lift"] > groups[2]["mean_lift"]
    assert mod.resolve_alpha(paired, 0.20) == 0.2
    assert mod.resolve_alpha(paired, 0.18) == 0.2


def _load_contrast_plot():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "extras" / "scripts" / "plot_self_gold_contrast.py"
    )
    spec = importlib.util.spec_from_file_location(
        "plot_self_gold_contrast", path,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sense_contrast_flips_from_generic_to_domain():
    mod = _load_contrast_plot()
    assert mod.sense_contrast(0.86, 0.48) == pytest.approx(0.38)
    assert mod.sense_contrast(0.32, 0.91) < 0
    items = []
    for alpha, correct, competing, hybrid_c, hybrid_g in (
        (0.0, 0.46, 1.0, 0.40, 1.0),
        (0.025, 0.32, 0.91, 0.269, 0.892),
        (0.20, 0.86, 0.48, 0.836, 0.402),
    ):
        q = "What is an event?"
        items.append({
            "index": 1, "kind": "direct", "question": q, "term": "event",
            "alpha": alpha, "pair": "gold_vs_answer",
            "dense": correct, "bm25": 0.0, "hybrid": hybrid_c,
        })
        items.append({
            "index": 1, "kind": "direct", "question": q, "term": "event",
            "alpha": alpha, "pair": "steered_vs_baseline",
            "dense": competing, "bm25": 0.0, "hybrid": hybrid_g,
        })
    rows = mod.contrast_items(items)
    by_a = {r["alpha"]: r for r in rows}
    assert by_a[0.025]["contrast_dense"] < 0
    assert by_a[0.2]["contrast_dense"] > 0
    assert by_a[0.2]["contrast_hybrid"] == pytest.approx(0.434)
    summary = mod.summarize_contrast(rows)
    assert mod.sense_transition(summary, "direct") == 0.2
    ssa20 = next(r for r in summary if r["alpha"] == 0.2)
    assert ssa20["ssa_dense"] == 1.0
    ssa0 = next(r for r in summary if r["alpha"] == 0.0)
    assert ssa0["ssa_dense"] == 0.0
    mapped = mod.as_similarity_summary(summary)
    gold20 = next(
        r for r in mapped
        if r["pair"] == "gold_vs_answer" and r["alpha"] == 0.2
    )
    assert gold20["hybrid_mean"] == pytest.approx(0.434)


def test_openai_backend_ignores_sentence_transformer_id():
    model, route = resolve_embed_route(
        "sentence-transformers/all-mpnet-base-v2", backend="openai",
    )
    assert route == "openai"
    assert model == DEFAULT_EMBED_MODEL


def test_openai_backend_keeps_text_embedding_id():
    model, route = resolve_embed_route(
        "openai/text-embedding-3-small", backend="openai",
    )
    assert route == "openai"
    assert model == "text-embedding-3-small"


def test_domain_plot_dir_is_per_domain(tmp_path):
    out = domain_plot_dir(tmp_path, "llama", "3101-biochemistry")
    assert out == tmp_path / "plots" / "llama" / "3101-biochemistry"


def test_domain_slug_handles_both_layouts():
    nested = Path("plots/llama/3101-biochemistry/similarity.json")
    flat = Path("plots/llama/3101-biochemistry.json")
    assert domain_slug_from_plot_path(nested) == "3101-biochemistry"
    assert domain_slug_from_plot_path(flat) == "3101-biochemistry"


def test_iter_scored_plot_json_finds_nested_and_flat(tmp_path):
    model = tmp_path / "llama"
    (model / "3101-biochemistry").mkdir(parents=True)
    nested = model / "3101-biochemistry" / "similarity.json"
    nested.write_text("{}", encoding="utf-8")
    flat = model / "4012-fluids.json"
    flat.write_text("{}", encoding="utf-8")
    (model / "overview-direct.json").write_text("{}", encoding="utf-8")
    found = sorted(p.name for p in iter_scored_plot_json_paths(tmp_path))
    assert found == ["4012-fluids.json", "similarity.json"]


def test_plot_json_identity_from_nested_path():
    path = Path("plots/llama/3101-biochemistry/similarity.json")
    model, cluster_id, cluster = plot_json_identity(path, {})
    assert model == "llama"
    assert cluster_id == "3101"
    assert cluster == "3101-biochemistry"


def test_assign_key_argmax_no_floor():
    assert assign_key(0.56, 0.54) == "llama"
    assert assign_key(0.40, 0.38) == "llama"
    assert assign_key(0.62, 0.68) == "gpt"
    assert assign_key(0.66, 0.66) == "tie"
    assert assign_key(None, 0.5) == "tie"


def test_assignment_rows_vote_per_alpha():
    scored = []
    for alpha, llama, gpt in (
        (0.0, 0.55, 0.52),
        (0.20, 0.72, 0.60),
        (0.30, 0.40, 0.39),
    ):
        scored.append({
            "index": 1, "kind": "direct", "term": "mask",
            "question": "What does a mask do?", "alpha": alpha,
            "pair": "contextual_vs_answer", "dense": llama, "bm25": 0.1,
            "hybrid": llama,
        })
        scored.append({
            "index": 1, "kind": "direct", "term": "mask",
            "question": "What does a mask do?", "alpha": alpha,
            "pair": "gold_vs_answer", "dense": gpt, "bm25": 0.1,
            "hybrid": gpt,
        })
    rows = assignment_rows(scored, field="dense")
    by_a = {r["alpha"]: r["assign"] for r in rows}
    assert by_a[0.0] == "llama"
    assert by_a[0.2] == "llama"
    assert by_a[0.3] == "llama"
    summary = assignment_summary(rows)
    at20 = next(s for s in summary if s["kind"] == "direct" and s["alpha"] == 0.2)
    assert at20["n_llama"] == 1
    assert at20["n_gpt"] == 0
    assert "n_neither" not in at20


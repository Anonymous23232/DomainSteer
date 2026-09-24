"""Tests for the isolated self-gold sweep — no models, no API."""

import json

import pytest

from domainsteer.pairs import ContrastivePair
from domainsteer.self_gold import (
    DEFAULT_CLUSTER_IDS, SELF_GOLD_CONTRAST_ID, alpha_grid,
    contrast_for_positive, expert_texts, gold_follows_baseline_sense,
    gold_is_refusal, gold_system_prompt, gold_user_message,
    items_with_model_gold, model_gold_pairs,
)


def test_default_ten_domains_are_unique():
    from domainsteer.clusters import load_clusters
    assert len(DEFAULT_CLUSTER_IDS) == 10
    assert len(set(DEFAULT_CLUSTER_IDS)) == 10
    assert "3001" in DEFAULT_CLUSTER_IDS
    known = {c.cluster_id for c in load_clusters()}
    assert set(DEFAULT_CLUSTER_IDS) <= known


def test_select_all_is_the_full_registry():
    from domainsteer.clusters import load_clusters
    runner = _load_runner()
    clusters = load_clusters()
    chosen = runner._select(clusters, None, all_clusters=True)
    assert len(chosen) == len(clusters) == 144
    assert {c.cluster_id for c in chosen} == {c.cluster_id for c in clusters}


def test_select_skip_ids_drops_a_default_domain():
    from domainsteer.clusters import load_clusters
    runner = _load_runner()
    clusters = load_clusters()
    chosen = runner._select(clusters, None, skip_ids=["3101"])
    ids = [c.cluster_id for c in chosen]
    assert "3101" not in ids
    assert "3001" in ids
    assert len(ids) == 9


def test_alpha_grid_always_runs_through_0_3():
    expected = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    assert alpha_grid(0.16) == expected
    assert alpha_grid(0.05) == expected
    assert alpha_grid(0.20) == expected
    assert alpha_grid(None) == expected
    assert alpha_grid(0.0) == expected


def test_gold_system_prompt_names_the_domain():
    prompt = gold_system_prompt("Agricultural biotechnology")
    assert "Agricultural biotechnology" in prompt
    assert "1-2 short sentences" in prompt
    assert "2-4 sentences" not in prompt
    assert "distractor" in prompt


def test_eval_system_prompt_is_not_gold_coaching():
    from domainsteer.self_gold import eval_system_prompt
    from domainsteer.steering import DEFAULT_SYSTEM_PROMPT

    prompt = eval_system_prompt("Agricultural biotechnology")
    assert prompt == (
        "You are a Agricultural biotechnology practitioner. "
        "Answer in 1-2 short sentences. Do not use lists."
    )
    assert "distractor" not in prompt.lower()
    assert "everyday" not in prompt.lower()
    assert "trap" not in prompt.lower()
    assert DEFAULT_SYSTEM_PROMPT.replace("helpful assistant",
                                         "Agricultural biotechnology practitioner") == prompt


def test_short_system_prompt_is_not_gold_coaching():
    from domainsteer.self_gold import short_system_prompt

    prompt = short_system_prompt("Agricultural biotechnology")
    assert prompt == (
        "You are a Agricultural biotechnology practitioner. "
        "Answer in one short sentence. Do not use lists."
    )
    assert "distractor" not in prompt.lower()
    assert "everyday" not in prompt.lower()
    assert gold_system_prompt("Agricultural biotechnology") != prompt


def test_expert_texts_gpt_uses_exam_answer():
    items = [
        {"question": "What is a stack?", "answer": "GPT trait stack"},
        {"question": "How do you stack two events?",
         "gpt_answer": "GPT insertion events"},
    ]
    golds = ["Llama DNA stack", "Llama combine events"]
    assert expert_texts(items, golds, "self") == golds
    assert expert_texts(items, golds, "gpt") == [
        "GPT trait stack", "GPT insertion events",
    ]


def test_contrast_for_positive_is_isolated():
    from domainsteer.pairs import CONTRAST_ID, TRAP_CONTRAST_ID, TRAP_LOCAL_CONTRAST_ID
    from domainsteer.self_gold import GPT_GOLD_CONTRAST_ID
    assert contrast_for_positive("self") == SELF_GOLD_CONTRAST_ID
    assert contrast_for_positive("gpt") == GPT_GOLD_CONTRAST_ID
    assert GPT_GOLD_CONTRAST_ID not in {
        SELF_GOLD_CONTRAST_ID, CONTRAST_ID, TRAP_CONTRAST_ID,
        TRAP_LOCAL_CONTRAST_ID,
    }


def test_model_gold_pairs_gpt_positive_swaps_expert():
    items = [
        {"kind": "direct", "question": "What is a stack?",
         "answer": "GPT trait stack"},
        {"kind": "tricky", "question": "How do you stack two events?",
         "answer": "GPT insertion events"},
    ]
    golds = ["Llama DNA stack", "Llama combine events"]
    baselines = ["LIFO data structure", "stack of plates"]
    experts = expert_texts(items, golds, "gpt")
    pairs = model_gold_pairs(items, experts, baselines, min_pairs=2)
    assert pairs[0].expert_text == "GPT trait stack"
    assert pairs[0].nonexpert_text == "LIFO data structure"
    assert pairs[1].expert_text == "GPT insertion events"


def test_items_with_model_gold_keep_gpt_answer():
    items = [{"question": "What is a stack?", "answer": "GPT trait stack",
              "kind": "direct"}]
    out = items_with_model_gold(items, ["model DNA stack"])
    assert out[0]["gpt_answer"] == "GPT trait stack"
    assert out[0]["answer"] == "model DNA stack"


def test_gold_user_message_flags_tricky_as_competing_canon():
    item = {
        "kind": "tricky",
        "term": "ensemble",
        "question": "If one member of the ensemble leaves, does the show stop?",
    }
    text = gold_user_message("Climate change science", item)
    assert "competing-canon" in text
    assert "ensemble" in text
    assert item["question"] in text
    retry = gold_user_message("Climate change science", item, retry=True)
    assert "everyday reading is wrong" in retry
    direct = gold_user_message(
        "Climate change science",
        {"kind": "direct", "term": "ensemble", "question": "What is an ensemble?"},
    )
    assert direct == "What is an ensemble?"


def test_gold_follows_baseline_sense_catches_tv_ensemble():
    gold = (
        "No, the ensemble does not stop, ensemble refers to a group of "
        "individuals working together to achieve a common goal, in this "
        "case, a TV show, so if one member leaves, the show can continue "
        "with the remaining members."
    )
    gpt = (
        "Removing one simulation does not halt the analysis, but it reduces "
        "the sample used to estimate ranges and probabilities."
    )
    baseline = (
        "No, a show typically doesn't stop just because one member of the "
        "ensemble leaves. The show can continue with adjustments to the cast."
    )
    assert gold_follows_baseline_sense(gold, gpt, baseline)
    domain_gold = (
        "Removing one ensemble member does not stop the analysis; the "
        "remaining climate-model simulations still estimate spread and "
        "robust projected responses."
    )
    assert not gold_follows_baseline_sense(domain_gold, gpt, baseline)
    assert gold_is_refusal("I'm ready to assist. What's the question?")


def test_model_gold_pairs_use_all_kinds():
    items = [
        {"kind": "direct", "question": "What is a stack?"},
        {"kind": "tricky", "question": "How do you stack two events?"},
    ]
    golds = ["DNA stack of traits", "combine two insertion events"]
    baselines = ["LIFO data structure", "stack of plates"]
    pairs = model_gold_pairs(items, golds, baselines, min_pairs=2)
    assert len(pairs) == 2
    assert all(isinstance(p, ContrastivePair) for p in pairs)
    assert pairs[0].expert_text == "DNA stack of traits"
    assert pairs[0].nonexpert_text == "LIFO data structure"
    assert pairs[1].concept == "How do you stack two events?"


def test_model_gold_pairs_need_enough():
    with pytest.raises(ValueError, match="at least"):
        model_gold_pairs(
            [{"question": "What is a stack?"}], ["gold"], ["base"],
            min_pairs=2,
        )


def test_self_gold_contrast_id_is_not_trap_local():
    from domainsteer.pairs import CONTRAST_ID, TRAP_CONTRAST_ID, TRAP_LOCAL_CONTRAST_ID
    assert SELF_GOLD_CONTRAST_ID not in {
        CONTRAST_ID, TRAP_CONTRAST_ID, TRAP_LOCAL_CONTRAST_ID,
    }


def _load_runner():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "run_self_gold_sweep", root / "extras" / "scripts" / "run_self_gold_sweep.py",
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def test_use_pairwise_requires_explicit_flag():
    runner = _load_runner()
    assert runner._use_pairwise(False, False) is False
    assert runner._use_pairwise(False, True) is False
    assert runner._use_pairwise(True, True) is False
    assert runner._use_pairwise(True, False) is True


def test_need_calibrate_skips_when_alpha_override_and_layer(tmp_path):
    runner = _load_runner()

    class Cached:
        layer = 12
        max_alpha = None

    class MissingLayer:
        layer = None
        max_alpha = None

    steering = tmp_path / "steering.json"
    assert runner._need_calibrate(False, steering, Cached(), 0.25) is False
    assert runner._need_calibrate(False, steering, Cached(), None) is False
    assert runner._need_calibrate(True, steering, Cached(), 0.25) is True
    assert runner._need_calibrate(False, steering, MissingLayer(), 0.25) is True
    steering.write_text("{}", encoding="utf-8")

    class Ready:
        layer = 16
        max_alpha = 0.16

    assert runner._need_calibrate(False, steering, Ready(), None) is False

    class DirectionsOnly:
        layer = None
        max_alpha = None

        def extracted_layers(self):
            return [10, 12, 14]

    assert runner._need_calibrate(False, steering, DirectionsOnly(), None) is False
    assert runner._need_calibrate(True, steering, DirectionsOnly(), None) is True


def test_apply_alpha_override_writes_ceiling(tmp_path):
    runner = _load_runner()
    cache = tmp_path / "steering_config.json"
    cache.write_text(json.dumps({"layer": 12, "max_alpha": None}), encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    class Steerer:
        layer = 12
        max_alpha = None
        config_path = cache

    steerer = Steerer()
    runner._apply_alpha_override(steerer, 0.25, out_dir)
    assert steerer.max_alpha == 0.25
    written = json.loads((out_dir / "steering.json").read_text(encoding="utf-8"))
    cached = json.loads(cache.read_text(encoding="utf-8"))
    assert written["max_alpha"] == 0.25
    assert cached["max_alpha"] == 0.25
    assert written["max_alpha_source"] == "cli_override"
    assert written["positive"] == "self"
    assert written["contrast"] == SELF_GOLD_CONTRAST_ID


def test_items_for_extraction_fills_gpt_from_gold_json():
    runner = _load_runner()
    items = [{"question": "What is a mask?", "kind": "direct"}]
    payload = {"items": [
        {"question": "What is a mask?", "gpt_answer": "GPT raster mask",
         "gold": "Llama mask", "baseline": "Halloween"},
    ]}
    filled = runner._items_for_extraction(items, payload)
    assert filled[0]["answer"] == "GPT raster mask"
    assert filled[0]["gpt_answer"] == "GPT raster mask"


def test_write_extraction_pairs_gpt_positive(tmp_path):
    runner = _load_runner()
    from domainsteer.self_gold import GPT_GOLD_CONTRAST_ID
    items = [
        {"question": "What is a mask?", "answer": "GPT raster mask"},
        {"question": "If the mask fails, what is still mapped?",
         "answer": "GPT remaining coverage"},
    ]
    golds = ["Llama GIS mask", "Llama leftover pixels"]
    baselines = ["Halloween mask", "a mask hides the face"]
    pairs = runner._write_extraction_pairs(
        tmp_path, items, golds, baselines, "gpt", "test/model", min_pairs=2,
    )
    assert pairs[0].expert_text == "GPT raster mask"
    assert pairs[0].nonexpert_text == "Halloween mask"
    meta = json.loads((tmp_path / "pairs.meta.json").read_text(encoding="utf-8"))
    assert meta["contrast"] == GPT_GOLD_CONTRAST_ID
    assert meta["positive"] == "gpt"


def test_seed_texts_from_self_gold_copies_only_exam(tmp_path):
    runner = _load_runner()

    class Cluster:
        cluster_id = "3704"
        cluster = "Geoinformatics"

    src = (tmp_path / "self"
           / "meta-llama-llama-3-2-3b-instruct"
           / "3704-geoinformatics")
    src.mkdir(parents=True)
    (src / "gold.json").write_text(json.dumps({"n": 2}), encoding="utf-8")
    (src / "questions.json").write_text(json.dumps({"n": 2}), encoding="utf-8")
    (src / "steering.json").write_text("{}", encoding="utf-8")
    dest = tmp_path / "gpt" / "3704-geoinformatics"
    runner._seed_texts_from_self_gold(
        dest, "meta-llama/Llama-3.2-3B-Instruct", Cluster(),
        tmp_path / "self",
    )
    assert (dest / "gold.json").exists()
    assert (dest / "questions.json").exists()
    assert not (dest / "steering.json").exists()


def test_default_results_dir_splits_on_positive():
    runner = _load_runner()
    self_dir = runner._default_results_dir("self")
    gpt_dir = runner._default_results_dir("gpt")
    assert self_dir.name == "registry-self-gold"
    assert gpt_dir.name == "registry-self-gold-gpt"
    assert self_dir != gpt_dir


def test_sweep_done_requires_every_alpha(tmp_path):
    runner = _load_runner()
    assert runner._sweep_done(tmp_path / "missing.jsonl", 2, [0.0, 0.1]) is False
    path = tmp_path / "responses.jsonl"
    path.write_text(json.dumps({
        "index": 1, "gold": "g", "baseline": "b",
        "steered": {"0.1000": "A complete steered sentence."},
    }) + "\n" + json.dumps({
        "index": 2, "gold": "g2", "baseline": "b2",
        "steered": {"0.1000": "Another complete steered sentence."},
    }) + "\n", encoding="utf-8")
    assert runner._sweep_done(path, 2, [0.0, 0.1]) is True
    assert runner._sweep_done(path, 2, [0.0, 0.1, 0.2]) is False
    clipped = tmp_path / "clipped.jsonl"
    clipped.write_text(json.dumps({
        "index": 1, "gold": "g", "baseline": "b",
        "steered": {"0.1000": "The process involves the removal of any"},
    }) + "\n", encoding="utf-8")
    assert runner._sweep_done(clipped, 1, [0.0, 0.1]) is False


def test_layer_sweep_done_requires_every_layer_and_alpha(tmp_path):
    runner = _load_runner()
    path = tmp_path / "layer_sweep.jsonl"
    assert runner._layer_sweep_done(path, 1, [10, 12], [0.0, 0.1]) is False
    path.write_text(json.dumps({
        "index": 1, "gold": "g", "baseline": "b",
        "steered": {
            "10": {"0.1000": "Layer ten answer."},
            "12": {"0.1000": "Layer twelve answer."},
        },
    }) + "\n", encoding="utf-8")
    assert runner._layer_sweep_done(path, 1, [10, 12], [0.0, 0.1]) is True
    assert runner._layer_sweep_done(path, 1, [10, 12, 14], [0.0, 0.1]) is False
    assert runner._layer_sweep_done(path, 1, [10, 12], [0.0, 0.1, 0.2]) is False
    flat = tmp_path / "flat.jsonl"
    flat.write_text(json.dumps({
        "index": 1, "gold": "g", "baseline": "b",
        "steered": {"0.1000": "s"},
    }) + "\n", encoding="utf-8")
    assert runner._layer_sweep_done(flat, 1, [10], [0.0, 0.1]) is False


def test_exam_matches_gold_requires_same_questions():
    runner = _load_runner()
    items = [{"question": "What is an event?"},
             {"question": "When the event is over, does anything remain?"}]
    gold = {"items": [{"question": "What is an event?"},
                      {"question": "When the event is over, does anything remain?"}]}
    assert runner._exam_matches_gold(items, gold) is True
    gold["items"][1]["question"] = (
        "How do junction PCR and flanking-sequence analysis work?"
    )
    assert runner._exam_matches_gold(items, gold) is False
    assert runner._exam_matches_gold(items, None) is False


def test_records_by_question_ignores_index():
    runner = _load_runner()
    assert runner._qnorm("  What is an event? ") == "What is an event?"
    rec = {"question": "What is an event?", "gold": "insert",
           "baseline": "concert"}
    assert runner._same_gold_baseline(rec, "insert", "concert") is True
    assert runner._same_gold_baseline(rec, "insert", "festival") is False


def test_load_exam_reuses_questions_json_without_api(tmp_path):
    runner = _load_runner()

    class Cluster:
        cluster_id = "3101"
        cluster = "Biochemistry and cell biology"
        concepts = []

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "questions.json").write_text(json.dumps({
        "items": [
            {"term": "translation", "kind": "direct",
             "question": "What is translation?", "answer": "a"},
            {"term": "translation", "kind": "tricky",
             "question": "What decodes mRNA?", "answer": "b"},
        ],
    }), encoding="utf-8")
    items = runner._load_exam(
        Cluster(), n=2, force=False, cache_dir=tmp_path / "cache",
        out_dir=out_dir, allow_api=False,
    )
    assert len(items) == 2
    assert items[0]["question"] == "What is translation?"


def test_load_exam_without_cache_does_not_call_api(tmp_path):
    runner = _load_runner()

    class Cluster:
        cluster_id = "3101"
        cluster = "Biochemistry and cell biology"
        concepts = []

    with pytest.raises(RuntimeError, match="No cached questions"):
        runner._load_exam(
            Cluster(), n=2, force=False, cache_dir=tmp_path / "cache",
            out_dir=tmp_path / "out", allow_api=False,
        )


def _load_short_sys():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "run_short_sys_gold", root / "extras" / "scripts" / "run_short_sys_gold.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_short_sys_writes_field_and_patches_responses(tmp_path):
    from domainsteer.self_gold import short_system_prompt

    script = _load_short_sys()

    class Cluster:
        cluster_id = "3001"
        cluster = "Agricultural biotechnology"

    out_dir = tmp_path / "3001-agricultural-biotechnology"
    out_dir.mkdir()
    gold = {
        "cluster_id": "3001",
        "items": [
            {"term": "event", "question": "What is an event?",
             "gold": "long coaching gold", "baseline": "a concert"},
        ],
    }
    (out_dir / "gold.json").write_text(json.dumps(gold), encoding="utf-8")
    (out_dir / "steering.json").write_text(json.dumps({"layer": 12}),
                                           encoding="utf-8")
    (out_dir / "responses.json").write_text(json.dumps([
        {"question": "What is an event?", "gold": "long coaching gold"},
    ]), encoding="utf-8")

    class FakeSteerer:
        layer = 12
        system_prompt = None
        _steering = None
        _direction = object()
        directions_dir = tmp_path

        def __init__(self, *args, **kwargs):
            pass

        def extracted_layers(self):
            return [12]

        def use_layer(self, layer):
            self.layer = int(layer)

        def generate(self, question, alpha=0.0, max_new_tokens=40):
            return "An event is a unique insertion outcome. Extra clause here."

    script.DomainSteerer = FakeSteerer
    payload = script.generate_domain(
        Cluster(), "meta-llama/Llama-3.1-8B-Instruct", out_dir, tmp_path,
        None, None, 40, force=False,
    )
    assert payload["items"][0]["short_sys"] == "An event is a unique insertion outcome."
    assert payload["short_system_prompt"] == short_system_prompt(
        "Agricultural biotechnology")
    assert payload["short_sys_layer"] == 12
    responses = json.loads((out_dir / "responses.json").read_text(encoding="utf-8"))
    assert responses[0]["short_sys"] == "An event is a unique insertion outcome."

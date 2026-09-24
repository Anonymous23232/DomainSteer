"""Tests for the polysemy-trap exam — no models, API mocked."""

import json

import pytest

import extras.trap as trap_module
from extras.llm_judge import Comparison
from extras.judge import degeneration_metrics, token_ngram_repeat_ratio
from extras.trap import (
    BenchmarkTrap,
    TRAP_ITEMS_PROMPT,
    TRAP_SENSE_PROMPT,
    _clean_trap_terms,
    exam_direct_probes,
    exam_direct_questions,
    flatten_trap_terms,
    format_trap_benchmark,
    gold_vs_baseline_pairs,
    grade_trap_run,
    n_trap_terms,
)

DIRECT_Q = "What is forcing?"
TRICKY_Q = "If you stop the forcing, does everything go back to normal?"
GOLD = (
    "Forcing is the external driver imposed on the catchment water balance: "
    "precipitation, potential evaporation, and lateral inflows that set the "
    "boundary fluxes a routing or storage model must respond to."
)


def _term(term="forcing", direct=None, tricky=None):
    return {
        "term": term,
        "direct_question": direct or f"What is {term}?",
        "direct_answer": GOLD,
        "tricky_question": tricky or (
            f"If you stop the {term}, does everything go back to normal?"
        ),
        "tricky_answer": GOLD,
    }


def test_n_trap_terms_is_half_the_question_count():
    assert n_trap_terms(50) == 25
    assert n_trap_terms(10) == 5
    assert n_trap_terms(1) == 1


def test_trap_prompts_include_competing_canon_examples():
    for prompt in (TRAP_ITEMS_PROMPT, TRAP_SENSE_PROMPT):
        assert "Who is Spider-Man's girlfriend, and did she die?" in prompt
        assert "How does the routing work?" in prompt
        assert "If the routing fails halfway, does anything still arrive?" in prompt
        assert "What is forcing?" in prompt
        assert "If you stop the forcing, does everything go back to normal?" in prompt
        assert "Did the fetch work?" in prompt
        assert "How far did the fetch have to go, and did anything come back?" in prompt
        assert "What is your target metric for calibration?" in prompt
        assert "two different settings look equally right" in prompt
        assert "disturbance moves through a channel" not in prompt
        assert "growable array of values" not in prompt
    assert "competing canon" in TRAP_ITEMS_PROMPT
    assert "contain the term itself" in TRAP_ITEMS_PROMPT
    assert "After the wake, does everyone go home?" in TRAP_ITEMS_PROMPT
    assert "After the wake, is anything still moving?" in TRAP_ITEMS_PROMPT
    assert "context-free conceptual" not in TRAP_ITEMS_PROMPT
    assert "factually correct" in TRAP_SENSE_PROMPT
    assert "invented terms" in TRAP_SENSE_PROMPT


def test_clean_trap_terms_keeps_short_what_is_questions():
    out = _clean_trap_terms([_term()], "Hydrology")
    assert len(out) == 1
    assert out[0]["direct_question"] == DIRECT_Q


def test_clean_trap_terms_drops_domain_name_and_this_field():
    leaked = _term()
    leaked["direct_question"] = "What is forcing in hydrology?"
    identity = _term(term="stage")
    identity["direct_question"] = "What does this field mean by stage?"
    identity["tricky_question"] = "How is stage related to discharge?"
    out = _clean_trap_terms([leaked, identity, _term()], "Hydrology")
    assert [row["term"] for row in out] == ["forcing"]


def test_clean_trap_terms_drops_tricky_that_omit_the_term():
    riddle = _term(
        term="translation",
        direct="What is translation?",
        tricky="What process decodes an mRNA sequence into a polypeptide?",
    )
    kept = _term(
        term="translation",
        direct="What is translation?",
        tricky="Who does the translation, and is the original still there?",
    )
    out = _clean_trap_terms([riddle, kept], "Biochemistry and cell biology")
    assert len(out) == 1
    assert "translation" in out[0]["tricky_question"].lower()


def test_flatten_trap_terms_makes_two_items():
    items = flatten_trap_terms([_term()])
    assert [item["kind"] for item in items] == ["direct", "tricky"]
    assert items[0]["question"] == DIRECT_Q
    assert items[1]["question"] == TRICKY_Q


def test_exam_direct_probes_drop_tricky():
    questions, golds = exam_direct_probes(flatten_trap_terms([_term()]))
    assert questions == [DIRECT_Q]
    assert golds == [GOLD]


def test_trap_generator_caches_under_domain_slug(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake(prompt, api_key=None, model=None):
        calls["n"] += 1
        assert "Hydrology" in prompt
        assert "baseflow" in prompt
        assert "What is forcing?" in prompt
        assert "If you stop the forcing, does everything go back to normal?" in prompt
        assert "Who is Spider-Man's girlfriend, and did she die?" in prompt
        assert "field collocations" not in prompt
        assert "domain-committed system prompt" in prompt
        return {"terms": [_term()]}

    monkeypatch.setattr(trap_module, "request_json_completion", fake)
    gen = BenchmarkTrap("Hydrology", cache_dir=tmp_path)
    items = gen.generate(n=2, concepts=["baseflow"])
    assert calls["n"] == 1
    assert len(items) == 2
    assert gen.path == tmp_path / "hydrology" / "benchmark_trap.json"
    assert gen.path.exists()
    payload = json.loads(gen.path.read_text(encoding="utf-8"))
    assert payload["contrast"] == "polysemy_trap_v6"
    assert len(payload["terms"]) == 1

    again = BenchmarkTrap("Hydrology", cache_dir=tmp_path).generate(n=2)
    assert calls["n"] == 1
    assert again == items


def test_stale_exam_contrast_regenerates(tmp_path, monkeypatch):
    gen = BenchmarkTrap("Hydrology", cache_dir=tmp_path)
    gen.path.parent.mkdir(parents=True, exist_ok=True)
    gen.path.write_text(json.dumps({
        "contrast": "polysemy_trap_v2",
        "terms": [_term()],
    }), encoding="utf-8")
    calls = {"n": 0}

    def fake(prompt, api_key=None, model=None):
        calls["n"] += 1
        assert "competing canon" in prompt
        return {"terms": [_term(term="stage")]}

    monkeypatch.setattr(trap_module, "request_json_completion", fake)
    items = gen.generate(n=2, concepts=["baseflow"])
    assert calls["n"] == 1
    assert items[0]["term"] == "stage"
    payload = json.loads(gen.path.read_text(encoding="utf-8"))
    assert payload["contrast"] == "polysemy_trap_v6"


def test_trap_generator_top_up_when_first_batch_is_short(tmp_path, monkeypatch):
    calls = []

    def fake(prompt, api_key=None, model=None):
        calls.append(prompt)
        i = len(calls)
        return {"terms": [_term(
            term=f"head{i}",
            direct=f"What is head {i}?",
            tricky=f"If you cut off head{i}, what is left?",
        )]}

    monkeypatch.setattr(trap_module, "request_json_completion", fake)
    items = BenchmarkTrap("Hydrology", cache_dir=tmp_path).generate(
        n=4, concepts=["stage"], batch_size=1)
    assert len(items) == 4
    assert len(calls) == 2
    assert "Already written" in calls[1]


def test_grade_trap_run_records_sense_judge(tmp_path, monkeypatch):
    items = flatten_trap_terms([_term()])

    class FakeReport:
        def __init__(self):
            self.comparisons = [
                Comparison("a", True, "A", "A", ("domain sense", "")),
                Comparison("tie", False, "A", "A", ("unclear", "")),
            ]

        def as_dict(self):
            return {"n": 2, "wins": 1, "losses": 0, "ties": 1,
                    "win_rate": 0.75, "consistency": 0.5}

    class FakeJudge:
        def __init__(self, *a, **k):
            pass

        def compare_all(self, questions, steered, baselines, golds=None):
            assert golds == [GOLD, GOLD]
            assert "forcing" in questions[0].lower()
            return FakeReport()

    monkeypatch.setattr("extras.llm_judge.PairwiseJudge", FakeJudge)
    monkeypatch.setattr(
        "extras.benchmark.score_against_gold",
        lambda *a, **k: [
            {"baseline_gold": 0.2, "steered_gold": 0.8, "steered_closer": True,
             "baseline_correct": False, "steered_correct": True},
            {"baseline_gold": 0.3, "steered_gold": 0.7, "steered_closer": True,
             "baseline_correct": False, "steered_correct": True},
        ],
    )

    results = grade_trap_run(
        "Hydrology", items,
        ["IT forcing function", "choose a routing package"],
        ["catchment water-balance driver", "Muskingum storage routing"],
        alpha=0.15, cache_dir=tmp_path, similarity=False, llm_judge=True,
    )
    assert results["set"] == "trap"
    assert results["n"] == 2
    assert results["pairwise"]["wins"] == 1
    assert results["items"][0]["judge"]["winner"] == "wins"
    assert results["gold_metrics"]["steered_mean"] == pytest.approx(0.75)
    assert results["by_kind"]["direct"]["n"] == 1
    assert results["by_kind"]["tricky"]["n"] == 1
    assert results["items"][0]["degeneration"]["steered"]["n_words"] > 0
    report = format_trap_benchmark(results)
    assert "GOLD:" in report and DIRECT_Q in report
    summary = format_trap_benchmark(results, include_items=False)
    assert "GOLD:" not in summary
    assert "ambiguous terms" in summary
    assert "technical reasoning" in summary
    json.dumps(results)


def test_grade_trap_run_skips_llm_judge_by_default(tmp_path, monkeypatch):
    items = flatten_trap_terms([_term()])

    def boom(*a, **k):
        raise AssertionError("LLM judge should not run")

    monkeypatch.setattr("extras.llm_judge.PairwiseJudge", boom)
    monkeypatch.setattr(
        "extras.benchmark.score_against_gold",
        lambda *a, **k: [
            {"baseline_gold": 0.2, "steered_gold": 0.8, "steered_closer": True,
             "baseline_correct": False, "steered_correct": True},
            {"baseline_gold": 0.3, "steered_gold": 0.7, "steered_closer": True,
             "baseline_correct": False, "steered_correct": True},
        ],
    )
    results = grade_trap_run(
        "Hydrology", items, ["everyday", "routing package"],
        ["catchment driver", "Muskingum routing"],
        alpha=0.15, cache_dir=tmp_path, similarity=False,
    )
    assert results["pairwise"] == {}
    assert "judge" not in results["items"][0]


def test_token_ngram_repeat_ratio_flags_loops():
    loop = "T35S T35S T35S T35S T35S T35S".split()
    clean = "A construct is a DNA sequence designed for transformation.".split()
    assert token_ngram_repeat_ratio(loop) > 0.5
    assert token_ngram_repeat_ratio(clean) == 0.0
    assert degeneration_metrics(" ".join(loop))["collapsed"] is True
    assert degeneration_metrics(" ".join(clean))["collapsed"] is False


def test_exam_direct_questions_drops_tricky():
    items = flatten_trap_terms([_term(), _term(
        term="stage",
        direct="What is stage?",
        tricky="How frequently do you update the curve when the stage shifts?",
    )])
    assert exam_direct_questions(items) == [DIRECT_Q, "What is stage?"]


def test_gold_vs_baseline_pairs_use_directs_only():
    terms = [
        _term(term=f"t{i}", direct=f"What is t{i}?",
              tricky=f"How does t{i} segregate under hygromycin?")
        for i in range(10)
    ]
    items = flatten_trap_terms(terms)
    seen = []

    def generate(question):
        seen.append(question)
        return f"everyday reading of {question}"

    pairs = gold_vs_baseline_pairs(items, generate)
    assert len(pairs) == 10
    assert {p.concept for p in pairs} == {f"What is t{i}?" for i in range(10)}
    assert all(p.expert_text == GOLD for p in pairs)
    assert all(p.nonexpert_text.startswith("everyday reading") for p in pairs)
    assert all("hygromycin" not in q for q in seen)


def test_gold_vs_baseline_pairs_need_enough_directs():
    with pytest.raises(ValueError, match="at least"):
        gold_vs_baseline_pairs(flatten_trap_terms([_term()]), lambda q: "x")


def test_pair_stems_top_up_directs_to_n(tmp_path, monkeypatch):
    calls = []

    def fake(prompt, api_key=None, model=None):
        calls.append(prompt)
        if '"terms"' in prompt or "tricky_question" in prompt:
            return {"terms": [_term()]}
        return {"stems": [
            {"term": "fetch", "question": "Did the fetch work?"},
        ]}

    monkeypatch.setattr(trap_module, "request_json_completion", fake)
    gen = BenchmarkTrap("Hydrology", cache_dir=tmp_path)
    items = gen.generate(n=2, concepts=["baseflow"])
    assert exam_direct_questions(items) == [DIRECT_Q]
    stems = gen.pair_stems(n=2, exam_items=items, concepts=["baseflow"])
    assert stems == [DIRECT_Q, "Did the fetch work?"]
    assert TRICKY_Q not in stems
    assert gen.pair_stems_path.exists()
    assert "Did the fetch work?" in calls[-1]
    again = gen.pair_stems(n=2, exam_items=items)
    assert again == stems
    assert sum('"stems"' in c or "No conceptual" in c for c in calls) == 1

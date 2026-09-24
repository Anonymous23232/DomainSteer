"""Tests for the shared unnamed-question benchmark — no models, API mocked."""

import json

import numpy as np
import pytest

import extras.benchmark as bench
from extras.benchmark import (BenchmarkGold, BenchmarkQuestions,
                                   _clean_gold_items, _clean_questions,
                                   _parse_rubric, cosine,
                                   format_benchmark, format_gold_benchmark,
                                   grade_gold_run, grade_run,
                                   resolve_item_concepts, score_against_gold,
                                   score_gold_similarity)
from domainsteer.pairs import ContrastivePair, save_pairs


def test_clean_questions_drops_short_and_duplicates():
    raw = [
        "What do outsiders most often get wrong about this field?",
        "What do outsiders most often get wrong about this field?",
        "too short",
        "No question mark here at all in this reasonably long sentence.",
        42,
        "How is a typical problem framed before anyone collects data?",
    ]
    out = _clean_questions(raw)
    assert len(out) == 2
    assert out[0].startswith("What do outsiders")
    assert out[1].startswith("How is a typical")


def test_question_generator_caches(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake(prompt, api_key=None, model=None):
        calls["n"] += 1
        n = 5
        return {"questions": [
            f"What does a practitioner check first when diagnosing a problem {i}?"
            for i in range(n)
        ]}

    monkeypatch.setattr(bench, "request_json_completion", fake)
    gen = BenchmarkQuestions(cache_dir=tmp_path)
    questions = gen.generate(n=5)
    assert calls["n"] == 1
    assert len(questions) == 5
    assert gen.path.exists()

    again = BenchmarkQuestions(cache_dir=tmp_path).generate(n=5)
    assert calls["n"] == 1
    assert again == questions


def test_question_generator_top_up_when_first_batch_is_short(tmp_path, monkeypatch):
    calls = []

    def fake(prompt, api_key=None, model=None):
        calls.append(prompt)
        start = len(calls) * 2
        return {"questions": [
            f"What is treated as background that other fields treat as the question {start}?",
            f"When two explanations compete, what actually decides between them {start + 1}?",
        ]}

    monkeypatch.setattr(bench, "request_json_completion", fake)
    questions = BenchmarkQuestions(cache_dir=tmp_path).generate(n=4)
    assert len(questions) == 4
    assert len(calls) == 2


def test_cosine_is_one_for_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, v) == pytest.approx(1.0)
    assert cosine(v, np.zeros(3)) == 0.0


def test_grade_run_records_judge_and_skips_empty_similarity(tmp_path, monkeypatch):
    def fake(prompt, api_key=None, model=None):
        first = prompt.split("Answer A:\n")[1].split("\n\nAnswer B:")[0].strip()
        return {"winner": "A" if "water" in first else "B",
                "reason": "committed to the domain"}

    monkeypatch.setattr("extras.llm_judge.request_json_completion", fake)
    monkeypatch.setattr(bench, "concept_gaps", lambda *a, **k: [])

    results = grade_run(
        "hydrology",
        ["What do outsiders most often get wrong about this field?"],
        ["Art, science, and coding."],
        ["Outsiders treat water as uniformly mobile."],
        alpha=0.16,
        cache_dir=tmp_path,
        similarity=False,
    )
    assert results["n"] == 1
    assert results["pairwise"]["wins"] == 1
    assert results["items"][0]["judge"]["winner"] == "wins"
    report = format_benchmark(results)
    assert "BASELINE:" in report and "STEERED:" in report
    json.dumps(results)
    summary = format_benchmark(results, include_items=False)
    assert "BASELINE:" not in summary


def test_grade_run_can_skip_the_llm_judge(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "concept_gaps", lambda *a, **k: [])
    results = grade_run(
        "hydrology",
        ["What do outsiders most often get wrong about this field?"],
        ["Art, science, and coding."],
        ["Outsiders treat water as uniformly mobile."],
        alpha=0.16,
        cache_dir=tmp_path,
        similarity=False,
        llm_judge=False,
    )
    assert results["set"] == "unnamed"
    assert results["pairwise"] == {}
    assert "judge" not in results["items"][0]


def test_grade_run_similarity_against_pair_clouds(tmp_path, monkeypatch):
    save_pairs(
        [ContrastivePair("aquifer recharge and hydraulic head",
                         "art science coding business")],
        tmp_path / "hydrology" / "pairs.jsonl",
    )
    monkeypatch.setattr(
        "extras.llm_judge.request_json_completion",
        lambda prompt, api_key=None, model=None: {"winner": "A", "reason": "ok"},
    )
    monkeypatch.setattr(bench, "concept_gaps", lambda *a, **k: [])

    def fake_embed(texts, api_key=None, model=None):
        vecs = []
        for text in texts:
            if "aquifer" in text or "hydraulic" in text or "water" in text:
                vecs.append(np.array([1.0, 0.0], dtype=np.float32))
            else:
                vecs.append(np.array([0.0, 1.0], dtype=np.float32))
        return vecs

    monkeypatch.setattr(bench, "embed_texts", fake_embed)
    results = grade_run(
        "hydrology",
        ["What do outsiders most often get wrong about this field?"],
        ["art science coding"],
        ["Outsiders treat water as uniformly mobile."],
        alpha=0.16,
        cache_dir=tmp_path,
        similarity=True,
    )
    sim = results["items"][0]["similarity"]
    assert sim["steered_in_domain"] > sim["baseline_in_domain"]
    assert sim["baseline_survey"] > sim["steered_survey"]


GOLD_QUESTION = (
    "When two hypotheses for a display are equally consistent with the "
    "observed sequence, what evidence decides between them?"
)
GOLD_ANSWER = (
    "A zoologist uses audience, context, and fitness consequence, not the "
    "sequence alone: whether the display occurs without predators, who the "
    "receiver is, and whether it covaries with mating success or survival."
)


def test_clean_gold_items_drops_short_answers_and_duplicates():
    raw = [
        {"question": GOLD_QUESTION, "answer": GOLD_ANSWER,
         "concepts": ["Animal Behaviour"]},
        {"question": GOLD_QUESTION, "answer": GOLD_ANSWER},
        {"question": "Too short?", "answer": GOLD_ANSWER},
        {"question": GOLD_QUESTION.replace("them?", "them now?"),
         "answer": "too short"},
        "not a dict",
    ]
    out = _clean_gold_items(raw)
    assert len(out) == 1
    assert out[0]["question"] == GOLD_QUESTION
    assert out[0]["concepts"] == ["Animal Behaviour"]


def test_gold_generator_caches_under_domain_slug(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake(prompt, api_key=None, model=None):
        calls["n"] += 1
        assert "Zoology" in prompt
        assert "Animal Behaviour" in prompt
        return {"items": [{
            "question": GOLD_QUESTION,
            "answer": GOLD_ANSWER,
            "concepts": ["Animal Behaviour"],
        }]}

    monkeypatch.setattr(bench, "request_json_completion", fake)
    gen = BenchmarkGold("Zoology", cache_dir=tmp_path)
    items = gen.generate(n=1, concepts=["Animal Behaviour"])
    assert calls["n"] == 1
    assert items[0]["answer"] == GOLD_ANSWER
    assert gen.path == tmp_path / "zoology" / "benchmark_gold.json"
    assert gen.path.exists()

    again = BenchmarkGold("Zoology", cache_dir=tmp_path).generate(n=1)
    assert calls["n"] == 1
    assert again == items


def test_gold_generator_top_up_when_first_batch_is_short(tmp_path, monkeypatch):
    calls = []

    def fake(prompt, api_key=None, model=None):
        calls.append(prompt)
        i = len(calls)
        return {"items": [{
            "question": (
                f"What evidence would change an expert's mind about "
                f"display function in this population {i}?"
            ),
            "answer": GOLD_ANSWER,
            "concepts": ["Animal Behaviour"],
        }]}

    monkeypatch.setattr(bench, "request_json_completion", fake)
    items = BenchmarkGold("Zoology", cache_dir=tmp_path).generate(
        n=2, concepts=["Animal Behaviour"], batch_size=1)
    assert len(items) == 2
    assert len(calls) == 2
    assert "Already written" in calls[1]


def test_score_against_gold_prefers_matching_answer(monkeypatch):
    def fake_embed(texts, model=None):
        vecs = []
        for text in texts:
            if "anatomy" in text or "gold" in text:
                vecs.append(np.array([1.0, 0.0], dtype=np.float32))
            else:
                vecs.append(np.array([0.0, 1.0], dtype=np.float32))
        return vecs

    monkeypatch.setattr(bench, "embed_sentence_transformer", fake_embed)
    rows = score_against_gold(
        ["art science coding"],
        ["species anatomy behaviour"],
        ["gold anatomy of the animal"],
        tau=0.5,
    )
    assert rows[0]["steered_gold"] > rows[0]["baseline_gold"]
    assert rows[0]["steered_correct"] is True
    assert rows[0]["baseline_correct"] is False
    assert rows[0]["steered_closer"] is True


def test_resolve_item_concepts_uses_gold_and_named_terms():
    item = {
        "question": GOLD_QUESTION,
        "answer": GOLD_ANSWER + " Animal Behaviour is the relevant system.",
        "concepts": ["Life history"],
    }
    resolved = resolve_item_concepts(
        item, ["Animal Behaviour", "Darcy's law", "ab"])
    assert "Life history" in resolved
    assert "Animal Behaviour" in resolved
    assert "Darcy's law" not in resolved
    assert "ab" not in resolved


def test_score_gold_similarity_separates_gold_and_concepts(monkeypatch):
    def fake_embed(texts, model=None):
        vecs = []
        for text in texts:
            if "anatomy" in text or "gold" in text or "Animal Behaviour" in text:
                vecs.append(np.array([1.0, 0.0], dtype=np.float32))
            else:
                vecs.append(np.array([0.0, 1.0], dtype=np.float32))
        return vecs

    monkeypatch.setattr(bench, "embed_sentence_transformer", fake_embed)
    rows = score_gold_similarity(
        ["art science coding"],
        ["species anatomy behaviour Animal Behaviour"],
        ["gold anatomy of the animal"],
        [["Animal Behaviour"]],
        tau=0.5, concept_tau=0.4,
    )
    row = rows[0]
    assert row["steered_gold"] > row["baseline_gold"]
    assert row["steered_concepts"] > row["baseline_concepts"]
    assert row["steered_coverage"] == 1.0
    assert row["baseline_coverage"] == 0.0


def test_parse_rubric_clamps_and_fills_overall():
    parsed = _parse_rubric({
        "technical_accuracy": 6, "domain_expertise": -1,
        "analytical_approach": 3, "explanation": "partial",
    })
    assert parsed["technical_accuracy"] == 5.0
    assert parsed["domain_expertise"] == 0.0
    assert parsed["overall"] == pytest.approx(2.7)
    assert parsed["explanation"] == "partial"


def test_expert_rubric_is_not_reference_matching():
    assert "Reference" not in bench.EXPERT_RUBRIC_PROMPT
    assert "{gold}" not in bench.EXPERT_RUBRIC_PROMPT
    assert "1 = mostly incorrect" in bench.EXPERT_RUBRIC_PROMPT
    assert "Alternative valid" in bench.EXPERT_RUBRIC_PROMPT
    assert "Reference:" not in bench.GOLD_PAIRWISE_PROMPT
    assert "{gold}" not in bench.GOLD_PAIRWISE_PROMPT


def test_grade_gold_run_records_accuracy(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bench, "embed_sentence_transformer",
        lambda texts, model=None: [
            np.array([0.0, 1.0] if "art" in t else [1.0, 0.0], dtype=np.float32)
            for t in texts
        ],
    )
    results = grade_gold_run(
        "Zoology",
        [{"question": GOLD_QUESTION, "answer": GOLD_ANSWER,
          "concepts": ["Animal Behaviour"]}],
        ["art science coding tour"],
        ["species anatomy and life stage of the animal"],
        alpha=0.12,
        tau=0.5,
        cache_dir=tmp_path,
        llm_judge=False,
    )
    metrics = results["gold_metrics"]
    assert metrics["steered_accuracy"] == 1.0
    assert metrics["baseline_accuracy"] == 0.0
    assert metrics["steered_closer"] == 1
    assert results["items"][0]["gold"] == GOLD_ANSWER
    report = format_gold_benchmark(results)
    assert "GOLD:" in report
    assert "MPNet similarity (gold)" in report
    assert "Concept coverage" in report
    json.dumps(results)
    summary = format_gold_benchmark(results, include_items=False)
    assert "GOLD:" not in summary
    assert "LLM expert score" not in summary


def test_grade_gold_run_hybrid_reports_metrics_separately(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bench, "embed_sentence_transformer",
        lambda texts, model=None: [
            np.array([0.0, 1.0] if "art" in t else [1.0, 0.0], dtype=np.float32)
            for t in texts
        ],
    )

    def fake_llm(prompt, api_key=None, model=None):
        if "technical_accuracy" in prompt:
            assert "Gold answer:" not in prompt
            candidate = prompt.split("Candidate answer:")[-1]
            high = "anatomy" in candidate
            score = 4 if high else 1
            return {
                "technical_accuracy": score, "domain_expertise": score,
                "analytical_approach": score, "overall": score,
                "explanation": "ok",
            }
        first = prompt.split("Answer A:\n")[1].split("\n\nAnswer B:")[0]
        winner = "A" if "anatomy" in first else "B"
        if "Gold answer:" in prompt:
            return {"winner": winner, "reason": "closer to gold"}
        assert "Gold answer:" not in prompt
        return {"winner": winner, "reason": "expert"}

    monkeypatch.setattr(bench, "request_json_completion", fake_llm)
    monkeypatch.setattr("extras.llm_judge.request_json_completion", fake_llm)

    results = grade_gold_run(
        "Zoology",
        [{"question": GOLD_QUESTION, "answer": GOLD_ANSWER,
          "concepts": ["Animal Behaviour"]}],
        ["art science coding tour"],
        ["species anatomy and life stage of the animal"],
        alpha=0.12,
        cache_dir=tmp_path,
        llm_judge=True,
    )
    assert results["expert_mean"]["steered_overall"] > results["expert_mean"]["baseline_overall"]
    assert results["pairwise"]["wins"] == 1
    assert results["pairwise"]["loss_rate"] == 0.0
    assert results["open_gold"]["pairwise"]["wins"] == 1
    assert results["open_gold"]["weights"] == {"embedding": 0.6, "llm": 0.4}
    assert results["items"][0]["open_gold"]["winner"] == "wins"
    assert results["open_gold"]["steered"] > results["open_gold"]["baseline"]
    summary = format_gold_benchmark(results, include_items=False)
    assert "LLM expert score (/5)" in summary
    assert "Pairwise win rate" in summary
    assert "Open gold closer-to-key" in summary
    assert "Open gold blend" in summary
    assert "Optional blend" in summary
    assert "MPNet similarity (concepts)" in summary


def test_open_gold_score_is_sixty_forty():
    assert bench._open_gold_score(1.0, 0.0) == pytest.approx(0.6)
    assert bench._open_gold_score(0.0, 1.0) == pytest.approx(0.4)
    assert bench._open_gold_score(0.5, 0.5) == pytest.approx(0.5)


def test_apply_open_gold_puts_gold_in_the_prompt(monkeypatch):
    prompts = []

    def fake(prompt, api_key=None, model=None):
        prompts.append(prompt)
        first = prompt.split("Answer A:\n")[1].split("\n\nAnswer B:")[0]
        return {"winner": "A" if "steered" in first else "B", "reason": "match"}

    monkeypatch.setattr("extras.llm_judge.request_json_completion", fake)
    results = {
        "domain": "Zoology",
        "n": 1,
        "items": [{
            "question": GOLD_QUESTION,
            "gold": GOLD_ANSWER,
            "baseline": "baseline text here",
            "steered": "steered text here",
            "gold_score": {"baseline_gold": 0.2, "steered_gold": 0.8},
        }],
    }
    out = bench.apply_open_gold(results)
    assert prompts
    assert all("Gold answer:" in prompt for prompt in prompts)
    assert all(GOLD_ANSWER in prompt for prompt in prompts)
    assert all("Domain: Zoology" in prompt for prompt in prompts)
    assert out["open_gold"]["pairwise"]["wins"] == 1
    assert out["items"][0]["open_gold"]["score"]["steered"] == pytest.approx(
        0.6 * 0.8 + 0.4 * 1.0)
    assert out["items"][0]["open_gold"]["score"]["baseline"] == pytest.approx(
        0.6 * 0.2 + 0.4 * 0.0)

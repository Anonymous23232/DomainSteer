"""Tests for the pairwise LLM judge — no API calls, generation is mocked."""

import pytest

import extras.llm_judge as judge_module
from extras.llm_judge import PairwiseJudge

QUESTIONS = ["What controls recharge?", "What is baseflow?"]


def _answer_shown_first(prompt: str) -> str:
    """Recover the text the prompt placed in the 'Answer A' slot."""
    return prompt.split("Answer A:\n")[1].split("\n\nAnswer B:")[0].strip()


def fake_api(verdict_for):
    """Build a request_json_completion stand-in.

    `verdict_for(answer_shown_first) -> "A" | "B" | "tie"` decides each call,
    so tests can express position bias or genuine preference.
    """
    calls = []

    def fake(prompt, api_key=None, model=None):
        first = _answer_shown_first(prompt)
        calls.append(first)
        return {"winner": verdict_for(first), "reason": "because"}

    fake.calls = calls
    return fake


@pytest.fixture
def judge():
    return PairwiseJudge("hydrology")


# ------------------------------------------------------- debiasing behaviour

def test_consistent_preference_is_counted(judge, monkeypatch):
    """Judge genuinely prefers the expert text regardless of position."""
    monkeypatch.setattr(judge_module, "request_json_completion",
                        fake_api(lambda first: "A" if first == "expert" else "B"))

    result = judge.compare("q", "expert", "lay")
    assert result.winner == "a"
    assert result.consistent is True
    assert judge.calls == 2          # both orderings were asked


def test_position_bias_collapses_to_tie(judge, monkeypatch):
    """A judge that always picks whatever is shown first must not produce a
    winner — that is pure position bias, not a preference."""
    monkeypatch.setattr(judge_module, "request_json_completion",
                        fake_api(lambda first: "A"))

    result = judge.compare("q", "expert", "lay")
    assert result.winner == "tie"
    assert result.consistent is False


def test_b_wins_when_consistent(judge, monkeypatch):
    monkeypatch.setattr(judge_module, "request_json_completion",
                        fake_api(lambda first: "A" if first == "lay" else "B"))
    assert judge.compare("q", "expert", "lay").winner == "b"


def test_explicit_tie_is_a_tie(judge, monkeypatch):
    monkeypatch.setattr(judge_module, "request_json_completion",
                        fake_api(lambda first: "tie"))
    result = judge.compare("q", "expert", "lay")
    assert result.winner == "tie" and result.consistent is True


def test_unparseable_verdict_becomes_tie(judge, monkeypatch):
    monkeypatch.setattr(judge_module, "request_json_completion",
                        lambda prompt, api_key=None, model=None: {"winner": "???"})
    assert judge.compare("q", "a", "b").winner == "tie"


# -------------------------------------------------------------- aggregation

def test_compare_all_reports_win_rate_and_consistency(judge, monkeypatch):
    monkeypatch.setattr(judge_module, "request_json_completion",
                        fake_api(lambda first: "A" if first.startswith("expert") else "B"))

    report = judge.compare_all(QUESTIONS, ["expert1", "expert2"], ["lay1", "lay2"])
    assert (report.wins, report.losses, report.ties) == (2, 0, 0)
    assert report.win_rate == 1.0
    assert report.consistency == 1.0
    assert report.as_dict()["n"] == 2


def test_ties_count_as_half(judge, monkeypatch):
    monkeypatch.setattr(judge_module, "request_json_completion",
                        fake_api(lambda first: "A"))       # all position bias
    report = judge.compare_all(QUESTIONS, ["x1", "x2"], ["y1", "y2"])
    assert report.ties == 2
    assert report.win_rate == 0.5          # no detectable difference
    assert report.consistency == 0.0


def test_length_mismatch_raises(judge):
    with pytest.raises(ValueError, match="mismatch"):
        judge.compare_all(QUESTIONS, ["only one"], ["a", "b"])


def test_scores_vs_baseline_maps_to_numbers(judge, monkeypatch):
    monkeypatch.setattr(judge_module, "request_json_completion",
                        fake_api(lambda first: "A" if first.startswith("good") else "B"))
    scores = judge.scores_vs_baseline(QUESTIONS, ["good1", "good2"], ["bad1", "bad2"])
    assert scores == [1.0, 1.0]


def test_identical_comparisons_are_cached(judge, monkeypatch):
    fake = fake_api(lambda first: "A" if first == "expert" else "B")
    monkeypatch.setattr(judge_module, "request_json_completion", fake)

    judge.compare("q", "expert", "lay")
    judge.compare("q", "expert", "lay")      # same inputs -> served from cache
    assert judge.calls == 2                  # not 4


# ------------------------------------------------- judge-aware calibration
#
# These reach extras.calibrate, which imports torch via steering.

def _require_torch():
    pytest.importorskip("torch", exc_type=ImportError)


def test_probe_alphas_derive_from_ceiling():
    _require_torch()
    from extras.calibrate import probe_alphas_from_ceiling

    assert probe_alphas_from_ceiling(0.06) == [0.03, 0.06]
    assert probe_alphas_from_ceiling(0.2, fractions=(0.25, 0.5, 1.0)) == [0.05, 0.1, 0.2]
    assert probe_alphas_from_ceiling(0.01) == [0.005, 0.01]


class FakeSteering:
    """Coherent up to `breaks_above`; gibberish beyond it."""

    def __init__(self, breaks_above):
        self.breaks_above = breaks_above
        self.model = self.tokenizer = self.device = None

    def generate(self, prompt, alpha=0.0, max_new_tokens=150):
        if alpha > self.breaks_above:
            return "big " * 100
        return f"answer to {prompt} at alpha {alpha}"


def test_calibrate_alpha_pairwise_reports_ceiling_and_preference(monkeypatch):
    _require_torch()
    import extras.calibrate as calibrate_module
    from extras.calibrate import calibrate_alpha_pairwise

    monkeypatch.setattr(calibrate_module, "spike_perplexity",
                        lambda m, t, text, d: 500.0 if "big big" in text else 2.0)

    judge = PairwiseJudge("hydrology")
    # judge prefers alpha 0.04; 0.06 is coherent but less preferred
    monkeypatch.setattr(
        judge_module, "request_json_completion",
        fake_api(lambda first: "A" if "alpha 0.04" in first else "B"),
    )

    max_alpha, best_alpha, stats = calibrate_alpha_pairwise(
        FakeSteering(breaks_above=0.06), judge, ["q1"],
        candidate_alphas=[0.02, 0.04, 0.06, 0.09],
    )
    assert max_alpha == 0.06            # coherence ceiling, unchanged meaning
    assert best_alpha == 0.04           # what the judge actually preferred
    assert 0.09 not in stats            # degraded, never judged


def test_calibrate_alpha_pairwise_all_degraded(monkeypatch):
    _require_torch()
    import extras.calibrate as calibrate_module
    from extras.calibrate import calibrate_alpha_pairwise

    monkeypatch.setattr(calibrate_module, "spike_perplexity",
                        lambda m, t, text, d: 500.0)
    judge = PairwiseJudge("hydrology")
    monkeypatch.setattr(judge_module, "request_json_completion",
                        fake_api(lambda first: "tie"))

    max_alpha, best_alpha, stats = calibrate_alpha_pairwise(
        FakeSteering(breaks_above=0.0), judge, ["q1"], candidate_alphas=[0.02]
    )
    assert (max_alpha, best_alpha, stats) == (None, None, {})


def test_select_best_layer_pairwise_forwards_golds(monkeypatch):
    _require_torch()
    import numpy as np
    from extras.calibrate import select_best_layer_pairwise
    from domainsteer.extract import ExtractionResult
    from extras.llm_judge import PairwiseReport

    seen = {}

    class Judge:
        calls = 0

        def compare_all(self, questions, answers_a, answers_b, golds=None):
            seen["golds"] = golds
            seen["n"] = len(questions)
            return PairwiseReport(n=1, wins=1, losses=0, ties=0,
                                  consistency=1.0)

    class FakeAS:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt, alpha=0.0, max_new_tokens=150):
            return f"{prompt} a={alpha}"

    monkeypatch.setattr("extras.calibrate.ActivationSteering", FakeAS)
    extraction = ExtractionResult(
        model_name="x",
        directions={10: np.ones(4), 12: np.ones(4)},
        holdout_accuracy={10: 1.0, 12: 1.0},
        n_train_pairs=8, n_holdout_pairs=2,
    )
    best, stats = select_best_layer_pairwise(
        None, None, extraction, Judge(), ["What is a stack?"],
        probe_alphas=(0.08,), golds=["trait stack"],
    )
    assert seen["golds"] == ["trait stack"]
    assert best in (10, 12)
    assert set(stats) == {10, 12}

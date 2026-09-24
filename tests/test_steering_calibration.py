"""Tests for steering math, judging, and calibration — no model downloads."""

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

import math

from extras.calibrate import calibrate_max_alpha, default_test_prompts
from extras.judge import (ExpertiseJudge, degeneration_metrics,
                               repetition_rate, token_ngram_repeat_ratio,
                               worst_window_perplexity)
from domainsteer.steering import (DEFAULT_MAX_NEW_TOKENS, DEFAULT_SYSTEM_PROMPT,
                                  ActivationSteering, SteerWindow,
                                  answer_complete, steered_hidden,
                                  trim_hanging_clause)

GIBBERISH = "big " * 200
NORMAL = ("Aquifer recharge depends on soil permeability, antecedent moisture, "
          "and rainfall intensity, which together control infiltration rates.")


class _FakeInner(torch.nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Identity() for _ in range(n_layers)
        )
        self.embed_tokens = torch.nn.Embedding(4, hidden_dim)


class _FakeModel(torch.nn.Module):
    """Minimal Llama-shaped stand-in: model.model.layers plus a parameter."""

    def __init__(self, n_layers: int = 4, hidden_dim: int = 8):
        super().__init__()
        self.model = _FakeInner(n_layers, hidden_dim)


class _FakeTokenizer:
    chat_template = None


def _steering(hidden_dim: int = 8, layer: int = 1) -> ActivationSteering:
    direction = torch.nn.functional.normalize(torch.randn(hidden_dim), dim=0)
    return ActivationSteering(_FakeModel(hidden_dim=hidden_dim),
                              _FakeTokenizer(), direction, layer)


def test_steered_hidden_shifts_exactly_alpha_of_norm():
    h = torch.randn(2, 7, 16)
    direction = torch.nn.functional.normalize(torch.randn(16), dim=0)
    alpha = 0.1
    steered = steered_hidden(h, alpha, direction)
    relative_shift = (steered - h).norm(dim=-1) / h.norm(dim=-1)
    assert torch.allclose(relative_shift, torch.full_like(relative_shift, alpha), atol=1e-5)


def test_steered_hidden_zero_alpha_is_identity():
    h = torch.randn(1, 4, 8)
    direction = torch.randn(8)
    assert torch.equal(steered_hidden(h, 0.0, direction), h)


def test_direction_follows_hidden_states_not_model_device():
    """A model sharded with device_map="auto" runs the target block on a
    different device than the embeddings, so the direction must be placed
    from the hook's own activations."""
    steering = _steering()
    assert steering.direction.dtype == torch.float32
    hidden = torch.zeros(1, 3, 8, dtype=torch.bfloat16, device="meta")

    aligned = steering._direction_like(hidden)

    assert aligned.device == hidden.device
    assert aligned.dtype == hidden.dtype
    # one cached copy per (device, dtype) — no per-token transfer
    assert steering._direction_like(hidden) is aligned


def test_repetition_rate_detects_loops():
    assert repetition_rate(GIBBERISH) > 90
    assert repetition_rate(NORMAL) < 10
    assert token_ngram_repeat_ratio(GIBBERISH.lower().split()) > 0.9
    assert degeneration_metrics(GIBBERISH)["collapsed"] is True


def test_steer_window_drops_after_n_new_tokens():
    window = SteerWindow(2)
    assert window.observe(8) is True   # prefill
    assert window.observe(1) is True   # token 1
    assert window.observe(1) is True   # token 2
    assert window.observe(1) is False  # off
    full = SteerWindow(None)
    assert full.observe(8) is True
    assert full.observe(1) is True
    prompt_only = SteerWindow(0)
    assert prompt_only.observe(8) is True
    assert prompt_only.observe(1) is False
    assert repetition_rate("") == 0.0


def test_worst_window_equals_full_ppl_for_uniform_losses():
    nlls = [1.0] * 50
    assert worst_window_perplexity(nlls, window=5) == pytest.approx(math.exp(1.0))


def test_worst_window_catches_local_spike():
    """One corrupted token must spike the windowed metric even though the
    full-text average barely moves — the 'Plowng' failure mode."""
    nlls = [1.0] * 50
    nlls[25] = 10.0
    windowed = worst_window_perplexity(nlls, window=5)
    full_text = math.exp(sum(nlls) / len(nlls))
    assert windowed == pytest.approx(math.exp((4 * 1.0 + 10.0) / 5))
    assert windowed > 3 * full_text


def test_worst_window_edge_cases():
    assert worst_window_perplexity([]) == float("inf")
    # window larger than the sequence falls back to the whole sequence
    assert worst_window_perplexity([2.0], window=5) == pytest.approx(math.exp(2.0))


class FakeClassifier:
    """Stands in for the zero-shot pipeline: expert-sounding texts score high."""

    def __call__(self, texts, candidate_labels):
        results = []
        for text in texts:
            score = 0.9 if "aquifer" in text.lower() else 0.2
            results.append({
                "labels": list(candidate_labels),
                "scores": [score, 1 - score],
            })
        return results


def test_judge_scores_and_gates():
    judge = ExpertiseJudge("hydrology")
    judge._pipeline = FakeClassifier()

    scores = judge.score([NORMAL, GIBBERISH, "", "something vague and short here"])
    assert scores[0] == pytest.approx(0.9)   # expert text, high
    assert scores[1] == 0.0                  # gibberish gated, classifier never sees it
    assert scores[2] == 0.0                  # empty gated
    assert scores[3] == pytest.approx(0.2)   # coherent but vague, low
    assert judge.domain in judge.positive_label


def _fake_quality_pipeline(degrade_above):
    """generate/quality pair where quality collapses above a known alpha."""

    def generate(prompt, alpha):
        return GIBBERISH if alpha > degrade_above else NORMAL

    def quality(text):
        if text == GIBBERISH:
            return 500.0, repetition_rate(text)
        return 2.0, repetition_rate(text)

    return generate, quality


def test_calibrate_finds_threshold():
    generate, quality = _fake_quality_pipeline(degrade_above=0.1)
    max_alpha = calibrate_max_alpha(
        generate, quality, prompts=["p1", "p2"],
        candidate_alphas=[0.02, 0.06, 0.09, 0.12, 0.2],
    )
    assert max_alpha == 0.09


def test_calibrate_none_when_all_fail():
    generate, quality = _fake_quality_pipeline(degrade_above=0.0)
    assert calibrate_max_alpha(generate, quality, ["p"], [0.02, 0.04]) is None


def test_calibrate_all_pass_returns_largest():
    generate, quality = _fake_quality_pipeline(degrade_above=99.0)
    assert calibrate_max_alpha(generate, quality, ["p"], [0.02, 0.25]) == 0.25


def test_default_test_prompts_do_not_name_the_domain():
    prompts = default_test_prompts("hydrology")
    assert len(prompts) >= 4
    joined = " ".join(prompts).lower()
    assert "hydrology" not in joined
    assert "this field" in joined


def test_generation_defaults_keep_answers_short():
    assert "1-2 short sentences" in DEFAULT_SYSTEM_PROMPT
    assert "Do not use lists" in DEFAULT_SYSTEM_PROMPT
    assert DEFAULT_MAX_NEW_TOKENS == 96


def test_trim_hanging_clause_drops_clipped_tail():
    full = "Folding is the process by which a chain assumes its native fold."
    clipped = full + " This process involves the removal of any non-native amino acids, such"
    assert answer_complete(full)
    assert not answer_complete(clipped)
    assert trim_hanging_clause(clipped) == full
    assert trim_hanging_clause(full) == full
    assert trim_hanging_clause("no punctuation at all") == "no punctuation at all"

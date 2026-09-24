"""Behavioral layer selection and strength calibration.

Separation accuracy saturates (every layer can tell the personas apart), so
the layer is chosen behaviorally: steer real prompts at probe strengths and
keep the layer whose completions the expertise judge scores highest.

Strength calibration then finds `max_alpha`: the largest alpha whose
generations stay coherent (worst-window perplexity within a ratio of
baseline, repetition near zero). The worst-window metric catches a single
corrupted word that full-text perplexity would average away. The user-facing
expertise dial (0-1) maps onto `max_alpha`.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from domainsteer.extract import ExtractionResult
from extras.judge import ExpertiseJudge, repetition_rate, spike_perplexity
from domainsteer.pairs import CALIBRATION_STEMS
from domainsteer.steering import ActivationSteering, DEFAULT_MAX_NEW_TOKENS

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_ALPHAS = [0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25,
                            0.30, 0.40, 0.50]
# Probe strengths for layer selection: strong enough to visibly change the
# output (0.04/0.08 barely moved a 3B model), below typical calibrated maxima.
DEFAULT_PROBE_ALPHAS = (0.08, 0.16)
CALIBRATION_PPL_RATIO = 2.5
CALIBRATION_MAX_REPETITION = 10.0


def default_test_prompts(domain: str) -> List[str]:
    """Generic stems that do not name the domain.

    Used for layer probing and calibration. Naming the domain in the prompt
    already puts the model in that field, so steering would only make the
    answer more technical. These stems stay field-agnostic; the direction
    has to supply the domain reading. They are the first entries of
    GENERIC_STEMS, so pair extraction and calibration ask the same kind of
    question.
    """
    del domain
    return list(CALIBRATION_STEMS)


def select_best_layer(model, tokenizer, extraction: ExtractionResult,
                      judge: ExpertiseJudge, prompts: Sequence[str],
                      probe_alphas: Sequence[float] = DEFAULT_PROBE_ALPHAS,
                      max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> Tuple[int, Dict[int, float]]:
    """Steer `prompts` at `probe_alphas` for every extracted layer; return the
    layer with the highest mean judge score, plus all per-layer scores."""
    layer_scores: Dict[int, float] = {}
    for layer in sorted(extraction.directions):
        steering = ActivationSteering(model, tokenizer,
                                      extraction.directions[layer], layer)
        completions = [
            steering.generate(prompt, alpha=alpha, max_new_tokens=max_new_tokens)
            for prompt in prompts
            for alpha in probe_alphas
        ]
        layer_scores[layer] = float(np.mean(judge.score(completions)))
        logger.info(f"Layer {layer}: mean expertise {layer_scores[layer] * 100:.1f}%")

    best = max(layer_scores, key=layer_scores.get)
    logger.info(f"Best layer: {best} ({layer_scores[best] * 100:.1f}%)")
    return best, layer_scores


def select_best_layer_pairwise(model, tokenizer, extraction: ExtractionResult,
                               judge, prompts: Sequence[str],
                               probe_alphas: Sequence[float] = DEFAULT_PROBE_ALPHAS,
                               max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                               golds: Optional[Sequence[str]] = None,
                               ) -> Tuple[int, Dict[int, dict]]:
    """Layer selection by pairwise preference against each layer's own
    unsteered baseline — an alternative to `select_best_layer`, which scores
    steered text in isolation with the NLI judge.

    Two differences that matter. Each steered answer is compared against the
    *baseline answer to the same prompt*, so prompt difficulty cancels
    instead of dominating. And every probe strength is reported separately
    rather than averaged, so a layer that works at a low alpha and collapses
    at a high one is visible instead of being blended into one mediocre
    number.

    Every extracted layer is scored (middle-third even layers from
    extraction). `judge` is a `PairwiseJudge`; `golds` is forwarded for
    templates that include a reference reading (trap sense). Returns the
    best layer and, per layer, the per-alpha win rates plus the overall mean.
    """
    if golds is not None and len(golds) != len(prompts):
        raise ValueError(
            f"Length mismatch: {len(prompts)} prompts, {len(golds)} golds."
        )
    gold_list = None if golds is None else list(golds)
    baselines: Dict[str, str] = {}
    layer_stats: Dict[int, dict] = {}

    for layer in sorted(extraction.directions):
        steering = ActivationSteering(model, tokenizer,
                                      extraction.directions[layer], layer)
        if not baselines:
            baselines = {p: steering.generate(p, alpha=0.0,
                                              max_new_tokens=max_new_tokens)
                         for p in prompts}

        per_alpha = {}
        for alpha in probe_alphas:
            steered = [steering.generate(p, alpha=alpha,
                                         max_new_tokens=max_new_tokens)
                       for p in prompts]
            report = judge.compare_all(list(prompts), steered,
                                       [baselines[p] for p in prompts],
                                       golds=gold_list)
            per_alpha[alpha] = report.as_dict()
            logger.info(f"Layer {layer} alpha={alpha}: "
                        f"win rate {report.win_rate * 100:.1f}% "
                        f"(consistency {report.consistency * 100:.0f}%)")

        mean_win = float(np.mean([s["win_rate"] for s in per_alpha.values()]))
        layer_stats[layer] = {"per_alpha": per_alpha, "mean_win_rate": mean_win}
        logger.info(f"Layer {layer}: mean win rate {mean_win * 100:.1f}%")

    best = max(layer_stats, key=lambda k: layer_stats[k]["mean_win_rate"])
    logger.info(f"Best layer: {best} "
                f"({layer_stats[best]['mean_win_rate'] * 100:.1f}% win rate)")
    return best, layer_stats


def calibrate_max_alpha(generate: Callable[[str, float], str],
                        quality: Callable[[str], Tuple[float, float]],
                        prompts: Sequence[str],
                        candidate_alphas: Optional[Sequence[float]] = None,
                        ppl_ratio: float = CALIBRATION_PPL_RATIO,
                        max_repetition: float = CALIBRATION_MAX_REPETITION,
                        ) -> Optional[float]:
    """Largest alpha whose generations pass the quality gates on every prompt.

    `generate(prompt, alpha) -> text`; `quality(text) -> (perplexity,
    repetition%)` — the perplexity slot is compared against baseline at the
    same ratio whatever variant supplies it (the real pipeline uses
    worst-window perplexity). Candidates are tried in ascending order and the
    search stops at the first failure. Returns None if even the smallest
    fails.
    """
    if candidate_alphas is None:
        candidate_alphas = DEFAULT_CANDIDATE_ALPHAS

    baseline_ppls = [quality(generate(p, 0.0))[0] for p in prompts]

    max_alpha: Optional[float] = None
    for alpha in sorted(candidate_alphas):
        ok = True
        for prompt, base_ppl in zip(prompts, baseline_ppls):
            ppl, rep = quality(generate(prompt, alpha))
            if ppl > base_ppl * ppl_ratio or rep > max_repetition:
                ok = False
                break
        logger.info(f"  alpha={alpha}: {'ok' if ok else 'degraded'}")
        if not ok:
            break
        max_alpha = alpha

    if max_alpha is None:
        logger.warning("No candidate alpha passed the quality gates; "
                       "steering is unreliable for this model/direction.")
    elif max_alpha == max(candidate_alphas):
        logger.warning(
            f"Calibration hit the candidate ceiling ({max_alpha}) without "
            "finding degradation — the true maximum may be higher."
        )
    return max_alpha


def calibrate_steering(steering: ActivationSteering, prompts: Sequence[str],
                       candidate_alphas: Optional[Sequence[float]] = None,
                       max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> Optional[float]:
    """Convenience wrapper wiring `calibrate_max_alpha` to a real model."""
    model, tokenizer, device = steering.model, steering.tokenizer, steering.device

    def generate(prompt: str, alpha: float) -> str:
        return steering.generate(prompt, alpha=alpha, max_new_tokens=max_new_tokens)

    def quality(text: str) -> Tuple[float, float]:
        return (spike_perplexity(model, tokenizer, text, device),
                repetition_rate(text))

    return calibrate_max_alpha(generate, quality, prompts, candidate_alphas)


def probe_alphas_from_ceiling(ceiling: float,
                              fractions: Sequence[float] = (0.5, 1.0),
                              ) -> List[float]:
    """Probe strengths as fractions of a measured coherence ceiling.

    The fixed `DEFAULT_PROBE_ALPHAS` were tuned on one model and silently
    exceed the usable range on others — probing above the ceiling scores
    degradation rather than expertise, which flattens every layer to the same
    mediocre number.
    """
    alphas = sorted({round(f * ceiling, 4) for f in fractions if f * ceiling > 0})
    return alphas or [round(ceiling, 4)]


def calibrate_alpha_pairwise(steering: ActivationSteering, judge,
                             prompts: Sequence[str],
                             candidate_alphas: Optional[Sequence[float]] = None,
                             ppl_ratio: float = CALIBRATION_PPL_RATIO,
                             max_repetition: float = CALIBRATION_MAX_REPETITION,
                             max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                             golds: Optional[Sequence[str]] = None,
                             ) -> Tuple[Optional[float], Optional[float], Dict[float, dict]]:
    """Strength calibration with a pairwise judge in the loop.

    `calibrate_max_alpha` only asks "is this still coherent?", so `max_alpha`
    is the point just before breakage — which assumes, untested, that more
    steering is always better up to that point. This walks the same ascending
    candidates under the same coherence gates, but also judges each surviving
    strength head-to-head against the unsteered baseline.

    Returns `(max_alpha, best_alpha, per_alpha_stats)`: the coherence ceiling
    (unchanged meaning), the coherent strength the judge most prefers, and
    the win rate at every strength tried.
    """
    model, tokenizer, device = steering.model, steering.tokenizer, steering.device
    if candidate_alphas is None:
        candidate_alphas = DEFAULT_CANDIDATE_ALPHAS

    if golds is not None and len(golds) != len(prompts):
        raise ValueError(
            f"Length mismatch: {len(prompts)} prompts, {len(golds)} golds."
        )
    gold_list = None if golds is None else list(golds)
    baselines = [steering.generate(p, alpha=0.0, max_new_tokens=max_new_tokens)
                 for p in prompts]
    baseline_ppls = [spike_perplexity(model, tokenizer, t, device) for t in baselines]

    stats: Dict[float, dict] = {}
    max_alpha: Optional[float] = None
    for alpha in sorted(candidate_alphas):
        texts, ok = [], True
        for prompt, base_ppl in zip(prompts, baseline_ppls):
            text = steering.generate(prompt, alpha=alpha,
                                     max_new_tokens=max_new_tokens)
            texts.append(text)
            if (spike_perplexity(model, tokenizer, text, device) > base_ppl * ppl_ratio
                    or repetition_rate(text) > max_repetition):
                ok = False
                break
        if not ok:
            logger.info(f"  alpha={alpha}: degraded")
            break

        report = judge.compare_all(list(prompts), texts, baselines,
                                   golds=gold_list)
        stats[alpha] = report.as_dict()
        max_alpha = alpha
        logger.info(f"  alpha={alpha}: ok | win rate "
                    f"{report.win_rate * 100:.1f}% "
                    f"(consistency {report.consistency * 100:.0f}%)")

    if not stats:
        logger.warning("No candidate alpha passed the quality gates.")
        return None, None, stats

    best_alpha = max(stats, key=lambda a: stats[a]["win_rate"])
    if stats[best_alpha]["win_rate"] <= 0.5:
        logger.warning(
            f"No strength beat the unsteered baseline (best {best_alpha} at "
            f"{stats[best_alpha]['win_rate'] * 100:.1f}%) — steering is not "
            "improving answers on these prompts."
        )
    elif best_alpha != max_alpha:
        logger.info(f"Judge prefers alpha={best_alpha} over the coherence "
                    f"ceiling {max_alpha}.")
    return max_alpha, best_alpha, stats

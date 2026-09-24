"""Expertise scoring and generation-quality metrics.

The judge scores how strongly a text reads as *this domain's* writing, using a
zero-shot NLI classifier with labels templated from the domain string.
Gibberish (high repeated-n-gram rate) scores zero without reaching the
classifier, so degenerate steering can never win a sweep.

The labels are domain-specific versus field-neutral, not expert versus lay:
layer selection has to pick the direction that makes unnamed prompts sound
like this field, not merely more technical.

Quality metrics: `repetition_rate` catches loops; `spike_perplexity` scores
the worst sliding window of token losses, catching localized corruption (a
single mangled word) that full-text perplexity averages away.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

GIBBERISH_REPETITION_THRESHOLD = 30.0  # % repeated 3-grams
SPIKE_WINDOW = 5                       # tokens per worst-window

POSITIVE_LABEL = "an answer that is specifically about {domain}, using that field's concepts and reasoning"
NEGATIVE_LABEL = "a field-neutral answer that could apply to any discipline, with no commitment to {domain}"


def repetition_rate(text: str, n: int = 3) -> float:
    """Percentage of repeated n-grams (0 = all unique, 100 = stuck in a loop)."""
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return (1.0 - len(set(ngrams)) / len(ngrams)) * 100


def token_ngram_repeat_ratio(tokens: Sequence, n: int = 3) -> float:
    """Fraction of n-grams that are extra copies of an earlier n-gram (0-1).

    ``sum(count-1 for repeats) / n_grams``. A T35S loop sits near 1.0; a
    normal sentence sits near 0.
    """
    if len(tokens) < 2 * n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(grams)


def distinct_n(text: str, n: int) -> float:
    """Unique word n-grams / total n-grams (1.0 = no repeats)."""
    words = text.lower().split()
    if len(words) < n:
        return 1.0 if words else 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def degeneration_metrics(text: str) -> dict:
    """Length, repetition, and truncation flags for one completion."""
    words = text.split()
    stripped = text.rstrip()
    return {
        "n_words": len(words),
        "n_chars": len(text),
        "repetition_rate": repetition_rate(text),
        "token_repeat_ratio": token_ngram_repeat_ratio(words),
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
        "distinct_3": distinct_n(text, 3),
        "ends_sentence": stripped.endswith((".", "!", "?")),
        "collapsed": repetition_rate(text) >= GIBBERISH_REPETITION_THRESHOLD
        or token_ngram_repeat_ratio(words) >= 0.45,
    }


def token_nlls(model, tokenizer, text: str, device) -> List[float]:
    """Per-token negative log-likelihoods of `text` under the model."""
    import torch

    encodings = tokenizer(text, return_tensors="pt").to(device)
    ids = encodings.input_ids
    if ids.shape[1] < 2:
        return []
    with torch.no_grad():
        logits = model(ids).logits
    log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    nlls = -log_probs.gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(0).squeeze(-1)
    return nlls.tolist()


def worst_window_perplexity(nlls: Sequence[float],
                            window: int = SPIKE_WINDOW) -> float:
    """exp of the highest mean NLL over any `window` consecutive tokens.

    Equals full-text perplexity for uniform losses; spikes when any short
    span is corrupted, which the full-text average dilutes.
    """
    if not nlls:
        return float("inf")
    w = max(1, min(window, len(nlls)))
    worst = max(sum(nlls[i:i + w]) / w for i in range(len(nlls) - w + 1))
    return math.exp(worst)


def spike_perplexity(model, tokenizer, text: str, device,
                     window: int = SPIKE_WINDOW) -> float:
    """Worst-window perplexity of `text` under the (unsteered) model."""
    if not text.strip():
        return float("inf")
    return worst_window_perplexity(token_nlls(model, tokenizer, text, device),
                                   window=window)


class ExpertiseJudge:
    """Zero-shot domain-adherence scorer (0.0-1.0 per text).

    The class name is historical. The labels distinguish this field's
    writing from a field-neutral answer, which is what layer selection
    needs once the pairs themselves are in-domain versus field-neutral.
    """

    def __init__(self, domain: str, device: Optional[int] = None):
        self.domain = domain
        self.positive_label = POSITIVE_LABEL.format(domain=domain)
        self.negative_label = NEGATIVE_LABEL.format(domain=domain)
        self._device = device
        self._pipeline = None

    def _classifier(self):
        if self._pipeline is None:
            import torch
            from transformers import pipeline
            device = self._device
            if device is None:
                device = 0 if torch.cuda.is_available() else -1
            self._pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device,
            )
        return self._pipeline

    def score(self, texts: List[str]) -> List[float]:
        """Domain-adherence score per text; gibberish and empty texts score 0."""
        gated = [
            bool(t.strip()) and repetition_rate(t) <= GIBBERISH_REPETITION_THRESHOLD
            for t in texts
        ]
        coherent = [t for t, ok in zip(texts, gated) if ok]

        coherent_scores: List[float] = []
        if coherent:
            results = self._classifier()(
                coherent, candidate_labels=[self.positive_label, self.negative_label]
            )
            if isinstance(results, dict):
                results = [results]
            for r in results:
                idx = r["labels"].index(self.positive_label)
                coherent_scores.append(float(r["scores"][idx]))

        scores: List[float] = []
        it = iter(coherent_scores)
        for ok in gated:
            scores.append(next(it) if ok else 0.0)
        return scores

"""Self-gold alpha sweep: isolated from gold / trap / trap_local caches.

GPT still writes the polysemy questions (direct + tricky). The model under
test writes gold under a domain-committed system prompt, and a default-prompt
baseline on the same stems. Extraction is expert-text minus baseline-text:
Llama domain-prompt gold (``--positive self``) or the GPT exam answer
(``--positive gpt``). Layer and max_alpha come from the pairwise LLM judge.
The sweep then writes baseline + steered completions and stops — no
sentence-transformer grading.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from domainsteer.pairs import ContrastivePair

_REFUSAL_RE = re.compile(
    r"(i('m| am) ready to assist"
    r"|what'?s the question\?"
    r"|how can i (help|assist)"
    r"|as an ai( language model)?"
    r"|i (do not|don't) have enough)",
    re.I,
)
_WORD_RE = re.compile(r"[a-z0-9]+", re.I)

SELF_GOLD_CONTRAST_ID = "self_gold_expert_vs_default_v1"
GPT_GOLD_CONTRAST_ID = "self_gold_gpt_vs_default_v1"
SELF_GOLD_GPT_VARIANT = "self_gold_gpt"
POSITIVE_SELF = "self"
POSITIVE_GPT = "gpt"
POSITIVES = (POSITIVE_SELF, POSITIVE_GPT)

DEFAULT_MODELS = (
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
)

# Spread across biological / chemical / earth / engineering / ecological /
# CS / social / physical. Override with --ids.
DEFAULT_CLUSTER_IDS = (
    "3001",  # Agricultural biotechnology
    "3101",  # Biochemistry and cell biology
    "3401",  # Analytical chemistry
    "3702",  # Climate change science
    "4012",  # Fluid mechanics and thermal engineering
    "4102",  # Ecological applications
    "4602",  # Artificial intelligence
    "3501",  # Accounting, auditing and accountability
    "4407",  # Policy and administration
    "5101",  # Astronomical sciences
)

SWEEP_ALPHAS = (0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
MIN_PAIRS = 10

SUMMARY_FIELDS = (
    "model_name", "cluster_id", "cluster", "division", "knowledge_domain",
    "status", "n", "n_direct", "n_tricky", "layer", "max_alpha", "best_alpha",
    "alphas", "error",
)


def gold_system_prompt(domain: str) -> str:
    return (
        f"You are a {domain} practitioner. Every question uses a term that "
        f"has an operational meaning in {domain}. Answer that meaning even "
        "when the wording sounds like everyday life, media, sports, law, or "
        "another field — those readings are distractors, not the answer. "
        "Answer in 1-2 short sentences. Do not use lists. Do not explain "
        "other fields' meanings of the term."
    )


def eval_system_prompt(domain: str) -> str:
    """Fair Rimsky-style prompt arm: names the field, no trap coaching.

    Matched in length and format to ``DEFAULT_SYSTEM_PROMPT``. Do not use
    ``gold_system_prompt`` here — that text is extraction scaffolding.
    """
    return (
        f"You are a {domain} practitioner. Answer in 1-2 short sentences. "
        "Do not use lists."
    )


def short_system_prompt(domain: str) -> str:
    """Normal short domain prompt used as an alternate gold.

    Names the field only. No trap coaching, no distractor instructions.
    One sentence so the answer matches GPT exam length.
    """
    return (
        f"You are a {domain} practitioner. Answer in one short sentence. "
        "Do not use lists."
    )


def gold_user_message(domain: str, item: dict, *, retry: bool = False) -> str:
    """User turn for gold only. Steered eval still sees the bare question."""
    question = " ".join(str(item.get("question") or "").split()).strip()
    term = " ".join(str(item.get("term") or "").split()).strip()
    named = f' of "{term}"' if term else ""
    if retry:
        return (
            f"The everyday reading is wrong. You are a {domain} specialist. "
            f"Answer only with {domain}'s operational meaning{named}.\n\n"
            f"{question}"
        )
    kind = str(item.get("kind") or "").strip()
    if kind != "tricky":
        return question
    return (
        f"This is a competing-canon trap. Answer only with {domain}'s "
        f"operational meaning{named}. Do not follow the everyday, media, "
        "sports, legal, or other-field reading the wording suggests.\n\n"
        f"{question}"
    )


def _tokens(text: str) -> set:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text or "")}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def gold_is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text or "")) or len(_tokens(text)) < 4


def gold_follows_baseline_sense(gold: str, gpt_answer: str,
                                baseline: str) -> bool:
    """True when gold tracks the unsteered reading more than GPT's domain answer."""
    g, t, b = _tokens(gold), _tokens(gpt_answer), _tokens(baseline)
    if gold_is_refusal(gold):
        return True
    vs_base = _jaccard(g, b)
    vs_gpt = _jaccard(g, t)
    return vs_base >= vs_gpt and vs_base >= 0.18


def alpha_grid(max_alpha: Optional[float] = None,
               steps: Sequence[float] = SWEEP_ALPHAS) -> List[float]:
    """0 plus every fixed step through 0.30. Calibrated max_alpha is ignored."""
    vals = [0.0]
    for step in steps:
        if float(step) > 0:
            vals.append(float(step))
    return vals


def contrast_for_positive(positive: str) -> str:
    if positive == POSITIVE_GPT:
        return GPT_GOLD_CONTRAST_ID
    return SELF_GOLD_CONTRAST_ID


def expert_texts(items: Sequence[dict], golds: Sequence[str],
                 positive: str = POSITIVE_SELF) -> List[str]:
    """Positive-class strings for v: Llama gold, or GPT exam answers."""
    if len(items) != len(golds):
        raise ValueError("items and golds must be the same length.")
    if positive == POSITIVE_GPT:
        out = []
        for item in items:
            text = item.get("answer") or item.get("gpt_answer") or ""
            out.append(" ".join(str(text).split()).strip())
        return out
    return [" ".join(str(gold or "").split()).strip() for gold in golds]


def items_with_model_gold(items: Sequence[dict],
                          golds: Sequence[str]) -> List[dict]:
    """Keep GPT's answer aside; put this model's gold in `answer`."""
    if len(items) != len(golds):
        raise ValueError("items and golds must be the same length.")
    out = []
    for item, gold in zip(items, golds):
        row = dict(item)
        row["gpt_answer"] = item.get("answer")
        row["answer"] = gold
        out.append(row)
    return out


def model_gold_pairs(items: Sequence[dict], golds: Sequence[str],
                     baselines: Sequence[str],
                     min_pairs: int = MIN_PAIRS) -> List[ContrastivePair]:
    """Expert text minus this model's default-prompt baseline.

    `golds` is already the positive class (Llama domain gold or GPT exam
    answers). Every exam item (direct and tricky) goes into the mean.
    Collapsed golds are retried at generation time; they are not dropped
    from v.
    """
    if not (len(items) == len(golds) == len(baselines)):
        raise ValueError("items, golds and baselines must be the same length.")
    pairs: List[ContrastivePair] = []
    for item, gold, baseline in zip(items, golds, baselines):
        question = " ".join(str(item.get("question") or "").split()).strip()
        expert = " ".join(str(gold or "").split()).strip()
        other = " ".join(str(baseline or "").split()).strip()
        if not question or not expert or not other:
            continue
        pairs.append(ContrastivePair(expert, other, question))
    if len(pairs) < min_pairs:
        raise ValueError(
            f"Need at least {min_pairs} gold-vs-baseline pairs, got "
            f"{len(pairs)}. Use a larger --n."
        )
    return pairs

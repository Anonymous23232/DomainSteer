"""Pairwise LLM-as-judge: which of two answers to the same question reads as
more expert?

Absolute scoring asks a model "how expert is this text, 0 to 1", which is
poorly calibrated and dominated by how hard the question was. Pairwise
preference asks the strictly easier question — "which of these two?" — with
the question held fixed, so difficulty cancels and only the contrast matters.

Every comparison runs in **both orders**. LLM judges have a well-documented
position bias toward whichever answer appears first, so a preference is only
counted when the two orderings agree on the same underlying answer;
disagreement is recorded as a tie rather than silently resolved. The
`consistency` rate that falls out is itself a diagnostic — if it sits near
50%, the judge is guessing and its verdicts mean nothing.

This does not replace `ExpertiseJudge` (zero-shot NLI: free, offline,
deterministic). It is a second, higher-quality instrument for reporting, and
it costs two API calls per comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from domainsteer.pairs import DEFAULT_GENERATION_MODEL, request_json_completion

logger = logging.getLogger(__name__)

WIN, LOSS, TIE = 1.0, 0.0, 0.5

COMPARISON_PROMPT = """\
You are evaluating which of two answers was written by a genuine {domain} \
expert.

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Judge only on domain expertise:
- Precise, correct use of {domain} terminology and reasoning.
- Technical accuracy. Penalise confident-sounding claims that are wrong, \
invented, or misattributed, even when they sound sophisticated. A vaguer but \
correct answer beats a specific but incorrect one.
- Depth of genuine understanding rather than surface vocabulary.

Ignore length, formatting, markdown, headings, and numbering entirely. \
Ignore which answer appears first.

Return strict JSON:
{{"winner": "A" or "B" or "tie", "reason": "<one sentence>"}}"""


@dataclass
class Comparison:
    """One debiased verdict. `winner` is 'a', 'b', or 'tie'."""

    winner: str
    consistent: bool
    forward: str                       # raw verdict with a shown first
    reverse: str                       # raw verdict with b shown first
    reasons: Tuple[str, str] = ("", "")

    @property
    def score_for_a(self) -> float:
        return {"a": WIN, "b": LOSS}.get(self.winner, TIE)


@dataclass
class PairwiseReport:
    n: int
    wins: int
    losses: int
    ties: int
    consistency: float                 # fraction where both orderings agreed
    comparisons: List[Comparison] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        """Ties count as half, so 0.5 means 'no detectable difference'."""
        return (self.wins + 0.5 * self.ties) / self.n if self.n else 0.5

    def as_dict(self) -> dict:
        return {"n": self.n, "wins": self.wins, "losses": self.losses,
                "ties": self.ties, "win_rate": self.win_rate,
                "loss_rate": self.losses / self.n if self.n else 0.0,
                "tie_rate": self.ties / self.n if self.n else 0.0,
                "consistency": self.consistency}


class PairwiseJudge:
    """Compare answers via an API model, debiased against answer ordering.

    Verdicts are cached in memory on (question, answer_a, answer_b), so
    repeating a comparison inside one run is free.
    """

    def __init__(self, domain: str, api_key: Optional[str] = None,
                 model: str = DEFAULT_GENERATION_MODEL,
                 prompt_template: Optional[str] = None):
        self.domain = domain
        self.api_key = api_key
        self.model = model
        self.prompt_template = prompt_template or COMPARISON_PROMPT
        self._cache: Dict[tuple, str] = {}
        self.calls = 0

    # ------------------------------------------------------------- one verdict

    def _ask(self, question: str, answer_a: str, answer_b: str,
             gold: str = "") -> Tuple[str, str]:
        """One ordering. Returns (verdict in {'A','B','tie'}, reason)."""
        key = (question, answer_a, answer_b, gold)
        if key in self._cache:
            return self._cache[key], ""

        prompt = self.prompt_template.format(
            domain=self.domain, question=question,
            answer_a=answer_a, answer_b=answer_b, gold=gold,
        )
        payload = request_json_completion(prompt, api_key=self.api_key,
                                          model=self.model)
        self.calls += 1

        verdict = str(payload.get("winner", "")).strip().upper()
        if verdict not in {"A", "B"}:
            verdict = "TIE"
        reason = str(payload.get("reason", "")).strip()
        self._cache[key] = verdict
        return verdict, reason

    def compare(self, question: str, answer_a: str, answer_b: str,
                gold: str = "") -> Comparison:
        """Both orderings; a preference only counts when they agree."""
        forward, r1 = self._ask(question, answer_a, answer_b, gold=gold)
        reverse, r2 = self._ask(question, answer_b, answer_a, gold=gold)

        # forward: a is shown as "A". reverse: b is shown as "A".
        picked_forward = {"A": "a", "B": "b"}.get(forward, "tie")
        picked_reverse = {"A": "b", "B": "a"}.get(reverse, "tie")

        consistent = picked_forward == picked_reverse
        winner = picked_forward if consistent else "tie"
        return Comparison(winner=winner, consistent=consistent,
                          forward=forward, reverse=reverse, reasons=(r1, r2))

    # ---------------------------------------------------------- many verdicts

    def compare_all(self, questions: Sequence[str],
                    answers_a: Sequence[str],
                    answers_b: Sequence[str],
                    golds: Optional[Sequence[str]] = None) -> PairwiseReport:
        """Compare two answer sets question by question.

        `answers_a` is the arm under test; `answers_b` is the reference. A
        win_rate above 0.5 means A is preferred. `golds` is interpolated into
        templates that contain `{gold}` (ignored by the default template).
        """
        if not (len(questions) == len(answers_a) == len(answers_b)):
            raise ValueError(
                f"Length mismatch: {len(questions)} questions, "
                f"{len(answers_a)} A answers, {len(answers_b)} B answers."
            )
        if golds is None:
            gold_list: Sequence[str] = [""] * len(questions)
        else:
            if len(golds) != len(questions):
                raise ValueError(
                    f"Length mismatch: {len(questions)} questions, "
                    f"{len(golds)} gold answers."
                )
            gold_list = golds

        comparisons = [self.compare(q, a, b, gold=g)
                       for q, a, b, g in zip(
                           questions, answers_a, answers_b, gold_list)]
        wins = sum(c.winner == "a" for c in comparisons)
        losses = sum(c.winner == "b" for c in comparisons)
        ties = sum(c.winner == "tie" for c in comparisons)
        n = len(comparisons)
        consistency = (sum(c.consistent for c in comparisons) / n) if n else 0.0

        report = PairwiseReport(n=n, wins=wins, losses=losses, ties=ties,
                                consistency=consistency, comparisons=comparisons)
        if n and consistency < 0.6:
            logger.warning(
                f"Judge agreed with itself on only {consistency * 100:.0f}% of "
                "order-swapped comparisons — verdicts are close to guessing."
            )
        logger.info(f"Pairwise: {wins}W/{losses}L/{ties}T of {n} "
                    f"(win rate {report.win_rate * 100:.1f}%, "
                    f"consistency {consistency * 100:.0f}%)")
        return report

    def scores_vs_baseline(self, questions: Sequence[str],
                           answers: Sequence[str],
                           baselines: Sequence[str]) -> List[float]:
        """Per-item 1.0 / 0.5 / 0.0 against a baseline answer set."""
        return [c.score_for_a
                for c in self.compare_all(questions, answers, baselines).comparisons]

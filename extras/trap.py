"""Polysemy-trap exam: overloaded terms, domain sense vs other-field readings.

Each domain gets its own bank of *direct* ("What is forcing?") and *tricky*
questions. Tricky still uses the overloaded name, but asks a follow-up fact \
whose default-canon answer is wrong for this field (Spider-Man's girlfriend \
died vs did not). The domain name is not in the question; gold is this \
field's reading. Contrastive pairs use the direct questions only. Tricky \
items stay on the benchmark.

Cache files (`benchmark_trap.json`, `benchmark_trap_pair_stems.json`,
`pairs-trap.jsonl`, `directions-trap/`) are separate from gold and unnamed
artifacts. `trap_local` keeps the GPT exam, then builds `v` from this
model's activations: gold answers minus unsteered completions on the same
directs (`pairs-trap-local.jsonl`, `directions-trap_local/`).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence, Union

from extras.judge import degeneration_metrics
from domainsteer.pairs import (
    DEFAULT_GENERATION_MODEL,
    ContrastivePair,
    PairGenerator,
    TRAP_CONTRAST_ID,
    TRAP_LOCAL_CONTRAST_ID,
    TRAP_LOCAL_PAIRS_META_NAME,
    TRAP_LOCAL_PAIRS_NAME,
    TRAP_POLYSEMY_EXAMPLES,
    _match_key,
    load_pairs,
    request_json_completion,
    save_pairs,
)

logger = logging.getLogger(__name__)

SET_TRAP = "trap"
TRAP_EXAM_CONTRAST = "polysemy_trap_v6"
TRAP_DIRECT_STEMS_CONTRAST = "trap_direct_stems_v1"
BENCHMARK_TRAP_NAME = "benchmark_trap.json"
TRAP_DIRECT_STEMS_NAME = "benchmark_trap_pair_stems.json"
DEFAULT_N_QUESTIONS = 50
DEFAULT_TRAP_PAIR_STEMS = 100
DEFAULT_TRAP_BATCH = 5
DEFAULT_TRAP_MIN_WORDS = 15
DEFAULT_TRAP_MAX_WORDS = 40
MIN_DIRECT_QUESTION_CHARS = 8
MIN_ANSWER_WORDS = 8
MIN_GOLD_BASELINE_PAIRS = 10

TRAP_ITEMS_PROMPT = """\
You are a highly knowledgeable {domain} expert writing a polysemy trap exam.

The intended pattern is a competing canon, not a definition riddle:

    Domain = Tobey Maguire's Spider-Man (never written in the question)
    direct: Who is Spider-Man?
    tricky: Who is Spider-Man's girlfriend, and did she die?
      unsteered: Gwen Stacy, yes (The Amazing Spider-Man)
      steered: Mary Jane Watson, and she is not dead

The tricky question still says "Spider-Man". It does not name Tobey Maguire, \
Raimi, or any other canon. A generalist answers the more famous or recent \
reading. Steering, not extra jargon in the question, selects {domain}.

Do not name {domain} (or any discipline) in the questions. Do not put \
{domain} mechanisms, assays, proteins, instruments, or experimental setup \
in the question. Gold answers are written later with a domain-committed \
system prompt.

""" + TRAP_POLYSEMY_EXAMPLES + """\

Write {n} overloaded terms used in {domain} AND in everyday speech, IT, \
pop culture, business, or another specialist field.

Seed ideas (invent more; do not write one item per bullet, and do not reuse \
the worked-example terms unless they also genuinely belong in {domain}):
{concept_lines}

For each term write:

- term:
  The overloaded word or short collocation.

- direct_question:
  A short, natural question containing the term itself.
  Another field or an ordinary speaker could plausibly interpret it \
differently.

  Follow the style of the worked examples:
  "What is forcing?"
  "How does the routing work?"
  "Who is Spider-Man?"

  Do not name {domain} or any discipline.

- direct_answer:
  Answer only in the {domain}-specific sense.
  Write {min_words}-{max_words} words (about 1-2 short sentences).
  Explain the relevant mechanism, meaning, or practice.
  Do not discuss alternative meanings.

- tricky_question:
  Keep the overloaded term in the question. Ask a follow-up fact, \
consequence, or "what happened to X" that a generalist will answer from \
the everyday / pop / IT canon, while a {domain}-steered model answers \
from this field.

  Same shape as:
  "Who is Spider-Man's girlfriend, and did she die?"
  "If you stop the forcing, does everything go back to normal?"
  "How far did the fetch have to go, and did anything come back?"

  Both canons must be able to answer the SAME sentence. Do not import a
  second scene's props, people, or punchline that only exist in the
  everyday reading.

  BAD (funeral-only; fluids has no "everyone" or "home"):
  "After the wake, does everyone go home?"
  BAD (movie-only): "When the film ends, do the credits roll?"
  BAD (laundry-only): "When the cycle ends, are the clothes dry?"
  BAD (legal-only): "If the charge is dropped, does the case get dismissed?"
  GOOD (funeral guests still moving vs residual flow):
  "After the wake, is anything still moving?"
  GOOD: "If the wake dies out, is everything back to how it was?"

  The tricky question MUST:
  * contain the term itself
  * be a question a layperson would ask about that word
  * have a plausible default-canon answer that is WRONG for {domain}
  * still be interpretable as a question about the {domain} sense of the
    term, without rejecting the wording

  The tricky question MUST NOT:
  * name {domain}, Tobey Maguire-style canon labels, "this field", or \
"this work"
  * describe the {domain} mechanism (no mRNA, ribosomes, wind over water, \
hydraulic head, spliceosome, cell cycle, ...)
  * be a riddle whose only solution is the term
    (BAD: "What process decodes an mRNA sequence into a polypeptide?")
  * contain a long scenario, experiment, named gene/drug/disease, assay, \
or numbers
  * say "rather than a ship", "rather than an API", or similar lay tails
  * simply repeat the direct "What is X?"
  * mention people going home, credits, albums, clothes, court cases, \
apps, files, hearts, or other props that cannot be read in {domain}

  One sentence. Everyday wording.

- tricky_answer:
  Answer the tricky question using the {domain} canon only.
  Write {min_words}-{max_words} words (about 1-2 short sentences).
  Resolve the question; do not repeat it.

GENERAL REQUIREMENTS:

1. The term must genuinely be overloaded.
2. Direct and tricky questions for one term must be substantially different.
3. Direct tests the ambiguity of the WORD.
4. Tricky tests a FACT that splits by canon, with the term still in the \
question.
5. Domain context lives in the system prompt / steering, not the question.
6. Prefer one precise sentence.
7. Terms must not be paraphrases of one another.
8. Do not reuse a worked-example term unless it belongs in {domain}.
9. Do not mention {domain}, "this field", or "this work" in the questions.
10. Answers resolve the question rather than repeating it.
11. Return JSON only.

{already}

Return strict JSON:
{{"terms": [{{"term": "...", "direct_question": "...", \
"direct_answer": "...", "tricky_question": "...", \
"tricky_answer": "..."}}]}}
"""

TRAP_SENSE_PROMPT = """\
You are evaluating which of two answers uses the {domain} sense of the \
terms in the question.

The question is written so several fields could claim it. A committed \
answer uses {domain}'s meaning (mechanisms, measurements, practice). A \
survey or other-field answer uses everyday, IT, business, or another \
specialist's reading, lists several fields, or refuses to pick one.

""" + TRAP_POLYSEMY_EXAMPLES + """\
Gold {domain} reading (for your reference):
{gold}

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Pick the answer that both (1) uses the {domain} sense of the terms and \
(2) is factually correct for that sense. Penalize invented terms, wrong \
mechanisms, and confident false detail even if they sound like {domain}. \
A shorter correct {domain} answer beats a longer invented one. Ignore \
length, formatting, and which answer appears first. A survey / everyday / \
other-field reading loses even if it is fluent. For the worked examples, \
hydrology sense wins over tech/AI or everyday sense; here the target \
sense is {domain}.

Return strict JSON:
{{"winner": "A" or "B" or "tie", "reason": "<one sentence>"}}"""

TRAP_DIRECT_STEMS_PROMPT = """\
You are a highly knowledgeable {domain} expert writing short polysemy traps.

Each question must be readable as several fields at once. Do not name \
{domain} (or any discipline) in the questions.

""" + TRAP_POLYSEMY_EXAMPLES + """\
Write {n} overloaded terms used in {domain} AND in everyday speech, IT, \
business, or another specialist field. Seed ideas (invent more; do not \
write one item per bullet, and do not reuse the worked-example terms unless \
they also belong in {domain}):
{concept_lines}

For each term write only a short direct question — "What is X?", \
"What does X mean?", "How does X work?", or "Define X." No conceptual \
role questions. No "this field" / "this work".

{already}

Return strict JSON:
{{"stems": [{{"term": "...", "question": "..."}}]}}"""


def n_trap_terms(n_questions: int) -> int:
    """`--n` is question count; each term yields a direct and a tricky item."""
    if n_questions < 1:
        raise ValueError(f"Need at least 1 question, got {n_questions}.")
    return max(1, n_questions // 2)


def flatten_trap_terms(terms: Sequence[dict]) -> List[dict]:
    """One cached term → two benchmark items (direct, then tricky)."""
    items: List[dict] = []
    for term in terms:
        for kind in ("direct", "tricky"):
            items.append({
                "term": term["term"],
                "kind": kind,
                "question": term[f"{kind}_question"],
                "answer": term[f"{kind}_answer"],
            })
    return items


def exam_direct_questions(items: Sequence[dict]) -> List[str]:
    """Definition / 'What is X?' questions only — the usable pair stems."""
    return [item["question"] for item in items if item.get("kind") == "direct"]


def exam_direct_probes(items: Sequence[dict]) -> tuple:
    """Exam directs with golds — layer probes for the sense judge.

    Tricky items already unpack the concept, so they do not discriminate
    layers. Returns `(questions, golds)`.
    """
    directs = [item for item in items if item.get("kind") == "direct"]
    return ([item["question"] for item in directs],
            [item["answer"] for item in directs])


def exam_direct_items(items: Sequence[dict]) -> List[dict]:
    """Direct exam rows (question + gold), tricky held out."""
    return [item for item in items if item.get("kind") == "direct"]


def gold_vs_baseline_pairs(items: Sequence[dict], generate,
                           min_pairs: int = MIN_GOLD_BASELINE_PAIRS
                           ) -> List[ContrastivePair]:
    """One pair per exam direct: GPT gold vs this model's unsteered answer.

    `generate(question) -> str` is the local model at alpha=0. Tricky items
    are skipped because they already state the concept, so baseline and
    gold may share a sense and `v` would collapse.
    """
    pairs: List[ContrastivePair] = []
    for item in exam_direct_items(items):
        question = " ".join(str(item.get("question") or "").split()).strip()
        gold = " ".join(str(item.get("answer") or "").split()).strip()
        if not question or not gold:
            continue
        baseline = " ".join(str(generate(question) or "").split()).strip()
        if not baseline:
            logger.warning(f"Skipping '{question[:80]}': empty baseline.")
            continue
        pairs.append(ContrastivePair(gold, baseline, question))
    if len(pairs) < min_pairs:
        raise ValueError(
            f"Need at least {min_pairs} exam directs with gold and a local "
            f"baseline, got {len(pairs)}. Use --n 20 or more (each term "
            "yields one direct)."
        )
    return pairs


def local_pairs_paths(model_dir) -> tuple:
    """Model-specific gold-vs-baseline cache (baselines are not portable)."""
    root = Path(model_dir)
    return root / TRAP_LOCAL_PAIRS_NAME, root / TRAP_LOCAL_PAIRS_META_NAME


def save_gold_vs_baseline_pairs(pairs: Sequence[ContrastivePair], model_dir,
                                model_name: str) -> Path:
    pairs_path, meta_path = local_pairs_paths(model_dir)
    save_pairs(list(pairs), pairs_path)
    meta_path.write_text(
        json.dumps({
            "contrast": TRAP_LOCAL_CONTRAST_ID,
            "model": model_name,
            "n": len(pairs),
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(f"Cached {len(pairs)} gold-vs-baseline pairs at {pairs_path}")
    return pairs_path


def load_gold_vs_baseline_pairs(model_dir) -> List[ContrastivePair]:
    pairs_path, meta_path = local_pairs_paths(model_dir)
    if not pairs_path.exists() or not meta_path.exists():
        return []
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if meta.get("contrast") != TRAP_LOCAL_CONTRAST_ID:
        return []
    return load_pairs(pairs_path)


def _clean_direct_stems(raw, domain: str,
                        blocked: Optional[Sequence[str]] = None) -> List[dict]:
    """Keep short overloaded-term definition questions."""
    blocked_keys = {_match_key(q) for q in (blocked or [])}
    if not isinstance(raw, list):
        return []
    seen_terms = set()
    seen_questions = set(blocked_keys)
    out: List[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        term = " ".join(str(item.get("term") or "").split()).strip()
        question = " ".join(str(item.get("question") or "").split()).strip()
        if len(term) < 2 or _mentions_domain(term, domain):
            continue
        if len(question) < MIN_DIRECT_QUESTION_CHARS or "?" not in question:
            continue
        if _question_leaks_identity(question, domain):
            continue
        term_key = _match_key(term)
        q_key = _match_key(question)
        if term_key in seen_terms or q_key in seen_questions:
            continue
        seen_terms.add(term_key)
        seen_questions.add(q_key)
        out.append({"term": term, "question": question})
    return out


def _mentions_domain(text: str, domain: str) -> bool:
    needle = domain.strip()
    if not needle:
        return False
    return re.search(rf"\b{re.escape(needle)}\b", text, re.I) is not None


def _question_uses_term(question: str, term: str) -> bool:
    """Tricky items must still name the overloaded word, like 'Spider-Man'."""
    q = _match_key(question)
    t = _match_key(term)
    if not t or not q:
        return False
    if t in q:
        return True
    for tok in t.split():
        if len(tok) >= 4 and tok in q:
            return True
    return False


def _question_leaks_identity(question: str, domain: str) -> bool:
    lowered = question.lower()
    if "this field" in lowered or "this work" in lowered:
        return True
    return _mentions_domain(question, domain)


def _clean_trap_terms(raw, domain: str) -> List[dict]:
    """Keep terms with a real overloaded word and two usable Q/A pairs."""
    if not isinstance(raw, list):
        return []
    seen_terms = set()
    seen_questions = set()
    out: List[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        term = " ".join(str(item.get("term") or "").split()).strip()
        if len(term) < 2 or _mentions_domain(term, domain):
            continue
        term_key = _match_key(term)
        if term_key in seen_terms:
            continue
        row = {"term": term}
        ok = True
        for kind in ("direct", "tricky"):
            question = " ".join(
                str(item.get(f"{kind}_question") or "").split()
            ).strip()
            answer = " ".join(
                str(item.get(f"{kind}_answer") or "").split()
            ).strip()
            min_q = MIN_DIRECT_QUESTION_CHARS if kind == "direct" else 12
            if len(question) < min_q or "?" not in question:
                ok = False
                break
            if _question_leaks_identity(question, domain):
                ok = False
                break
            if kind == "tricky" and not _question_uses_term(question, term):
                ok = False
                break
            if len(answer.split()) < MIN_ANSWER_WORDS:
                ok = False
                break
            q_key = _match_key(question)
            if q_key in seen_questions:
                ok = False
                break
            row[f"{kind}_question"] = question
            row[f"{kind}_answer"] = answer
            row[f"_{kind}_key"] = q_key
        if not ok:
            continue
        if row["direct_question"] == row["tricky_question"]:
            continue
        seen_terms.add(term_key)
        seen_questions.add(row.pop("_direct_key"))
        seen_questions.add(row.pop("_tricky_key"))
        out.append(row)
    return out


class BenchmarkTrap:
    """Per-domain polysemy items, cached under the domain slug."""

    def __init__(self, domain: str,
                 cache_dir: Optional[Union[str, Path]] = None,
                 api_key: Optional[str] = None,
                 generation_model: str = DEFAULT_GENERATION_MODEL):
        self.domain = domain
        self.api_key = api_key
        self.generation_model = generation_model
        self._pairs = PairGenerator(
            domain, cache_dir=cache_dir, api_key=api_key,
            generation_model=generation_model, contrast=TRAP_CONTRAST_ID,
        )
        self.path = self._pairs.domain_dir / BENCHMARK_TRAP_NAME
        self.pair_stems_path = self._pairs.domain_dir / TRAP_DIRECT_STEMS_NAME

    def cached_items(self) -> List[dict]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if payload.get("contrast") not in (None, TRAP_EXAM_CONTRAST):
            return []
        return flatten_trap_terms(_clean_trap_terms(payload.get("terms", []),
                                                    self.domain))

    def cached_pair_stems(self) -> List[str]:
        """Exam directs plus any cached extra definition stems."""
        stems = exam_direct_questions(self.cached_items())
        extras = self._load_extra_stems(blocked=stems)
        for row in extras:
            if row["question"] not in stems:
                stems.append(row["question"])
        return stems

    def pair_stems(self, n: int = DEFAULT_TRAP_PAIR_STEMS,
                   exam_items: Sequence[dict] = (),
                   force: bool = False,
                   concepts: Optional[Sequence[str]] = None,
                   batch_size: int = DEFAULT_TRAP_BATCH) -> List[str]:
        """Direct exam questions, topped up to `n` definition traps (default 100).

        Tricky conceptual items stay on the benchmark; they are not pair stems.
        """
        directs = exam_direct_questions(exam_items)
        if n < 1:
            raise ValueError(f"Need at least 1 pair stem, got {n}.")
        need = max(0, n - len(directs))
        extras = []
        if need:
            extras = self._extra_direct_stems(
                need, blocked=directs, force=force, concepts=concepts,
                batch_size=batch_size,
            )
        stems = list(directs)
        have = {_match_key(q) for q in stems}
        for row in extras:
            key = _match_key(row["question"])
            if key in have:
                continue
            stems.append(row["question"])
            have.add(key)
            if len(stems) >= n:
                break
        if len(stems) < n:
            raise ValueError(
                f"Trap pair stems: {len(directs)} exam directs + "
                f"{len(extras)} extras; need {n}."
            )
        return stems[:n]

    def _load_extra_stems(self, blocked: Sequence[str]) -> List[dict]:
        if not self.pair_stems_path.exists():
            return []
        try:
            payload = json.loads(self.pair_stems_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if payload.get("contrast") not in (None, TRAP_DIRECT_STEMS_CONTRAST):
            return []
        return _clean_direct_stems(payload.get("stems", []), self.domain,
                                   blocked=blocked)

    def _extra_direct_stems(self, n: int, blocked: Sequence[str],
                            force: bool,
                            concepts: Optional[Sequence[str]],
                            batch_size: int) -> List[dict]:
        if not force:
            cached = self._load_extra_stems(blocked=blocked)
            if len(cached) >= n:
                logger.info(
                    f"Loading {n} cached extra trap pair stems from "
                    f"{self.pair_stems_path}"
                )
                return cached[:n]
        grounding = self._pairs._grounding_terms(concepts)
        collected = self._generate_direct_stems(
            n, grounding, blocked=blocked, batch_size=batch_size,
        )
        if len(collected) < n:
            raise ValueError(
                f"API returned {len(collected)} extra direct stems; need {n}."
            )
        self.pair_stems_path.parent.mkdir(parents=True, exist_ok=True)
        self.pair_stems_path.write_text(
            json.dumps({
                "contrast": TRAP_DIRECT_STEMS_CONTRAST,
                "domain": self.domain,
                "n": n,
                "model": self.generation_model,
                "stems": collected,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Cached {len(collected)} extra trap pair stems at "
                    f"{self.pair_stems_path}")
        return collected[:n]

    def _generate_direct_stems(self, n: int, grounding: Sequence[str],
                               blocked: Sequence[str],
                               batch_size: int) -> List[dict]:
        collected: List[dict] = []
        concept_lines = (
            "\n".join(f"- {term}" for term in grounding)
            if grounding else "- (none listed — invent overloaded terms "
            "from the domain name.)"
        )
        already_terms = list(blocked)
        while len(collected) < n:
            need = min(batch_size, n - len(collected))
            already = ""
            listed = already_terms + [row["question"] for row in collected]
            if listed:
                already = (
                    "Already written — do not repeat or paraphrase these "
                    "questions:\n"
                    + "\n".join(f"- {q}" for q in listed)
                    + "\n"
                )
            payload = request_json_completion(
                TRAP_DIRECT_STEMS_PROMPT.format(
                    domain=self.domain, n=need,
                    concept_lines=concept_lines, already=already,
                ),
                api_key=self.api_key, model=self.generation_model,
            )
            batch = _clean_direct_stems(
                payload.get("stems", []), self.domain,
                blocked=listed,
            )
            if not batch:
                break
            have = {_match_key(row["question"]) for row in collected}
            have.update(_match_key(row["term"]) for row in collected)
            for row in batch:
                q_key = _match_key(row["question"])
                t_key = _match_key(row["term"])
                if q_key in have or t_key in have:
                    continue
                collected.append(row)
                have.add(q_key)
                have.add(t_key)
                if len(collected) >= n:
                    break
        return collected

    def generate(self, n: int = DEFAULT_N_QUESTIONS,
                 force: bool = False,
                 concepts: Optional[Sequence[str]] = None,
                 batch_size: int = DEFAULT_TRAP_BATCH) -> List[dict]:
        n_terms = n_trap_terms(n)
        n_items = n_terms * 2
        if self.path.exists() and not force:
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if payload.get("contrast") not in (None, TRAP_EXAM_CONTRAST):
                logger.info(
                    f"Stale trap exam contrast {payload.get('contrast')!r} "
                    f"at {self.path}; regenerating as {TRAP_EXAM_CONTRAST}."
                )
            else:
                terms = _clean_trap_terms(payload.get("terms", []), self.domain)
                items = flatten_trap_terms(terms)
                if len(items) >= n_items:
                    logger.info(
                        f"Loading {n_items} cached trap items from {self.path}"
                    )
                    return items[:n_items]

        grounding = self._pairs._grounding_terms(concepts)
        terms = self._generate(n_terms, grounding, batch_size=batch_size)
        items = flatten_trap_terms(terms)
        if len(items) < n_items:
            raise ValueError(
                f"API returned {len(terms)} usable trap terms "
                f"({len(items)} items); need {n_terms} terms / {n_items} items."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({
                "contrast": TRAP_EXAM_CONTRAST,
                "domain": self.domain,
                "n_terms": n_terms,
                "n": n_items,
                "model": self.generation_model,
                "terms": terms,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Cached {len(terms)} trap terms "
                    f"({len(items)} items) at {self.path}")
        return items[:n_items]

    def _generate(self, n_terms: int, grounding: Sequence[str],
                  batch_size: int) -> List[dict]:
        collected: List[dict] = []
        concept_lines = (
            "\n".join(f"- {term}" for term in grounding)
            if grounding else "- (none listed — invent overloaded terms "
            "from the domain name.)"
        )
        while len(collected) < n_terms:
            need = min(batch_size, n_terms - len(collected))
            already = ""
            if collected:
                already = (
                    "Already written — do not repeat or paraphrase these "
                    "terms or questions:\n"
                    + "\n".join(
                        f"- {row['term']}: {row['direct_question']} / "
                        f"{row['tricky_question']}"
                        for row in collected
                    )
                    + "\n"
                )
            payload = request_json_completion(
                TRAP_ITEMS_PROMPT.format(
                    domain=self.domain, n=need,
                    concept_lines=concept_lines, already=already,
                    min_words=DEFAULT_TRAP_MIN_WORDS,
                    max_words=DEFAULT_TRAP_MAX_WORDS,
                ),
                api_key=self.api_key, model=self.generation_model,
            )
            batch = _clean_trap_terms(payload.get("terms", []), self.domain)
            if not batch:
                break
            have = {_match_key(row["term"]) for row in collected}
            have_q = {_match_key(row["direct_question"]) for row in collected}
            have_q.update(_match_key(row["tricky_question"]) for row in collected)
            for row in batch:
                term_key = _match_key(row["term"])
                d_key = _match_key(row["direct_question"])
                t_key = _match_key(row["tricky_question"])
                if term_key in have or d_key in have_q or t_key in have_q:
                    continue
                collected.append(row)
                have.add(term_key)
                have_q.add(d_key)
                have_q.add(t_key)
                if len(collected) >= n_terms:
                    break
        return collected


def _trap_pair_clouds(domain: str, cache_dir: Optional[Union[str, Path]]
                      ) -> tuple:
    gen = PairGenerator(domain, cache_dir=cache_dir, contrast=TRAP_CONTRAST_ID)
    if not gen.pairs_path.exists():
        return [], []
    pairs = load_pairs(gen.pairs_path)
    return ([p.expert_text for p in pairs],
            [p.nonexpert_text for p in pairs])


def _mean_floats(values) -> Optional[float]:
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def _pairwise_from_rows(rows: Sequence[dict]) -> dict:
    wins = sum((r.get("judge") or {}).get("winner") == "wins" for r in rows)
    losses = sum((r.get("judge") or {}).get("winner") == "loses" for r in rows)
    ties = sum((r.get("judge") or {}).get("winner") == "tie" for r in rows)
    decided = wins + losses
    return {
        "n": len(rows),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": (wins / decided) if decided else 0.5,
    }


def _kind_summary(rows: Sequence[dict]) -> dict:
    deg = [r.get("degeneration") or {} for r in rows]
    return {
        "n": len(rows),
        "pairwise": _pairwise_from_rows(rows),
        "gold_baseline": _mean_floats(
            (r.get("gold_score") or {}).get("baseline_gold") for r in rows),
        "gold_steered": _mean_floats(
            (r.get("gold_score") or {}).get("steered_gold") for r in rows),
        "repetition_baseline": _mean_floats(
            (d.get("baseline") or {}).get("repetition_rate") for d in deg),
        "repetition_steered": _mean_floats(
            (d.get("steered") or {}).get("repetition_rate") for d in deg),
        "collapsed_baseline": _mean_floats(
            float(bool((d.get("baseline") or {}).get("collapsed"))) for d in deg),
        "collapsed_steered": _mean_floats(
            float(bool((d.get("steered") or {}).get("collapsed"))) for d in deg),
        "n_words_baseline": _mean_floats(
            (d.get("baseline") or {}).get("n_words") for d in deg),
        "n_words_steered": _mean_floats(
            (d.get("steered") or {}).get("n_words") for d in deg),
    }


def grade_trap_run(domain: str,
                   items: Sequence[dict],
                   baselines: Sequence[str],
                   steered: Sequence[str],
                   alpha: float,
                   cache_dir: Optional[Union[str, Path]] = None,
                   api_key: Optional[str] = None,
                   generation_model: str = DEFAULT_GENERATION_MODEL,
                   similarity: bool = True,
                   llm_judge: bool = False,
                   tau: float = 0.50,
                   st_model: Optional[str] = None,
                   pair_path: Optional[Union[str, Path]] = None,
                   ) -> dict:
    """Gold cosine + pair-cloud cosine + degeneration. No LLM judge."""
    from extras.benchmark import (
        DEFAULT_ST_MODEL, _comparison_view, _gold_metrics, _mean_optional,
        score_against_gold, score_similarity,
    )
    from extras.llm_judge import PairwiseJudge

    questions = [item["question"] for item in items]
    golds = [item["answer"] for item in items]
    if not (len(questions) == len(baselines) == len(steered) == len(golds)):
        raise ValueError("items, baselines and steered must be the same length.")

    encoder = st_model or DEFAULT_ST_MODEL
    gold_rows = score_against_gold(
        baselines, steered, golds, tau=tau, st_model=encoder,
    )

    report = None
    comparisons = [None] * len(questions)
    if llm_judge:
        judge = PairwiseJudge(domain, api_key=api_key, model=generation_model,
                              prompt_template=TRAP_SENSE_PROMPT)
        report = judge.compare_all(questions, steered, baselines, golds=golds)
        comparisons = report.comparisons

    expert_texts, survey_texts = _trap_pair_clouds(domain, cache_dir)
    if pair_path:
        local_path = Path(pair_path)
        if local_path.exists():
            local_pairs = load_pairs(local_path)
            expert_texts = [p.expert_text for p in local_pairs]
            survey_texts = [p.nonexpert_text for p in local_pairs]
    sim_rows: List[dict] = [{} for _ in questions]
    if similarity and expert_texts:
        sim_rows = score_similarity(baselines, steered, expert_texts,
                                    survey_texts, api_key=api_key)

    scored = []
    for item, base, steer, gold_score, comparison, sim in zip(
            items, baselines, steered, gold_rows, comparisons, sim_rows):
        row = {
            "term": item.get("term"),
            "kind": item.get("kind"),
            "question": item["question"],
            "gold": item["answer"],
            "baseline": base,
            "steered": steer,
            "gold_score": gold_score,
            "similarity": sim,
            "degeneration": {
                "baseline": degeneration_metrics(base),
                "steered": degeneration_metrics(steer),
            },
        }
        if comparison is not None:
            row["judge"] = _comparison_view(comparison)
        scored.append(row)

    sim_mean = {
        key: _mean_optional(sim_rows, key)
        for key in ("baseline_in_domain", "steered_in_domain",
                    "baseline_survey", "steered_survey")
    }
    by_kind = {
        kind: _kind_summary([row for row in scored if row.get("kind") == kind])
        for kind in ("direct", "tricky")
        if any(row.get("kind") == kind for row in scored)
    }
    return {
        "domain": domain,
        "set": SET_TRAP,
        "n": len(questions),
        "n_terms": len({item.get("term") for item in items if item.get("term")}),
        "alpha": alpha,
        "questions_path": str(
            PairGenerator(domain, cache_dir=cache_dir,
                          contrast=TRAP_CONTRAST_ID).domain_dir
            / BENCHMARK_TRAP_NAME
        ),
        "pairwise": report.as_dict() if report is not None else {},
        "gold_metrics": _gold_metrics(gold_rows, tau=tau, st_model=encoder),
        "similarity_mean": sim_mean,
        "degeneration_mean": _kind_summary(scored),
        "by_kind": by_kind,
        "items": scored,
    }


def format_trap_benchmark(results: dict, include_items: bool = True) -> str:
    from extras.benchmark import _fmt_pair

    pairwise = results.get("pairwise") or {}
    metrics = results.get("gold_metrics") or {}
    sim = results.get("similarity_mean") or {}
    n_terms = results.get("n_terms") or ""
    lines = [
        f"Trap sense eval  domain={results['domain']}  "
        f"n={results['n']}  terms={n_terms}  alpha={results['alpha']:.3f}",
        f"Items: {results.get('questions_path', '')}",
        "",
    ]
    if pairwise:
        lines.append(
            f"LLM judge (steered vs baseline, {results['domain']} sense): "
            f"win rate {pairwise['win_rate'] * 100:.1f}%  "
            f"({pairwise['wins']}W/{pairwise['losses']}L/{pairwise['ties']}T, "
            f"consistency {pairwise['consistency'] * 100:.0f}%)"
        )
    if metrics.get("steered_mean") is not None:
        lines.append(_fmt_pair("MPNet similarity (gold)",
                               metrics.get("baseline_mean"),
                               metrics.get("steered_mean")))
    if sim.get("steered_in_domain") is not None:
        lines.append(
            "Embedding cosine to in-domain trap-pair cloud: "
            f"baseline {sim.get('baseline_in_domain') or 0:.3f} → "
            f"steered {sim['steered_in_domain']:.3f}"
        )
    if sim.get("steered_survey") is not None:
        lines.append(
            "Embedding cosine to survey trap-pair cloud: "
            f"baseline {sim.get('baseline_survey') or 0:.3f} → "
            f"steered {sim['steered_survey']:.3f}"
        )
    deg = results.get("degeneration_mean") or {}
    if deg.get("repetition_steered") is not None:
        lines.append(
            "Repetition % (word 3-gram): "
            f"baseline {deg.get('repetition_baseline') or 0:.1f} → "
            f"steered {deg['repetition_steered']:.1f}   "
            f"collapsed {deg.get('collapsed_baseline') or 0:.0%} → "
            f"{deg.get('collapsed_steered') or 0:.0%}"
        )
    for kind, slice_ in (results.get("by_kind") or {}).items():
        pw = slice_.get("pairwise") or {}
        label = "ambiguous terms" if kind == "direct" else "technical reasoning"
        lines.append(
            f"{kind} ({label}, n={slice_.get('n', 0)}): "
            f"sense {pw.get('win_rate', 0) * 100:.1f}%  "
            f"({pw.get('wins', 0)}W/{pw.get('losses', 0)}L/"
            f"{pw.get('ties', 0)}T)  "
            f"gold {slice_.get('gold_baseline') or 0:.3f} → "
            f"{slice_.get('gold_steered') or 0:.3f}  "
            f"collapsed {slice_.get('collapsed_steered') or 0:.0%}"
        )
    if not include_items:
        return "\n".join(lines)
    lines.append("")
    for i, item in enumerate(results["items"], 1):
        kind = item.get("kind") or ""
        term = item.get("term") or ""
        label = f"{kind} / {term}".strip(" /")
        lines += [
            f"----- {i}/{results['n']}  {label} -----",
            f"Q: {item['question']}",
            "",
            "GOLD:",
            item.get("gold") or "",
            "",
            "BASELINE:",
            item["baseline"],
            "",
            "STEERED:",
            item["steered"],
            "",
        ]
        judge = item.get("judge")
        if judge:
            lines.append(f"judge: steered {judge['winner']}  ({judge['reason']})")
        gold = item.get("gold_score") or {}
        if gold.get("steered_gold") is not None:
            lines.append(
                f"gold cosine  base={gold['baseline_gold']:.3f}  "
                f"steered={gold['steered_gold']:.3f}"
            )
        s = item.get("similarity") or {}
        if s.get("steered_in_domain") is not None:
            lines.append(
                f"sim in-domain  base={s['baseline_in_domain']:.3f}  "
                f"steered={s['steered_in_domain']:.3f}   "
                f"survey  base={s.get('baseline_survey') or 0:.3f}  "
                f"steered={s.get('steered_survey') or 0:.3f}"
            )
        deg = item.get("degeneration") or {}
        bdeg, sdeg = deg.get("baseline") or {}, deg.get("steered") or {}
        if sdeg:
            lines.append(
                f"degen  words {bdeg.get('n_words', 0)}→{sdeg.get('n_words', 0)}  "
                f"rep% {bdeg.get('repetition_rate', 0):.0f}→"
                f"{sdeg.get('repetition_rate', 0):.0f}  "
                f"collapsed {bool(bdeg.get('collapsed'))}→"
                f"{bool(sdeg.get('collapsed'))}"
            )
        lines.append("")
    return "\n".join(lines)

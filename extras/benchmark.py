"""Open-ended benchmarks: unnamed identity stems, or per-domain gold items.

`--set unnamed` (default) is a shared bank of field-neutral questions cached
at the cache root. Every domain is tested on the same stems. An LLM judge
scores domain identity, and embedding cosine measures closeness to the
in-domain vs survey pair clouds.

`--set gold` is a per-domain exam: specialist questions plus reference
answers. Grading is a hybrid pipeline whose metrics are reported separately
(not averaged into one headline number):

1. Sentence-transformer cosine to the gold answer, and to the item's concept
   set (terms named on the item plus lexicon terms evidenced in the gold).
2. Concept coverage: fraction of that set present in the candidate.
3. Dimensional LLM rubric (0-5) for expertise — no gold in the prompt.
   Gold is used only for embedding similarity and concept coverage.
4. Pairwise LLM comparison of baseline vs steered (expertise, not
   reference-matching).
5. Open gold: pairwise LLM with the gold key in the prompt (which answer is
   closer to the reference), blended 0.60 MPNet gold-cosine + 0.40 judge.

An optional weighted blend is stored in JSON for convenience; the printed
table is the four metrics above plus open gold.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from domainsteer.pairs import (DEFAULT_GENERATION_MODEL, PairGenerator,
                               _match_key, load_pairs, request_json_completion)

logger = logging.getLogger(__name__)

DEFAULT_N_QUESTIONS = 50
BENCHMARK_QUESTIONS_NAME = "benchmark_questions.json"
BENCHMARK_GOLD_NAME = "benchmark_gold.json"
GOLD_CONTRAST = "open_gold_v1"
SET_UNNAMED = "unnamed"
SET_GOLD = "gold"
SET_TRAP = "trap"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_ST_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_GOLD_TAU = 0.50
DEFAULT_CONCEPT_TAU = 0.40
DEFAULT_GOLD_BATCH = 10
DEFAULT_GOLD_MIN_WORDS = 40
DEFAULT_GOLD_MAX_WORDS = 80
MIN_CONCEPT_CHARS = 4
RUBRIC_KEYS = (
    "technical_accuracy",
    "domain_expertise",
    "analytical_approach",
)
# Optional JSON blend only — not the printed headline.
COMPOSITE_WEIGHTS = {
    "embedding": 0.30,
    "llm_expert": 0.40,
    "pairwise": 0.30,
}
OPEN_GOLD_WEIGHTS = {
    "embedding": 0.60,
    "llm": 0.40,
}

QUESTIONS_PROMPT = """\
Write {n} distinct open-ended questions that a practitioner in ANY specialist \
field could be asked.

Requirements:
- Do not name any discipline, industry, method, or school (not science, art, \
medicine, engineering, business, transport, hydrology, zoology, chemistry, \
or any other field).
- Refer to the unnamed domain as "this field" or "this work" when a noun is \
needed.
- Each question must admit a different expert reading in unrelated fields \
(for example water science, animal biology, performing arts, and logistics).
- One sentence per question. No yes/no questions, no multiple choice, no \
lists of sub-questions.
- Questions must not be paraphrases of each other.

{already}

Return strict JSON:
{{"questions": ["...", "..."]}}"""

GOLD_ITEMS_PROMPT = """\
You are a highly knowledgeable {domain} expert writing an open-ended exam \
with answer keys.

Write {n} items. Each item is a question a practitioner must reason about, \
plus a correct answer.

Requirements for questions:
- Unmistakably about {domain}. Use its vocabulary.
- Require a mechanism, diagnostic, trade-off, or measurement.
- One sentence. No yes/no, no multiple choice, no stacked sub-questions.
- Not a definition ("What is X?"). Not "what do outsiders get wrong".
- Not paraphrases of each other.
- Draw on these concepts where they fit; do not write one "What is X?" per \
bullet:
{concept_lines}

Requirements for answers:
- {min_words}-{max_words} words. State the field's correct reading.
- Use mechanisms, measurements, and practice. Do not mention other disciplines.
- Do not say you are a generalist. Do not list unrelated fields.
- The answer must actually resolve the question, not restate it.

{already}

Return strict JSON:
{{"items": [{{"question": "...", "answer": "...", "concepts": ["..."]}}, ...]}}"""

ADHERENCE_PROMPT = """\
You are evaluating which of two answers treats the unnamed field in the \
question as {domain}.

The question does not name a discipline. A committed answer interprets \
"this field" or "this work" as {domain} and stays inside it. A survey \
answer lists several unrelated fields, asks which field was meant, or \
answers as a generalist. Committing to a different specialist field \
(not {domain}) also loses.

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Judge only on domain identity — not fluency, length, or formatting.

Return strict JSON:
{{"winner": "A" or "B" or "tie", "reason": "<one sentence>"}}"""

CONCEPT_GAP_PROMPT = """\
Domain: {domain}

Registry concepts (the official partition of this domain):
{concept_lines}

Steered answers to unnamed questions (the model was not told the domain name):
{answer_block}

List specialist concept names that (1) are clearly used in the steered \
answers, (2) belong inside {domain}, and (3) are NOT already in the registry \
list above (ignore case and minor wording). Do not invent concepts that are \
not evidenced. Do not list other disciplines.

Return strict JSON:
{{"missing": [{{"name": "...", "evidence": "<short quote or paraphrase>"}}]}}"""

EXPERT_RUBRIC_PROMPT = """\
You are an expert {domain} practitioner.

Evaluate the candidate answer independently. Do not score wording similarity \
to any particular solution, and do not assume there is only one valid method. \
Alternative valid {domain} approaches should be rewarded. Use the full 0-5 \
range; the intermediate scores are defined below.

Question:
{question}

Candidate answer:
{response}

Score each category from 0 to 5.

technical_accuracy — is the content factually right for {domain}?
0 = incorrect
1 = mostly incorrect
2 = partially correct
3 = substantially correct
4 = correct with minor omissions
5 = expert-level: precise and reliable

domain_expertise — appropriate use of {domain} concepts, terminology, and theory.
0 = absent
1 = lay vocabulary, no field concepts
2 = a few field terms used loosely
3 = appropriate concepts with gaps in theory
4 = solid use of concepts, terminology, and theory
5 = expert-level

analytical_approach — quality of the diagnostic process, measurements, \
calculations, and interpretation.
0 = absent
1 = no diagnostic process
2 = a method named without justification
3 = a plausible diagnostic path with gaps
4 = coherent measurements, calculations, and interpretation
5 = expert-level

overall is the mean of the three, rounded to one decimal.

Return strict JSON:
{{"technical_accuracy": 0, "domain_expertise": 0, \
"analytical_approach": 0, "overall": 0.0, \
"explanation": "<one sentence>"}}"""

GOLD_PAIRWISE_PROMPT = """\
You are an expert {domain} practitioner.

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Which answer demonstrates stronger {domain} expertise?

Consider correctness, use of {domain} concepts, and the quality of the \
diagnostic or analytical approach. Alternative valid methods both count as \
expert. Ignore length, formatting, wording overlap, and which answer appears \
first. A fluent but wrong answer loses to a vaguer but correct one.

Return strict JSON:
{{"winner": "A" or "B" or "tie", "reason": "<one sentence>"}}"""


OPEN_GOLD_PROMPT = """\
You are evaluating which of two {domain} answers is closer to the gold answer.

Domain: {domain}

Question:
{question}

Gold answer:
{gold}

Answer A:
{answer_a}

Answer B:
{answer_b}

Judge only closeness to the gold answer — the same mechanisms, measurements, \
and conclusions. Alternative wording is fine if the content matches. Ignore \
length, formatting, and which answer appears first.

Return strict JSON:
{{"winner": "A" or "B" or "tie", "reason": "<one sentence>"}}"""


def _questions_path(cache_dir: Optional[Union[str, Path]]) -> Path:
    base = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "domainsteer"
    return base / BENCHMARK_QUESTIONS_NAME


def _clean_questions(raw) -> List[str]:
    if not isinstance(raw, list):
        return []
    seen = set()
    out: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split()).strip()
        if len(text) < 20 or "?" not in text:
            continue
        key = _match_key(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _clean_gold_items(raw) -> List[dict]:
    """Keep items with a real question and a long-enough reference answer."""
    if not isinstance(raw, list):
        return []
    seen = set()
    out: List[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        question = " ".join(str(item.get("question") or "").split()).strip()
        answer = " ".join(str(item.get("answer") or "").split()).strip()
        if len(question) < 20 or "?" not in question:
            continue
        if len(answer.split()) < 20:
            continue
        key = _match_key(question)
        if key in seen:
            continue
        seen.add(key)
        concepts = item.get("concepts") or []
        if isinstance(concepts, str):
            concepts = [concepts]
        names = [str(c).strip() for c in concepts if str(c).strip()]
        out.append({"question": question, "answer": answer, "concepts": names})
    return out


class BenchmarkQuestions:
    """Generate and cache a shared bank of unnamed open-ended stems."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None,
                 api_key: Optional[str] = None,
                 generation_model: str = DEFAULT_GENERATION_MODEL):
        self.cache_dir = cache_dir
        self.api_key = api_key
        self.generation_model = generation_model
        self.path = _questions_path(cache_dir)

    def generate(self, n: int = DEFAULT_N_QUESTIONS,
                 force: bool = False) -> List[str]:
        if n < 1:
            raise ValueError(f"Need at least 1 question, got {n}.")
        if self.path.exists() and not force:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            questions = _clean_questions(payload.get("questions", []))
            if len(questions) >= n:
                logger.info(f"Loading {n} cached benchmark questions from {self.path}")
                return questions[:n]

        questions = self._generate(n)
        if len(questions) < n:
            raise ValueError(
                f"API returned {len(questions)} usable questions; need {n}."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"n": n, "model": self.generation_model,
                        "questions": questions}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Cached {len(questions)} benchmark questions at {self.path}")
        return questions[:n]

    def _generate(self, n: int) -> List[str]:
        collected: List[str] = []
        while len(collected) < n:
            need = n - len(collected)
            already = ""
            if collected:
                already = (
                    "Already written — do not repeat or paraphrase:\n"
                    + "\n".join(f"- {q}" for q in collected)
                    + "\n"
                )
            payload = request_json_completion(
                QUESTIONS_PROMPT.format(n=need, already=already),
                api_key=self.api_key, model=self.generation_model,
            )
            batch = _clean_questions(payload.get("questions", []))
            if not batch:
                break
            have = {_match_key(q) for q in collected}
            for question in batch:
                if _match_key(question) not in have:
                    collected.append(question)
                    have.add(_match_key(question))
                if len(collected) >= n:
                    break
        return collected


class BenchmarkGold:
    """Per-domain open-ended items with reference answers, cached under the
    domain slug so Zoology and hydrology never share a bank."""

    def __init__(self, domain: str,
                 cache_dir: Optional[Union[str, Path]] = None,
                 api_key: Optional[str] = None,
                 generation_model: str = DEFAULT_GENERATION_MODEL):
        self.domain = domain
        self.api_key = api_key
        self.generation_model = generation_model
        self._pairs = PairGenerator(domain, cache_dir=cache_dir, api_key=api_key,
                                    generation_model=generation_model)
        self.path = self._pairs.domain_dir / BENCHMARK_GOLD_NAME

    def generate(self, n: int = DEFAULT_N_QUESTIONS,
                 force: bool = False,
                 concepts: Optional[Sequence[str]] = None,
                 batch_size: int = DEFAULT_GOLD_BATCH) -> List[dict]:
        if n < 1:
            raise ValueError(f"Need at least 1 item, got {n}.")
        if self.path.exists() and not force:
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if payload.get("contrast") not in (None, GOLD_CONTRAST):
                raise ValueError(
                    f"Cached gold items at {self.path} were built under "
                    f"contrast {payload.get('contrast')!r}, not {GOLD_CONTRAST}. "
                    "Pass --force-questions to rebuild."
                )
            items = _clean_gold_items(payload.get("items", []))
            if len(items) >= n:
                logger.info(f"Loading {n} cached gold items from {self.path}")
                return items[:n]

        terms = self._pairs._grounding_terms(concepts)
        items = self._generate(n, terms, batch_size=batch_size)
        if len(items) < n:
            raise ValueError(
                f"API returned {len(items)} usable gold items; need {n}."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({
                "contrast": GOLD_CONTRAST,
                "domain": self.domain,
                "n": n,
                "model": self.generation_model,
                "items": items,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Cached {len(items)} gold items at {self.path}")
        return items[:n]

    def _generate(self, n: int, terms: Sequence[str],
                  batch_size: int) -> List[dict]:
        collected: List[dict] = []
        concept_lines = (
            "\n".join(f"- {term}" for term in terms)
            if terms else "- (none listed — use the domain name.)"
        )
        while len(collected) < n:
            need = min(batch_size, n - len(collected))
            already = ""
            if collected:
                already = (
                    "Already written — do not repeat or paraphrase these "
                    "questions:\n"
                    + "\n".join(f"- {item['question']}" for item in collected)
                    + "\n"
                )
            payload = request_json_completion(
                GOLD_ITEMS_PROMPT.format(
                    domain=self.domain, n=need,
                    concept_lines=concept_lines, already=already,
                    min_words=DEFAULT_GOLD_MIN_WORDS,
                    max_words=DEFAULT_GOLD_MAX_WORDS,
                ),
                api_key=self.api_key, model=self.generation_model,
            )
            batch = _clean_gold_items(payload.get("items", []))
            if not batch:
                break
            have = {_match_key(item["question"]) for item in collected}
            for item in batch:
                key = _match_key(item["question"])
                if key not in have:
                    collected.append(item)
                    have.add(key)
                if len(collected) >= n:
                    break
        return collected


def embed_texts(texts: Sequence[str], api_key: Optional[str] = None,
                model: str = EMBEDDING_MODEL) -> List[np.ndarray]:
    """OpenAI embeddings for `texts`, one vector per string."""
    import os
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "Embeddings need the openai package: pip install -e \".[generation]\""
        ) from e
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("No OpenAI API key for embeddings.")
    if not texts:
        return []
    client = OpenAI(api_key=key)
    response = client.embeddings.create(model=model, input=list(texts))
    by_index = {item.index: item.embedding for item in response.data}
    return [np.array(by_index[i], dtype=np.float32) for i in range(len(texts))]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def mean_vector(vectors: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    if not vectors:
        return None
    stacked = np.stack(vectors)
    return stacked.mean(axis=0)


_st_models: Dict[str, object] = {}


def embed_sentence_transformer(
        texts: Sequence[str],
        model: str = DEFAULT_ST_MODEL) -> List[np.ndarray]:
    """Local sentence-transformer embeddings, L2-normalized."""
    if not texts:
        return []
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Gold scoring needs sentence-transformers: pip install -e \".[eval]\""
        ) from e
    encoder = _st_models.get(model)
    if encoder is None:
        encoder = SentenceTransformer(model)
        _st_models[model] = encoder
    vectors = encoder.encode(list(texts), normalize_embeddings=True,
                             show_progress_bar=False)
    return [np.asarray(row, dtype=np.float32) for row in vectors]


def score_against_gold(baselines: Sequence[str], steered: Sequence[str],
                       golds: Sequence[str], tau: float = DEFAULT_GOLD_TAU,
                       st_model: str = DEFAULT_ST_MODEL) -> List[dict]:
    """Per-answer cosine to the gold reference, plus thresholded accuracy."""
    unique: List[str] = []
    index: Dict[str, int] = {}

    def add(text: str) -> None:
        if text not in index:
            index[text] = len(unique)
            unique.append(text)

    for text in list(baselines) + list(steered) + list(golds):
        add(text)
    vectors = embed_sentence_transformer(unique, model=st_model)
    rows = []
    for base, steer, gold in zip(baselines, steered, golds):
        s_base = cosine(vectors[index[base]], vectors[index[gold]])
        s_steer = cosine(vectors[index[steer]], vectors[index[gold]])
        rows.append({
            "baseline_gold": s_base,
            "steered_gold": s_steer,
            "baseline_correct": s_base >= tau,
            "steered_correct": s_steer >= tau,
            "steered_closer": s_steer > s_base,
        })
    return rows


def _domain_lexicon(domain: str,
                    cache_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """Grounding terms plus registry concept names for this domain."""
    names: List[str] = []
    gen = PairGenerator(domain, cache_dir=cache_dir)
    names.extend(gen._grounding_terms(None))
    names.extend(_registry_concepts(domain))
    seen = set()
    out = []
    for name in names:
        text = str(name).strip()
        key = _match_key(text)
        if len(text) < MIN_CONCEPT_CHARS or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def resolve_item_concepts(item: dict, lexicon: Sequence[str]) -> List[str]:
    """Question-level concept set: named concepts plus lexicon terms in gold."""
    named = [str(c).strip() for c in (item.get("concepts") or []) if str(c).strip()]
    haystack = _match_key(
        f"{item.get('answer') or ''} {item.get('question') or ''}"
    )
    extra = []
    for term in sorted(lexicon, key=len, reverse=True):
        key = _match_key(term)
        if len(term) < MIN_CONCEPT_CHARS or key not in haystack:
            continue
        if any(key in _match_key(s) and key != _match_key(s) for s in extra + named):
            continue
        extra.append(term)
    seen = {_match_key(t) for t in named}
    out = list(named)
    for term in extra:
        key = _match_key(term)
        if key not in seen:
            seen.add(key)
            out.append(term)
    return out


def _concept_covered(concept: str, answer: str, answer_vec: np.ndarray,
                     concept_vec: np.ndarray, tau: float) -> bool:
    if _match_key(concept) in _match_key(answer):
        return True
    return cosine(answer_vec, concept_vec) >= tau


def _embed_index(texts: Sequence[str],
                 st_model: str) -> Tuple[Dict[str, int], List[np.ndarray]]:
    unique: List[str] = []
    index: Dict[str, int] = {}
    for text in texts:
        if text not in index:
            index[text] = len(unique)
            unique.append(text)
    return index, embed_sentence_transformer(unique, model=st_model)


def score_gold_similarity(baselines: Sequence[str], steered: Sequence[str],
                          golds: Sequence[str],
                          concept_lists: Sequence[Sequence[str]],
                          tau: float = DEFAULT_GOLD_TAU,
                          concept_tau: float = DEFAULT_CONCEPT_TAU,
                          st_model: str = DEFAULT_ST_MODEL) -> List[dict]:
    """Stage 1: gold cosine, concept-cloud cosine, and concept coverage."""
    concept_texts = [c for concepts in concept_lists for c in concepts]
    index, vectors = _embed_index(
        list(baselines) + list(steered) + list(golds) + concept_texts,
        st_model,
    )
    rows = []
    for base, steer, gold, concepts in zip(
            baselines, steered, golds, concept_lists):
        base_vec = vectors[index[base]]
        steer_vec = vectors[index[steer]]
        gold_vec = vectors[index[gold]]
        s_base = cosine(base_vec, gold_vec)
        s_steer = cosine(steer_vec, gold_vec)
        row = {
            "baseline_gold": s_base,
            "steered_gold": s_steer,
            "baseline_correct": s_base >= tau,
            "steered_correct": s_steer >= tau,
            "steered_closer": s_steer > s_base,
            "baseline_concepts": None,
            "steered_concepts": None,
            "baseline_coverage": None,
            "steered_coverage": None,
            "concepts": list(concepts),
            "baseline_covered": [],
            "steered_covered": [],
        }
        if concepts:
            cloud = mean_vector([vectors[index[c]] for c in concepts])
            if cloud is not None:
                row["baseline_concepts"] = cosine(base_vec, cloud)
                row["steered_concepts"] = cosine(steer_vec, cloud)
            covered_base, covered_steer = [], []
            for concept in concepts:
                cvec = vectors[index[concept]]
                if _concept_covered(concept, base, base_vec, cvec, concept_tau):
                    covered_base.append(concept)
                if _concept_covered(concept, steer, steer_vec, cvec, concept_tau):
                    covered_steer.append(concept)
            row["baseline_covered"] = covered_base
            row["steered_covered"] = covered_steer
            n_c = len(concepts)
            row["baseline_coverage"] = len(covered_base) / n_c
            row["steered_coverage"] = len(covered_steer) / n_c
        rows.append(row)
    return rows


def _pair_clouds(domain: str, cache_dir: Optional[Union[str, Path]]
                 ) -> Tuple[List[str], List[str]]:
    gen = PairGenerator(domain, cache_dir=cache_dir)
    if not gen.pairs_path.exists():
        return [], []
    pairs = load_pairs(gen.pairs_path)
    return ([p.expert_text for p in pairs],
            [p.nonexpert_text for p in pairs])


def _registry_concepts(domain: str) -> List[str]:
    try:
        from domainsteer.clusters import find_cluster, load_clusters
        cluster = find_cluster(load_clusters(), domain)
    except (OSError, ValueError):
        return []
    return [c.name for c in cluster.concepts if c.name.strip()]


def concept_gaps(domain: str, steered: Sequence[str],
                 concepts: Sequence[str],
                 api_key: Optional[str] = None,
                 generation_model: str = DEFAULT_GENERATION_MODEL) -> List[dict]:
    """Concepts evidenced in steered answers but absent from the registry."""
    if not steered or not concepts:
        return []
    payload = request_json_completion(
        CONCEPT_GAP_PROMPT.format(
            domain=domain,
            concept_lines="\n".join(f"- {name}" for name in concepts),
            answer_block="\n\n".join(
                f"{i}. {text[:500]}" for i, text in enumerate(steered, 1)
            ),
        ),
        api_key=api_key, model=generation_model,
    )
    missing = payload.get("missing", [])
    if not isinstance(missing, list):
        return []
    known = {_match_key(name) for name in concepts}
    out = []
    for item in missing:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or _match_key(name) in known:
            continue
        out.append({
            "name": name,
            "evidence": str(item.get("evidence") or "").strip(),
        })
    return out


def score_similarity(baselines: Sequence[str], steered: Sequence[str],
                     expert_texts: Sequence[str], survey_texts: Sequence[str],
                     api_key: Optional[str] = None,
                     ) -> List[dict]:
    """Per-answer cosine to the in-domain and survey pair clouds."""
    unique: List[str] = []
    index: Dict[str, int] = {}

    def add(text: str) -> None:
        if text not in index:
            index[text] = len(unique)
            unique.append(text)

    for text in list(baselines) + list(steered) + list(expert_texts) + list(survey_texts):
        add(text)
    vectors = embed_texts(unique, api_key=api_key)
    expert_mean = mean_vector([vectors[index[t]] for t in expert_texts])
    survey_mean = mean_vector([vectors[index[t]] for t in survey_texts])
    rows = []
    for base, steer in zip(baselines, steered):
        row = {"baseline_in_domain": None, "steered_in_domain": None,
               "baseline_survey": None, "steered_survey": None}
        if expert_mean is not None:
            row["baseline_in_domain"] = cosine(vectors[index[base]], expert_mean)
            row["steered_in_domain"] = cosine(vectors[index[steer]], expert_mean)
        if survey_mean is not None:
            row["baseline_survey"] = cosine(vectors[index[base]], survey_mean)
            row["steered_survey"] = cosine(vectors[index[steer]], survey_mean)
        rows.append(row)
    return rows


def format_item(index: int, n: int, question: str, baseline: str,
                steered: str, alpha: float) -> str:
    """One question's live inference dump."""
    return (
        f"===== {index}/{n} =====\n"
        f"Q: {question}\n\n"
        f"--- BASELINE ---\n{baseline}\n\n"
        f"--- STEERED (alpha={alpha:.3f}) ---\n{steered}"
    )


def format_benchmark(results: dict, include_items: bool = True) -> str:
    """Per-question baseline/steered dump plus summary scores."""
    lines = [
        f"Benchmark  domain={results['domain']}  "
        f"n={results['n']}  alpha={results['alpha']:.3f}",
        f"Questions: {results['questions_path']}",
        "",
    ]
    pairwise = results.get("pairwise") or {}
    if pairwise:
        lines.append(
            f"LLM judge (steered vs baseline, domain identity): "
            f"win rate {pairwise['win_rate'] * 100:.1f}%  "
            f"({pairwise['wins']}W/{pairwise['losses']}L/{pairwise['ties']}T, "
            f"consistency {pairwise['consistency'] * 100:.0f}%)"
        )
    sim = results.get("similarity_mean") or {}
    if sim.get("steered_in_domain") is not None:
        lines.append(
            "Embedding cosine to in-domain pair cloud: "
            f"baseline {sim.get('baseline_in_domain') or 0:.3f} → "
            f"steered {sim['steered_in_domain']:.3f}"
        )
    if sim.get("steered_survey") is not None:
        lines.append(
            "Embedding cosine to survey pair cloud: "
            f"baseline {sim.get('baseline_survey') or 0:.3f} → "
            f"steered {sim['steered_survey']:.3f}"
        )
    missing = results.get("missing_concepts") or []
    if missing:
        lines.append("Concepts in steered answers not in the registry:")
        for item in missing:
            lines.append(f"  - {item['name']}: {item['evidence']}")
    else:
        lines.append("No extra registry-level concepts proposed from steered answers.")
    if not include_items:
        return "\n".join(lines)
    lines.append("")
    for i, item in enumerate(results["items"], 1):
        lines += [
            f"----- {i}/{results['n']} -----",
            f"Q: {item['question']}",
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
        s = item.get("similarity") or {}
        if s.get("steered_in_domain") is not None:
            lines.append(
                f"sim in-domain  base={s['baseline_in_domain']:.3f}  "
                f"steered={s['steered_in_domain']:.3f}   "
                f"survey  base={s.get('baseline_survey') or 0:.3f}  "
                f"steered={s.get('steered_survey') or 0:.3f}"
            )
        lines.append("")
    return "\n".join(lines)


def _mean_optional(rows: Sequence[dict], key: str) -> Optional[float]:
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(sum(values) / len(values)) if values else None


def _clamp_score(value, lo: float = 0.0, hi: float = 5.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, number))


def _parse_rubric(payload: dict) -> dict:
    scores = {key: _clamp_score(payload.get(key)) for key in RUBRIC_KEYS}
    overall = payload.get("overall")
    if overall is None or overall == "":
        scores["overall"] = round(sum(scores.values()) / len(RUBRIC_KEYS), 1)
    else:
        scores["overall"] = _clamp_score(overall)
    scores["explanation"] = str(payload.get("explanation") or "").strip()
    return scores


def score_expert_rubric(domain: str, questions: Sequence[str],
                        responses: Sequence[str],
                        api_key: Optional[str] = None,
                        generation_model: str = DEFAULT_GENERATION_MODEL
                        ) -> List[dict]:
    """Stage 2: 0-5 expertise scores. Gold is not shown to this judge."""
    rows = []
    for question, response in zip(questions, responses):
        payload = request_json_completion(
            EXPERT_RUBRIC_PROMPT.format(
                domain=domain, question=question, response=response,
            ),
            api_key=api_key, model=generation_model,
        )
        if not isinstance(payload, dict):
            payload = {}
        rows.append(_parse_rubric(payload))
    return rows


def grade_run(domain: str,
              questions: Sequence[str],
              baselines: Sequence[str],
              steered: Sequence[str],
              alpha: float,
              cache_dir: Optional[Union[str, Path]] = None,
              api_key: Optional[str] = None,
              generation_model: str = DEFAULT_GENERATION_MODEL,
              similarity: bool = True,
              llm_judge: bool = True,
              ) -> dict:
    """LLM identity judge + embedding scores for an already-generated pair set.

    The pairwise prompt asks which answer treats the unnamed field as
    ``domain`` (committed vs survey), not whether it matches a gold key.
    """
    from extras.llm_judge import PairwiseJudge

    if not (len(questions) == len(baselines) == len(steered)):
        raise ValueError("questions, baselines and steered must be the same length.")

    report = None
    comparisons = [None] * len(questions)
    if llm_judge:
        judge = PairwiseJudge(domain, api_key=api_key, model=generation_model,
                              prompt_template=ADHERENCE_PROMPT)
        report = judge.compare_all(questions, steered, baselines)
        comparisons = report.comparisons

    expert_texts, survey_texts = _pair_clouds(domain, cache_dir)
    sim_rows: List[dict] = [{} for _ in questions]
    if similarity and expert_texts:
        sim_rows = score_similarity(baselines, steered, expert_texts,
                                    survey_texts, api_key=api_key)

    concepts = _registry_concepts(domain)
    missing: List[dict] = []
    if llm_judge:
        missing = concept_gaps(domain, steered, concepts, api_key=api_key,
                               generation_model=generation_model)

    items = []
    for question, base, steer, comparison, sim in zip(
            questions, baselines, steered, comparisons, sim_rows):
        row = {
            "question": question,
            "baseline": base,
            "steered": steer,
            "similarity": sim,
        }
        if comparison is not None:
            reason = comparison.reasons[0] if comparison.winner == "a" else comparison.reasons[1]
            if comparison.winner == "tie":
                reason = comparison.reasons[0] or comparison.reasons[1]
            row["judge"] = {
                "winner": {"a": "wins", "b": "loses", "tie": "tie"
                           }.get(comparison.winner, "tie"),
                "consistent": comparison.consistent,
                "reason": reason,
            }
        items.append(row)

    sim_mean = {
        key: _mean_optional(sim_rows, key)
        for key in ("baseline_in_domain", "steered_in_domain",
                    "baseline_survey", "steered_survey")
    }
    return {
        "domain": domain,
        "set": SET_UNNAMED,
        "n": len(questions),
        "alpha": alpha,
        "questions_path": str(_questions_path(cache_dir)),
        "pairwise": report.as_dict() if report is not None else {},
        "similarity_mean": sim_mean,
        "missing_concepts": missing,
        "items": items,
    }


def _gold_metrics(rows: Sequence[dict], tau: float, st_model: str,
                  concept_tau: float = DEFAULT_CONCEPT_TAU) -> dict:
    n = len(rows)
    closer = sum(1 for row in rows if row.get("steered_closer"))
    return {
        "tau": tau,
        "concept_tau": concept_tau,
        "st_model": st_model,
        "baseline_mean": _mean_optional(rows, "baseline_gold"),
        "steered_mean": _mean_optional(rows, "steered_gold"),
        "baseline_concept_sim": _mean_optional(rows, "baseline_concepts"),
        "steered_concept_sim": _mean_optional(rows, "steered_concepts"),
        "baseline_coverage": _mean_optional(rows, "baseline_coverage"),
        "steered_coverage": _mean_optional(rows, "steered_coverage"),
        "baseline_accuracy": (
            sum(1 for row in rows if row.get("baseline_correct")) / n
            if n else None
        ),
        "steered_accuracy": (
            sum(1 for row in rows if row.get("steered_correct")) / n
            if n else None
        ),
        "steered_closer": closer,
        "steered_closer_rate": closer / n if n else None,
    }


def _rubric_means(rows: Sequence[dict]) -> dict:
    if not rows:
        return {}
    keys = list(RUBRIC_KEYS) + ["overall"]
    return {key: _mean_optional(rows, key) for key in keys}


def _pairwise_win_score(comparison) -> float:
    if comparison is None:
        return None
    return {"a": 1.0, "b": 0.0}.get(comparison.winner, 0.5)


def _comparison_view(comparison) -> dict:
    reason = comparison.reasons[0] if comparison.winner == "a" else comparison.reasons[1]
    if comparison.winner == "tie":
        reason = comparison.reasons[0] or comparison.reasons[1]
    return {
        "winner": {"a": "wins", "b": "loses", "tie": "tie"
                   }.get(comparison.winner, "tie"),
        "consistent": comparison.consistent,
        "reason": reason,
    }


def _open_gold_score(cosine, llm_score) -> Optional[float]:
    """0.60 MPNet gold cosine + 0.40 open-gold pairwise (1/0.5/0)."""
    if cosine is None and llm_score is None:
        return None
    if cosine is None:
        return float(llm_score)
    if llm_score is None:
        return float(cosine)
    return (OPEN_GOLD_WEIGHTS["embedding"] * float(cosine)
            + OPEN_GOLD_WEIGHTS["llm"] * float(llm_score))


def apply_open_gold(results: dict,
                    api_key: Optional[str] = None,
                    generation_model: str = DEFAULT_GENERATION_MODEL) -> dict:
    """Add or replace the open-gold judge on an already-scored gold run."""
    from extras.llm_judge import PairwiseJudge

    items = results.get("items") or []
    if not items:
        raise ValueError("open gold needs items with question, gold, baseline, steered.")
    questions, golds, baselines, steered = [], [], [], []
    for item in items:
        question = item.get("question")
        gold = item.get("gold")
        base = item.get("baseline")
        steer = item.get("steered")
        if not (question and gold and base and steer):
            raise ValueError("open gold needs question, gold, baseline, and steered "
                             "on every item.")
        questions.append(question)
        golds.append(gold)
        baselines.append(base)
        steered.append(steer)

    domain = results.get("domain") or ""
    logger.info(f"Open gold: judging {len(items)} items against the gold key")
    print("Open gold: which answer is closer to the gold key...\n", flush=True)
    judge = PairwiseJudge(domain, api_key=api_key, model=generation_model,
                          prompt_template=OPEN_GOLD_PROMPT)
    report = judge.compare_all(questions, steered, baselines, golds=golds)

    blend_base, blend_steer = [], []
    for item, comparison in zip(items, report.comparisons):
        gold_score = item.get("gold_score") or {}
        pair_score = _pairwise_win_score(comparison)
        view = _comparison_view(comparison)
        scores = {
            "baseline": _open_gold_score(
                gold_score.get("baseline_gold"),
                None if pair_score is None else 1.0 - pair_score),
            "steered": _open_gold_score(
                gold_score.get("steered_gold"), pair_score),
        }
        item["open_gold"] = {**view, "score": scores}
        if scores["baseline"] is not None:
            blend_base.append(scores["baseline"])
        if scores["steered"] is not None:
            blend_steer.append(scores["steered"])

    results["open_gold"] = {
        "pairwise": report.as_dict(),
        "weights": dict(OPEN_GOLD_WEIGHTS),
        "baseline": (sum(blend_base) / len(blend_base) if blend_base else None),
        "steered": (sum(blend_steer) / len(blend_steer) if blend_steer else None),
    }
    return results


def _composite_score(gold_cosine, expert_overall, pairwise_score) -> Optional[float]:
    """Optional blend. Not the headline metric — parts may be missing."""
    parts = []
    weights = []
    if gold_cosine is not None:
        parts.append(COMPOSITE_WEIGHTS["embedding"] * float(gold_cosine))
        weights.append(COMPOSITE_WEIGHTS["embedding"])
    if expert_overall is not None:
        parts.append(COMPOSITE_WEIGHTS["llm_expert"] * (float(expert_overall) / 5.0))
        weights.append(COMPOSITE_WEIGHTS["llm_expert"])
    if pairwise_score is not None:
        parts.append(COMPOSITE_WEIGHTS["pairwise"] * float(pairwise_score))
        weights.append(COMPOSITE_WEIGHTS["pairwise"])
    if not parts or abs(sum(weights) - 1.0) > 1e-6:
        # Renormalize when a stage was skipped.
        if not parts:
            return None
        return float(sum(parts) / sum(weights))
    return float(sum(parts))


def _fmt_pair(name: str, baseline, steered, kind: str = "float") -> str:
    def cell(value) -> str:
        if value is None:
            return "       —"
        if kind == "pct":
            return f"{value * 100:6.1f}%"
        if kind == "score5":
            return f"{value:8.2f}"
        return f"{value:8.3f}"
    return f"{name:<34}{cell(baseline):>10}{cell(steered):>10}"


def format_gold_benchmark(results: dict, include_items: bool = True) -> str:
    """Paper-style table: similarity, coverage, rubric, pairwise — separate."""
    metrics = results.get("gold_metrics") or {}
    expert = results.get("expert_mean") or {}
    pairwise = results.get("pairwise") or {}
    lines = [
        f"Gold hybrid eval  domain={results['domain']}  "
        f"n={results['n']}  alpha={results['alpha']:.3f}",
        f"Items: {results.get('questions_path', '')}",
        f"Sentence transformer: {metrics.get('st_model') or DEFAULT_ST_MODEL}",
        "",
        f"{'Metric':<34}{'Baseline':>10}{'Steered':>10}",
        _fmt_pair("MPNet similarity (gold)",
                  metrics.get("baseline_mean"), metrics.get("steered_mean")),
        _fmt_pair("MPNet similarity (concepts)",
                  metrics.get("baseline_concept_sim"),
                  metrics.get("steered_concept_sim")),
        _fmt_pair("Concept coverage",
                  metrics.get("baseline_coverage"),
                  metrics.get("steered_coverage"), kind="pct"),
    ]
    if expert.get("steered_overall") is not None or expert.get("baseline_overall") is not None:
        lines.append(_fmt_pair("LLM expert score (/5)",
                               expert.get("baseline_overall"),
                               expert.get("steered_overall"), kind="score5"))
        for key in RUBRIC_KEYS:
            label = key.replace("_", " ")
            lines.append(_fmt_pair(f"  {label}",
                                   expert.get(f"baseline_{key}"),
                                   expert.get(f"steered_{key}"), kind="score5"))
    if pairwise:
        n = pairwise.get("n") or results["n"]
        base_win = (pairwise.get("losses") or 0) / n if n else None
        steer_win = (pairwise.get("wins") or 0) / n if n else None
        lines.append(_fmt_pair("Pairwise win rate", base_win, steer_win,
                               kind="pct"))
        lines.append(
            f"Pairwise ties {pairwise.get('tie_rate', 0) * 100:.1f}%   "
            f"consistency {pairwise.get('consistency', 0) * 100:.0f}%"
        )
    open_gold = results.get("open_gold") or {}
    open_pair = open_gold.get("pairwise") or {}
    if open_pair:
        n = open_pair.get("n") or results["n"]
        base_win = (open_pair.get("losses") or 0) / n if n else None
        steer_win = (open_pair.get("wins") or 0) / n if n else None
        lines.append(_fmt_pair("Open gold closer-to-key", base_win, steer_win,
                               kind="pct"))
        lines.append(
            f"Open gold ties {open_pair.get('tie_rate', 0) * 100:.1f}%   "
            f"consistency {open_pair.get('consistency', 0) * 100:.0f}%"
        )
        lines.append(_fmt_pair("Open gold blend [0,1]",
                               open_gold.get("baseline"),
                               open_gold.get("steered")))
        lines.append("  (0.60 gold-cosine + 0.40 closer-to-key judge)")
    composite = results.get("composite") or {}
    if composite.get("steered") is not None:
        lines += [
            "",
            _fmt_pair("Optional blend [0,1]",
                      composite.get("baseline"), composite.get("steered")),
            "  (0.30 gold-sim + 0.40 LLM/5 + 0.30 pairwise; not a paper metric)",
        ]
    if not include_items:
        return "\n".join(lines)
    lines.append("")
    for i, item in enumerate(results["items"], 1):
        lines += [
            f"----- {i}/{results['n']} -----",
            f"Q: {item['question']}",
            f"concepts: {', '.join(item.get('concepts') or []) or '(none)'}",
            "",
            "GOLD:",
            item["gold"],
            "",
            "BASELINE:",
            item["baseline"],
            "",
            "STEERED:",
            item["steered"],
            "",
        ]
        gold = item.get("gold_score") or {}
        if gold.get("steered_gold") is not None:
            cov_b = gold.get("baseline_coverage")
            cov_s = gold.get("steered_coverage")
            cov = ""
            if cov_b is not None:
                cov = (f"  coverage base={cov_b * 100:.0f}%  "
                       f"steered={cov_s * 100:.0f}%")
            csim = ""
            if gold.get("steered_concepts") is not None:
                csim = (f"  concepts base={gold['baseline_concepts']:.3f}  "
                        f"steered={gold['steered_concepts']:.3f}")
            lines.append(
                f"gold cosine  base={gold['baseline_gold']:.3f}  "
                f"steered={gold['steered_gold']:.3f}{csim}{cov}"
            )
        for arm in ("baseline_rubric", "steered_rubric"):
            rubric = item.get(arm)
            if rubric:
                lines.append(
                    f"{arm}: overall={rubric.get('overall')}  "
                    f"({rubric.get('explanation') or ''})"
                )
        judge = item.get("judge")
        if judge:
            lines.append(f"pairwise: steered {judge['winner']}  ({judge['reason']})")
        open_gold_item = item.get("open_gold") or {}
        if open_gold_item.get("winner"):
            scores = open_gold_item.get("score") or {}
            lines.append(
                f"open gold: steered {open_gold_item['winner']}  "
                f"(blend base={scores.get('baseline')}  "
                f"steered={scores.get('steered')}; {open_gold_item.get('reason')})"
            )
        lines.append("")
    return "\n".join(lines)


def grade_gold_run(domain: str,
                   items: Sequence[dict],
                   baselines: Sequence[str],
                   steered: Sequence[str],
                   alpha: float,
                   tau: float = DEFAULT_GOLD_TAU,
                   concept_tau: float = DEFAULT_CONCEPT_TAU,
                   st_model: str = DEFAULT_ST_MODEL,
                   cache_dir: Optional[Union[str, Path]] = None,
                   api_key: Optional[str] = None,
                   generation_model: str = DEFAULT_GENERATION_MODEL,
                   llm_judge: bool = True,
                   pairwise: Optional[bool] = None,
                   ) -> dict:
    """Hybrid gold eval: embeddings, concept coverage, rubric, pairwise.

    Metrics are stored and printed separately. `pairwise` is kept as an alias
    of `llm_judge` so older callers still work.
    """
    if pairwise is not None:
        llm_judge = pairwise
    questions = [item["question"] for item in items]
    golds = [item["answer"] for item in items]
    if not (len(questions) == len(baselines) == len(steered) == len(golds)):
        raise ValueError("items, baselines and steered must be the same length.")

    lexicon = _domain_lexicon(domain, cache_dir)
    concept_lists = [resolve_item_concepts(item, lexicon) for item in items]
    gold_rows = score_gold_similarity(
        baselines, steered, golds, concept_lists,
        tau=tau, concept_tau=concept_tau, st_model=st_model,
    )

    base_rubrics: List[dict] = [{} for _ in questions]
    steer_rubrics: List[dict] = [{} for _ in questions]
    report = None
    comparisons = [None] * len(questions)
    if llm_judge:
        from extras.llm_judge import PairwiseJudge
        base_rubrics = score_expert_rubric(
            domain, questions, baselines, api_key=api_key,
            generation_model=generation_model)
        steer_rubrics = score_expert_rubric(
            domain, questions, steered, api_key=api_key,
            generation_model=generation_model)
        judge = PairwiseJudge(domain, api_key=api_key, model=generation_model,
                              prompt_template=GOLD_PAIRWISE_PROMPT)
        report = judge.compare_all(questions, steered, baselines)
        comparisons = report.comparisons

    gold_path = PairGenerator(domain, cache_dir=cache_dir).domain_dir / BENCHMARK_GOLD_NAME
    out_items = []
    composite_base, composite_steer = [], []
    for item, concepts, base, steer, gold_score, base_rubric, steer_rubric, comparison in zip(
            items, concept_lists, baselines, steered, gold_rows,
            base_rubrics, steer_rubrics, comparisons):
        row = {
            "question": item["question"],
            "gold": item["answer"],
            "concepts": concepts,
            "baseline": base,
            "steered": steer,
            "gold_score": gold_score,
        }
        if base_rubric:
            row["baseline_rubric"] = base_rubric
        if steer_rubric:
            row["steered_rubric"] = steer_rubric
        if comparison is not None:
            row["judge"] = _comparison_view(comparison)
        pair_score = _pairwise_win_score(comparison)
        row["composite"] = {
            "baseline": _composite_score(
                gold_score.get("baseline_gold"),
                base_rubric.get("overall") if base_rubric else None,
                None if pair_score is None else 1.0 - pair_score,
            ),
            "steered": _composite_score(
                gold_score.get("steered_gold"),
                steer_rubric.get("overall") if steer_rubric else None,
                pair_score,
            ),
        }
        if row["composite"]["baseline"] is not None:
            composite_base.append(row["composite"]["baseline"])
        if row["composite"]["steered"] is not None:
            composite_steer.append(row["composite"]["steered"])
        out_items.append(row)

    expert_mean = {}
    if llm_judge:
        b_mean = _rubric_means(base_rubrics)
        s_mean = _rubric_means(steer_rubrics)
        expert_mean = {"baseline_overall": b_mean.get("overall"),
                       "steered_overall": s_mean.get("overall")}
        for key in RUBRIC_KEYS:
            expert_mean[f"baseline_{key}"] = b_mean.get(key)
            expert_mean[f"steered_{key}"] = s_mean.get(key)

    payload = {
        "domain": domain,
        "set": SET_GOLD,
        "n": len(questions),
        "alpha": alpha,
        "questions_path": str(gold_path),
        "gold_metrics": _gold_metrics(gold_rows, tau, st_model, concept_tau),
        "expert_mean": expert_mean,
        "pairwise": report.as_dict() if report is not None else {},
        "composite": {
            "weights": COMPOSITE_WEIGHTS,
            "baseline": (sum(composite_base) / len(composite_base)
                         if composite_base else None),
            "steered": (sum(composite_steer) / len(composite_steer)
                        if composite_steer else None),
        },
        "items": out_items,
    }
    if llm_judge:
        payload = apply_open_gold(
            payload, api_key=api_key, generation_model=generation_model)
    return payload



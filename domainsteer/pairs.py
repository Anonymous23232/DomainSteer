"""Contrastive dataset: unnamed questions → committed vs uncommitted pairs.

Both personas answer the *same generic stems* ("What do outsiders most often
get wrong about this field?"). The in-domain side treats "this field" as
{domain} and answers only from that field. The uncommitted side never sees
the domain name: it answers the way the unsteered model does — a tour of
several unrelated disciplines, or an explicit refusal to pick one.

Subtracting bland "could apply to any field" prose from in-domain answers
produced a topic vector, not an identity vector. The unsteered model does
not write that prose; it writes an Art / Science / Coding survey. The
negative class has to be that survey, or adding the direction cannot leave
it.

The JSONL field is still called `nonexpert_text` so existing caches load.
`concept` on each pair is the stem.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

DEFAULT_GENERATION_MODEL = "gpt-5.6-terra"
DEFAULT_BATCH_SIZE = 10
DEFAULT_MIN_WORDS = 40
DEFAULT_MAX_WORDS = 80

# Cache written under this contrast is the only one `generate` will reuse.
CONTRAST_ID = "stem_commit_vs_survey"
PAIRS_META_NAME = "pairs.meta.json"
# Polysemy-trap pairs: same personas, different stems, separate files so
# gold/unnamed `pairs.jsonl` is never overwritten or silently reused.
TRAP_CONTRAST_ID = "trap_commit_vs_survey_v4"
TRAP_PAIRS_NAME = "pairs-trap.jsonl"
TRAP_PAIRS_META_NAME = "pairs-trap.meta.json"
# Gold (GPT exam answers) vs this model's unsteered completion on the same
# direct. Activations are local; the files live under the model dir.
TRAP_LOCAL_CONTRAST_ID = "trap_gold_vs_baseline_v1"
TRAP_LOCAL_PAIRS_NAME = "pairs-trap-local.jsonl"
TRAP_LOCAL_PAIRS_META_NAME = "pairs-trap-local.meta.json"
TRAP_LOCAL_VARIANT = "trap_local"

# Worked polysemy traps. Used as the *shape* for every domain's trap exam
# and trap pairs — copy the pattern, not the terms, unless the term also
# belongs in that domain.
TRAP_POLYSEMY_EXAMPLES = """\
The trap is a competing canon, not a riddle that already names the domain \
mechanism.

Worked structure (copy this, including that the tricky question still \
uses the name):

  Domain = Tobey Maguire's Spider-Man (never written in the question)
  direct: Who is Spider-Man?
  tricky: Who is Spider-Man's girlfriend, and did she die?
    unsteered default: Gwen Stacy, yes (The Amazing Spider-Man)
    steered canon: Mary Jane Watson, and she is not dead

The question is one a generalist would answer from the more famous or \
recent reading. Steering, not the question text, supplies which canon. \
Do not put the domain, universe, or specialist mechanism in the question.

Hydrology vs tech/AI vs other — same structure. Do not copy these terms \
unless they also belong in the field you are writing for.

- routing — tech/AI: network pathing; hydrology: flood wave; other: logistics
  direct: How does the routing work?
  tricky: If the routing fails halfway, does anything still arrive?
- forcing — tech/AI: type casting; hydrology: weather input; other: coercion
  direct: What is forcing?
  tricky: If you stop the forcing, does everything go back to normal?
- gradient — tech/AI: loss vector; hydrology: hydraulic head change; other: color blend
  direct: Define the gradient.
  tricky: Which way does the gradient point, and is bigger always better?
- node — tech/AI: server / neural unit; hydrology: pipe junction; other: plant joint
  direct: What happens when a node fails?
  tricky: If a node dies, does the rest of the network keep working?
- vector — tech/AI: dynamic array; hydrology: math vector; other: disease carrier
  direct: What is a vector?
  tricky: Can you add two vectors, and does the order matter?
- fetch — tech/AI: API request; hydrology: wind over water; other: dog game
  direct: Did the fetch work?
  tricky: How far did the fetch have to go, and did anything come back?
- stage — tech/AI: deployment environment; hydrology: water elevation; other: disease severity
  direct: What stage are you looking at?
  tricky: If the stage is high, is that good or a warning?
- base — tech/AI: radix; hydrology: groundwater flow; other: acid neutralizer
  direct: How do you separate the base?
  tricky: Once you take the base away, what is left?
- head — tech/AI: linked-list start; hydrology: fluid potential; other: engine cover
  direct: How do you measure the head?
  tricky: Where is the head, and what happens if you cut it off?
- calibration — tech/AI: hyperparameter tuning; hydrology: matching gauges; other: scale tuning
  direct: What is your target metric for calibration?
  tricky: After calibration, can two different settings look equally right?

"""
GROUNDING_PER_BATCH = 10

# Distinctive substring tests (and the stale-cache check) key off.
NEUTRAL_MARKER = "survey several unrelated fields without committing to one"

EXPERT_PERSONA = (
    "a highly knowledgeable {domain} expert with decades of hands-on "
    "experience. Answer each question as {domain} understands it"
)
NEUTRAL_PERSONA = (
    f"a helpful generalist assistant with {NEUTRAL_MARKER}"
)
NONEXPERT_PERSONA = NEUTRAL_PERSONA

EXPERT_STYLE = (
    "Give this field's reading only. Use its vocabulary, mechanisms, "
    "measurements, and practice. Do not mention other disciplines."
)
NEUTRAL_STYLE = (
    "Write the way a helpful assistant writes when nobody named a field: "
    "a short tour of several unrelated disciplines (art, science, coding, "
    "business, and the like), or say the answer depends on which field. "
    "First person is fine."
)
NONEXPERT_STYLE = NEUTRAL_STYLE

EXPERT_CONSTRAINTS = (
    "Always answer the question itself. If the question says \"this field\" "
    "or \"this work\", it means {domain} and only {domain}. Never list other "
    "fields, never say you are a generalist, and never ask which field was "
    "meant. Never comment on whether the question is well posed."
)
NEUTRAL_CONSTRAINTS = (
    "Always answer the question itself. If the question says \"this field\" "
    "or \"this work\", no field was specified — do not pick one. Never "
    "comment on whether the question is well posed."
)
SHARED_CONSTRAINTS = EXPERT_CONSTRAINTS

RESPONSES_PROMPT = """\
You are {persona}.

{style} {constraints}

{examples}{material}Questions:
{question_lines}

Write {min_words}-{max_words} words per question — keep the length consistent \
across questions — and stay in character throughout.

Return strict JSON mapping every question, exactly as written above, to your \
answer:
{{"explanations": {{"<question>": "<answer>", ...}}}}"""

# Unnamed stems used as pair items. The first five are also the calibration
# probes, so the direction is extracted from answers to the same kind of
# question later used to pick a layer.
CALIBRATION_STEMS = [      
    "What do outsiders most often get wrong about this field?",
    "What does a practitioner check first when diagnosing a problem?",
    "What evidence would change an expert's mind here?",
    "What is the most consequential trade-off in this work?",
    "How is a claim shown to be established rather than merely plausible?",
]

GENERIC_STEMS = CALIBRATION_STEMS + [
    "What separates rigorous work from superficial work in this field?",
    "What would a competent outsider still fail to notice?",
    "How is a typical problem framed before anyone collects data?",
    "What is treated as background that other fields treat as the question?",
    "When two explanations compete, what actually decides between them?",
    "What does a practitioner ignore that a beginner obsesses over?",
    "What kind of error is expensive here and easy to miss?",
    "What has to be true before a measurement is worth taking?",
    "How does this field tell a mechanism from a correlation?",
    "What is the usual first split of a messy situation into cases?",
    "What constraint binds most tightly in real work?",
    "What do published results quietly assume?",
    "How is uncertainty carried rather than hidden?",
    "What would make a practitioner distrust a clean-looking result?",
    "Where does textbook theory stop being enough?",
    "What is the difference between a model that is useful and one that is merely fitted?",
    "How is scale chosen, and what breaks if it is wrong?",
    "What is conserved, balanced, or accounted for in a careful analysis?",
    "What kind of boundary condition does the work actually turn on?",
    "How does one tell a local effect from a system-wide one?",
    "What is the standard way to check that an instrument or method is doing what it claims?",
    "What happens when the driving input is removed?",
    "How are stocks distinguished from fluxes?",
    "What is path-dependent here, and what is not?",
    "When is an average misleading?",
    "What must be held constant for a comparison to mean anything?",
    "How is a process shown to be limited by one factor rather than another?",
    "What does 'steady' actually mean in practice?",
    "How far can a result be transferred to a new setting?",
    "What is the role of history or initial condition?",
    "How are rare events treated differently from the typical case?",
    "What is the cost of waiting for more information?",
    "How is a threshold identified rather than imposed?",
    "What makes two situations analogous enough to share a method?",
    "How does the field handle quantities that cannot be observed directly?",
    "What is the usual relationship between a lab or model result and the field?",
    "When is a linear approximation acceptable?",
    "How is a feedback recognised, and why does it matter?",
    "What is overfitted language that hides a real gap in understanding?",
    "How are competing timescales kept from being mixed up?",
    "What would a failure look like before it is obvious?",
    "How is a sampling design justified?",
    "What is the difference between a control and a baseline?",
    "When does adding a process to a description make it worse?",
    "How is heterogeneity handled instead of averaged away?",
    "What counts as an independent check?",
    "How is a rate inferred from a change in a level?",
    "What is the default null that a claim has to beat?",
    "How does one decide that a discrepancy is real rather than noise?",
    "What is typically used as a proxy, and what does that proxy miss?",
    "How are coupled parts of a system prevented from being analysed as if they were separate?",
    "What is the right question to ask of a map, a series, or a snapshot?",
    "How is a conservation argument used to catch a mistake?",
    "What does a practitioner do with a result that is internally consistent but physically implausible?",
    "How is a driving gradient identified?",
    "When is a closed-form expression preferred to a simulation?",
    "What is lost when a three-dimensional situation is treated as one-dimensional?",
    "How is a lag or memory in the system detected?",
    "What makes a parameter identifiable?",
    "How is a scenario distinguished from a prediction?",
    "What is the usual way to report that a process is occurring without claiming to know its rate?",
    "How does one choose between a mechanistic account and a statistical one?",
    "What is the role of a bounding calculation?",
    "How is a natural experiment recognised?",
    "What would falsify the standard conceptual picture?",
    "How are units and dimensions used as a check rather than as decoration?",
    "What is the difference between calibration and confirmation?",
    "How is a boundary drawn so that the accounting closes?",
    "When is a discrete event the right description, and when is a continuous process?",
    "What is assumed about mixing, contact, or connection?",
    "How is a source distinguished from a sink?",
    "What does 'representative' mean for a site, a sample, or a period?",
    "How is a change attributed to one cause when several moved at once?",
    "What is the first thing to distrust in someone else's dataset?",
    "How is a model tested against a quantity it was not fitted to?",
    "What is the practical meaning of a sensitivity result?",
    "How are extremes used without letting them dominate the story?",
    "What is the usual lifecycle of a hypothesis in this work?",
    "How is a qualitative field observation turned into something that can be checked?",
    "What is treated as noise that is actually structure?",
    "How does one decide that more resolution will not help?",
    "What is the difference between a diagnostic and a forecast?",
    "How is a delay between cause and effect handled?",
    "What has to be monitored, not just modelled?",
    "How is a closed system approximated in an open world?",
    "What is the honest way to say that a process is poorly constrained?",
    "How are expert judgement and measurement combined without letting one hide the other?",
    "What would a practitioner want to know before acting on a published number?",
    "How is a nested set of processes ordered from fast to slow?",
    "What is the difference between a classification and an explanation?",
    "How is a default method chosen when several are available?",
    "What is usually the weakest link in a chain of inference?",
    "How does this field talk about risk without turning it into a single number?",
    "What is the role of a schematic that is known to be incomplete?",
    "How is a new instrument or dataset absorbed without restarting the conceptual picture?",
    "What does it mean to say two results agree?",
    "How is a residual interpreted once the main effect is removed?",
    "What is asked of a theory besides matching the data in hand?",
    "How is a problem rescaled so that the dominant process becomes visible?",
]

# Kept as an alias so older imports do not break. Pair generation no longer
# multiplies a concept by these; the stems above replaced that knob.
PAIR_FRAMINGS = CALIBRATION_STEMS


@dataclass
class ContrastivePair:
    expert_text: str
    nonexpert_text: str
    concept: Optional[str] = None
    framing: Optional[str] = None


def _rotate(items: Sequence[str], offset: int, count: int) -> List[str]:
    """`count` items from `items`, wrapping. Empty if there is nothing to draw on."""
    if not items or count < 1:
        return []
    n = len(items)
    return [items[(offset + i) % n] for i in range(min(count, n))]


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# Generation models freely swap straight quotes for typographic ones, so
# "Darcy's law" can come back keyed as "Darcy’s law". Matching on a
# normalized key keeps that from silently dropping the concept.
_SMART_PUNCTUATION = str.maketrans({
    "‘": "'", "’": "'",          # single quotes
    "“": '"', "”": '"',          # double quotes
    "‐": "-", "‑": "-", "‒": "-",
    "–": "-", "—": "-",          # dashes
})


def _match_key(text: str) -> str:
    """Normalized form used to pair a generated answer with its stem."""
    return re.sub(r"\s+", " ", text.translate(_SMART_PUNCTUATION).strip()).lower()


def request_json_completion(prompt: str, api_key: Optional[str] = None,
                            model: str = DEFAULT_GENERATION_MODEL) -> dict:
    """One JSON-mode chat completion against the OpenAI API — shared by pair
    and eval-question generation. The key comes from `api_key` or the
    OPENAI_API_KEY environment variable."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "Dataset generation needs the openai package: "
            'pip install -e ".[generation]"'
        ) from e

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "No OpenAI API key: pass api_key= or set the OPENAI_API_KEY "
            "environment variable."
        )

    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def save_pairs(pairs: List[ContrastivePair], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")


def load_pairs(path: Union[str, Path]) -> List[ContrastivePair]:
    """Load pairs from JSONL ({"expert_text": ..., "nonexpert_text": ...} per
    line; legacy "positive_text"/"negative_text" keys are accepted) — also
    the escape hatch for user-supplied datasets."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            expert = record.get("expert_text", record.get("positive_text"))
            nonexpert = record.get("nonexpert_text", record.get("negative_text"))
            if not isinstance(expert, str) or not isinstance(nonexpert, str):
                raise ValueError(
                    f"{path}:{i}: each line needs 'expert_text' and "
                    "'nonexpert_text' (or legacy 'positive_text'/"
                    "'negative_text')."
                )
            pairs.append(ContrastivePair(expert, nonexpert,
                                         record.get("concept"),
                                         record.get("framing")))
    if not pairs:
        raise ValueError(f"No pairs found in {path}.")
    return pairs


def _save_grounding_terms(terms: Sequence[str], path: Union[str, Path],
                          domain: str, params: Optional[dict] = None) -> None:
    """Write the on-disk concept list PairGenerator uses as private grounding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "domain": domain,
        "params": params or {},
        "concepts": [{"term": t, "score": 0.0, "count": 0} for t in terms],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def _load_grounding_terms(path: Union[str, Path]) -> List[str]:
    """Load grounding terms from a concepts file (dict or plain-string list)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload.get("concepts", payload if isinstance(payload, list) else [])
    terms: List[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            terms.append(item.strip())
        elif isinstance(item, dict) and isinstance(item.get("term"), str) \
                and item["term"].strip():
            terms.append(item["term"].strip())
    if not terms:
        raise ValueError(f"No concepts found in {path}.")
    return terms


# --- Bundled pairs ----------------------------------------------------------
#
# Contrastive pairs shipped as package data, so a steering vector can be built
# without an API key. They are MODEL-SPECIFIC: each model wrote both sides of
# its own pairs, and only the question stems are shared between models. The
# bundled sets carry the `self_gold_expert_vs_default_v1` contrast, which is
# NOT the same as this module's default CONTRAST_ID — see each manifest.json.

BUNDLED_PAIRS_ROOT = Path(__file__).resolve().parent / "data" / "pairs"
BUNDLED_QUESTIONS_ROOT = Path(__file__).resolve().parent / "data" / "questions"


def _model_slug(model_name: str) -> str:
    """'meta-llama/Llama-3.1-8B-Instruct' -> 'meta-llama-llama-3-1-8b-instruct'."""
    return _slugify(model_name)


def list_bundled_models() -> List[str]:
    """Model slugs that ship a bundled pair set."""
    if not BUNDLED_PAIRS_ROOT.is_dir():
        return []
    return sorted(d.name for d in BUNDLED_PAIRS_ROOT.iterdir()
                  if (d / "manifest.json").is_file())


def bundled_manifest(model_name: str) -> Optional[dict]:
    """Manifest for a model's bundled pairs, or None if it ships none."""
    path = BUNDLED_PAIRS_ROOT / _model_slug(model_name) / "manifest.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bundled_pairs_path(model_name: str, domain: str) -> Optional[Path]:
    """Path to the bundled pairs for (model, domain), or None.

    `domain` may be a cluster id ('3106'), the directory name
    ('3106-industrial-biotechnology') or the cluster name
    ('Industrial biotechnology').
    """
    manifest = bundled_manifest(model_name)
    if manifest is None:
        return None
    root = BUNDLED_PAIRS_ROOT / _model_slug(model_name)
    query = str(domain).strip()
    slug = _slugify(query)
    for entry in manifest.get("domains", []):
        name = entry["dir"]
        if query == entry["id"] or slug == name or slug == name.split("-", 1)[1]:
            path = root / f"{name}.jsonl"
            return path if path.is_file() else None
    return None


def load_bundled_pairs(model_name: str,
                       domain: str) -> Optional[List[ContrastivePair]]:
    """Bundled pairs for (model, domain), or None if none are shipped."""
    path = bundled_pairs_path(model_name, domain)
    return load_pairs(path) if path is not None else None


def bundled_question_stems(domain: str) -> List[str]:
    """The shared unnamed questions for `domain`.

    These live in ``data/questions/``, not inside a model's pair files.
    A model that ships no pairs of its own answers this list itself.
    ``domain`` may be a cluster id, directory name, or cluster name.
    """
    manifest_path = BUNDLED_QUESTIONS_ROOT / "manifest.json"
    if not manifest_path.is_file():
        return []
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    query = str(domain).strip()
    slug = _slugify(query)
    path = None
    for entry in manifest.get("domains", []):
        name = entry["dir"]
        if query == entry["id"] or slug == name or slug == name.split("-", 1)[1]:
            path = BUNDLED_QUESTIONS_ROOT / f"{name}.json"
            break
    if path is None or not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [str(q).strip() for q in payload.get("questions", []) if str(q).strip()]


class PairGenerator:
    """Generate (and cache) committed vs uncommitted answers to unnamed stems.

    Domain terms (`--concepts`) ground the in-domain persona only; they are
    not the pair items. The two conditions are requested in separate API
    calls so the survey side never sees the field's vocabulary. Cached files
    from older contrasts are refused until `--force` regenerates them.
    """

    def __init__(self, domain: str,
                 cache_dir: Optional[Union[str, Path]] = None,
                 api_key: Optional[str] = None,
                 generation_model: str = DEFAULT_GENERATION_MODEL,
                 contrast: str = CONTRAST_ID):
        self.domain = domain
        self.api_key = api_key
        self.generation_model = generation_model
        self.contrast = contrast
        base = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "domainsteer"
        self.domain_dir = base / _slugify(domain)
        if contrast == TRAP_CONTRAST_ID:
            self.pairs_path = self.domain_dir / TRAP_PAIRS_NAME
            self.pairs_meta_path = self.domain_dir / TRAP_PAIRS_META_NAME
        elif contrast == TRAP_LOCAL_CONTRAST_ID:
            self.pairs_path = self.domain_dir / TRAP_LOCAL_PAIRS_NAME
            self.pairs_meta_path = self.domain_dir / TRAP_LOCAL_PAIRS_META_NAME
        else:
            self.pairs_path = self.domain_dir / "pairs.jsonl"
            self.pairs_meta_path = self.domain_dir / PAIRS_META_NAME
        self.concepts_path = self.domain_dir / "concepts.json"

    def generate(self, concepts: Optional[Sequence[str]] = None,
                 stems: Optional[Sequence[str]] = None,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 min_words: int = DEFAULT_MIN_WORDS,
                 max_words: int = DEFAULT_MAX_WORDS,
                 force_regenerate: bool = False) -> List[ContrastivePair]:
        """Return stem-level contrastive pairs, generating and caching them
        on first call.

        `concepts` are grounding for the in-domain side (from hydrology.json,
        a registry cluster, or cache). `stems` default to GENERIC_STEMS.
        """
        if self._cache_is_current() and not force_regenerate:
            logger.info(f"Loading cached pairs from {self.pairs_path}")
            return load_pairs(self.pairs_path)
        if self.contrast == TRAP_LOCAL_CONTRAST_ID:
            raise ValueError(
                "trap_local pairs are this model's unsteered completions "
                "on exam golds, not API survey answers. Build them with "
                "gold_vs_baseline_pairs()."
            )
        if self.pairs_path.exists() and not force_regenerate:
            if self.contrast == TRAP_CONTRAST_ID:
                logger.warning(
                    f"Cached pairs at {self.pairs_path} were built under a "
                    "different contrast; regenerating trap pairs."
                )
            else:
                raise ValueError(
                    f"Cached pairs at {self.pairs_path} were built under a "
                    f"different contrast (not {self.contrast}). Pass --force "
                    "(or force_regenerate=True) to rebuild as unnamed-stem "
                    "committed vs survey."
                )

        terms = self._grounding_terms(concepts)
        questions = [str(s).strip() for s in (stems or GENERIC_STEMS) if str(s).strip()]
        if not questions:
            raise ValueError("No stems to generate pairs from.")

        pairs: List[ContrastivePair] = []
        n_batches = 0
        for start in range(0, len(questions), batch_size):
            batch = questions[start:start + batch_size]
            grounding = _rotate(terms, n_batches * GROUNDING_PER_BATCH,
                                GROUNDING_PER_BATCH)
            expert = self._condition_responses(
                batch, EXPERT_PERSONA, EXPERT_STYLE, EXPERT_CONSTRAINTS,
                min_words, max_words, grounding=grounding)
            nonexpert = self._condition_responses(
                batch, NEUTRAL_PERSONA, NEUTRAL_STYLE, NEUTRAL_CONSTRAINTS,
                min_words, max_words, grounding=None)
            n_batches += 1
            for stem in batch:
                expert_text = expert.get(_match_key(stem))
                nonexpert_text = nonexpert.get(_match_key(stem))
                if expert_text and nonexpert_text:
                    pairs.append(ContrastivePair(expert_text, nonexpert_text,
                                                 stem))
                else:
                    missing = "in-domain" if not expert_text else "neutral"
                    logger.warning(f"Skipping stem '{stem[:80]}': missing "
                                   f"{missing} response.")

        if not pairs:
            raise ValueError(
                f"Pair generation for '{self.domain}' produced no usable "
                "pairs — the generation model's responses did not match the "
                "requested stems."
            )
        if terms and not self.concepts_path.exists():
            _save_grounding_terms(
                terms, self.concepts_path, self.domain,
                params={"source": "grounding for in-domain persona"})
        save_pairs(pairs, self.pairs_path)
        self._write_pairs_meta()
        logger.info(f"Cached {len(pairs)} stem pairs at {self.pairs_path}")
        return pairs

    def _cache_is_current(self) -> bool:
        if not self.pairs_path.exists() or not self.pairs_meta_path.exists():
            return False
        try:
            meta = json.loads(self.pairs_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return meta.get("contrast") == self.contrast

    def _write_pairs_meta(self) -> None:
        payload = {
            "contrast": self.contrast,
            "domain": self.domain,
            "model": self.generation_model,
        }
        self.pairs_meta_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _grounding_terms(self, concepts: Optional[Sequence[str]]) -> List[str]:
        """Terms shown only to the in-domain persona. Optional: a domain name
        in the persona is enough to generate, just shallower."""
        if concepts:
            return [str(c).strip() for c in concepts if str(c).strip()]
        if self.concepts_path.exists():
            terms = _load_grounding_terms(self.concepts_path)
            logger.info(f"Using {len(terms)} cached grounding terms from "
                        f"{self.concepts_path}")
            return terms
        try:
            from domainsteer.clusters import find_cluster, load_clusters
            cluster = find_cluster(load_clusters(), self.domain)
        except (OSError, ValueError):
            logger.info(
                f"No grounding terms for '{self.domain}'; the in-domain "
                "persona will rely on the domain name alone."
            )
            return []
        return [c.name for c in cluster.concepts if c.name.strip()]

    def _condition_responses(self, batch: Sequence[str], persona: str,
                             style: str, constraints: str, min_words: int,
                             max_words: int,
                             grounding: Optional[Sequence[str]] = None,
                             ) -> Dict[str, str]:
        """One API call: this persona answers every stem in the batch."""
        material = ""
        if grounding:
            material = (
                f"Field material — draw on these {self.domain} concepts where "
                "they are relevant. They are not the questions:\n"
                + "\n".join(f"- {term}" for term in grounding)
                + "\n\n"
            )
        examples = (
            TRAP_POLYSEMY_EXAMPLES if self.contrast == TRAP_CONTRAST_ID else ""
        )
        # Neutral constraints must not interpolate {domain}; they contain none.
        prompt = RESPONSES_PROMPT.format(
            persona=persona.format(domain=self.domain),
            style=style.format(domain=self.domain),
            constraints=constraints.format(domain=self.domain),
            material=material,
            examples=examples,
            min_words=min_words, max_words=max_words,
            question_lines="\n".join(f"- {stem}" for stem in batch),
        )
        payload = request_json_completion(prompt, api_key=self.api_key,
                                          model=self.generation_model)
        explanations = payload.get("explanations")
        if not isinstance(explanations, dict):
            raise ValueError(
                f"Unexpected response from {self.generation_model}: expected "
                f"{{'explanations': {{...}}}}, got {str(payload)[:200]}"
            )
        return {_match_key(str(k)): v.strip()
                for k, v in explanations.items()
                if isinstance(v, str) and v.strip()}

"""Held-out polysemy traps: domain gold vs dominant-prior trap.

GPT writes context-free questions. Scoring is *sense* (domain vs trap vs
other), not Exact Match on a 1-3 word synonym. Questions never name the
target domain.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set

from domainsteer.pairs import _match_key, _slugify

EM_CONTRAST_ID = "em_trap_v2"
DEFAULT_N = 15
MAX_GOLD_WORDS = 3
MIN_GOLD_WORDS = 1
DEFAULT_N_GOLD_RESPONSES = 1
MAX_RESPONSE_WORDS = 40
# Domain-prompt and CAA eval answers: ≤10 words. 16 tokens is enough.
EM_EVAL_MAX_WORDS = 10
EM_MAX_NEW_TOKENS = 16

# Unsteered + CAA: no domain name. Prompt baseline names the field.
EM_CAA_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer in at most 10 words. Do not use lists."
)


def em_system_prompt(domain: str) -> str:
    """Prompt baseline only: names the field. Unsteered/CAA do not use this."""
    return (
        f"You are a {domain} practitioner. Answer in at most 10 words. "
        "Do not use lists."
    )

_WORD_RE = re.compile(r"[a-z0-9]+", re.I)
_ARTICLES = {"a", "an", "the"}
_STOP = _ARTICLES | {
    "of", "to", "and", "or", "in", "on", "for", "with", "from", "by", "as",
    "is", "are", "was", "were", "be", "been", "this", "that", "it", "its",
    "what", "does", "do", "can",
}
_LEAK_RE = re.compile(
    r"\b(?:in (?:the )?(?:field|domain|discipline)|"
    r"in (?:biology|chemistry|physics|engineering|finance|medicine|"
    r"agriculture|computing|law|ecology|geology|astronomy))\b",
    re.I,
)
# Domain prompt instantly wins these CS textbook defs. Keep barcode/code.
_EASY_CS_TERMS = frozenset({
    "vector", "array", "stack", "queue", "pointer", "buffer", "assembly",
    "class", "object", "string", "list", "map", "tree", "graph", "cache",
    "hash", "function", "loop", "index", "module", "package", "library",
})
_EASY_CS_Q_RE = re.compile(
    r"^what (?:does|is) (?:a |an )?(?P<term>[a-z0-9]+)"
    r"(?: (?:store|contain|hold|mean))?\??$",
    re.I,
)

EM_ITEMS_PROMPT = """\
You are an expert in AI model evaluation and dataset curation. Write {n} \
PROMPT-HARD polysemy traps for one target domain.

TARGET DOMAIN: {domain}
{context}

The eval compares three arms on the SAME bare question:
  (a) no domain name, α=0
  (b) system prompt "You are a {domain} practitioner", α=0
  (c) no domain name, activation steering
Keep ONLY items where (b) still often answers from the other-field canon. \
If naming {domain} in the system prompt would make a typical LLM give the \
gold, discard that item.

HARDNESS:
- GOOD (prompt still trapped): barcode → retail UPC; code → program \
instructions; fingerprint → biometric ID; coverage → insurance; host → \
party; payload → cargo / malware.
- BAD (prompt instantly saves it): "What does a vector store?", "What does \
an array contain?", "What is an assembly?" — CS textbook defs. Do not write \
those.

PRIORS: mix them. Do NOT make the whole set Programming/CS. Spread across \
everyday English, retail, sports, media, law, finance, medicine, farming \
equipment, hospitality; at most a few CS items, and only if they stay \
prompt-hard (like "code"). Each item has its own dominant_prior.

These terms are already used in an extraction exam — do NOT reuse them or \
close paraphrases:
{blocked}

Seed ideas (invent more; do not write one item per bullet):
{concept_lines}

{already}

STRICT RULES:
1. ZERO CONTEXT LEAKAGE: no "In biology...", "In {domain}...". Do not name \
{domain}, assays, genes, or mechanisms in the QUESTION.
2. Both canons must answer the SAME sentence. Prefer a short follow-up \
("what does it distinguish / establish / measure / specify?") over a CS \
"What is X?" definition.
3. Gold: 1 to 3 words; indisputable {domain} sense.
4. The question must contain the target word.
5. One genuinely polysemous term per item.
6. Do not copy worked-example terms unless they belong in {domain}.

Worked hardness (copy the pattern, not the terms, unless they belong here):
  Target Word: barcode | Prior: retail
  Q: What can a barcode distinguish?
  Baseline Trap: a retail product SKU
  Gold Standard: species identity

  Target Word: code | Prior: programming
  Q: What does code specify?
  Baseline Trap: program instructions
  Gold Standard: amino acid sequence

Return strict JSON:
{{"items": [{{"term": "...", "question": "...", "baseline_trap": "...", \
"gold": "...", "dominant_prior": "..."}}]}}
"""

EM_GOLD_VARIANTS_PROMPT = """\
You are writing the exam KEY for held-out polysemy traps.

TARGET DOMAIN: {domain}
{context}

Each item is a bare question (no domain name). Eval answers are at most \
10 words, no lists. The 1-3 word gold is only a gloss — write ONE full \
{domain} answer in the same format as eval.

For EVERY item below write:
- gold_response: one answer of at most 10 words a {domain} practitioner \
would give to the BARE question. Do not name {domain}. Do not mention \
the other-field reading.

Items:
{item_lines}

Return strict JSON:
{{"items": [{{"term": "...", "gold_response": "..."}}]}}
"""


def gold_words(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def normalize_em(text: str) -> str:
    words = [w for w in gold_words(text) if w not in _ARTICLES]
    return " ".join(words)


def gold_ok(text: str) -> bool:
    n = len(gold_words(text))
    return MIN_GOLD_WORDS <= n <= MAX_GOLD_WORDS and bool(normalize_em(text))


def response_ok(text: str) -> bool:
    n = len(gold_words(text))
    return 4 <= n <= MAX_RESPONSE_WORDS and bool(normalize_em(text))


def _phrase_list(value) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out, seen = [], set()
    for raw in value:
        text = " ".join(str(raw or "").split()).strip()
        key = normalize_em(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def clean_aliases(raw) -> List[str]:
    return [x for x in _phrase_list(raw) if gold_ok(x)]


def clean_responses(raw) -> List[str]:
    return [x for x in _phrase_list(raw) if response_ok(x)]


def item_phrases(item: dict, *keys: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for key in keys:
        for text in _phrase_list(item.get(key)):
            norm = normalize_em(text)
            if norm in seen:
                continue
            seen.add(norm)
            out.append(text)
    return out


def item_gold_keys(item: dict) -> List[str]:
    return [x for x in item_phrases(item, "gold", "gold_aliases") if gold_ok(x)]


def item_trap_keys(item: dict) -> List[str]:
    return item_phrases(item, "baseline_trap", "trap_aliases")


def item_gold_text(item: dict) -> str:
    return " ".join(item_phrases(
        item, "gold", "gold_aliases", "gold_responses"))


def item_trap_text(item: dict) -> str:
    return " ".join(item_phrases(
        item, "baseline_trap", "trap_aliases", "trap_responses"))


def item_gold_response(item: dict) -> str:
    """The single frozen reference sentence scored against."""
    raw = item.get("gold_response")
    if raw:
        texts = clean_responses([raw])
        if texts:
            return texts[0]
    texts = clean_responses(item.get("gold_responses"))
    return texts[0] if texts else ""


def item_has_gold_responses(item: dict) -> bool:
    return bool(item_gold_response(item))


def question_leaks(question: str, domain: str,
                   extra: Optional[Sequence[str]] = None) -> bool:
    text = question or ""
    if _LEAK_RE.search(text):
        return True
    needles = [domain, *(extra or [])]
    lowered = text.lower()
    for needle in needles:
        name = " ".join(str(needle or "").lower().split())
        if len(name) >= 5 and name in lowered:
            return True
        for token in name.split():
            if len(token) >= 8 and token in lowered:
                return True
    return False


def term_in_question(term: str, question: str) -> bool:
    t = normalize_em(term)
    q = normalize_em(question)
    if not t:
        return False
    return t in q or all(w in q.split() for w in t.split())


def prompt_easy_cs_def(term: str, question: str) -> bool:
    """True if this is a CS textbook def a domain system prompt instantly solves."""
    t = (term or "").strip().lower()
    q = " ".join((question or "").split())
    m = _EASY_CS_Q_RE.match(q)
    if not m:
        return False
    word = m.group("term").lower()
    return t in _EASY_CS_TERMS or word in _EASY_CS_TERMS


def em_hit(prediction: str, gold: str) -> dict:
    pred = normalize_em(prediction)
    key = normalize_em(gold)
    exact = bool(key) and pred == key
    contained = bool(key) and (key in pred or all(w in pred.split() for w in key.split()))
    return {
        "exact": exact,
        "contained": contained,
        "normalized_pred": pred,
        "normalized_gold": key,
    }


def em_hit_any(prediction: str, golds: Sequence[str]) -> dict:
    best = em_hit(prediction, "")
    for gold in golds:
        hit = em_hit(prediction, gold)
        if hit["exact"]:
            return hit
        if hit["contained"] and not best.get("contained"):
            best = hit
    return best


_SENTENCE_SPLIT = re.compile(r'(?<=[.!?…])["\'”’)\]]*\s+')


def clip_eval_answer(text: str, max_sentences: int = 1,
                     max_words: int = EM_EVAL_MAX_WORDS) -> str:
    """Keep eval answers ≤ ``max_words`` (default 10)."""
    text = " ".join((text or "").split()).strip()
    if not text:
        return ""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    kept = " ".join(parts[:max(1, max_sentences)]) if parts else text
    words = kept.split()
    if len(words) > max_words:
        kept = " ".join(words[:max_words]).rstrip(" ,;:")
    return kept


def format_item(item: dict) -> str:
    prior = item.get("dominant_prior")
    prior_line = f"\n* **Prior:** {prior}" if prior else ""
    aliases = item_gold_keys(item)
    alias_line = ""
    extra = [a for a in aliases if normalize_em(a) != normalize_em(item.get("gold") or "")]
    if extra:
        alias_line = f"\n* **Gold aliases:** {'; '.join(extra[:6])}"
    gold_text = item_gold_response(item)
    resp_line = f"\n* **Gold response:** {gold_text}" if gold_text else ""
    return (
        f"* **Target Word:** {item.get('term')}{prior_line}\n"
        f"* **Q:** {item.get('question')}\n"
        f"* **Baseline Trap:** {item.get('baseline_trap')}\n"
        f"* **Gold Standard:** {item.get('gold')}{alias_line}{resp_line}"
    )


def _plural_phrases(phrase: str) -> Set[str]:
    k = normalize_em(phrase)
    if not k:
        return set()
    out = {k}
    words = k.split()
    last = words[-1]
    if not last.endswith("s"):
        out.add(" ".join(words[:-1] + [last + "s"]) if len(words) > 1 else last + "s")
    elif last.endswith("s") and not last.endswith("ss") and len(last) > 3:
        out.add(" ".join(words[:-1] + [last[:-1]]) if len(words) > 1 else last[:-1])
    return out


_FILLER_TAIL = {
    "meaning", "identity", "order", "status", "class", "profile",
    "assignment", "standard", "presence", "resistance", "tolerance",
}
_COMMON_HEAD = {
    "protein", "sequence", "genetic", "genome", "target", "plant", "trait",
    "dna", "rna", "gene", "data", "line", "product", "expression",
    "translation", "quality", "stress", "field", "whole", "seed", "base",
    "read", "identity", "meaning", "status", "order", "class", "profile",
    "material", "sample", "organism", "crop", "code", "host", "tag", "run",
    "call", "bank", "stack", "draft", "stock", "coverage", "payload",
    "release", "cassette", "specific", "functional", "molecular", "desired",
}


def key_phrases(key: str) -> Set[str]:
    """Gold *and* alias strings that count as a mention.

    ``amino acid sequence`` → ``amino acid`` / ``amino acids``.
    ``codon meaning`` → ``codon`` / ``codons``.
    ``herbicide tolerance`` / ``pest resistance`` → ``herbicide`` / ``pest``.
    ``taxon ID`` and ``preliminary assembly`` stay whole phrases.
    """
    k = normalize_em(key)
    if not k:
        return set()
    out = set(_plural_phrases(k))
    words = k.split()
    if len(words) >= 3:
        out |= _plural_phrases(" ".join(words[:-1]))
    elif len(words) == 2:
        head, tail = words[0], words[1]
        # Only drop a filler tail (meaning, resistance, …). Do not turn every
        # 2-word alias into its first word.
        if tail in _FILLER_TAIL and head not in _COMMON_HEAD and len(head) >= 4:
            out |= _plural_phrases(head)
    return out


def mentions_key(text: str, key: str) -> bool:
    """True if ``text`` contains the gold/alias as a phrase."""
    blob = f" {normalize_em(text)} "
    return any(f" {phrase} " in blob for phrase in key_phrases(key))


def mention_hits(text: str, keys: Sequence[str]) -> List[str]:
    hits, seen = [], set()
    for key in keys:
        if not mentions_key(text, key):
            continue
        norm = normalize_em(key)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        hits.append(key)
    return hits


def gold_mentioned(text: str, item: dict) -> List[str]:
    """Which gold standard or gold-alias phrases appear in ``text``."""
    keys = item_phrases(item, "gold", "gold_aliases")
    if not keys:
        keys = [k for k in [item.get("gold")] if k]
    return mention_hits(text, keys)


def apply_variants(item: dict, raw: Optional[dict] = None) -> dict:
    """Merge a GPT gold expansion onto one trap item.

    Scoring uses a single ``gold_responses[0]``. Older files may still carry
    aliases; they are kept if present but are not generated.
    """
    src = raw if isinstance(raw, dict) else {}
    out = dict(item)
    aliases = clean_aliases(
        list(_phrase_list(item.get("gold_aliases")))
        + list(_phrase_list(src.get("gold_aliases")))
    )
    gold_key = normalize_em(out.get("gold") or "")
    if aliases:
        out["gold_aliases"] = [a for a in aliases if normalize_em(a) != gold_key][:8]
    golds = clean_responses(
        list(_phrase_list(src.get("gold_response")))
        + list(_phrase_list(src.get("gold_responses")))
        + list(_phrase_list(item.get("gold_responses")))
        + list(_phrase_list(item.get("gold_response")))
    )
    if golds:
        out["gold_responses"] = golds[:DEFAULT_N_GOLD_RESPONSES]
    trap_aliases = clean_aliases(
        list(_phrase_list(item.get("trap_aliases")))
        + list(_phrase_list(src.get("trap_aliases")))
    )
    trap_key = normalize_em(out.get("baseline_trap") or "")
    if trap_aliases:
        out["trap_aliases"] = [
            a for a in trap_aliases if normalize_em(a) != trap_key
        ][:6]
    traps = clean_responses(
        list(_phrase_list(item.get("trap_responses")))
        + list(_phrase_list(src.get("trap_responses")))
    )
    if traps:
        out["trap_responses"] = traps
    return out


def clean_em_items(raw, domain: str, *, blocked: Optional[Iterable[str]] = None,
                   extra_leak: Optional[Sequence[str]] = None) -> List[dict]:
    blocked_keys: Set[str] = {_match_key(x) for x in (blocked or []) if x}
    out = []
    seen = set()
    if not isinstance(raw, list):
        return out
    for row in raw:
        if not isinstance(row, dict):
            continue
        term = " ".join(str(row.get("term") or "").split()).strip()
        question = " ".join(str(row.get("question") or "").split()).strip()
        trap = " ".join(str(row.get("baseline_trap") or "").split()).strip()
        gold = " ".join(str(row.get("gold") or "").split()).strip()
        if not (term and question and trap and gold):
            continue
        if not gold_ok(gold):
            continue
        if question_leaks(question, domain, extra_leak):
            continue
        if not term_in_question(term, question):
            continue
        if prompt_easy_cs_def(term, question):
            continue
        key = _match_key(term)
        qkey = _match_key(question)
        if key in blocked_keys or qkey in blocked_keys or key in seen or qkey in seen:
            continue
        seen.add(key)
        seen.add(qkey)
        prior = " ".join(str(row.get("dominant_prior") or "").split()).strip()
        out.append(apply_variants({
            "term": term,
            "question": question,
            "baseline_trap": trap,
            "gold": gold,
            "dominant_prior": prior,
            "gold_aliases": row.get("gold_aliases"),
            "gold_response": row.get("gold_response"),
            "gold_responses": row.get("gold_responses"),
            "trap_aliases": row.get("trap_aliases"),
            "trap_responses": row.get("trap_responses"),
        }))
    return out


# --- Bundled benchmark ------------------------------------------------------
#
# The polysemy-trap exam shipped as package data: 144 domains x 15 items. Each
# item is a bare question whose overloaded term has to be read in the domain
# sense. `gold_responses` holds exactly ONE reference per item — the single
# frozen target every arm is scored against.

BUNDLED_BENCHMARK_ROOT = Path(__file__).resolve().parent / "data" / "benchmark"


def benchmark_manifest() -> Optional[dict]:
    """Manifest for the bundled benchmark, or None if it is not installed."""
    path = BUNDLED_BENCHMARK_ROOT / "manifest.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_benchmark_domains() -> List[dict]:
    """The benchmark's domains: id, dir, cluster, division, n_items."""
    manifest = benchmark_manifest()
    return list(manifest.get("domains", [])) if manifest else []


def benchmark_path(domain: str) -> Optional[Path]:
    """Path to one domain's benchmark file, or None.

    `domain` may be a cluster id ('3106'), the file stem
    ('3106-industrial-biotechnology') or the cluster name
    ('Industrial biotechnology').
    """
    manifest = benchmark_manifest()
    if manifest is None:
        return None
    query = str(domain).strip()
    slug = _slugify(query)
    for entry in manifest.get("domains", []):
        name = entry["dir"]
        if query == entry["id"] or slug == name or slug == name.split("-", 1)[1]:
            path = BUNDLED_BENCHMARK_ROOT / f"{name}.json"
            return path if path.is_file() else None
    return None


def load_benchmark(domain: str) -> Optional[dict]:
    """One domain's benchmark: metadata plus `items`, or None if absent."""
    path = benchmark_path(domain)
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_benchmark() -> Iterator[dict]:
    """Every bundled domain in id order."""
    for entry in list_benchmark_domains():
        data = load_benchmark(entry["id"])
        if data is not None:
            yield data

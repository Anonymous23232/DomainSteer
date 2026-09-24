"""Gold-vs-steered similarity for the self-gold sweep.

Default plotted score is a current dense+sparse hybrid:

- dense — cosine of a current embedding model (default OpenAI
  text-embedding-3-large; local option google/embeddinggemma-300m)
- sparse — BM25 between the two answers, IDF fit on the domain texts
- hybrid — 0.7 dense + 0.3 length-normalized BM25

That fusion is the usual 2024–2026 hybrid-search recipe (dense embedding +
BM25), not a 2006–2017 lexical/Word2Vec mix.

Each item is scored twice:
    gold_vs_answer — baseline at alpha=0, steered at alpha>0
    steered_vs_baseline — how far steering moved the unsteered answer
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
DEFAULT_EMBED_MODEL = "text-embedding-3-large"
LOCAL_EMBED_MODEL = "google/embeddinggemma-300m"
DENSE_WEIGHT = 0.7
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _alpha_key(value) -> str:
    return f"{float(value):.4f}"


def nested_steered(steered) -> bool:
    """True when steered is {layer: {alpha: text}} rather than {alpha: text}."""
    if not isinstance(steered, dict) or not steered:
        return False
    first = next(iter(steered.values()))
    return isinstance(first, dict)


def _steered_texts(steered) -> List[str]:
    texts: List[str] = []
    if not isinstance(steered, dict):
        return texts
    for value in steered.values():
        if isinstance(value, dict):
            texts.extend(str(text or "") for text in value.values())
        else:
            texts.append(str(value or ""))
    return texts


def collect_texts(records: Sequence[dict]) -> List[str]:
    texts: List[str] = []
    for row in records:
        texts.append(str(row.get("gold") or ""))
        texts.append(str(row.get("baseline") or ""))
        texts.append(str(row.get("gpt_answer") or ""))
        texts.append(str(row.get("short_sys") or ""))
        texts.append(str(row.get("unsteered") or ""))
        texts.append(str(row.get("contextual") or ""))
        texts.extend(_steered_texts(row.get("steered") or {}))
    return [t for t in texts if t.strip()]


def _qnorm(text) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def attach_short_sys(records: Sequence[dict], domain_dir: Path) -> List[dict]:
    """Copy missing ``short_sys`` fields from the domain's gold.json."""
    by_q: Dict[str, str] = {}
    gold_path = Path(domain_dir) / "gold.json"
    if gold_path.exists():
        payload = json.loads(gold_path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else payload.get("items") or []
        for item in items:
            text = " ".join(str(item.get("short_sys") or "").split()).strip()
            key = _qnorm(item.get("question"))
            if key and text:
                by_q[key] = text
    out: List[dict] = []
    for row in records:
        view = dict(row)
        existing = " ".join(str(view.get("short_sys") or "").split()).strip()
        if not existing:
            found = by_q.get(_qnorm(view.get("question")), "")
            if found:
                view["short_sys"] = found
        out.append(view)
    return out


def as_gpt_gold_records(records: Sequence[dict]) -> List[dict]:
    """GPT exam answer as gold; Llama domain-prompt gold as contextual.

    The unsteered default stays in `baseline`. Llama's domain-prompt answer
    is `contextual`. Scoring then reports baseline vs both golds, and GPT
    gold vs steered at each alpha.
    """
    out: List[dict] = []
    for row in records:
        gpt = " ".join(str(row.get("gpt_answer") or "").split()).strip()
        llama_gold = " ".join(str(row.get("gold") or "").split()).strip()
        if not gpt or not llama_gold:
            continue
        view = dict(row)
        view["gold"] = gpt
        view["contextual"] = llama_gold
        view["gpt_answer"] = gpt
        out.append(view)
    return out


def as_short_sys_records(records: Sequence[dict]) -> List[dict]:
    """GPT exam answer as gold; Llama short-system-prompt answer as contextual.

    The everyday unsteered answer stays in `baseline`, so alpha 0 on the
    GPT-vs-steered curve is still the unnamed default. Scoring also reports
    GPT vs short-sys (constant) and short-sys vs steered at each alpha.
    """
    out: List[dict] = []
    for row in records:
        gpt = " ".join(str(row.get("gpt_answer") or "").split()).strip()
        short = " ".join(str(row.get("short_sys") or "").split()).strip()
        if not gpt or not short:
            continue
        view = dict(row)
        view["gold"] = gpt
        view["contextual"] = short
        view["gpt_answer"] = gpt
        view["short_sys"] = short
        out.append(view)
    return out


def flatten_layer_records(records: Sequence[dict]) -> List[dict]:
    """One record per layer when steered is nested layer → alpha → text."""
    out: List[dict] = []
    for row in records:
        steered = row.get("steered") or {}
        if not nested_steered(steered):
            out.append(row)
            continue
        for layer, by_alpha in steered.items():
            view = dict(row)
            try:
                view["layer"] = int(layer)
            except (TypeError, ValueError):
                continue
            view["steered"] = by_alpha if isinstance(by_alpha, dict) else {}
            out.append(view)
    return out


def hash_embeddings(texts: Sequence[str], dim: int = 32) -> Dict[str, np.ndarray]:
    """Deterministic stand-in vectors for tests. Do not use for plots."""
    out: Dict[str, np.ndarray] = {}
    for text in texts:
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=dim).astype(np.float64)
        norm = float(np.linalg.norm(vec))
        out[text] = vec / norm if norm else vec
    return out


@dataclass
class BM25Index:
    idf: Dict[str, float]
    avgdl: float
    n_docs: int
    k1: float = BM25_K1
    b: float = BM25_B

    @classmethod
    def fit(cls, corpus: Sequence[Sequence[str]],
            k1: float = BM25_K1, b: float = BM25_B) -> "BM25Index":
        n_docs = max(len(corpus), 1)
        df: Counter = Counter()
        lengths = []
        for doc in corpus:
            lengths.append(len(doc))
            for term in set(doc):
                df[term] += 1
        avgdl = (sum(lengths) / max(len(lengths), 1)) or 1.0
        idf = {
            term: math.log(1.0 + (n_docs - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }
        return cls(idf=idf, avgdl=avgdl, n_docs=n_docs, k1=k1, b=b)

    def score(self, query: Sequence[str], doc: Sequence[str]) -> float:
        if not query or not doc:
            return 0.0
        freqs = Counter(doc)
        dl = len(doc)
        total = 0.0
        for term in query:
            freq = freqs.get(term, 0)
            if freq <= 0:
                continue
            idf = self.idf.get(term, 0.0)
            denom = freq + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            total += idf * (freq * (self.k1 + 1.0)) / denom
        return total


def bm25_similarity(toks_a: Sequence[str], toks_b: Sequence[str],
                    index: BM25Index) -> float:
    """Symmetric BM25, each side divided by the self-score so identical=1."""
    aa = index.score(toks_a, toks_a)
    bb = index.score(toks_b, toks_b)
    ab = index.score(toks_a, toks_b)
    ba = index.score(toks_b, toks_a)
    left = min(ab / aa, 1.0) if aa > 0 else 0.0
    right = min(ba / bb, 1.0) if bb > 0 else 0.0
    return 0.5 * (left + right)


@dataclass
class Encoders:
    embeddings: Dict[str, np.ndarray]
    bm25: BM25Index
    embed_model: str = "hash"
    dense_weight: float = DENSE_WEIGHT
    _empty: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))

    def vector(self, text: str) -> np.ndarray:
        vec = self.embeddings.get(text)
        if vec is None:
            return self._empty
        return vec


def fit_encoders(records: Sequence[dict],
                 embeddings: Optional[Dict[str, np.ndarray]] = None,
                 embed_model: str = "hash",
                 dense_weight: float = DENSE_WEIGHT) -> Encoders:
    texts = collect_texts(records)
    tokenized = [tokenize(t) for t in texts]
    if embeddings is None:
        embeddings = hash_embeddings(texts)
    empty_dim = 1
    if embeddings:
        empty_dim = next(iter(embeddings.values())).shape[0]
    return Encoders(
        embeddings=embeddings,
        bm25=BM25Index.fit(tokenized),
        embed_model=embed_model,
        dense_weight=dense_weight,
        _empty=np.zeros(empty_dim, dtype=np.float64),
    )


def hybrid_similarity(left: str, right: str, enc: Encoders) -> dict:
    dense = cosine(enc.vector(left), enc.vector(right))
    dense = max(0.0, min(1.0, dense))
    sparse = bm25_similarity(tokenize(left), tokenize(right), enc.bm25)
    hybrid = enc.dense_weight * dense + (1.0 - enc.dense_weight) * sparse
    return {
        "dense": dense,
        "bm25": sparse,
        "hybrid": hybrid,
    }


def alphas_for_record(row: dict) -> List[float]:
    values = [0.0]
    steered = row.get("steered") or {}
    if nested_steered(steered):
        for by_alpha in steered.values():
            if isinstance(by_alpha, dict):
                values.extend(float(k) for k in by_alpha)
    elif isinstance(steered, dict):
        values.extend(float(k) for k in steered)
    listed = row.get("alphas") or []
    for raw in listed:
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    uniq = sorted({round(a, 4) for a in values})
    return uniq


def candidate_text(row: dict, alpha: float) -> str:
    if abs(alpha) < 1e-12:
        return str(row.get("unsteered") or row.get("baseline") or "")
    steered = row.get("steered") or {}
    if not isinstance(steered, dict):
        return ""
    key = _alpha_key(alpha)
    if key in steered:
        return str(steered[key] or "")
    for raw, text in steered.items():
        try:
            if abs(float(raw) - alpha) < 1e-9:
                return str(text or "")
        except (TypeError, ValueError):
            continue
    return ""


def score_records(records: Sequence[dict], enc: Encoders) -> List[dict]:
    scored = []
    for row in flatten_layer_records(records):
        gold = str(row.get("gold") or "")
        baseline = str(row.get("baseline") or "")
        contextual = str(row.get("contextual") or "")
        if contextual:
            scored.append(_score_row(
                row, 0.0, "baseline_vs_contextual",
                hybrid_similarity(baseline, contextual, enc),
            ))
            scored.append(_score_row(
                row, 0.0, "gold_vs_contextual",
                hybrid_similarity(gold, contextual, enc),
            ))
        for alpha in alphas_for_record(row):
            answer = candidate_text(row, alpha)
            gold_metrics = hybrid_similarity(gold, answer, enc)
            scored.append(_score_row(row, alpha, "gold_vs_answer", gold_metrics))
            vs_base = hybrid_similarity(answer, baseline, enc)
            scored.append(_score_row(row, alpha, "steered_vs_baseline", vs_base))
            if contextual:
                scored.append(_score_row(
                    row, alpha, "contextual_vs_answer",
                    hybrid_similarity(contextual, answer, enc),
                ))
    return scored


def _score_row(row: dict, alpha: float, pair: str, metrics: dict) -> dict:
    payload = {
        "index": row.get("index"),
        "term": row.get("term"),
        "kind": row.get("kind") or "unknown",
        "question": row.get("question"),
        "alpha": round(float(alpha), 4),
        "pair": pair,
        "dense": metrics["dense"],
        "bm25": metrics["bm25"],
        "hybrid": metrics["hybrid"],
    }
    layer = row.get("layer")
    if layer is not None:
        try:
            payload["layer"] = int(layer)
        except (TypeError, ValueError):
            pass
    chosen = row.get("chosen_layer")
    if chosen is not None:
        try:
            payload["chosen_layer"] = int(chosen)
        except (TypeError, ValueError):
            pass
    return payload


ASSIGN_LLAMA = "llama"
ASSIGN_GPT = "gpt"
ASSIGN_TIE = "tie"
ASSIGN_LABELS = (ASSIGN_TIE, ASSIGN_LLAMA, ASSIGN_GPT)
GPT_SCORE_PAIR = "gold_vs_answer"
LLAMA_SCORE_PAIR = "contextual_vs_answer"


def assign_key(llama_score: Optional[float], gpt_score: Optional[float]) -> str:
    """Closer gold wins. Tie only when the two scores are exactly equal."""
    if llama_score is None or gpt_score is None:
        return ASSIGN_TIE
    if float(llama_score) > float(gpt_score):
        return ASSIGN_LLAMA
    if float(gpt_score) > float(llama_score):
        return ASSIGN_GPT
    return ASSIGN_TIE


def assignment_rows(scored: Sequence[dict], field: str = "dense") -> List[dict]:
    """One row per question × alpha: Llama vs GPT similarity vote."""
    by_q: Dict[Tuple, dict] = {}
    for row in scored:
        pair = row.get("pair")
        if pair not in (GPT_SCORE_PAIR, LLAMA_SCORE_PAIR):
            continue
        key = (row.get("index"), row.get("kind"), row.get("question"))
        item = by_q.setdefault(key, {
            "index": row.get("index"),
            "kind": row.get("kind") or "unknown",
            "term": row.get("term") or "",
            "question": row.get("question") or "",
            "gpt": {},
            "llama": {},
        })
        alpha = round(float(row["alpha"]), 4)
        bucket = item["gpt"] if pair == GPT_SCORE_PAIR else item["llama"]
        bucket[alpha] = {
            "dense": float(row["dense"]),
            "bm25": float(row["bm25"]),
            "hybrid": float(row["hybrid"]),
        }
    out: List[dict] = []
    for item in by_q.values():
        alphas = sorted(set(item["gpt"]) | set(item["llama"]))
        for alpha in alphas:
            gpt_pt = item["gpt"].get(alpha) or {}
            llama_pt = item["llama"].get(alpha) or {}
            gpt_s = gpt_pt.get(field)
            llama_s = llama_pt.get(field)
            out.append({
                "index": item["index"],
                "kind": item["kind"],
                "term": item["term"],
                "question": item["question"],
                "alpha": alpha,
                "field": field,
                "gpt_score": gpt_s,
                "llama_score": llama_s,
                "margin": (llama_s - gpt_s
                           if llama_s is not None and gpt_s is not None else None),
                "assign": assign_key(llama_s, gpt_s),
            })
    out.sort(key=lambda r: (str(r["kind"]), int(r["index"] or 0), r["alpha"]))
    return out


def assignment_summary(rows: Sequence[dict]) -> List[dict]:
    groups: Dict[Tuple, List[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["alpha"], row["kind"])].append(row)
        groups[(row["alpha"], "all")].append(row)
    out = []
    for (alpha, kind), items in sorted(groups.items()):
        n = len(items)
        counts = {label: sum(1 for r in items if r["assign"] == label)
                  for label in ASSIGN_LABELS}
        rec = {
            "alpha": alpha,
            "kind": kind,
            "n": n,
            **{f"n_{label}": counts[label] for label in ASSIGN_LABELS},
            **{f"p_{label}": (counts[label] / n if n else None)
               for label in ASSIGN_LABELS},
        }
        out.append(rec)
    return out


def summarize(scored: Sequence[dict]) -> List[dict]:
    groups: Dict[Tuple[float, str, str], List[dict]] = defaultdict(list)
    for row in scored:
        pair = row.get("pair") or "gold_vs_answer"
        groups[(row["alpha"], row["kind"], pair)].append(row)
        groups[(row["alpha"], "all", pair)].append(row)
    out = []
    for (alpha, kind, pair), rows in sorted(groups.items()):
        out.append({
            "alpha": alpha,
            "kind": kind,
            "pair": pair,
            "n": len(rows),
            "hybrid_mean": float(np.mean([r["hybrid"] for r in rows])),
            "hybrid_std": float(np.std([r["hybrid"] for r in rows])),
            "dense_mean": float(np.mean([r["dense"] for r in rows])),
            "bm25_mean": float(np.mean([r["bm25"] for r in rows])),
        })
    return out


def summarize_by_layer(scored: Sequence[dict]) -> List[dict]:
    groups: Dict[Tuple[int, float, str, str], List[dict]] = defaultdict(list)
    for row in scored:
        if row.get("layer") is None:
            continue
        pair = row.get("pair") or "gold_vs_answer"
        layer = int(row["layer"])
        groups[(layer, row["alpha"], row["kind"], pair)].append(row)
        groups[(layer, row["alpha"], "all", pair)].append(row)
    out = []
    for (layer, alpha, kind, pair), rows in sorted(groups.items()):
        out.append({
            "layer": layer,
            "alpha": alpha,
            "kind": kind,
            "pair": pair,
            "n": len(rows),
            "hybrid_mean": float(np.mean([r["hybrid"] for r in rows])),
            "hybrid_std": float(np.std([r["hybrid"] for r in rows])),
            "dense_mean": float(np.mean([r["dense"] for r in rows])),
            "bm25_mean": float(np.mean([r["bm25"] for r in rows])),
        })
    return out


def heatmap_matrix(summary: Sequence[dict], kind: str,
                   pair: str = "gold_vs_answer"
                   ) -> Tuple[List[int], List[float], np.ndarray]:
    rows = [
        r for r in summary
        if r.get("kind") == kind and r.get("pair") == pair
        and r.get("layer") is not None
    ]
    layers = sorted({int(r["layer"]) for r in rows})
    alphas = sorted({round(float(r["alpha"]), 4) for r in rows})
    grid = np.full((len(layers), len(alphas)), np.nan, dtype=np.float64)
    lookup = {
        (int(r["layer"]), round(float(r["alpha"]), 4)): float(r["hybrid_mean"])
        for r in rows
    }
    for i, layer in enumerate(layers):
        for j, alpha in enumerate(alphas):
            value = lookup.get((layer, alpha))
            if value is not None:
                grid[i, j] = value
    return layers, alphas, grid


def average_heatmaps(
    grids: Sequence[Tuple[List[int], List[float], np.ndarray]],
) -> Tuple[List[int], List[float], np.ndarray]:
    """Mean hybrid over domains; missing cells stay NaN."""
    layers = sorted({layer for ls, _, _ in grids for layer in ls})
    alphas = sorted({alpha for _, als, _ in grids for alpha in als})
    acc = np.zeros((len(layers), len(alphas)), dtype=np.float64)
    counts = np.zeros_like(acc)
    layer_i = {layer: i for i, layer in enumerate(layers)}
    alpha_j = {alpha: j for j, alpha in enumerate(alphas)}
    for ls, als, grid in grids:
        for i, layer in enumerate(ls):
            for j, alpha in enumerate(als):
                value = grid[i, j]
                if np.isnan(value):
                    continue
                acc[layer_i[layer], alpha_j[alpha]] += value
                counts[layer_i[layer], alpha_j[alpha]] += 1
    mean = np.full_like(acc, np.nan)
    np.divide(acc, counts, out=mean, where=counts > 0)
    return layers, alphas, mean


_EMBEDDER_CACHE: Dict[str, object] = {}


def load_embedder(model: str = LOCAL_EMBED_MODEL):
    if model in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[model]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required: pip install -e \".[eval]\""
        ) from exc
    encoder = SentenceTransformer(model)
    _EMBEDDER_CACHE[model] = encoder
    return encoder


def encode_texts(texts: Sequence[str], encoder) -> Dict[str, np.ndarray]:
    unique = list(dict.fromkeys(t for t in texts if t))
    if not unique:
        return {}
    vectors = encoder.encode(
        unique, normalize_embeddings=True, show_progress_bar=False, batch_size=16,
    )
    return {
        text: np.asarray(vec, dtype=np.float64)
        for text, vec in zip(unique, vectors)
    }


def openai_embed_model_id(model: str) -> Optional[str]:
    """Return an OpenAI embedding id, or None if `model` is a local embedder."""
    name = model.split("/", 1)[-1] if model.startswith("openai/") else model
    if name.startswith("text-embedding-"):
        return name
    return None


def resolve_embed_route(model: str, backend: str = "auto") -> Tuple[str, str]:
    """Pick (model_id, 'openai'|'local') from --embed-model and --backend."""
    openai_id = openai_embed_model_id(model)
    if backend == "openai":
        used = openai_id or DEFAULT_EMBED_MODEL
        return used, "openai"
    if backend == "local":
        if openai_id:
            raise ValueError(
                f"Local backend cannot use OpenAI embedder {model!r}"
            )
        return model, "local"
    if backend == "auto":
        if openai_id:
            return openai_id, "openai"
        return model, "local"
    raise ValueError(f"Unknown embedding backend {backend!r}")


def _openai_embeddings(texts: Sequence[str], model: str) -> Dict[str, np.ndarray]:
    from extras.benchmark import embed_texts
    unique = list(dict.fromkeys(t for t in texts if t))
    out: Dict[str, np.ndarray] = {}
    chunk = 256
    name = openai_embed_model_id(model) or DEFAULT_EMBED_MODEL
    for start in range(0, len(unique), chunk):
        batch = unique[start:start + chunk]
        vectors = embed_texts(batch, model=name)
        for text, vec in zip(batch, vectors):
            out[text] = np.asarray(vec, dtype=np.float64)
    return out


def embeddings_for_texts(texts: Sequence[str], model: str = DEFAULT_EMBED_MODEL,
                         backend: str = "auto"
                         ) -> Tuple[Dict[str, np.ndarray], str]:
    """Return {text: vector} and the model id actually used."""
    unique = [t for t in dict.fromkeys(texts) if t]
    model, route = resolve_embed_route(model, backend)
    if not unique:
        return {}, model
    if route == "local":
        try:
            print(f"Encoding {len(unique)} texts with local {model}")
            return encode_texts(unique, load_embedder(model)), model
        except Exception as exc:
            if backend == "local":
                raise
            print(f"Local embedder unavailable ({exc}). Using OpenAI.")
            model = DEFAULT_EMBED_MODEL
            route = "openai"
    print(f"Encoding {len(unique)} texts with OpenAI {model}")
    return _openai_embeddings(unique, model), model


def domain_plot_dir(results_dir: Path, model_slug: str, domain_name: str,
                    plots_name: str = "plots") -> Path:
    """<plots_name>/<model>/<domain-slug>/ — per-domain scores and figures."""
    return Path(results_dir) / plots_name / model_slug / domain_name


def iter_scored_plot_json_paths(plots_root: Path) -> Iterable[Path]:
    """Yield domain score JSON: flat ``<model>/<domain>.json`` or nested
    ``<model>/<domain>/*.json``. Skip overview dumps."""
    root = Path(plots_root)
    if not root.exists():
        return
    seen = set()
    for pattern in ("*/*.json", "*/*/*.json"):
        for path in sorted(root.glob(pattern)):
            if path.name.startswith("overview"):
                continue
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            yield path


def domain_slug_from_plot_path(path: Path) -> str:
    """Domain folder name for a scored plot JSON, either layout."""
    path = Path(path)
    if path.stem in {"similarity", "heatmap", "scores"}:
        return path.parent.name
    return path.stem


def plot_json_identity(path: Path, payload: dict) -> Tuple[str, str, str]:
    """Return (model, cluster_id, cluster) from payload, with path fallbacks."""
    nested = len(path.parts) >= 2 and path.stem in {
        "similarity", "heatmap", "scores",
    }
    parent = path.parent
    model = str(payload.get("model") or (
        parent.parent.name if nested else parent.name
    ))
    cluster_name = parent.name if nested else path.stem
    cluster = str(payload.get("cluster") or cluster_name)
    cluster_id = str(
        payload.get("cluster_id") or cluster_name.split("-")[0]
    )
    return model, cluster_id, cluster


def load_domain_records(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return payload["items"]
    raise ValueError(f"Unexpected responses format at {path}")


def _iter_result_records(results_root: Path, names: Sequence[str]
                         ) -> Iterable[Tuple[str, Path, List[dict]]]:
    root = Path(results_root)
    if not root.exists():
        return
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if (model_dir.name in {"cache", "plots", "controls"}
                or model_dir.name.startswith("plots")):
            continue
        for domain_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            path = None
            for name in names:
                candidate = domain_dir / name
                if candidate.exists():
                    path = candidate
                    break
            if path is None:
                continue
            if path.suffix == ".jsonl":
                records = [json.loads(line) for line in path.read_text(
                    encoding="utf-8").splitlines() if line.strip()]
            else:
                records = load_domain_records(path)
            if records:
                yield model_dir.name, domain_dir, records


def iter_self_gold_domains(results_root: Path
                           ) -> Iterable[Tuple[str, Path, List[dict]]]:
    yield from _iter_result_records(
        results_root, ("responses.json", "responses.jsonl"),
    )


def iter_self_gold_layer_domains(results_root: Path
                                 ) -> Iterable[Tuple[str, Path, List[dict]]]:
    yield from _iter_result_records(
        results_root, ("layer_sweep.json", "layer_sweep.jsonl"),
    )

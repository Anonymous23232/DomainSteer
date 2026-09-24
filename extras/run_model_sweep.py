"""Reproduce the four-arm layer x strength grid on any model, from the package alone.

The reported results cover two Llama models because those are the two whose
contrastive pairs ship in `domainsteer/data/pairs/`. This script adds a third
(or any other) model by having it write its own pairs first, then running the
same grid and the same scoring, so the new numbers are directly comparable.

    stage pairs       the target model answers each self-gold stem twice --
                      once under the expert prompt, once under the default one
                      -- and the difference becomes its contrastive pairs
    stage directions  one unit vector per layer, from those pairs
    stage sweep       the four arms on the bundled 2,160-item trap benchmark
    stage score       all-mpnet-base-v2 cosine to each item's single gold

Everything it needs comes from the installed package:

    domainsteer.pairs        bundled stems (shared across models, see below)
    domainsteer.em_traps     the trap benchmark, prompts, clipping
    domainsteer.self_gold    the expert prompt used to elicit the positive side
    domainsteer.extract      DirectionExtractor, load_model
    domainsteer.steering     ActivationSteering, DEFAULT_SYSTEM_PROMPT

Why the stems can be reused: the bundled pair sets are model-specific in their
*text* -- each model wrote both sides -- but their question stems are identical
across the shipped models (verified for all 144 domains). The stems are the
self-gold exam; only the answers are the model's own. So a new model answers the
same stems, and its vector is built the same way.

    python extras/run_model_sweep.py --dry-run
    python extras/run_model_sweep.py --ids 3106 3107        # pilot
    python extras/run_model_sweep.py --all                  # full grid
    python extras/run_model_sweep.py --all --shard 0/2      # one per GPU
    python extras/run_model_sweep.py --stages score         # rescore only

Resume-safe: every stage writes after each domain (the sweep, after each item)
and re-runs skip work already on disk.

Output, under --out (default results/model-sweep/<model-slug>/):

    <id>-<slug>/pairs.jsonl        the model's own contrastive pairs
    <id>-<slug>/directions/        direction_layer_<L>.npy + meta.json
    <id>-<slug>/responses.json     items[i][arm][layer][alpha] = answer
    scores.csv                     one row per (domain, arm, layer, alpha)
    summary.txt                    the four-arm table
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # repo root (extras/ -> repo)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domainsteer.clusters import find_cluster, load_clusters
from domainsteer.em_traps import (EM_CAA_SYSTEM_PROMPT, EM_EVAL_MAX_WORDS,
                                  EM_MAX_NEW_TOKENS, benchmark_manifest,
                                  clip_eval_answer, em_system_prompt, load_benchmark)
from domainsteer.pairs import (ContrastivePair, _slugify, list_bundled_models,
                               load_bundled_pairs, load_pairs, save_pairs)
from domainsteer.self_gold import SELF_GOLD_CONTRAST_ID, gold_system_prompt

logger = logging.getLogger("model_sweep")

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_LAYERS = list(range(10, 21))                       # 11 layers
DEFAULT_ALPHAS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]    # 7 strengths
STEERED_ARMS = ("caa", "prompt_caa")
ALL_STAGES = ("pairs", "directions", "sweep", "score")
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
PAIR_MAX_NEW_TOKENS = 96          # pair answers are 1-2 sentences, not exam-length
CONTRAST_ID = "model_sweep_v1"


# ----------------------------------------------------------------- utilities

def akey(alpha: float) -> str:
    return f"{alpha:g}"


def qkey(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def load_json(path: Path):
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def fmt_hours(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def stem_source(preferred: Optional[str]) -> str:
    """Which bundled set to take the self-gold stems from."""
    models = list_bundled_models()
    if not models:
        raise SystemExit(
            "No bundled pair sets found. This script reads the self-gold stems "
            "from domainsteer/data/pairs/; reinstall the package."
        )
    if preferred:
        slug = _slugify(preferred)
        if slug not in models:
            raise SystemExit(f"{preferred!r} is not bundled. Have: {', '.join(models)}")
        return preferred
    return models[0]


def domain_stems(source_model: str, cluster_id: str) -> List[str]:
    pairs = load_bundled_pairs(source_model, cluster_id)
    if not pairs:
        return []
    seen, stems = set(), []
    for pair in pairs:
        stem = " ".join(str(pair.concept or "").split()).strip()
        if stem and stem.lower() not in seen:
            seen.add(stem.lower())
            stems.append(stem)
    return stems


def select_clusters(args) -> list:
    clusters = load_clusters()
    if args.ids:
        picked = []
        for ident in args.ids:
            cluster = find_cluster(clusters, ident)
            if cluster is None:
                raise SystemExit(f"No domain matches {ident!r}.")
            picked.append(cluster)
    else:
        picked = list(clusters)
    if args.limit:
        picked = picked[: args.limit]
    if args.shard:
        index, total = (int(x) for x in args.shard.split("/"))
        picked = [c for i, c in enumerate(picked) if i % total == index]
    return picked


class Paths:
    def __init__(self, out_root: Path, cluster):
        self.domain = out_root / f"{cluster.cluster_id}-{_slugify(cluster.cluster)}"
        self.pairs = self.domain / "pairs.jsonl"
        self.pairs_meta = self.domain / "pairs.meta.json"
        self.directions = self.domain / "directions"
        self.responses = self.domain / "responses.json"


# -------------------------------------------------------------------- model

class Runner:
    """One loaded model; ActivationSteering cached per (layer, system prompt)."""

    def __init__(self, model_name: str, mock: bool = False):
        self.model_name = model_name
        self.mock = mock
        self.model = self.tokenizer = None
        self.env = {"mock": mock}
        if not mock:
            import torch
            import transformers
            from domainsteer.extract import load_model
            self.model, self.tokenizer = load_model(model_name)
            self.env.update({
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "device": (torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else "cpu"),
            })
        self.directions: Dict[int, np.ndarray] = {}
        self._steering: Dict[tuple, object] = {}

    def set_directions(self, directions: Dict[int, np.ndarray]) -> None:
        self.directions = directions
        self._steering = {}

    def _steerer(self, layer: int, system_prompt: str):
        key = (layer, system_prompt)
        steering = self._steering.get(key)
        if steering is None:
            from domainsteer.steering import ActivationSteering
            steering = ActivationSteering(self.model, self.tokenizer,
                                          self.directions[layer], layer,
                                          system_prompt=system_prompt)
            self._steering[key] = steering
        return steering

    def plain(self, question: str, system_prompt: str, max_new_tokens: int) -> str:
        """Unsteered generation, used for the pair stage (no direction yet)."""
        if self.mock:
            return f"mock[{system_prompt.split()[3]}] {question}"
        from domainsteer.steering import ActivationSteering
        key = ("plain", system_prompt)
        steering = self._steering.get(key)
        if steering is None:
            zero = np.zeros(self.model.config.hidden_size, dtype=np.float32)
            zero[0] = 1.0            # unit vector; never applied, alpha stays 0
            steering = ActivationSteering(self.model, self.tokenizer, zero, 0,
                                          system_prompt=system_prompt)
            self._steering[key] = steering
        return steering.generate(question, alpha=0.0, max_new_tokens=max_new_tokens)

    def exam(self, question: str, system_prompt: str, layer: int, alpha: float,
             max_new_tokens: int, max_words: int, max_sentences: int) -> str:
        clip = dict(max_sentences=max_sentences, max_words=max_words)
        if self.mock:
            return clip_eval_answer(
                f"mock L{layer} a{alpha:g} {question}", **clip)
        steering = self._steerer(layer, system_prompt)
        return clip_eval_answer(
            steering.generate(question, alpha=alpha, max_new_tokens=max_new_tokens),
            **clip)


# --------------------------------------------------------------- stage: pairs

def stage_pairs(cluster, paths: Paths, runner: Runner, args, source_model: str) -> bool:
    if paths.pairs.is_file() and not args.force_pairs:
        return True
    stems = domain_stems(source_model, cluster.cluster_id)
    if not stems:
        print(f"  [{cluster.cluster_id}] no bundled stems; skipped", flush=True)
        return False

    expert_sys = gold_system_prompt(cluster.cluster)
    from domainsteer.steering import DEFAULT_SYSTEM_PROMPT

    pairs: List[ContrastivePair] = []
    for i, stem in enumerate(stems, 1):
        expert = " ".join(runner.plain(stem, expert_sys, PAIR_MAX_NEW_TOKENS).split())
        default = " ".join(runner.plain(stem, DEFAULT_SYSTEM_PROMPT,
                                        PAIR_MAX_NEW_TOKENS).split())
        if not expert or not default or expert == default:
            continue
        pairs.append(ContrastivePair(expert, default, stem))
        if args.verbose:
            print(f"    [{cluster.cluster_id}] stem {i}/{len(stems)}", flush=True)

    if len(pairs) < args.min_pairs:
        print(f"  [{cluster.cluster_id}] only {len(pairs)} usable pairs "
              f"(need {args.min_pairs}); skipped", flush=True)
        return False

    save_pairs(pairs, paths.pairs)
    write_json(paths.pairs_meta, {
        "contrast": SELF_GOLD_CONTRAST_ID,
        "positive": "self",
        "model": runner.model_name,
        "n": len(pairs),
        "stems_from": source_model,
        "expert_system_prompt": expert_sys,
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
        "max_new_tokens": PAIR_MAX_NEW_TOKENS,
    })
    print(f"  [{cluster.cluster_id}] wrote {len(pairs)} pairs", flush=True)
    return True


# ---------------------------------------------------------- stage: directions

def stage_directions(cluster, paths: Paths, runner: Runner, args) -> bool:
    have = paths.directions / "meta.json"
    if have.is_file() and not args.force_directions:
        meta = load_json(have) or {}
        if set(args.layers).issubset({int(L) for L in meta.get("layers", [])}):
            return True
    if not paths.pairs.is_file():
        return False
    if runner.mock:
        paths.directions.mkdir(parents=True, exist_ok=True)
        dim = 8
        for layer in args.layers:
            v = np.zeros(dim, dtype=np.float32)
            v[layer % dim] = 1.0
            np.save(paths.directions / f"direction_layer_{layer}.npy", v)
        write_json(paths.directions / "meta.json",
                   {"model_name": runner.model_name, "layers": list(args.layers),
                    "estimator": "mock", "holdout_accuracy": {}, "holdout_margin": {},
                    "n_train_pairs": 0, "n_holdout_pairs": 0})
        return True

    from domainsteer.extract import DirectionExtractor
    pairs = load_pairs(paths.pairs)
    extractor = DirectionExtractor(runner.model, runner.tokenizer)
    result = extractor.extract(pairs, layers=list(args.layers),
                               model_name=runner.model_name,
                               estimator=args.estimator)
    result.save(paths.directions)
    accs = result.holdout_accuracy
    best = max(accs, key=accs.get) if accs else None
    print(f"  [{cluster.cluster_id}] directions for {len(result.directions)} layers"
          + (f"; best holdout acc layer {best} = {accs[best]:.2f}" if best else ""),
          flush=True)
    return True


# --------------------------------------------------------------- stage: sweep

def missing_cells(row: dict, layers: Sequence[int], alphas: Sequence[float],
                  arms: Sequence[str]) -> int:
    n = int(not row.get("unsteered")) + int(not row.get("prompt"))
    for arm in arms:
        cells = row.get(arm) or {}
        for layer in layers:
            cell = cells.get(str(layer)) or {}
            n += sum(1 for a in alphas if not str(cell.get(akey(a)) or "").strip())
    return n


def stage_sweep(cluster, paths: Paths, runner: Runner, args, clock: dict) -> bool:
    bench = load_benchmark(cluster.cluster_id)
    if bench is None:
        print(f"  [{cluster.cluster_id}] not in the bundled benchmark; skipped",
              flush=True)
        return False
    items = bench["items"]

    meta = load_json(paths.directions / "meta.json") or {}
    have_layers = {int(L) for L in meta.get("layers", [])}
    layers = [L for L in args.layers if L in have_layers]
    if not layers:
        print(f"  [{cluster.cluster_id}] no directions; run --stages directions",
              flush=True)
        return False
    runner.set_directions({
        L: np.load(paths.directions / f"direction_layer_{L}.npy") for L in layers})

    prev = {} if args.force_sweep else {
        qkey(r.get("question")): r
        for r in (load_json(paths.responses) or {}).get("items") or []}
    rows = {qkey(it["question"]): prev.get(qkey(it["question"])) or {} for it in items}

    helper = EM_CAA_SYSTEM_PROMPT
    prompt_sys = em_system_prompt(cluster.cluster)
    arms = args.arms
    mnt = args.max_new_tokens

    def save():
        write_json(paths.responses, {
            "cluster_id": cluster.cluster_id,
            "cluster": cluster.cluster,
            "model_name": runner.model_name,
            "contrast": CONTRAST_ID,
            "benchmark_contrast": bench.get("contrast"),
            "layers": sorted(layers),
            "alphas": sorted(args.alphas),
            "arms": list(arms),
            "max_new_tokens": mnt,
            "decoding": (f"greedy; clip_eval_answer ({args.max_sentences} sentence(s), "
                         f"<={args.max_words} words)"),
            "unsteered_system_prompt": helper,
            "prompt_system_prompt": prompt_sys,
            "caa_system_prompt": helper,
            "prompt_caa_system_prompt": prompt_sys,
            "env": runner.env,
            "n": len(items),
            "items": [rows[qkey(it["question"])] for it in items
                      if rows.get(qkey(it["question"]))],
        })

    def gen(question, system_prompt, layer, alpha):
        return runner.exam(question, system_prompt, layer, alpha, mnt,
                           args.max_words, args.max_sentences)

    base_layer = layers[0]
    for i, item in enumerate(items, 1):
        question = item["question"]
        row = rows[qkey(question)]
        for field in ("term", "question", "baseline_trap", "gold",
                      "gold_aliases", "gold_responses", "dominant_prior"):
            if item.get(field) is not None:
                row[field] = item[field]
        todo = missing_cells(row, layers, args.alphas, arms)
        if todo == 0:
            continue
        t0 = time.time()
        # alpha = 0 registers no hook, so the layer is irrelevant for these two
        if not row.get("unsteered"):
            row["unsteered"] = gen(question, helper, base_layer, 0.0)
        if not row.get("prompt"):
            row["prompt"] = gen(question, prompt_sys, base_layer, 0.0)
        for arm in arms:
            system_prompt = helper if arm == "caa" else prompt_sys
            cells = row.setdefault(arm, {})
            for layer in layers:
                cell = cells.setdefault(str(layer), {})
                for alpha in args.alphas:
                    if not str(cell.get(akey(alpha)) or "").strip():
                        cell[akey(alpha)] = gen(question, system_prompt, layer, alpha)
        rows[qkey(question)] = row
        save()
        clock["done"] += todo
        clock["busy"] += time.time() - t0
        rate = clock["busy"] / max(1, clock["done"])
        left = max(0, clock["total"] - clock["done"])
        print(f"  [{cluster.cluster_id}] item {i:2d}/{len(items)}  +{todo} gens  "
              f"{rate:.2f} s/gen  {clock['done']}/{clock['total']}  "
              f"ETA {fmt_hours(left * rate)}", flush=True)
    save()
    return True


# --------------------------------------------------------------- stage: score

def stage_score(clusters, out_root: Path, args) -> None:
    from sentence_transformers import SentenceTransformer

    files = []
    for cluster in clusters:
        path = Paths(out_root, cluster).responses
        data = load_json(path)
        if data and data.get("items"):
            files.append((cluster, data))
    if not files:
        raise SystemExit("Nothing to score: no responses.json found under --out.")

    texts = set()
    for _, data in files:
        for item in data["items"]:
            refs = item.get("gold_responses") or []
            if refs:
                texts.add(refs[0])
            for arm in ("unsteered", "prompt"):
                if item.get(arm):
                    texts.add(item[arm])
            for arm in STEERED_ARMS:
                for cell in (item.get(arm) or {}).values():
                    for answer in cell.values():
                        if answer:
                            texts.add(answer)
    texts = sorted(t for t in texts if t and t.strip())
    print(f"embedding {len(texts)} unique strings with {EMBED_MODEL} ...", flush=True)
    encoder = SentenceTransformer(EMBED_MODEL)
    matrix = encoder.encode(texts, batch_size=args.batch_size,
                            convert_to_numpy=True, normalize_embeddings=True,
                            show_progress_bar=not args.quiet)
    vec = {t: matrix[i] for i, t in enumerate(texts)}

    def sim(answer, reference):
        a, b = vec.get(answer or ""), vec.get(reference or "")
        return float(a @ b) if a is not None and b is not None else None

    rows = []
    for cluster, data in files:
        for item in data["items"]:
            refs = item.get("gold_responses") or []
            if not refs:
                continue
            gold = refs[0]
            base = {"cluster_id": cluster.cluster_id, "cluster": cluster.cluster,
                    "term": item.get("term"), "question": item.get("question")}
            unsteered = sim(item.get("unsteered"), gold)
            prompt = sim(item.get("prompt"), gold)
            for arm in ("unsteered", "prompt"):
                score = unsteered if arm == "unsteered" else prompt
                if score is not None:
                    rows.append({**base, "arm": arm, "layer": "", "alpha": "",
                                 "sim": score, "unsteered_sim": unsteered,
                                 "prompt_sim": prompt})
            for arm in STEERED_ARMS:
                for layer, cell in (item.get(arm) or {}).items():
                    for alpha, answer in cell.items():
                        score = sim(answer, gold)
                        if score is None:
                            continue
                        rows.append({**base, "arm": arm, "layer": int(layer),
                                     "alpha": float(alpha), "sim": score,
                                     "unsteered_sim": unsteered, "prompt_sim": prompt})

    csv_path = out_root / "scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}: {len(rows)} rows", flush=True)

    summary = summarize(rows, files)
    (out_root / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)


def summarize(rows, files) -> str:
    """Per-domain best cell, and how often it beats the two reference arms."""
    by_domain: Dict[str, Dict] = {}
    for row in rows:
        d = by_domain.setdefault(row["cluster_id"], {
            "cluster": row["cluster"], "cells": {}, "unsteered": [], "prompt": []})
        if row["arm"] == "unsteered":
            d["unsteered"].append(row["sim"])
        elif row["arm"] == "prompt":
            d["prompt"].append(row["sim"])
        else:
            key = (row["arm"], row["layer"], row["alpha"])
            cell = d["cells"].setdefault(key, [])
            cell.append((row["sim"], row["unsteered_sim"], row["prompt_sim"]))

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    lines = []
    lines.append("=" * 78)
    lines.append(f"FOUR-ARM GRID  {len(by_domain)} domains")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Each domain scored at its best of the layer x strength cells, chosen")
    lines.append("on the same items it is then reported on -- an upper bound on the")
    lines.append("direction, not a deployable policy. Read it as such.")
    lines.append("")

    for arm, reference in (("caa", "unsteered"), ("caa", "prompt"),
                           ("prompt_caa", "prompt")):
        wins = beaten = 0
        best_alphas, best_layers = [], []
        for cid, d in by_domain.items():
            cells = {k: v for k, v in d["cells"].items() if k[0] == arm}
            if not cells:
                continue
            beaten += 1
            best_key = max(cells, key=lambda k: mean([s for s, _, _ in cells[k]]))
            triples = cells[best_key]
            majority = sum(
                1 for s, u, p in triples
                if s is not None and (u if reference == "unsteered" else p) is not None
                and s > (u if reference == "unsteered" else p))
            if majority * 2 > len(triples):
                wins += 1
            best_layers.append(best_key[1])
            best_alphas.append(best_key[2])
        if not beaten:
            continue
        lo, hi = (min(best_alphas), max(best_alphas)) if best_alphas else (0, 0)
        lines.append(
            f"  {arm:11} vs {reference:10}  majority of items closer to gold in "
            f"{wins}/{beaten} domains")
        lines.append(
            f"  {'':11}    {'':10}  best strengths {lo:g}-{hi:g}, "
            f"layers {min(best_layers)}-{max(best_layers)}")
        lines.append("")

    lines.append("Mean gold similarity, pooled over every item and domain:")
    for arm in ("unsteered", "prompt"):
        vals = [r["sim"] for r in rows if r["arm"] == arm]
        lines.append(f"  {arm:11} {mean(vals):.4f}   ({len(vals)} answers)")
    for arm in STEERED_ARMS:
        vals = [r["sim"] for r in rows if r["arm"] == arm]
        if vals:
            lines.append(f"  {arm:11} {mean(vals):.4f}   "
                         f"({len(vals)} answers over all cells)")
    lines.append("")
    models = {data.get("model_name") for _, data in files}
    lines.append(f"model(s): {', '.join(sorted(m for m in models if m))}")
    lines.append(f"scored against one gold reference per item ({EMBED_MODEL})")
    return "\n".join(lines)


# ---------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=None,
                    help="default results/model-sweep/<model-slug>/")
    ap.add_argument("--ids", nargs="+", help="cluster ids or names; default all")
    ap.add_argument("--all", action="store_true", help="every domain (the default)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", help="i/n, e.g. 0/2 to split across two GPUs")
    ap.add_argument("--stages", default=",".join(ALL_STAGES),
                    help=f"comma-separated subset of {ALL_STAGES}")
    ap.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    ap.add_argument("--arms", default=",".join(STEERED_ARMS),
                    help=f"comma-separated subset of {STEERED_ARMS}")
    ap.add_argument("--estimator", default="diff_means", choices=("diff_means", "rfm"))
    ap.add_argument("--stem-model", default=None,
                    help="bundled set to take self-gold stems from "
                         "(default: the first; stems are identical across them)")
    ap.add_argument("--max-new-tokens", type=int, default=EM_MAX_NEW_TOKENS)
    ap.add_argument("--max-words", type=int, default=EM_EVAL_MAX_WORDS)
    ap.add_argument("--max-sentences", type=int, default=1)
    ap.add_argument("--min-pairs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--force-pairs", action="store_true")
    ap.add_argument("--force-directions", action="store_true")
    ap.add_argument("--force-sweep", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and check inputs; load no model")
    ap.add_argument("--mock", action="store_true",
                    help="run the whole pipeline with stub generations")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(message)s")
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = set(args.arms) - set(STEERED_ARMS)
    if bad:
        raise SystemExit(f"Unknown arm(s): {sorted(bad)}; choose from {STEERED_ARMS}")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = set(stages) - set(ALL_STAGES)
    if bad:
        raise SystemExit(f"Unknown stage(s): {sorted(bad)}; choose from {ALL_STAGES}")

    source_model = stem_source(args.stem_model)
    out_root = args.out or (ROOT / "results" / "model-sweep" / _slugify(args.model))
    clusters = select_clusters(args)
    manifest = benchmark_manifest()
    if manifest is None:
        raise SystemExit("The bundled benchmark is missing; reinstall the package.")

    per_item = 2 + len(args.arms) * len(args.layers) * len(args.alphas)
    n_items = sum(len(load_benchmark(c.cluster_id)["items"])
                  for c in clusters if load_benchmark(c.cluster_id))
    total_gens = per_item * n_items
    pair_gens = 2 * sum(len(domain_stems(source_model, c.cluster_id)) for c in clusters)

    print(f"model          {args.model}")
    print(f"out            {out_root}")
    print(f"domains        {len(clusters)}")
    print(f"stems from     {source_model} (identical across bundled models)")
    print(f"benchmark      {manifest['n_items_total']} items, "
          f"{manifest['n_domains']} domains, gold = {manifest['gold_selection']}")
    print(f"grid           {len(args.layers)} layers x {len(args.alphas)} strengths "
          f"= {len(args.layers) * len(args.alphas)} settings per arm")
    print(f"arms           unsteered, prompt, {', '.join(args.arms)}")
    print(f"generations    {pair_gens} for pairs + {total_gens} for the sweep")
    print(f"stages         {', '.join(stages)}")
    if args.dry_run:
        missing = [c.cluster_id for c in clusters
                   if not domain_stems(source_model, c.cluster_id)]
        print(f"domains without bundled stems: {missing or 'none'}")
        return 0

    need_model = any(s in stages for s in ("pairs", "directions", "sweep"))
    runner = Runner(args.model, mock=args.mock) if need_model else None
    clock = {"done": 0, "busy": 0.0, "total": total_gens}

    for n, cluster in enumerate(clusters, 1):
        paths = Paths(out_root, cluster)
        print(f"[{n}/{len(clusters)}] {cluster.cluster_id} {cluster.cluster}", flush=True)
        if "pairs" in stages and not stage_pairs(cluster, paths, runner, args,
                                                 source_model):
            continue
        if "directions" in stages and not stage_directions(cluster, paths, runner, args):
            continue
        if "sweep" in stages:
            stage_sweep(cluster, paths, runner, args, clock)

    if "score" in stages:
        stage_score(clusters, out_root, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

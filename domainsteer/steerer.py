"""DomainSteerer: the end-to-end user API.

Pipeline: domain (registry cluster / explicit concepts) → contrastive
in-domain / field-neutral pair generation → direction extraction →
steered generation with a raw `alpha`. Pass `layer=` to `build()`; that
path is the installed package.

NLI layer pick and the `expertise` dial live in extras/ and are not in
the wheel. `build()` without `layer=` lazy-imports them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

from domainsteer.clusters import Domain
from domainsteer.directions import DEFAULT_ESTIMATOR, ESTIMATORS
from domainsteer.pairs import (CONTRAST_ID, DEFAULT_GENERATION_MODEL,
                               PairGenerator, TRAP_CONTRAST_ID,
                               TRAP_LOCAL_CONTRAST_ID, TRAP_LOCAL_VARIANT,
                               _slugify)
from domainsteer.self_gold import SELF_GOLD_GPT_VARIANT

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "steering_config.json"


def _import_extras(module: str):
    """Load an extras.* module. Not shipped in the installed wheel."""
    try:
        return __import__(f"extras.{module}", fromlist=["*"])
    except ImportError as exc:
        raise ImportError(
            f"This path needs extras.{module}, which is not part of the "
            "installed package. Pass layer= to build() and call "
            "generate(..., alpha=...), or run from a checkout that "
            "includes extras/."
        ) from exc


def _artifact_suffix(estimator: str, variant: Optional[str] = None) -> str:
    """Per-estimator / per-variant cache suffix; empty for the gold default.

    The default estimator keeps writing to `directions/` and
    `steering_config.json` exactly as before, so switching estimators is a
    side-by-side comparison rather than a destructive migration — nothing
    already built is overwritten or silently reused under a new name.
    Trap directions live in `directions-trap/` (and `directions-trap-rfm/`
    if the estimator is not the default). Gold-vs-baseline trap directions
    live in `directions-trap_local/`.
    """
    bits = []
    if variant:
        bits.append(variant)
    if estimator != DEFAULT_ESTIMATOR:
        bits.append(estimator)
    return ("-" + "-".join(bits)) if bits else ""


class DomainSteerer:
    """
    Steer `model_name` toward expertise in `domain`.

    Usage:
        steerer = DomainSteerer(
            model_name=...,
            domain=Domain("hydrology", concepts=["aquifer", "baseflow"]),
        )
        steerer.build(layer=15)              # no-op if already built (cached)
        steerer.generate(prompt, alpha=0.25)
        steerer.compare(prompt, alpha=0.25)

    `domain` is a Domain (name + explicit concepts) or a bare name string —
    the latter resolves against the domain-cluster registry, or uses the
    domain's cached pairs when it has them. All artifacts (concepts, pairs,
    directions, config) live under `cache_dir/<domain-slug>/` and are reused
    across runs.

    For the 144 registry domains on Llama-3.1-8B-Instruct and
    Llama-3.2-3B-Instruct, contrastive pairs ship with the package, so
    `build(layer=…)` needs no API key. See
    `domainsteer.pairs.list_bundled_models`.
    """

    def __init__(self, model_name: str, domain: Union[str, Domain],
                 cache_dir: Optional[Union[str, Path]] = None,
                 api_key: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 generation_model: str = DEFAULT_GENERATION_MODEL,
                 estimator: str = DEFAULT_ESTIMATOR,
                 variant: Optional[str] = None,
                 use_bundled_pairs: bool = True):
        if estimator not in ESTIMATORS:
            raise ValueError(f"Unknown estimator '{estimator}'; choose from "
                             f"{sorted(ESTIMATORS)}.")
        if variant not in (None, "", "trap", TRAP_LOCAL_VARIANT,
                           SELF_GOLD_GPT_VARIANT):
            raise ValueError(
                f"Unknown variant {variant!r}; use None, 'trap', "
                f"'{TRAP_LOCAL_VARIANT}', or '{SELF_GOLD_GPT_VARIANT}'."
            )
        self.domain_spec = domain if isinstance(domain, Domain) else Domain(domain)
        self.model_name = model_name
        self.domain = self.domain_spec.name
        self.system_prompt = system_prompt
        self.estimator = estimator
        self.variant = variant or None

        self._cache_dir = cache_dir
        self._api_key = api_key
        self._use_bundled_pairs = use_bundled_pairs
        if self.variant == TRAP_LOCAL_VARIANT:
            contrast = TRAP_LOCAL_CONTRAST_ID
        elif self.variant == "trap":
            contrast = TRAP_CONTRAST_ID
        else:
            contrast = CONTRAST_ID
        self._pair_generator = PairGenerator(
            self.domain, cache_dir=cache_dir, api_key=api_key,
            generation_model=generation_model, contrast=contrast,
        )
        self.domain_dir = self._pair_generator.domain_dir
        self.model_dir = self.domain_dir / _slugify(model_name)
        if self.variant == TRAP_LOCAL_VARIANT:
            trap = _import_extras("trap")
            pairs_path, meta_path = trap.local_pairs_paths(self.model_dir)
            self._pair_generator.pairs_path = pairs_path
            self._pair_generator.pairs_meta_path = meta_path

        suffix = _artifact_suffix(estimator, self.variant)
        self.directions_name = f"directions{suffix}"
        self.directions_dir = self.model_dir / self.directions_name
        self.config_path = self.model_dir / f"steering_config{suffix}.json"

        self.layer: Optional[int] = None
        self.max_alpha: Optional[float] = None
        self.best_alpha: Optional[float] = None
        self._direction = None
        self._model = None
        self._tokenizer = None
        self._steering = None
        self._expert_steering = None
        self._trap_config_ok = True
        self._loaded_judge: Optional[str] = None

        if self.config_path.exists():
            self._load_config()

    # ------------------------------------------------------------------ build

    def build(self, force: bool = False,
              pairwise: Optional[bool] = None, pairwise_judge=None,
              force_pairs: bool = False,
              stems: Optional[list] = None,
              probe_prompts: Optional[list] = None,
              pairwise_golds: Optional[list] = None,
              pairs: Optional[list] = None,
              layer: Optional[int] = None) -> None:
        """Run (or resume) the full pipeline; cached stages are skipped.

        `force` redoes the model-side work — direction extraction, layer
        selection, calibration — and always reuses cached pairs, because they
        cost API calls. Regenerating them is opt-in via `force_pairs` (or
        `domainsteer pairs --force`).

        Pair resolution order: cached pairs under `cache_dir`, then the pairs
        bundled as package data for this (model, domain), then this model
        answering the shared 30 stems (no API call), then API generation.
        The bundled sets are model-specific — each model wrote both sides of
        its own pairs — and carry the `self_gold_expert_vs_default_v1`
        contrast. Turn them off with `use_bundled_pairs=False`.

        Pass `layer=` to skip extras entirely: extract directions and use
        that layer. That is the installed-package path; `generate` then
        takes a raw `alpha`. Without `layer=`, `build()` lazy-imports
        extras/ for NLI (or pairwise) layer pick and the expertise dial.

        `stems` and `probe_prompts` are for the trap variant: extract from
        answers to the trap questions, then probe/calibrate on those same
        stems instead of the unnamed identity bank.

        `pairs` skips API generation and extracts from the supplied
        contrastive pairs (used by trap_local: gold vs this model's baseline).

        `pairwise=True` swaps the zero-shot NLI judge for a `PairwiseJudge`
        in both layer selection and strength calibration. It also measures a
        coherence ceiling *before* the layer sweep and probes at fractions of
        it, rather than at fixed constants that may exceed what the model
        tolerates. Trap defaults to pairwise (sense judge + golds);
        gold/unnamed stay on NLI unless this is passed explicitly.
        """
        if pairwise is None:
            pairwise = self.variant in ("trap", TRAP_LOCAL_VARIANT)
        if self._build_is_current(force=force, pairwise=pairwise, layer=layer):
            logger.info(f"Already built (layer {self.layer}, "
                        f"max_alpha {self.max_alpha}); use force=True to redo.")
            return

        from domainsteer.extract import DirectionExtractor, ExtractionResult

        resolved = pairs if pairs is not None else self._resolve_pairs(
            force=force_pairs, stems=stems)
        self._ensure_model()

        directions_dir = self.directions_dir
        reuse_directions = (directions_dir / "meta.json").exists() and not force
        if self.variant in ("trap", TRAP_LOCAL_VARIANT) and not self._trap_config_ok:
            reuse_directions = False
        want = int(layer) if layer is not None else None
        extraction = None
        if reuse_directions:
            logger.info(f"Reusing extracted directions from {directions_dir}")
            extraction = ExtractionResult.load(directions_dir)
            if extraction.estimator != self.estimator:
                raise ValueError(
                    f"{directions_dir} holds '{extraction.estimator}' "
                    f"directions but '{self.estimator}' was requested. Delete "
                    "that directory or rebuild with force=True."
                )
        if extraction is None or (want is not None and want not in extraction.directions):
            # build(layer=15) must extract 15. The default sweep is the even
            # middle-third layers (10, 12, 14, 16, 18 on Llama-3.1-8B), so a
            # requested odd layer is extracted on its own and merged in.
            extractor = DirectionExtractor(self._model, self._tokenizer)
            fresh = extractor.extract(
                resolved, layers=None if want is None else [want],
                model_name=self.model_name, estimator=self.estimator)
            if extraction is None:
                extraction = fresh
            else:
                extraction.directions[want] = fresh.directions[want]
                extraction.holdout_accuracy[want] = fresh.holdout_accuracy.get(want, 0.0)
                if want in fresh.holdout_margin:
                    extraction.holdout_margin[want] = fresh.holdout_margin[want]
            extraction.save(directions_dir)

        extra: dict = {}
        if layer is not None:
            best_layer = int(layer)
            if best_layer not in extraction.directions:
                raise ValueError(
                    f"No extracted direction for layer {best_layer}; "
                    f"have {sorted(extraction.directions)}"
                )
            layer_scores = {
                L: extraction.holdout_accuracy.get(L, 0.0)
                for L in extraction.directions
            }
            max_alpha = None
            extra["judge"] = "none"
            steering = self._make_steering(
                extraction.directions[best_layer], best_layer)
        else:
            cal = _import_extras("calibrate")
            judge_mod = _import_extras("judge")
            prompts = list(probe_prompts) if probe_prompts else []
            if not prompts and self.variant in ("trap", TRAP_LOCAL_VARIANT):
                prompts = [p.concept for p in resolved[:5] if p.concept]
            if not prompts:
                prompts = cal.default_test_prompts(self.domain)

            if pairwise:
                best_layer, layer_scores, max_alpha, extra = self._build_pairwise(
                    extraction, prompts, pairwise_judge, golds=pairwise_golds,
                )
                steering = self._make_steering(
                    extraction.directions[best_layer], best_layer)
            else:
                judge = judge_mod.ExpertiseJudge(self.domain)
                best_layer, layer_scores = cal.select_best_layer(
                    self._model, self._tokenizer, extraction, judge, prompts
                )
                steering = self._make_steering(
                    extraction.directions[best_layer], best_layer)
                max_alpha = cal.calibrate_steering(steering, prompts[:2])

        self.layer = best_layer
        self.max_alpha = max_alpha
        self._direction = extraction.directions[best_layer]
        self._steering = steering
        self._loaded_judge = extra.get("judge") or (
            "none" if layer is not None else ("pairwise" if pairwise else "nli")
        )

        config = {
            "domain": self.domain,
            "model_name": self.model_name,
            "layer": best_layer,
            "layer_scores": {str(k): v for k, v in layer_scores.items()},
            "max_alpha": max_alpha,
            "direction_file":
                f"{self.directions_name}/direction_layer_{best_layer}.npy",
            "steering_scale": "hidden_norm_relative",
            "estimator": self.estimator,
            "holdout_margin": {str(k): v
                               for k, v in extraction.holdout_margin.items()},
            "judge": self._loaded_judge,
            "variant": self.variant or "",
            "contrast": self._pair_generator.contrast,
            **extra,
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Built: layer {best_layer}, max_alpha {max_alpha} "
                    f"→ {self.config_path}")

    # --------------------------------------------------------------- generate

    def generate(self, prompt: str, expertise: Optional[float] = None,
                 alpha: Optional[float] = None,
                 max_new_tokens: int = 96,
                 steer_new_tokens: Optional[int] = None) -> str:
        """
        Generate with domain-expertise steering. Pass exactly one of:
            expertise — 0.0 (baseline) to 1.0 (this model's calibrated
                        maximum); negative steers away from expertise.
            alpha     — raw norm-relative strength (typical 0-0.25).
        """
        if (expertise is None) == (alpha is None):
            raise ValueError("Pass exactly one of `expertise` or `alpha`.")
        if self.layer is None:
            raise ValueError("Not built for this model/domain yet — run .build().")

        if expertise is not None:
            if self.max_alpha is None:
                raise ValueError(
                    "No calibrated max_alpha (calibration failed); "
                    "pass a raw `alpha` instead."
                )
            alpha = expertise * self.max_alpha

        self._ensure_steering()
        return self._steering.generate(prompt, alpha=alpha,
                                       max_new_tokens=max_new_tokens,
                                       steer_new_tokens=steer_new_tokens)

    def compare(self, prompt: str, expertise: Optional[float] = None,
                alpha: Optional[float] = None,
                max_new_tokens: int = 96,
                steer_new_tokens: Optional[int] = None) -> dict:
        """Baseline and steered response for the same prompt. Defaults to
        expertise=1.0. Returns {"prompt", "baseline", "steered", "expertise",
        "alpha"}."""
        if expertise is None and alpha is None:
            expertise = 1.0

        baseline = self.generate(prompt, alpha=0.0, max_new_tokens=max_new_tokens)
        if alpha is not None:
            steered = self.generate(prompt, alpha=alpha,
                                    max_new_tokens=max_new_tokens,
                                    steer_new_tokens=steer_new_tokens)
        else:
            steered = self.generate(prompt, expertise=expertise,
                                    max_new_tokens=max_new_tokens,
                                    steer_new_tokens=steer_new_tokens)
        effective_alpha = alpha if alpha is not None else expertise * self.max_alpha
        return {
            "prompt": prompt,
            "baseline": baseline,
            "steered": steered,
            "expertise": expertise,
            "alpha": effective_alpha,
        }

    def benchmark(self, n: int = 50, expertise: float = 1.0,
                  alpha: Optional[float] = None,
                  max_new_tokens: int = 96,
                  force_questions: bool = False,
                  similarity: bool = True,
                  question_set: Optional[str] = None,
                  tau: float = 0.50,
                  st_model: Optional[str] = None,
                  pairwise: bool = True,
                  steer_new_tokens: Optional[int] = None) -> dict:
        """Baseline vs steered on unnamed identity, gold exam, or trap exam."""
        if self.layer is None:
            raise ValueError("Not built for this model/domain yet — run .build().")
        if alpha is None:
            if self.max_alpha is None:
                raise ValueError(
                    "No calibrated max_alpha (calibration failed); "
                    "pass a raw `alpha` instead."
                )
            alpha = expertise * self.max_alpha

        bench = _import_extras("benchmark")
        trap = _import_extras("trap")

        if question_set is None:
            question_set = trap.SET_TRAP if self.variant in (
                "trap", TRAP_LOCAL_VARIANT) else bench.SET_UNNAMED
        if question_set not in (bench.SET_UNNAMED, bench.SET_GOLD, trap.SET_TRAP):
            raise ValueError(
                f"Unknown benchmark set {question_set!r}; "
                f"use {bench.SET_UNNAMED!r}, {bench.SET_GOLD!r}, or {trap.SET_TRAP!r}."
            )

        golds = None
        gold_items = []
        trap_items = []
        if question_set == bench.SET_GOLD:
            gold_items = bench.BenchmarkGold(
                self.domain, cache_dir=self._cache_dir, api_key=self._api_key,
                generation_model=self._pair_generator.generation_model,
            ).generate(n=n, force=force_questions)
            questions = [item["question"] for item in gold_items]
            golds = [item["answer"] for item in gold_items]
        elif question_set == trap.SET_TRAP:
            trap_items = trap.BenchmarkTrap(
                self.domain, cache_dir=self._cache_dir, api_key=self._api_key,
                generation_model=self._pair_generator.generation_model,
            ).generate(n=n, force=force_questions)
            questions = [item["question"] for item in trap_items]
            golds = [item["answer"] for item in trap_items]
        else:
            questions = bench.BenchmarkQuestions(
                cache_dir=self._cache_dir, api_key=self._api_key,
                generation_model=self._pair_generator.generation_model,
            ).generate(n=n, force=force_questions)

        self._ensure_steering()
        baselines, steered = self._generate_baseline_steered(
            questions, alpha, max_new_tokens, golds=golds,
            steer_new_tokens=steer_new_tokens)

        if question_set == bench.SET_GOLD:
            print("Hybrid eval: embeddings, concept coverage"
                  + (", LLM rubric + pairwise + open gold" if pairwise else "")
                  + "...\n", flush=True)
            results = bench.grade_gold_run(
                self.domain, gold_items, baselines, steered, alpha,
                tau=tau, st_model=st_model or bench.DEFAULT_ST_MODEL,
                cache_dir=self._cache_dir, api_key=self._api_key,
                generation_model=self._pair_generator.generation_model,
                llm_judge=pairwise,
            )
            out_name = "benchmark_gold.json"
        elif question_set == trap.SET_TRAP:
            print("Trap eval: gold cosine"
                  + (" + pair-cloud embeddings" if similarity else "")
                  + " (no LLM judge)...\n", flush=True)
            results = trap.grade_trap_run(
                self.domain, trap_items, baselines, steered, alpha,
                cache_dir=self._cache_dir, api_key=self._api_key,
                generation_model=self._pair_generator.generation_model,
                similarity=similarity, llm_judge=False,
                tau=tau, st_model=st_model,
                pair_path=self._pair_generator.pairs_path
                if self.variant == TRAP_LOCAL_VARIANT else None,
            )
            out_name = "benchmark_trap.json"
        else:
            print("Grading with LLM judge"
                  + (" and embeddings" if similarity else "") + "...\n",
                  flush=True)
            results = bench.grade_run(
                self.domain, questions, baselines, steered, alpha,
                cache_dir=self._cache_dir, api_key=self._api_key,
                generation_model=self._pair_generator.generation_model,
                similarity=similarity,
            )
            out_name = "benchmark.json"

        results["model_name"] = self.model_name
        results["n_generated"] = len(questions)
        results["set"] = question_set

        self.model_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.model_dir / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved benchmark to {out_path}")
        return results

    def _generate_baseline_steered(self, questions, alpha: float,
                                   max_new_tokens: int,
                                   golds: Optional[list] = None,
                                   steer_new_tokens: Optional[int] = None):
        """Print and collect unsteered then steered answers for each question."""
        n_q = len(questions)
        print(f"Generating {n_q} baseline + steered answers "
              f"(alpha={alpha:.3f}). Each pair can take a minute.\n",
              flush=True)
        baselines, steered = [], []
        for i, question in enumerate(questions, 1):
            print(f"===== {i}/{n_q} =====\nQ: {question}\n", flush=True)
            if golds is not None:
                print(f"--- GOLD ---\n{golds[i - 1]}\n", flush=True)
            print("... generating baseline", flush=True)
            baseline = self.generate(question, alpha=0.0,
                                     max_new_tokens=max_new_tokens)
            print(f"--- BASELINE ---\n{baseline}\n", flush=True)
            print("... generating steered", flush=True)
            steered_text = self.generate(question, alpha=alpha,
                                         max_new_tokens=max_new_tokens,
                                         steer_new_tokens=steer_new_tokens)
            print(f"--- STEERED (alpha={alpha:.3f}) ---\n{steered_text}\n",
                  flush=True)
            baselines.append(baseline)
            steered.append(steered_text)
        return baselines, steered

    # -------------------------------------------------------------- internals

    def _build_is_current(self, force: bool, pairwise: bool,
                          layer: Optional[int] = None) -> bool:
        """Cached layer/alpha match what this build would use."""
        if force or self.layer is None:
            return False
        if layer is not None:
            return int(self.layer) == int(layer) and self._direction is not None
        if self.max_alpha is None:
            return False
        loaded = self._loaded_judge or "nli"
        wanted = "pairwise" if pairwise else "nli"
        return loaded == wanted

    def _build_pairwise(self, extraction, prompts, pairwise_judge, golds=None):
        """Layer selection and calibration with a pairwise LLM judge.

        Order matters here. A coherence ceiling is measured first on a middle
        candidate layer, and the layer sweep then probes at fractions of it —
        otherwise the judge spends calls comparing degraded output, which is
        how a sweep ends up flat.
        """
        cal = _import_extras("calibrate")

        if pairwise_judge is None:
            llm_judge = _import_extras("llm_judge")
            template = None
            if self.variant in ("trap", TRAP_LOCAL_VARIANT):
                template = _import_extras("trap").TRAP_SENSE_PROMPT
            pairwise_judge = llm_judge.PairwiseJudge(
                self.domain, api_key=self._api_key, prompt_template=template,
            )

        gold_list = list(golds) if golds else None
        cal_prompts = list(prompts[:2])
        cal_golds = None if gold_list is None else gold_list[:2]

        layers = sorted(extraction.directions)
        probe_layer = layers[len(layers) // 2]
        ceiling = cal.calibrate_steering(
            self._make_steering(extraction.directions[probe_layer], probe_layer),
            cal_prompts,
        )
        if ceiling is None:
            logger.warning("No coherent strength found while probing layer "
                           f"{probe_layer}; falling back to fixed probe alphas.")
            probe = None
        else:
            probe = cal.probe_alphas_from_ceiling(ceiling)
            logger.info(f"Coherence ceiling ~{ceiling} (layer {probe_layer}); "
                        f"probing {len(layers)} middle layers at {probe}")

        kwargs = {"probe_alphas": probe} if probe else {}
        best_layer, layer_stats = cal.select_best_layer_pairwise(
            self._model, self._tokenizer, extraction, pairwise_judge,
            prompts, golds=gold_list, **kwargs
        )
        layer_scores = {k: v["mean_win_rate"] for k, v in layer_stats.items()}

        steering = self._make_steering(extraction.directions[best_layer], best_layer)
        max_alpha, best_alpha, alpha_stats = cal.calibrate_alpha_pairwise(
            steering, pairwise_judge, cal_prompts, golds=cal_golds,
        )

        extra = {
            "best_alpha": best_alpha,
            "coherence_ceiling_probe": ceiling,
            "probe_alphas": probe,
            "alpha_scores": {str(k): v for k, v in alpha_stats.items()},
            "layer_detail": {str(k): v for k, v in layer_stats.items()},
            "judge_calls": pairwise_judge.calls,
        }
        return best_layer, layer_scores, max_alpha, extra

    def _resolve_pairs(self, force: bool, stems=None):
        """Cached pairs, or generate from stems / unnamed GENERIC_STEMS."""
        from domainsteer.pairs import load_pairs

        gen = self._pair_generator
        concepts = list(self.domain_spec.concepts) or None
        if self.variant == TRAP_LOCAL_VARIANT:
            if gen.pairs_path.exists() and not force:
                return load_pairs(gen.pairs_path)
            raise ValueError(
                "trap_local steering needs gold-vs-baseline pairs under "
                f"{gen.pairs_path} (run trap_local_calibrate)."
            )
        if stems:
            return gen.generate(concepts=concepts, stems=list(stems),
                                force_regenerate=force)
        if gen.pairs_path.exists() and not force:
            return gen.generate()
        bundled = self._bundled_pairs()
        if bundled is not None:
            return bundled
        if self.variant is None and not stems:
            local = self._pairs_from_this_model(force=force)
            if local is not None:
                return local
        if self.variant == "trap":
            trap_stems = self._trap_stems()
            if not trap_stems:
                raise ValueError(
                    "Trap steering needs trap questions in cache "
                    "(run trap_questions) or explicit stems."
                )
            return gen.generate(concepts=concepts, stems=trap_stems,
                                force_regenerate=force)
        return gen.generate(concepts=concepts, force_regenerate=force)

    def _bundled_pairs(self):
        """Pairs shipped as package data for (this model, this domain).

        Only for the default variant: `trap` and `trap_local` need their own
        contrast, and nothing is bundled for them. Returns None when the flag
        is off, the model ships no set, or the domain is not in it — the
        caller then has this model answer the shared stems, and only calls
        the API when those stems are not in the package.
        """
        if not self._use_bundled_pairs or self.variant is not None:
            return None
        from domainsteer.pairs import bundled_manifest, load_bundled_pairs

        query = self.domain_spec.name
        pairs = load_bundled_pairs(self.model_name, query)
        if pairs is None:
            return None
        manifest = bundled_manifest(self.model_name) or {}
        # The bundled sets are self-gold pairs, not this steerer's default
        # contrast, so say so rather than letting it pass unremarked.
        logger.info(
            "Using %d bundled pairs for %r (%s, contrast=%s) instead of "
            "generating them; pass use_bundled_pairs=False or "
            "force_pairs=True to regenerate through the API.",
            len(pairs), query, manifest.get("model", self.model_name),
            manifest.get("contrast", "unknown"),
        )
        return pairs

    def _model_pairs_path(self) -> Path:
        return self.model_dir / "pairs.jsonl"

    def _pairs_from_this_model(self, force: bool):
        """Have this model answer the shared 30 stems. No API call.

        Used when `model_name` has no bundled pairs. The questions come from
        ``data/questions/``; both answers are this model's own text, written
        with the hook off. Cached under `model_dir/pairs.jsonl` so a later
        model does not reuse them.
        """
        from domainsteer.pairs import (ContrastivePair, bundled_question_stems,
                                       load_pairs, save_pairs)
        from domainsteer.self_gold import SELF_GOLD_CONTRAST_ID, gold_system_prompt

        path = self._model_pairs_path()
        if path.exists() and not force:
            logger.info("Using %s written earlier by this model.", path)
            return load_pairs(path)

        stems = bundled_question_stems(self.domain)
        if not stems:
            return None

        self._ensure_model()
        from domainsteer.steering import DEFAULT_SYSTEM_PROMPT, ActivationSteering
        import numpy as np

        hidden = int(self._model.config.hidden_size)
        zero = np.zeros(hidden, dtype=np.float32)
        zero[0] = 1.0
        expert_sys = gold_system_prompt(self.domain)
        writers = {
            expert_sys: ActivationSteering(
                self._model, self._tokenizer, zero, 0, system_prompt=expert_sys),
            DEFAULT_SYSTEM_PROMPT: ActivationSteering(
                self._model, self._tokenizer, zero, 0,
                system_prompt=DEFAULT_SYSTEM_PROMPT),
        }

        def answer(stem: str, system: str) -> str:
            # alpha 0 registers no hook; the dummy direction is unused.
            text = writers[system].generate(stem, alpha=0.0, max_new_tokens=96)
            return " ".join(text.split())

        pairs = []
        for stem in stems:
            expert = answer(stem, expert_sys)
            default = answer(stem, DEFAULT_SYSTEM_PROMPT)
            if expert and default and expert != default:
                pairs.append(ContrastivePair(expert, default, stem))
        if len(pairs) < 10:
            raise ValueError(
                f"Need at least 10 pairs from {self.model_name}, got {len(pairs)}. "
                "The two prompts produced the same text, or generation was empty."
            )
        save_pairs(pairs, path)
        meta = {
            "contrast": SELF_GOLD_CONTRAST_ID,
            "positive": "self",
            "model": self.model_name,
            "n": len(pairs),
            "stems_from": "data/questions",
            "expert_system_prompt": expert_sys,
            "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
        }
        path.with_name("pairs.meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        logger.info(
            "Wrote %d pairs for %s from the shared stems (no API call) → %s",
            len(pairs), self.model_name, path,
        )
        return pairs

    def _trap_stems(self):
        trap = _import_extras("trap").BenchmarkTrap(
            self.domain, cache_dir=self._cache_dir, api_key=self._api_key,
            generation_model=self._pair_generator.generation_model,
        )
        return trap.cached_pair_stems()

    def _load_config(self) -> None:
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("steering_scale") != "hidden_norm_relative":
            logger.warning("Config has an unknown steering scale; rebuilding "
                           "is recommended.")
            return
        if self.variant in ("trap", TRAP_LOCAL_VARIANT):
            stored = config.get("contrast")
            expected = (TRAP_LOCAL_CONTRAST_ID if self.variant == TRAP_LOCAL_VARIANT
                        else TRAP_CONTRAST_ID)
            if stored not in (None, expected):
                logger.warning(
                    f"Trap config at {self.config_path} was built under "
                    f"contrast {stored!r}, not {expected}; "
                    "ignoring it so extract can rebuild."
                )
                self._trap_config_ok = False
                return
        self._loaded_judge = config.get("judge") or "nli"
        self.layer = config.get("layer")
        self.max_alpha = config.get("max_alpha")
        self.best_alpha = config.get("best_alpha")
        direction_path = self.model_dir / config.get("direction_file", "")
        if direction_path.exists():
            import numpy as np
            self._direction = np.load(direction_path)
            logger.info(f"Loaded config: layer {self.layer}, "
                        f"max_alpha {self.max_alpha}")
        else:
            logger.warning(f"Direction file missing at {direction_path}; "
                           "run .build().")
            self.layer = None

    def _ensure_model(self) -> None:
        if self._model is None:
            from domainsteer.extract import load_model
            self._model, self._tokenizer = load_model(self.model_name)

    def _make_steering(self, direction, layer, system_prompt=None):
        from domainsteer.steering import ActivationSteering, DEFAULT_SYSTEM_PROMPT
        return ActivationSteering(
            self._model, self._tokenizer, direction, layer,
            system_prompt=system_prompt or self.system_prompt or DEFAULT_SYSTEM_PROMPT,
        )

    def _ensure_steering(self) -> None:
        if self._steering is None:
            if self._direction is None or self.layer is None:
                raise ValueError("Not built — run .build().")
            self._ensure_model()
            self._steering = self._make_steering(self._direction, self.layer)

    def extracted_layers(self) -> List[int]:
        """Layers with a cached direction vector, not only the judge pick."""
        directory = self.directions_dir
        layers: List[int] = []
        if directory.exists():
            for path in directory.glob("direction_layer_*.npy"):
                try:
                    layers.append(int(path.stem.rsplit("_", 1)[-1]))
                except ValueError:
                    continue
        if layers:
            return sorted(set(layers))
        return [self.layer] if self.layer is not None else []

    def use_layer(self, layer: int) -> None:
        """Switch the active direction to another extracted layer."""
        import numpy as np

        layer = int(layer)
        path = self.directions_dir / f"direction_layer_{layer}.npy"
        if not path.exists():
            raise ValueError(
                f"No extracted direction for layer {layer} at {path}"
            )
        if self.layer == layer and self._direction is not None:
            return
        self.layer = layer
        self._direction = np.load(path)
        self._steering = None
        self._expert_steering = None

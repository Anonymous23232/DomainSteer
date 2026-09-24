# DomainSteer

Contrastive activation addition for scientific domains. DomainSteer builds one steering vector per field and adds it to a causal language model's residual stream while the model generates.

Pairs for **144 scientific domains** ship with the package, so building a vector needs no API key. The **2,160-item polysemy-trap benchmark** ships too.

```bash
pip install domainsteer
```

```python
from domainsteer import DomainSteerer

steerer = DomainSteerer(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    domain="Industrial biotechnology",
)
steerer.build(layer=15)
steerer.generate("What is a strain?", alpha=0.25)
steerer.compare(
    "If the assembly falls apart, do you get back the original pieces?",
    alpha=0.25,
)
```

Steered, "strain" is a microbial variant. Unsteered, it is a pulled muscle. Each bundled pair is that contrast: the same unnamed question, answered once in the domain sense and once in the everyday sense.

`build(layer=15)` extracts the vector at decoder block 15 and caches it. Later calls reuse the cache. `generate` takes a raw strength `alpha` (typical range 0.10–0.40). The shift at each token is `h ← h + α‖h‖v`.

Directions are written under `~/.cache/domainsteer/<domain>/<model>/directions/`. Pass `cache_dir=` to put them somewhere else.

## Bundled pairs

| Model | Domains | Pairs |
|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | 144 | 4,320 |
| `meta-llama/Llama-3.2-3B-Instruct` | 144 | 4,320 |

```python
from domainsteer import list_bundled_models, bundled_manifest, load_bundled_pairs

list_bundled_models()
load_bundled_pairs("meta-llama/Llama-3.1-8B-Instruct", "3106")
load_bundled_pairs("meta-llama/Llama-3.1-8B-Instruct", "Industrial biotechnology")
```

Pairs are model-specific. Both shipped models share the question stems and do not share the answers. A vector built from 8B pairs is not applied to 3B. If the model you name has no bundled set, `load_bundled_pairs` returns `None`.

`DomainSteerer` looks for pairs in this order:

1. A cache file under `cache_dir/<domain-slug>/pairs.jsonl`
2. The bundled pairs for that model and domain
3. This model answering the 30 shared questions in `data/questions/` itself, with the hook off. No API call. Used when the model is not one of the two shipped sets (for example Gemma).
4. API generation through `PairGenerator` (`pip install "domainsteer[generation]"` and an API key), only when the domain has no shipped questions

Turn the bundle off with `use_bundled_pairs=False`. Regenerate with `build(force_pairs=True)`.

## Benchmark

The polysemy-trap exam is 144 domains × 15 items = **2,160** questions. The question does not name the field. The overloaded term has to be read in the domain sense.

```python
from domainsteer import load_benchmark, iter_benchmark

item = load_benchmark("3106")["items"][0]
item["question"]        # 'What can a fingerprint distinguish?'
item["gold"]            # 'product identity'
item["baseline_trap"]   # "a person's identity"
item["gold_responses"]  # one frozen reference sentence

sum(len(d["items"]) for d in iter_benchmark())  # 2160
```

`gold_responses` holds exactly one reference sentence per item.

## What `build` does

1. Resolve the domain against the registry of 144 ANZSRC groups, ten concepts each.
2. Load the 30 contrastive pairs for that model and domain.
3. Replay both answers under the helper system prompt. The domain prompt is not in this forward pass.
4. Mean-pool the assistant-token hidden states at the layer you passed.
5. Save `v = unit(mean expert − mean default)`.

Pass `estimator="rfm"` to use a Recursive Feature Machine instead of the difference of means. Those vectors are cached separately.

Calling `build()` with no `layer` asks an NLI judge to pick the layer and to map an `expertise` dial onto `alpha`. That judge is not installed by `pip install domainsteer`. Use `build(layer=…)` and a raw `alpha`.

## Optional installs

| Extra | Install | What it adds |
|---|---|---|
| `generation` | `pip install "domainsteer[generation]"` | API pair generation (`anthropic`, `openai`) |
| `eval` | `pip install "domainsteer[eval]"` | Cosine scoring with `sentence-transformers` |
| `dev` | `pip install "domainsteer[dev]"` | `pytest`, `ruff` |

## License

MIT.

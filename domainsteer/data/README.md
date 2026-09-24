# domainsteer/data

Package data — installed with the library and shipped in the wheel. Everything
here is loaded through the library, never by path from user code.

```
domain_clusters.json          144 domains x 10 concepts (the registry)
questions/
  manifest.json               domain list
  <id>-<domain-slug>.json     the 30 shared questions for that domain
pairs/<model-slug>/
  manifest.json               model, contrast, domain list, pair counts
  <id>-<domain-slug>.jsonl    30 contrastive pairs for that domain
benchmark/
  manifest.json               contrast, domain list, item counts
  <id>-<domain-slug>.json     15 polysemy-trap items for that domain
```

## The registry

`domain_clusters.json` moved here from the repo root so it resolves from an
installed wheel, not just a source checkout. `load_clusters()` finds it
automatically; a repo-root copy is still honoured if one is present.

## The contrastive pairs

Two complete sets, 144 domains x 30 pairs = 4,320 pairs each:

| Model | Slug | Pairs |
|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | `meta-llama-llama-3-1-8b-instruct` | 4,320 |
| `meta-llama/Llama-3.2-3B-Instruct` | `meta-llama-llama-3-2-3b-instruct` | 4,320 |

One JSONL record per pair:

```json
{"expert_text": "...", "nonexpert_text": "...", "concept": "What is a strain?", "framing": null}
```

`expert_text` reads the stem in the domain sense, `nonexpert_text` in the
everyday sense — for `3106` (Industrial biotechnology), a microbial variant
versus a pulled muscle. The difference in mean hidden state over these two
sides is the steering direction.

### Two things to know

**They are model-specific.** Each model wrote *both* sides of its own pairs.
Only the question stems are shared between the two sets; the texts are not, and
the per-model `manifest.json` records which model produced them. Those stems
are stored on their own in `questions/`, so a model that ships no pair set
answers that list itself. The 8B answers are not reused for the 3B model.

**They carry the `self_gold_expert_vs_default_v1` contrast**, which is *not*
`domainsteer.pairs.CONTRAST_ID` (`stem_commit_vs_survey`), the contrast
`PairGenerator` produces from the API. `DomainSteerer` logs which contrast it
used whenever it falls back to a bundled set.

## Using them

```python
from domainsteer import list_bundled_models, bundled_manifest, load_bundled_pairs

list_bundled_models()
# ['meta-llama-llama-3-1-8b-instruct', 'meta-llama-llama-3-2-3b-instruct']

bundled_manifest("meta-llama/Llama-3.1-8B-Instruct")["contrast"]
# 'self_gold_expert_vs_default_v1'

# id, directory name, or cluster name all resolve
load_bundled_pairs("meta-llama/Llama-3.1-8B-Instruct", "3106")
load_bundled_pairs("meta-llama/Llama-3.1-8B-Instruct", "Industrial biotechnology")
```

`DomainSteerer` uses them automatically, so `build()` needs no API key for these
144 domains on these two models. Resolution order:

1. cached pairs under `cache_dir/<domain-slug>/pairs.jsonl`
2. **these bundled pairs**, for a matching (model, domain)
3. this model answering the 30 questions in `questions/`, when it ships no pairs
4. API generation via `PairGenerator`, when the domain has no question file

Opt out with `DomainSteerer(..., use_bundled_pairs=False)`, or regenerate with
`build(force_pairs=True)`. The `trap` and `trap_local` variants never use them:
they need their own contrast, and none is bundled.

## The benchmark

The polysemy-trap exam: **144 domains x 15 items = 2,160** bare questions whose
overloaded term has to be read in the domain sense. One file per domain.

```json
{
  "term": "fingerprint",
  "question": "What can a fingerprint distinguish?",
  "baseline_trap": "a person's identity",
  "gold": "product identity",
  "dominant_prior": "biometrics",
  "gold_aliases": ["batch identity", "strain identity", "..."],
  "gold_responses": ["It separates closely related fermentation products by their chemical signatures."],
  "trap_aliases": ["personal identity", "..."],
  "trap_responses": ["..."]
}
```

`question` is asked with no context, so nothing but the steering direction (or a
named domain in the prompt) can select the reading. `gold` is the domain sense,
`baseline_trap` the everyday one the unsteered model falls into.

### One gold reference per item

`gold_responses` holds **exactly one** frozen reference sentence. Every arm is
scored against that same sentence. Extra paraphrases are not shipped; keeping
them invites scoring against a per-arm nearest match, which is a different
(and easier) measurement.

```python
from domainsteer import load_benchmark, iter_benchmark, list_benchmark_domains

load_benchmark("3106")["items"][0]["question"]
load_benchmark("Industrial biotechnology")          # id, stem or cluster name
sum(len(d["items"]) for d in iter_benchmark())      # 2160
```

## Provenance

**Pairs.** Produced by the self-gold run behind the paper: each model answered
the same question stems twice, once holding the domain and once not, and both
sides were kept. Every record was validated on the way in, and
`tests/test_bundled_pairs.py` re-checks the manifests against the files.

**Benchmark.** Written by `gpt-5.6-terra` under the `em_trap_v2` contrast, with
exam terms held out of the pairs so the bank is not seen during vector
extraction. The single gold reference per item is the frozen sentence used in
the reported scoring. `tests/test_bundled_benchmark.py` re-checks every item.

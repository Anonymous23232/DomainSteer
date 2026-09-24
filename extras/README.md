# extras

Not part of `pip install domainsteer`. Research checkout only.

| Path | What it is |
|---|---|
| `run_model_sweep.py` | Four-arm layer × strength grid (the paper experiment driver) |
| `concepts.py` | Concept JSON I/O (registry folding now lives in `domainsteer.clusters`) |
| `judge.py` | NLI expertise judge, perplexity/repetition metrics |
| `calibrate.py` | NLI layer pick and `max_alpha` / expertise dial |
| `llm_judge.py` | Pairwise LLM-as-judge |
| `benchmark.py` | Open unnamed / gold exam (not `em_trap_v2`) |
| `trap.py` | Older `polysemy_trap_v6` exam |
| `self_gold_sim.py` | Hybrid BM25+dense scoring for the old self-gold sweep |
| `iclr2027/` | ICLR 2027 paper sources and figure scripts |

`DomainSteerer.build()` without `layer=` lazy-imports from here. A wheel
install without this folder will raise unless you pass `layer=` and a raw
`alpha`.

```bash
python extras/run_model_sweep.py --dry-run
python extras/run_model_sweep.py --ids 3106 --stages pairs,directions,sweep,score
```

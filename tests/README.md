# tests

CPU-only tests for the `domainsteer` library. Nothing here downloads or runs a
model.

```bash
pip install -e ".[dev]"
pytest -q
```

## Skipped in this distribution

Four files cover a library module *and* load an experiment script by path from
`extras/scripts/`. Those scripts are part of the research repo, not the package,
so [conftest.py](conftest.py) skips the **30** tests that need them and runs the
**48** library tests in the same files:

| File | Skipped | Still run |
|---|---|---|
| `test_em_traps.py` | 2 | 13 |
| `test_prompt_caa.py` | 4 | 1 |
| `test_self_gold.py` | 16 | 14 |
| `test_self_gold_sim.py` | 8 | 20 |

The guard looks for `"extras" / "scripts"` in the test's own source and in any
module helper it calls, so importing archived modules from `extras/*.py` does
not skip them. In a checkout that does have `extras/scripts/`, `conftest.py`
does nothing and everything runs.

## Bundled data

[test_bundled_pairs.py](test_bundled_pairs.py) checks the package data: both
manifests against the files on disk, all 144 registry domains resolving by
cluster id and by name for each model, lookup by id / directory name / cluster
name, unknown models and domains returning `None`, and every one of the 8,640
shipped records being well formed.

[test_bundled_benchmark.py](test_bundled_benchmark.py) checks the benchmark: the
manifest against the files, all 144 domains loading with 15 items each, lookup by
id / stem / cluster name, and — for every one of the 2,160 items — the required
fields, a domain reading distinct from the trap reading, and **exactly one** gold
reference.

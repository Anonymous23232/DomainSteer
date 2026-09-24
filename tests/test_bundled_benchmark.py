"""The polysemy-trap benchmark shipped as package data.

144 domains x 15 items = 2,160 bare questions whose overloaded term has to be
read in the domain sense. CPU-only: nothing here loads a model.
"""
import json

from domainsteer.em_traps import (BUNDLED_BENCHMARK_ROOT, benchmark_manifest,
                                  benchmark_path, iter_benchmark,
                                  list_benchmark_domains, load_benchmark)

REQUIRED_FIELDS = ("term", "question", "gold", "baseline_trap")


def test_manifest_matches_files_on_disk():
    manifest = benchmark_manifest()
    assert manifest["name"] == "polysemy-traps"
    assert manifest["contrast"] == "em_trap_v2"
    assert manifest["n_domains"] == 144
    assert manifest["n_items_total"] == 2160
    assert manifest["items_per_domain"] == [15]
    assert manifest["gold_selection"] == "single"

    files = [p for p in BUNDLED_BENCHMARK_ROOT.glob("*.json") if p.name != "manifest.json"]
    assert len(files) == manifest["n_domains"]


def test_every_domain_loads_with_fifteen_items():
    total = 0
    for entry in list_benchmark_domains():
        data = load_benchmark(entry["id"])
        assert data is not None, entry["id"]
        assert len(data["items"]) == entry["n_items"] == 15
        assert str(data["cluster_id"]) == entry["id"]
        total += len(data["items"])
    assert total == 2160


def test_lookup_accepts_id_stem_and_cluster_name():
    found = {
        benchmark_path(q)
        for q in ("3106", "3106-industrial-biotechnology", "Industrial biotechnology")
    }
    assert len(found) == 1
    assert found.pop().name == "3106-industrial-biotechnology.json"


def test_unknown_domain_returns_none():
    assert benchmark_path("Competitive Yodelling") is None
    assert load_benchmark("Competitive Yodelling") is None


def test_exactly_one_gold_reference_per_item():
    """The single frozen gold scored against — not a set of paraphrases."""
    for data in iter_benchmark():
        for item in data["items"]:
            refs = item["gold_responses"]
            assert isinstance(refs, list)
            assert len(refs) == 1, (data["cluster_id"], item["term"])
            assert isinstance(refs[0], str) and refs[0].strip()


def test_every_item_is_well_formed():
    seen = 0
    for data in iter_benchmark():
        for item in data["items"]:
            for field in REQUIRED_FIELDS:
                value = item.get(field)
                assert isinstance(value, str) and value.strip(), (field, item)
            # the trap reading must differ from the domain reading
            assert item["gold"].strip().lower() != item["baseline_trap"].strip().lower()
            seen += 1
    assert seen == 2160


def test_files_are_valid_json_with_stable_shape():
    for path in sorted(BUNDLED_BENCHMARK_ROOT.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if path.name == "manifest.json":
            assert "domains" in data
            continue
        assert path.stem.startswith(str(data["cluster_id"]))
        assert data["contrast"] == "em_trap_v2"
        assert isinstance(data["items"], list)


def test_iter_benchmark_covers_every_domain_in_id_order():
    ids = [str(d["cluster_id"]) for d in iter_benchmark()]
    assert len(ids) == 144
    assert ids == sorted(ids)
    assert len(set(ids)) == 144

"""Contrastive pairs shipped as package data (domainsteer/data/pairs/).

CPU-only: nothing here loads a model.
"""
import json

import pytest

from domainsteer.clusters import load_clusters
from domainsteer.pairs import (BUNDLED_PAIRS_ROOT, ContrastivePair,
                               bundled_manifest, bundled_pairs_path,
                               list_bundled_models, load_bundled_pairs)

MODEL_8B = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_3B = "meta-llama/Llama-3.2-3B-Instruct"
BUNDLED_CONTRAST = "self_gold_expert_vs_default_v1"


def test_both_model_sets_ship():
    assert list_bundled_models() == [
        "meta-llama-llama-3-1-8b-instruct",
        "meta-llama-llama-3-2-3b-instruct",
    ]


@pytest.mark.parametrize("model", [MODEL_8B, MODEL_3B])
def test_manifest_matches_files_on_disk(model):
    manifest = bundled_manifest(model)
    assert manifest["contrast"] == BUNDLED_CONTRAST
    assert manifest["n_domains"] == 144
    assert manifest["model"] == model

    root = BUNDLED_PAIRS_ROOT / manifest["model_slug"]
    assert len(list(root.glob("*.jsonl"))) == manifest["n_domains"]

    total = 0
    for entry in manifest["domains"]:
        path = root / f"{entry['dir']}.jsonl"
        assert path.is_file(), entry["dir"]
        n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        assert n == entry["n_pairs"]
        total += n
    assert total == manifest["n_pairs_total"]


@pytest.mark.parametrize("model", [MODEL_8B, MODEL_3B])
def test_every_registry_domain_resolves_by_name_and_id(model):
    clusters = load_clusters()
    assert len(clusters) == 144
    for cluster in clusters:
        assert bundled_pairs_path(model, cluster.cluster) is not None, cluster.cluster
        assert bundled_pairs_path(model, cluster.cluster_id) is not None, cluster.cluster_id


def test_lookup_accepts_id_dirname_and_name():
    found = {
        bundled_pairs_path(MODEL_8B, q)
        for q in ("3106", "3106-industrial-biotechnology", "Industrial biotechnology")
    }
    assert len(found) == 1
    assert found.pop().name == "3106-industrial-biotechnology.jsonl"


def test_unknown_model_and_domain_return_none():
    assert bundled_manifest("mistralai/Mistral-7B-Instruct") is None
    assert load_bundled_pairs("mistralai/Mistral-7B-Instruct", "3106") is None
    assert load_bundled_pairs(MODEL_8B, "Competitive Yodelling") is None


def test_loaded_pairs_are_usable_contrastive_pairs():
    pairs = load_bundled_pairs(MODEL_8B, "Industrial biotechnology")
    assert len(pairs) == 30
    assert all(isinstance(p, ContrastivePair) for p in pairs)
    for p in pairs:
        assert p.expert_text.strip()
        assert p.nonexpert_text.strip()
        assert p.expert_text != p.nonexpert_text


def test_pairs_are_model_specific_not_shared():
    """Each model wrote both sides itself; only the stems are shared."""
    eight = load_bundled_pairs(MODEL_8B, "Industrial biotechnology")
    three = load_bundled_pairs(MODEL_3B, "Industrial biotechnology")
    assert [p.concept for p in eight] == [p.concept for p in three]
    assert [p.expert_text for p in eight] != [p.expert_text for p in three]


@pytest.mark.parametrize("model", [MODEL_8B, MODEL_3B])
def test_every_shipped_record_is_well_formed(model):
    root = BUNDLED_PAIRS_ROOT / bundled_manifest(model)["model_slug"]
    for path in sorted(root.glob("*.jsonl")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            assert isinstance(record.get("expert_text"), str), f"{path}:{i}"
            assert isinstance(record.get("nonexpert_text"), str), f"{path}:{i}"
            assert isinstance(record.get("concept"), str), f"{path}:{i}"

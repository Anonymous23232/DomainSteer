"""Tests for the concepts record and its on-disk format."""

import pytest

from extras.concepts import Concept, _dedup_key, load_concepts, save_concepts


def test_save_load_roundtrip(tmp_path):
    concepts = [Concept("aquifer", 3.2, 12), Concept("baseflow", 2.1, 7)]
    path = tmp_path / "concepts.json"
    save_concepts(concepts, path, domain="hydrology", params={"source": "curated"})
    assert load_concepts(path) == concepts


def test_load_accepts_plain_string_list(tmp_path):
    path = tmp_path / "handwritten.json"
    path.write_text('{"concepts": ["infiltration", "runoff"]}',
                    encoding="utf-8")
    loaded = load_concepts(path)
    assert [c.term for c in loaded] == ["infiltration", "runoff"]


def test_load_ignores_unknown_keys(tmp_path):
    """Generated files carry 'category'; hand-edited ones carry anything."""
    path = tmp_path / "generated.json"
    path.write_text(
        '{"concepts": [{"term": "aquifer", "category": "groundwater",'
        ' "note": "keep"}]}',
        encoding="utf-8")
    assert [c.term for c in load_concepts(path)] == ["aquifer"]


def test_load_rejects_empty(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text('{"concepts": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="No concepts"):
        load_concepts(path)


def test_dedup_key_folds_plurals():
    assert _dedup_key("piezometers") == _dedup_key("piezometer")
    assert _dedup_key("confining beds") == _dedup_key("confining bed")
    assert _dedup_key("analysis") == "analysis"

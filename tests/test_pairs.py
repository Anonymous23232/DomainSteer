"""Tests for contrastive pair generation. No API calls — generation is mocked."""

import json

import pytest

import domainsteer.pairs as pairs_module
from domainsteer.pairs import (
    CONTRAST_ID,
    ContrastivePair,
    GENERIC_STEMS,
    NEUTRAL_MARKER,
    PairGenerator,
    TRAP_CONTRAST_ID,
    TRAP_LOCAL_CONTRAST_ID,
    _rotate,
    load_pairs,
    save_pairs,
)

GROUNDING = ["groundwater recharge", "hydraulic conductivity", "baseflow"]
STEMS = [
    "What do outsiders most often get wrong about this field?",
    "What does a practitioner check first when diagnosing a problem?",
    "What evidence would change an expert's mind here?",
]


def _questions(prompt: str) -> list:
    """Stems listed under 'Questions:', not the expert's field-material bullets."""
    lines = prompt.splitlines()
    start = next(i for i, line in enumerate(lines) if line == "Questions:")
    items = []
    for line in lines[start + 1:]:
        if line.startswith("- "):
            items.append(line[2:])
        elif line.strip():
            break
    return items


def fake_completion_factory(calls):
    def fake(prompt, api_key=None, model=None):
        calls.append(prompt)
        neutral = NEUTRAL_MARKER in prompt
        return {"explanations": {
            q: (f"a field-neutral account of {q}" if neutral
                else f"a precise technical account of {q}")
            for q in _questions(prompt)
        }}

    return fake


@pytest.fixture
def fake_api(monkeypatch):
    calls = []
    monkeypatch.setattr(pairs_module, "request_json_completion",
                        fake_completion_factory(calls))
    return calls


def test_generate_pairs_one_per_stem(tmp_path, fake_api):
    gen = PairGenerator("hydrology", cache_dir=tmp_path)
    pairs = gen.generate(concepts=GROUNDING, stems=STEMS)

    assert len(pairs) == len(STEMS)
    assert [p.concept for p in pairs] == STEMS
    for pair in pairs:
        assert "precise technical" in pair.expert_text
        assert "field-neutral" in pair.nonexpert_text
    assert len(fake_api) == 2


def test_generator_caches_and_reuses(tmp_path, fake_api):
    gen = PairGenerator("hydrology", cache_dir=tmp_path)
    pairs = gen.generate(concepts=GROUNDING, stems=STEMS)
    assert gen.pairs_path.exists()
    assert len(fake_api) == 2

    gen2 = PairGenerator("hydrology", cache_dir=tmp_path)
    pairs2 = gen2.generate()
    assert len(fake_api) == 2
    assert pairs2 == pairs


def test_generator_force_regenerate(tmp_path, fake_api):
    gen = PairGenerator("hydrology", cache_dir=tmp_path)
    gen.generate(concepts=GROUNDING, stems=STEMS)
    new_stems = ["What is the most consequential trade-off in this work?"]
    regenerated = gen.generate(concepts=GROUNDING, stems=new_stems,
                               force_regenerate=True)
    assert len(regenerated) == 1
    assert regenerated[0].concept == new_stems[0]


def test_generator_batches_stems(tmp_path, fake_api):
    gen = PairGenerator("hydrology", cache_dir=tmp_path)
    pairs = gen.generate(concepts=GROUNDING, stems=STEMS, batch_size=2)
    assert len(pairs) == len(STEMS)
    assert len(fake_api) == 4   # 2 batches x 2 conditions


def test_missing_response_skips_stem(tmp_path, monkeypatch):
    def partial(prompt, api_key=None, model=None):
        questions = _questions(prompt)
        explanations = {q: f"text about {q}" for q in questions}
        if NEUTRAL_MARKER in prompt:
            explanations.pop(STEMS[2], None)
        return {"explanations": explanations}

    monkeypatch.setattr(pairs_module, "request_json_completion", partial)
    gen = PairGenerator("hydrology", cache_dir=tmp_path)
    pairs = gen.generate(concepts=GROUNDING, stems=STEMS)
    assert [p.concept for p in pairs] == STEMS[:2]


def test_typographic_punctuation_still_matches(tmp_path, monkeypatch):
    stems = ["What is a model's role here?", "What is Manning's equation for?"]

    def curly(prompt, api_key=None, model=None):
        return {"explanations": {
            "What is a model’s role here?": "answer one",
            "What is Manning’s  equation for?": "answer two",
        }}

    monkeypatch.setattr(pairs_module, "request_json_completion", curly)
    gen = PairGenerator("hydrology", cache_dir=tmp_path)
    pairs = gen.generate(stems=stems)
    assert [p.concept for p in pairs] == stems


def test_unknown_domain_still_generates_from_stems(tmp_path, fake_api):
    """Stems do not require a registry match; grounding is optional."""
    gen = PairGenerator("underwater basket weaving", cache_dir=tmp_path)
    pairs = gen.generate(stems=STEMS)
    assert len(pairs) == len(STEMS)
    for prompt in fake_api:
        if NEUTRAL_MARKER in prompt:
            assert "underwater basket weaving" not in prompt.lower()
        else:
            assert "underwater basket weaving" in prompt.lower()


def test_registry_cluster_grounds_in_domain_only(tmp_path, monkeypatch):
    import domainsteer.clusters as clusters_module
    from domainsteer.clusters import ClusterConcept, DomainCluster

    cluster = DomainCluster(
        cluster_id="9001", cluster="Test Hydrology",
        concepts=[ClusterConcept("Aquifer Properties", "storage"),
                  ClusterConcept("Well Hydraulics", "drawdown")],
        domain="Natural Sciences", division="Earth Sciences",
    )
    monkeypatch.setattr(clusters_module, "load_clusters",
                        lambda path=None: [cluster])
    monkeypatch.setattr(pairs_module, "request_json_completion",
                        fake_completion_factory([]))

    gen = PairGenerator("Test Hydrology", cache_dir=tmp_path)
    calls = []
    monkeypatch.setattr(pairs_module, "request_json_completion",
                        fake_completion_factory(calls))
    pairs = gen.generate(stems=STEMS)

    assert len(pairs) == len(STEMS)
    in_domain = [p for p in calls if NEUTRAL_MARKER not in p]
    neutral = [p for p in calls if NEUTRAL_MARKER in p]
    assert any("Aquifer Properties" in p for p in in_domain)
    assert all("Aquifer Properties" not in p for p in neutral)
    assert all("Well Hydraulics" not in p for p in neutral)


def test_concepts_fall_back_to_cached_file_as_grounding(tmp_path, fake_api):
    from extras.concepts import Concept, save_concepts

    gen = PairGenerator("hydrology", cache_dir=tmp_path)
    save_concepts([Concept("infiltration", 1.0, 5)], gen.concepts_path,
                  domain="hydrology")
    pairs = gen.generate(stems=STEMS)
    assert [p.concept for p in pairs] == STEMS
    in_domain = [p for p in fake_api if NEUTRAL_MARKER not in p]
    assert any("infiltration" in p for p in in_domain)
    assert all("infiltration" not in p for p in fake_api if NEUTRAL_MARKER in p)


def test_jsonl_roundtrip(tmp_path):
    pairs = [ContrastivePair("expert one", "lay one", "concept a"),
             ContrastivePair("expert two", "lay two", "concept b", "Why it matters.")]
    path = tmp_path / "pairs.jsonl"
    save_pairs(pairs, path)
    assert load_pairs(path) == pairs


def test_both_sides_receive_the_same_stems(tmp_path, fake_api):
    gen = PairGenerator("hydrology", cache_dir=tmp_path)
    gen.generate(concepts=GROUNDING, stems=STEMS)

    expert = [_questions(p) for p in fake_api if NEUTRAL_MARKER not in p]
    neutral = [_questions(p) for p in fake_api if NEUTRAL_MARKER in p]
    assert expert == neutral == [STEMS]


def test_grounding_terms_never_reach_the_neutral_prompt(tmp_path, fake_api):
    gen = PairGenerator("hydrology", cache_dir=tmp_path)
    gen.generate(concepts=GROUNDING, stems=STEMS)

    in_domain = [p for p in fake_api if NEUTRAL_MARKER not in p]
    neutral = [p for p in fake_api if NEUTRAL_MARKER in p]
    assert in_domain and neutral
    for prompt in in_domain:
        assert "it means hydrology" in prompt
        assert "Never list other fields" in prompt
        for term in GROUNDING:
            assert term in prompt
    for prompt in neutral:
        assert "hydrology" not in prompt.lower()
        assert NEUTRAL_MARKER in prompt
        for term in GROUNDING:
            assert term not in prompt


def test_default_stems_are_the_generic_bank(tmp_path, fake_api):
    gen = PairGenerator("hydrology", cache_dir=tmp_path)
    pairs = gen.generate(concepts=GROUNDING)
    assert len(pairs) == len(GENERIC_STEMS)
    assert pairs[0].concept == GENERIC_STEMS[0]


def test_rotate_wraps_without_padding_past_the_list():
    assert _rotate(["a", "b", "c"], 0, 10) == ["a", "b", "c"]
    assert _rotate(["a", "b", "c"], 2, 2) == ["c", "a"]
    assert _rotate([], 0, 10) == []


def test_load_pairs_accepts_legacy_keys(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps({"positive_text": "expert side",
                                "negative_text": "lay side"}) + "\n")
    pairs = load_pairs(path)
    assert pairs == [ContrastivePair("expert side", "lay side", None)]


def test_load_pairs_rejects_bad_schema(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"expert_text": "only one side"}) + "\n")
    with pytest.raises(ValueError, match="nonexpert_text"):
        load_pairs(path)


def test_load_pairs_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="No pairs"):
        load_pairs(path)


def test_missing_api_key_is_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gen = PairGenerator("hydrology", cache_dir=tmp_path)
    with pytest.raises((ValueError, ImportError), match="(?i)api key|openai"):
        gen.generate(concepts=GROUNDING, stems=STEMS)


def test_domain_slug_isolation(tmp_path):
    gen_a = PairGenerator("hydrology", cache_dir=tmp_path)
    gen_b = PairGenerator("Tax Law (US)", cache_dir=tmp_path)
    assert gen_a.domain_dir != gen_b.domain_dir
    assert gen_b.domain_dir.name == "tax-law-us"


def test_stale_concept_matched_cache_is_refused(tmp_path, fake_api):
    gen = PairGenerator("hydrology", cache_dir=tmp_path)
    gen.pairs_path.parent.mkdir(parents=True, exist_ok=True)
    gen.pairs_path.write_text(
        json.dumps({"expert_text": "old", "nonexpert_text": "lay",
                    "concept": "aquifer"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different contrast"):
        gen.generate(concepts=GROUNDING, stems=STEMS)
    assert not fake_api

    rebuilt = gen.generate(concepts=GROUNDING, stems=STEMS,
                           force_regenerate=True)
    assert rebuilt
    assert json.loads(gen.pairs_meta_path.read_text(encoding="utf-8"))[
        "contrast"] == pairs_module.CONTRAST_ID


def test_trap_contrast_uses_separate_files(tmp_path, fake_api):
    gold = PairGenerator("hydrology", cache_dir=tmp_path)
    gold.generate(concepts=GROUNDING, stems=STEMS)
    trap = PairGenerator("hydrology", cache_dir=tmp_path,
                         contrast=TRAP_CONTRAST_ID)
    trap_stems = ["What is forcing?", "What calculation tracks a disturbance through a channel?"]
    pairs = trap.generate(concepts=GROUNDING, stems=trap_stems)

    assert gold.pairs_path == tmp_path / "hydrology" / "pairs.jsonl"
    assert trap.pairs_path == tmp_path / "hydrology" / "pairs-trap.jsonl"
    assert trap.pairs_path.exists()
    assert gold.pairs_path.exists()
    assert trap.pairs_path != gold.pairs_path
    assert [p.concept for p in pairs] == trap_stems
    assert json.loads(trap.pairs_meta_path.read_text(encoding="utf-8"))[
        "contrast"] == TRAP_CONTRAST_ID
    assert json.loads(gold.pairs_meta_path.read_text(encoding="utf-8"))[
        "contrast"] == CONTRAST_ID
    gold_again = PairGenerator("hydrology", cache_dir=tmp_path).generate()
    assert [p.concept for p in gold_again] == STEMS
    trap_prompts = fake_api[2:]
    assert trap_prompts
    for prompt in trap_prompts:
        assert "What is forcing?" in prompt
        assert "If the routing fails halfway, does anything still arrive?" in prompt
        assert "two different settings look equally right" in prompt
    for prompt in fake_api[:2]:
        assert "If the routing fails halfway" not in prompt


def test_trap_stale_pairs_regenerate_without_force(tmp_path, fake_api):
    trap = PairGenerator("hydrology", cache_dir=tmp_path,
                         contrast=TRAP_CONTRAST_ID)
    trap.pairs_path.parent.mkdir(parents=True, exist_ok=True)
    trap.pairs_path.write_text(
        json.dumps({"expert_text": "old", "nonexpert_text": "lay",
                    "concept": "What is forcing?"}) + "\n",
        encoding="utf-8",
    )
    trap.pairs_meta_path.write_text(
        json.dumps({"contrast": "trap_commit_vs_survey_v3"}),
        encoding="utf-8",
    )
    stems = ["What is forcing?", "What does scale mean?"]
    rebuilt = trap.generate(stems=stems)
    assert [p.concept for p in rebuilt] == stems
    assert json.loads(trap.pairs_meta_path.read_text(encoding="utf-8"))[
        "contrast"] == TRAP_CONTRAST_ID


def test_trap_local_generate_refuses_api(tmp_path, fake_api):
    gen = PairGenerator("hydrology", cache_dir=tmp_path,
                        contrast=TRAP_LOCAL_CONTRAST_ID)
    assert gen.pairs_path.name == "pairs-trap-local.jsonl"
    with pytest.raises(ValueError, match="unsteered"):
        gen.generate(stems=STEMS)
    assert not fake_api


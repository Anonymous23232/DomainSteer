"""Tests for the domain-cluster registry — loading, resolution, invariants.

The invariant checks are the point of most of these. The registry's promises —
exactly ten concepts per domain, no concept shared between domains — are only
worth making if something enforces them, and `verify_clusters` is what a build
and the test suite both lean on.
"""

import json

import pytest

from domainsteer.clusters import (CONCEPTS_PER_CLUSTER, ClusterConcept, Domain,
                                  DomainCluster, find_cluster, find_collisions,
                                  find_duplicates, is_restatement,
                                  is_valid_concept_name, load_clusters,
                                  save_clusters, select_clusters,
                                  verify_clusters)


def make_cluster(cluster_id="3406", name="Inorganic Chemistry",
                 concepts=None, **kwargs):
    return DomainCluster(
        cluster_id=cluster_id,
        cluster=name,
        concepts=[ClusterConcept(c) for c in (concepts or ["Structure & Bonding"])],
        domain=kwargs.pop("domain", "Natural Sciences"),
        division=kwargs.pop("division", "Chemical Sciences"),
        **kwargs,
    )


TEN = ["Structure & Bonding", "Coordination Chemistry", "Main-Group Chemistry",
       "Organometallic Chemistry", "Solid-State Materials", "Reaction Pathways",
       "Bioinorganic Chemistry", "Spectroscopic Characterisation",
       "Cluster Compounds", "Lanthanide Actinide Chemistry"]


# ------------------------------------------------------------------- Domain

def test_domain_requires_a_name():
    with pytest.raises(ValueError, match="non-empty name"):
        Domain("   ")


def test_domain_strips_and_defaults_concepts():
    domain = Domain("  hydrology ")
    assert domain.name == "hydrology"
    assert domain.concepts == []


# ------------------------------------------------------------------ loading

def write_registry(tmp_path, clusters):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"clusters": clusters}), encoding="utf-8")
    return path


def test_load_accepts_plain_string_concepts(tmp_path):
    path = write_registry(tmp_path, [
        {"cluster_id": "1", "cluster": "Inorganic Chemistry",
         "concepts": ["Structure & Bonding", "Coordination Chemistry"]},
    ])
    clusters = load_clusters(path)
    assert [c.name for c in clusters[0].concepts] == ["Structure & Bonding",
                                                      "Coordination Chemistry"]


def test_load_accepts_the_published_seed_shape(tmp_path):
    """The source classification's objects carry id/scope/prefix; all load."""
    path = write_registry(tmp_path, [
        {"cluster_id": "3101", "cluster": "Biochemistry and cell biology",
         "concepts": [{"id": "310106", "name": "Enzymes", "scope": "",
                       "prefix": "Take on the role of..."}]},
    ])
    concept = load_clusters(path)[0].concepts[0]
    assert (concept.name, concept.seed_id) == ("Enzymes", "310106")


def test_load_rejects_a_file_with_no_clusters(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"clusters": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty 'clusters'"):
        load_clusters(path)


def test_save_then_load_round_trips(tmp_path):
    original = make_cluster(concepts=TEN, boundary="Excludes organic frameworks.")
    path = tmp_path / "out.json"
    save_clusters([original], path)

    loaded = load_clusters(path)[0]
    assert loaded.cluster == original.cluster
    assert loaded.boundary == "Excludes organic frameworks."
    assert [c.name for c in loaded.concepts] == TEN


def test_siblings_come_from_the_division_listing():
    cluster = make_cluster(
        name="Nuclear and plasma physics",
        division_scope="Classical physics, Nuclear and plasma physics, "
                       "Quantum physics",
    )
    assert cluster.siblings() == ["Classical physics", "Quantum physics"]


# --------------------------------------------------------------- resolution

@pytest.fixture
def registry():
    return [
        make_cluster("3406", "Inorganic Chemistry", TEN),
        make_cluster("3407", "Theoretical and Computational Chemistry", TEN),
        make_cluster("5106", "Nuclear and Plasma Physics", TEN,
                     division="Physical Sciences"),
    ]


@pytest.mark.parametrize("query", [
    "3406",                       # cluster id
    "Inorganic Chemistry",        # exact name
    "inorganic chemistry",        # case-insensitive
    "Inorganic",                  # unique substring
])
def test_find_cluster_resolution_order(registry, query):
    assert find_cluster(registry, query).cluster_id == "3406"


def test_find_cluster_rejects_an_ambiguous_substring(registry):
    with pytest.raises(ValueError, match="matches 2 clusters"):
        find_cluster(registry, "Chemistry")


def test_find_cluster_suggests_when_nothing_matches(registry):
    with pytest.raises(ValueError, match="No cluster matches"):
        find_cluster(registry, "underwater basket weaving")


def test_find_cluster_rejects_an_empty_query(registry):
    with pytest.raises(ValueError, match="Empty domain name"):
        find_cluster(registry, "  ")


def test_select_by_id_prefix_and_division(registry):
    assert len(select_clusters(registry, ids=["34"])) == 2
    assert len(select_clusters(registry, divisions=["Physical Sciences"])) == 1


def test_select_searches_concept_names_too(registry):
    assert len(select_clusters(registry, search="organometallic")) == 3


def test_select_with_no_filters_returns_everything(registry):
    assert len(select_clusters(registry)) == 3


def test_select_raises_when_nothing_matches(registry):
    with pytest.raises(ValueError, match="No clusters matched"):
        select_clusters(registry, ids=["99"])


# --------------------------------------------------------------- restatement

@pytest.mark.parametrize("concept,cluster", [
    ("Theology", "Theology"),                                  # identical
    ("Nuclear physics", "Nuclear and plasma physics"),         # subset
    ("Plasma physics; fusion plasmas", "Nuclear and plasma physics"),  # segment
    ("Medical physics", "Medical and biological physics"),
])
def test_restatements_are_detected(concept, cluster):
    assert is_restatement(concept, cluster)


@pytest.mark.parametrize("concept,cluster", [
    ("Analytical biochemistry", "Biochemistry and cell biology"),
    ("Cell metabolism", "Biochemistry and cell biology"),
    ("Enzymes", "Biochemistry and cell biology"),
    ("Coordination Chemistry", "Inorganic Chemistry"),
])
def test_real_subareas_are_not_restatements(concept, cluster):
    assert not is_restatement(concept, cluster)


@pytest.mark.parametrize("name", [
    "Coordination Chemistry", "Solid-State & Materials Chemistry", "Enzymes",
])
def test_valid_concept_names_are_kept(name):
    assert is_valid_concept_name(name)


@pytest.mark.parametrize("name", [
    "",                                    # empty
    "X",                                   # too short
    "Group 13 Chemistry",                  # digits
    "a b c d e f g h i",                   # too many words
])
def test_malformed_concept_names_are_rejected(name):
    assert not is_valid_concept_name(name)


# ---------------------------------------------------------------- invariants

def test_disjoint_registry_passes_verification():
    clusters = [make_cluster("1", "Inorganic Chemistry", TEN),
                make_cluster("2", "Nuclear Physics", [f"Area {c}" for c in "ABCDEFGHIJ"])]
    # "Area A".."Area J" carry no digits and stay within the word limit.
    report = verify_clusters(clusters)
    assert report["ok"], report


def test_shared_concept_is_a_collision():
    shared = TEN[:9] + ["Reaction Pathways"]
    clusters = [make_cluster("1", "Inorganic Chemistry", TEN),
                make_cluster("2", "Organic Chemistry", shared)]
    collisions = find_collisions(clusters)
    assert "reaction pathway" in collisions
    assert set(collisions["reaction pathway"]) == {"1", "2"}


def test_plural_variants_collide():
    clusters = [make_cluster("1", "Alpha Domain", ["Reaction Mechanism"]),
                make_cluster("2", "Beta Domain", ["Reaction Mechanisms"])]
    assert len(find_collisions(clusters)) == 1


def test_a_repeat_within_one_cluster_is_a_duplicate_not_a_collision():
    """Adjudication cannot resolve these — there is no rival claimant — so
    they must not reach the resolver dressed as collisions."""
    clusters = [make_cluster("1", "Alpha Domain",
                             ["Reaction Mechanism", "Reaction Mechanisms"])]
    assert find_collisions(clusters) == {}
    assert find_duplicates(clusters) == {"1": ["reaction mechanism"]}
    assert not verify_clusters(clusters)["ok"]


def test_wrong_concept_count_is_reported():
    report = verify_clusters([make_cluster("1", "Inorganic Chemistry", TEN[:4])])
    assert report["wrong_count"] == {"1": 4}
    assert not report["ok"]


def test_strict_verification_raises():
    with pytest.raises(ValueError, match="without exactly"):
        verify_clusters([make_cluster("1", "Inorganic Chemistry", TEN[:4])],
                        strict=True)


def test_restatement_in_the_registry_fails_verification():
    concepts = TEN[:9] + ["Inorganic Chemistry"]
    report = verify_clusters([make_cluster("1", "Inorganic Chemistry", concepts)])
    assert report["restatements"] == [("1", "Inorganic Chemistry")]


def test_default_target_is_ten():
    """The dial the rest of the pipeline sizes itself against."""
    assert CONCEPTS_PER_CLUSTER == 10

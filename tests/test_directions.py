"""Tests for the direction estimators and their diagnostics.

Deliberately free of torch: `domainsteer.directions` is pure numpy, so these
run even where the torch install is broken — which is exactly where the rest
of the extraction tests skip.
"""

import numpy as np
import pytest

from domainsteer.directions import (DEFAULT_ESTIMATOR, ESTIMATORS,
                                    difference_of_means, direction_cosine,
                                    rfm_agop, separation_margin,
                                    shared_layer_cosines, split_by_concept)

DIM = 48
N = 90


def planted_clouds(seed=0, noise=0.3, separation=1.0, dim=DIM, n=N):
    """Two clouds offset by ±separation along one planted unit direction."""
    rng = np.random.default_rng(seed)
    planted = rng.normal(size=dim)
    planted /= np.linalg.norm(planted)
    base = rng.normal(size=(n, dim))
    expert = base + separation * planted + rng.normal(scale=noise, size=(n, dim))
    nonexpert = base - separation * planted + rng.normal(scale=noise, size=(n, dim))
    return expert, nonexpert, planted


@pytest.mark.parametrize("name", sorted(ESTIMATORS))
def test_every_estimator_returns_a_unit_vector(name):
    expert, nonexpert, _ = planted_clouds()
    direction = ESTIMATORS[name](expert, nonexpert)
    assert direction.shape == (DIM,)
    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("name", sorted(ESTIMATORS))
def test_every_estimator_recovers_the_planted_direction(name):
    expert, nonexpert, planted = planted_clouds()
    direction = ESTIMATORS[name](expert, nonexpert)
    assert abs(float(direction @ planted)) > 0.85


@pytest.mark.parametrize("name", sorted(ESTIMATORS))
def test_every_estimator_separates_the_clouds(name):
    expert, nonexpert, _ = planted_clouds()
    direction = ESTIMATORS[name](expert, nonexpert)
    assert abs(separation_margin(expert, nonexpert, direction)) > 1.0


def test_default_estimator_is_registered_and_is_difference_of_means():
    assert ESTIMATORS[DEFAULT_ESTIMATOR] is difference_of_means


def test_difference_of_means_is_exactly_the_normalized_mean_gap():
    """The default path must stay bit-for-bit what it always was."""
    expert, nonexpert, _ = planted_clouds()
    expected = expert.mean(axis=0) - nonexpert.mean(axis=0)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(difference_of_means(expert, nonexpert), expected)


def test_difference_of_means_rejects_identical_clouds():
    points = np.ones((12, DIM))
    with pytest.raises(ValueError, match="Degenerate"):
        difference_of_means(points, points)


# ---------------------------------------------------------------------- RFM

def test_rfm_is_deterministic():
    expert, nonexpert, _ = planted_clouds()
    np.testing.assert_allclose(rfm_agop(expert, nonexpert),
                               rfm_agop(expert, nonexpert))


def test_rfm_scales_to_hidden_dim_without_a_full_eigendecomposition():
    """AGOP is d x d (4096^2 for an 8B model) but its rank never exceeds the
    sample count, so cost tracks pairs rather than hidden size."""
    rng = np.random.default_rng(4)
    expert = rng.normal(size=(30, 2048)) + 0.4
    nonexpert = rng.normal(size=(30, 2048))
    direction = rfm_agop(expert, nonexpert, iterations=2)
    assert direction.shape == (2048,)
    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-8)


def test_rfm_rejects_tiny_input():
    with pytest.raises(ValueError, match="at least 4"):
        rfm_agop(np.ones((1, DIM)), np.ones((1, DIM)))


def test_rfm_survives_a_large_nuisance_direction():
    """A high-variance direction carrying no label information should not
    capture the estimate."""
    expert, nonexpert, planted = planted_clouds()
    rng = np.random.default_rng(7)
    nuisance = np.zeros(DIM)
    nuisance[0] = 1.0
    expert = expert + rng.normal(scale=8.0, size=(N, 1)) * nuisance
    nonexpert = nonexpert + rng.normal(scale=8.0, size=(N, 1)) * nuisance

    direction = rfm_agop(expert, nonexpert)
    assert abs(float(direction @ planted)) > 0.85
    assert abs(float(direction @ nuisance)) < 0.3


# -------------------------------------------------------------- diagnostics

def test_margin_keeps_discriminating_after_accuracy_saturates():
    """The whole reason the margin exists: two directions can both separate
    every held-out pair while one is plainly better."""
    strong_e, strong_n, _ = planted_clouds(seed=1, noise=0.1, separation=3.0)
    weak_e, weak_n, _ = planted_clouds(seed=1, noise=0.1, separation=1.0)

    strong_v = difference_of_means(strong_e, strong_n)
    weak_v = difference_of_means(weak_e, weak_n)

    strong_acc = float(((strong_e @ strong_v) > (strong_n @ strong_v)).mean())
    weak_acc = float(((weak_e @ weak_v) > (weak_n @ weak_v)).mean())
    assert strong_acc == weak_acc == 1.0          # accuracy cannot tell them apart

    assert (separation_margin(strong_e, strong_n, strong_v)
            > 2 * separation_margin(weak_e, weak_n, weak_v))


def test_margin_is_zero_for_an_orthogonal_direction():
    expert, nonexpert, planted = planted_clouds()
    orthogonal = np.zeros(DIM)
    orthogonal[0] = 1.0
    orthogonal -= (orthogonal @ planted) * planted
    orthogonal /= np.linalg.norm(orthogonal)
    assert abs(separation_margin(expert, nonexpert, orthogonal)) < 0.5


def test_margin_flips_sign_with_the_direction():
    expert, nonexpert, _ = planted_clouds()
    direction = difference_of_means(expert, nonexpert)
    assert separation_margin(expert, nonexpert, direction) == pytest.approx(
        -separation_margin(expert, nonexpert, -direction)
    )


def test_margin_handles_degenerate_input():
    single = np.ones((1, DIM))
    assert separation_margin(single, single, np.ones(DIM) / np.sqrt(DIM)) == 0.0


def test_cosine_of_identical_and_opposed_directions():
    vector = np.array([3.0, 4.0])
    assert direction_cosine(vector, vector) == pytest.approx(1.0)
    assert direction_cosine(vector, -vector) == pytest.approx(-1.0)
    assert direction_cosine(vector, np.array([-4.0, 3.0])) == pytest.approx(0.0)


def test_cosine_of_a_zero_vector_is_zero_not_nan():
    assert direction_cosine(np.zeros(3), np.ones(3)) == 0.0


def _isotropic_clouds(n, dim, seed=0, separation=2.0):
    """Independently drawn isotropic clouds — here the difference of means is
    the optimal discriminant, so both estimators target the same vector."""
    rng = np.random.default_rng(seed)
    planted = rng.normal(size=dim)
    planted /= np.linalg.norm(planted)
    return (rng.normal(size=(n, dim)) + separation * planted,
            rng.normal(size=(n, dim)) - separation * planted,
            planted)


def test_gate_reports_agreement_when_pairs_are_plentiful():
    """The cosine gate's purpose: with enough pairs to resolve the direction,
    both estimators recover it and the gate says a GPU A/B is not warranted."""
    expert, nonexpert, planted = _isotropic_clouds(n=400, dim=DIM)
    dm = difference_of_means(expert, nonexpert)
    rfm = rfm_agop(expert, nonexpert)

    assert abs(float(dm @ planted)) > 0.95        # both actually found it
    assert abs(float(rfm @ planted)) > 0.90
    assert abs(direction_cosine(dm, rfm)) > 0.95


def test_high_cosine_can_also_mean_too_few_pairs():
    """The failure mode the gate must not hide. When hidden size dwarfs the
    pair count, both estimators are dominated by the same sampling noise:
    they converge on each other while both sit far from the truth. A high
    cosine then says the pair count is the binding constraint, not the
    estimator."""
    expert, nonexpert, planted = _isotropic_clouds(n=147, dim=4096)
    dm = difference_of_means(expert, nonexpert)
    rfm = rfm_agop(expert, nonexpert, iterations=2)

    assert abs(direction_cosine(dm, rfm)) > 0.95   # they agree ...
    assert abs(float(dm @ planted)) < 0.7          # ... and both are wrong
    assert abs(float(rfm @ planted)) < 0.7


def test_rfm_beats_difference_of_means_on_margin_when_structure_is_paired():
    """And the converse. `planted_clouds` shares a base vector between the two
    sides, so the mean gap recovers the planted direction almost exactly —
    yet the planted direction is *not* the best separator under that
    covariance. RFM finds a better one, which is the whole reason to offer it,
    and the gate flags the disagreement."""
    expert, nonexpert, planted = planted_clouds(noise=0.1, separation=2.0)
    dm = difference_of_means(expert, nonexpert)
    rfm = rfm_agop(expert, nonexpert)

    assert abs(float(dm @ planted)) > 0.99          # mean gap nails the plant
    assert (separation_margin(expert, nonexpert, rfm)
            > separation_margin(expert, nonexpert, dm))
    assert abs(direction_cosine(dm, rfm)) < 0.95    # gate says "worth testing"


# ------------------------------------------------------- the holdout split

def test_split_keeps_every_framing_of_a_concept_on_one_side():
    """The leakage this exists to prevent: framings of one concept share most
    of their content, so a concept straddling the split would make holdout
    accuracy measure memorisation instead of generalisation."""
    concepts = [c for c in "abcdefghij" for _ in range(10)]
    holdout, train = split_by_concept(concepts, holdout_fraction=0.2, seed=0)

    held = {concepts[i] for i in holdout}
    trained = {concepts[i] for i in train}
    assert held and trained
    assert not (held & trained)
    assert held | trained == set("abcdefghij")


def test_split_covers_every_pair_exactly_once():
    concepts = [c for c in "abcdefghij" for _ in range(10)]
    holdout, train = split_by_concept(concepts, holdout_fraction=0.2, seed=1)
    assert sorted([*holdout, *train]) == list(range(len(concepts)))


def test_split_holds_out_the_requested_fraction_of_concepts():
    concepts = [c for c in "abcdefghij" for _ in range(10)]
    holdout, _ = split_by_concept(concepts, holdout_fraction=0.2, seed=0)
    assert len(holdout) == 20          # 2 of 10 concepts, 10 framings each


def test_split_is_seeded():
    concepts = [c for c in "abcdefghij" for _ in range(3)]
    first = split_by_concept(concepts, seed=7)[0]
    assert np.array_equal(first, split_by_concept(concepts, seed=7)[0])
    assert not np.array_equal(first, split_by_concept(concepts, seed=8)[0])


def test_unlabelled_pairs_split_individually():
    """A hand-written pairs file records no concepts, and must keep splitting
    per pair the way it always did."""
    holdout, train = split_by_concept([None] * 50, holdout_fraction=0.2, seed=0)
    assert len(holdout) == 10 and len(train) == 40


def test_too_few_concepts_to_hold_any_out_is_an_error():
    """Ten framings of one concept is ten pairs and zero generalisation
    evidence; saying so beats reporting accuracy on a leaked holdout."""
    with pytest.raises(ValueError, match="too few"):
        split_by_concept(["only concept"] * 10, holdout_fraction=0.2)


def test_shared_layer_cosines_only_on_overlap():
    rng = np.random.default_rng(0)
    a = {10: rng.normal(size=16), 12: rng.normal(size=16)}
    b = {12: a[12].copy(), 14: rng.normal(size=16)}
    cosines = shared_layer_cosines(a, b)
    assert list(cosines) == [12]
    assert cosines[12] == pytest.approx(1.0)

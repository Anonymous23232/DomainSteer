"""Direction estimators: two activation clouds → one steering direction.

Kept free of torch and transformers on purpose. This is the numerical core of
extraction — pure numpy over arrays of hidden states — so it is testable, and
usable, without loading a model. `extract.py` imports from here and re-exports
for backwards compatibility.

An estimator takes the expert and non-expert activation matrices of the
training split, each `(n_pairs, hidden_dim)`, and returns one unit vector.
Everything downstream — the layer sweep, calibration, the steering hook —
depends only on that contract, so estimators are interchangeable.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ESTIMATOR = "diff_means"
RFM_ITERATIONS = 5
RFM_RIDGE = 1e-3


# ------------------------------------------------------------------ splitting

def split_by_concept(concepts: Sequence[Optional[str]],
                     holdout_fraction: float = 0.2,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Train/holdout indices for a list of pairs, grouped by their concept.

    When a concept is asked under several framings it contributes several
    pairs, and splitting by pair index would put "Health Policy, explain the
    mechanism" into training and "Health Policy, explain what limits it" into
    holdout. Those two share a concept and most of their content, so holdout
    accuracy would be measuring memorisation — it would read high however poor
    the direction was, which is exactly the opposite of what that number is
    for.

    Grouping keeps every framing of a concept on one side. Pairs with no
    concept recorded each form their own group, so a hand-written pairs file
    splits per pair as it always did.

    Takes bare labels rather than pairs so this stays testable without torch,
    which is the same reason the estimators live here.
    """
    groups: Dict[str, List[int]] = {}
    for i, concept in enumerate(concepts):
        groups.setdefault(concept or f"__ungrouped_{i}", []).append(i)

    keys = list(groups)
    n_holdout = max(1, int(len(keys) * holdout_fraction))
    if n_holdout >= len(keys):
        raise ValueError(
            f"{len(keys)} distinct concept(s) is too few to hold any out at "
            f"holdout_fraction={holdout_fraction}. Adding framings of the same "
            "concepts will not help — the split is by concept — so add "
            "concepts."
        )

    order = np.random.default_rng(seed).permutation(len(keys))
    holdout: List[int] = []
    train: List[int] = []
    for rank, key_index in enumerate(order):
        (holdout if rank < n_holdout else train).extend(groups[keys[key_index]])
    return np.array(sorted(holdout)), np.array(sorted(train))


# --------------------------------------------------------------- estimators

def _unit(vector: np.ndarray, what: str) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError(f"Degenerate ({what}) direction.")
    return vector / norm


def difference_of_means(expert: np.ndarray, nonexpert: np.ndarray) -> np.ndarray:
    """Normalized difference of class means — the original estimator."""
    return _unit(expert.mean(axis=0) - nonexpert.mean(axis=0), "zero")


def _laplace_bandwidth(distances: np.ndarray) -> float:
    """Median heuristic, which adapts to the scale of raw hidden states."""
    off_diagonal = distances[~np.eye(len(distances), dtype=bool)]
    median = float(np.median(off_diagonal))
    return median if median > 0 else 1.0


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    sq = (points ** 2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (points @ points.T)
    return np.sqrt(np.maximum(d2, 0.0))


def rfm_agop(expert: np.ndarray, nonexpert: np.ndarray,
             iterations: int = RFM_ITERATIONS,
             ridge: float = RFM_RIDGE) -> np.ndarray:
    """Top eigenvector of the AGOP of a Recursive Feature Machine.

    Alternates two steps: fit kernel ridge regression predicting the label
    (expert 1, non-expert 0) under a Mahalanobis metric M, then reset M to
    the Average Gradient Outer Product of the fitted predictor,
    ``M = E[∇f ∇fᵀ]``. Directions in which the prediction actually moves get
    large eigenvalues; directions the predictor ignores decay away.

    The naive form needs an eigendecomposition of a `hidden_dim × hidden_dim`
    matrix each round — 4096² for an 8B model. It is avoided throughout:
    AGOP is ``GᵀG / n`` for the `n × d` gradient matrix G, so its rank never
    exceeds the sample count, and the SVD of G supplies both the Mahalanobis
    projection and the final eigenvector directly. Cost is set by the number
    of pairs, not by hidden size.
    """
    points = np.concatenate([expert, nonexpert]).astype(np.float64)
    labels = np.concatenate([np.ones(len(expert)), np.zeros(len(nonexpert))])
    n = len(points)
    if n < 4:
        raise ValueError(f"RFM needs at least 4 activations, got {n}.")

    projected = points                       # M = I on the first round
    basis: Optional[np.ndarray] = None       # M^½, once M stops being I
    right = np.empty((0, 0))

    for _ in range(max(1, iterations)):
        distances = _pairwise_distances(projected)
        gamma = 1.0 / _laplace_bandwidth(distances)

        kernel = np.exp(-gamma * distances)
        try:
            weights = np.linalg.solve(kernel + ridge * np.eye(n), labels)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(kernel + ridge * np.eye(n), labels,
                                      rcond=None)[0]

        # ∇f(x_i) = -gamma Σ_j w_j exp(-gamma r_ij)/r_ij · Mᵀ(z_i - z_j).
        # The Laplace kernel is non-differentiable at r = 0, so the self term
        # is excluded rather than dividing by zero.
        safe = np.where(distances > 0, distances, 1.0)
        coeff = weights[None, :] * kernel / safe
        np.fill_diagonal(coeff, 0.0)
        deltas = coeff.sum(axis=1)[:, None] * projected - coeff @ projected

        # M = basisᵀbasis, so M(x − x_j) = basisᵀ(z − z_j); on the first
        # round M is the identity and the deltas are already in input space.
        gradients = -gamma * (deltas if basis is None else deltas @ basis)

        # AGOP = GᵀG/n; its eigenvectors are G's right singular vectors.
        _, singular, right = np.linalg.svd(gradients, full_matrices=False)
        rank = int((singular > singular[0] * 1e-10).sum()) if singular[0] > 0 else 0
        if rank == 0:
            raise ValueError("Degenerate (zero gradient) RFM direction.")

        # M^½ = V diag(s)/√n Vᵀ, and ‖M^½u‖ = ‖basis · u‖ for this basis.
        basis = (singular[:rank, None] / np.sqrt(n)) * right[:rank]
        projected = points @ basis.T

    return _unit(right[0], "zero RFM")


ESTIMATORS = {
    "diff_means": difference_of_means,
    "rfm": rfm_agop,
}


# --------------------------------------------------------------- diagnostics

def separation_margin(expert: np.ndarray, nonexpert: np.ndarray,
                      direction: np.ndarray) -> float:
    """Standardized mean difference (Cohen's d) of projections onto `direction`.

    Holdout *accuracy* saturates — every layer can tell the personas apart, as
    `calibrate.py` notes — so it cannot rank two estimators that both separate
    cleanly, and on a 20% holdout of ~180 pairs it is scored on ~36 items
    anyway. This is the continuous version: it keeps climbing after accuracy
    pins at 1.0, so it can say *how much* better one direction is.
    """
    a = expert @ direction
    b = nonexpert @ direction
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def direction_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine between two directions — the cheap gate before spending GPU time.

    Two estimators agreeing above ~0.95 have found the same thing, so swapping
    one for the other should not change behaviour.

    Read a *high* cosine carefully, because there are two ways to get one.
    Either the estimators genuinely agree, or there are too few pairs for
    either to resolve the direction and both are dominated by the same
    sampling noise. Measured on isotropic synthetic clouds, with the true
    direction known:

        pairs   dim    cos(dm, rfm)   cos(dm, true)
          400    48           0.969           0.995
          147   512           0.978           0.832
          147  4096           0.997           0.472

    The last row is the shape of a real run — a few hundred pairs against a
    hidden size of thousands. The estimators converge on each other while
    both sit far from the truth. When `hidden_dim` greatly exceeds the pair
    count, a high cosine says the estimator is not the binding constraint;
    the pair count is.
    """
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else 0.0


def shared_layer_cosines(a: Dict[int, np.ndarray],
                         b: Dict[int, np.ndarray]) -> Dict[int, float]:
    """Per-layer cosine on the layers both maps share.

    The diagnostic for whether two *domains* produced the same direction:
    hydrology vs policy above ~0.8 at the same layer means both extracted
    "sounds technical", and steering will not supply a domain reading.
    Below ~0.3 there is domain content in the residual.
    """
    return {layer: direction_cosine(a[layer], b[layer])
            for layer in sorted(set(a) & set(b))}

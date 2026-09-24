"""
domainsteer — Steer any LLM toward user-specified domain expertise via
calibrated activation steering.
"""

__version__ = "1.0.1"

from domainsteer.clusters import (CONCEPTS_PER_CLUSTER, ClusterConcept, Domain,
                                  DomainCluster, find_cluster, load_clusters,
                                  select_clusters, verify_clusters)
from domainsteer.em_traps import (benchmark_manifest, benchmark_path,
                                  iter_benchmark, list_benchmark_domains,
                                  load_benchmark)
from domainsteer.pairs import (ContrastivePair, PairGenerator, bundled_manifest,
                               bundled_pairs_path, list_bundled_models,
                               load_bundled_pairs, load_pairs, save_pairs)
from domainsteer.steerer import DomainSteerer

__all__ = [
    "CONCEPTS_PER_CLUSTER",
    "ClusterConcept",
    "ContrastivePair",
    "Domain",
    "DomainCluster",
    "DomainSteerer",
    "PairGenerator",
    "benchmark_manifest",
    "benchmark_path",
    "bundled_manifest",
    "bundled_pairs_path",
    "find_cluster",
    "iter_benchmark",
    "list_benchmark_domains",
    "list_bundled_models",
    "load_benchmark",
    "load_bundled_pairs",
    "load_clusters",
    "load_pairs",
    "save_pairs",
    "select_clusters",
    "verify_clusters",
]

# Heavier modules (torch/transformers) are imported on demand:
#   domainsteer.extract    — DirectionExtractor, ExtractionResult, load_model
#   domainsteer.steering   — ActivationSteering
# Archived (not imported here): extras/ — not installed with the wheel.
#   NLI/pairwise calibration, older exams, LLM judge, grid sweep, paper.

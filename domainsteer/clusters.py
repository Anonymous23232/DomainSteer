"""Domain clusters: the registry of steerable domains and their concepts.

This is the front door of the pipeline. A **cluster** is one steerable domain
— an ANZSRC 4-digit research group — and its **concepts** are the sub-areas
that partition it. Naming a cluster is all a user has to do: the registry's
ten concept names ground pair generation, `pairs.py` turns those into
contrastive pairs, and everything downstream follows.

Two invariants define the registry, and both are checkable:

*Exactly ten concepts per cluster.* Uniform depth is the point. A cluster
trimmed from twenty-eight and a cluster grown from one must end up describing
their territory at the same granularity, or dataset sizes and concept
specificity vary by an accident of how the source classification happened to
subdivide that corner of knowledge — and cross-domain numbers stop being
comparable.

*Globally disjoint concepts.* No concept key appears under two clusters. This
is what makes a cross-domain benchmark clean (no leakage between the domains
being compared) and what a future on-topic gate would rest on. Note that it is
not required for a single domain's steering to work: the extracted direction
is a *domain* direction, in-domain versus field-neutral on the same term, so a term two
fields share still yields a valid contrast.

The ANZSRC 6-digit fields the source classification ships are kept as *seeds*
rather than used as the concepts themselves, because in narrow clusters they
are only the cluster's own name split into parts — cluster 5005 "Theology" has
the single entry "Theology", and 5106 "Nuclear and plasma physics" has
"Nuclear physics" and "Plasma physics". `is_restatement` detects exactly that
so `cluster_build` can discard them instead of generating sub-areas underneath
a heading that is the whole domain.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

CONCEPTS_PER_CLUSTER = 10


def _singularize(word: str) -> str:
    """Crude English plural folding, enough to collapse 'piezometers' onto
    'piezometer' and 'confining beds' onto 'confining bed'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _dedup_key(term: str) -> str:
    """Singular/plural variants of the same concept share a key."""
    words = term.split()
    return " ".join(words[:-1] + [_singularize(words[-1])])

# Installed as package data (domainsteer/data/domain_clusters.json) so the
# registry resolves from a wheel as well as from a source checkout. The
# repo-root copy is still honoured if present, for checkouts that predate the
# move.
_PACKAGE_CLUSTERS_PATH = Path(__file__).resolve().parent / "data" / "domain_clusters.json"
_LEGACY_CLUSTERS_PATH = Path(__file__).resolve().parent.parent / "domain_clusters.json"

DEFAULT_CLUSTERS_PATH = (_PACKAGE_CLUSTERS_PATH
                         if _PACKAGE_CLUSTERS_PATH.exists()
                         else _LEGACY_CLUSTERS_PATH)

MIN_CONCEPT_WORDS = 1
MAX_CONCEPT_WORDS = 8

# Letters plus the punctuation real sub-area names carry ("Solid-State &
# Materials Chemistry", "Darcy's law"). Digits and notation mark a citation
# fragment or a variable name rather than a sub-area.
_CONCEPT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z'’\-&/, ]*[A-Za-z]$")

# Dropped before comparing a concept name against its cluster's name, so
# "Nuclear physics" is recognised inside "Nuclear and plasma physics".
_STOPWORDS = {"and", "or", "of", "the", "in", "for", "with", "a", "an", "to",
              "on", "incl", "excl", "other", "not", "elsewhere", "classified"}


# ------------------------------------------------------------------- domain

@dataclass
class Domain:
    """A steering target: its name plus the concepts that define it.

    `concepts` are the terms contrastive pairs get generated for. A bare name
    is allowed when the domain resolves to a registry cluster, or when its
    pairs are already cached.
    """

    name: str
    concepts: List[str] = dataclass_field(default_factory=list)

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Domain needs a non-empty name.")
        self.name = self.name.strip()


# ------------------------------------------------------------------ records

@dataclass
class ClusterConcept:
    """One sub-area of a cluster — used as grounding for in-domain pair answers."""

    name: str
    scope: str = ""
    seed_id: str = ""          # source classification id, when it came from one

    def describe(self) -> str:
        """The form the expansion prompt lists concepts in."""
        return f"{self.name} — {self.scope}" if self.scope else self.name


@dataclass
class DomainCluster:
    """One steerable domain: a cluster, its ancestry, and its ten concepts."""

    cluster_id: str
    cluster: str
    concepts: List[ClusterConcept] = dataclass_field(default_factory=list)
    scope: str = ""
    domain: str = ""
    domain_id: str = ""
    division: str = ""
    division_id: str = ""
    division_scope: str = ""
    boundary: str = ""

    @property
    def name(self) -> str:
        """The name this cluster carries into the rest of the pipeline."""
        return self.cluster

    @property
    def path(self) -> str:
        """Human-readable ancestry, used in prompts and logs."""
        return f"{self.domain} > {self.division} > {self.cluster}"

    def siblings(self) -> List[str]:
        """The other clusters in this division.

        The source classification already stores them, comma-joined, in
        `division_scope` — so the neighbours a concept must not trespass on
        come free with the data rather than needing to be inferred.
        """
        others = [s.strip() for s in self.division_scope.split(",") if s.strip()]
        return [s for s in others if s.lower() != self.cluster.lower()]

    def context_block(self) -> str:
        """The ancestry as prompt context.

        Names alone are ambiguous across the tree — "Dynamics" means one thing
        under mechanics and another under meteorology — so prompts are built
        from the full path rather than the bare cluster name.
        """
        lines = [f"Knowledge domain: {self.domain}",
                 f"Division: {self.division}",
                 f"Cluster: {self.cluster} (id {self.cluster_id})"]
        if self.scope:
            lines.append(f"Scope: {self.scope}")
        if self.boundary:
            lines.append(f"Boundary: {self.boundary}")
        return "\n".join(lines)


# ------------------------------------------------------------------ loading

def _as_concept(raw, index: int) -> Optional[ClusterConcept]:
    """One concept from either registry generation.

    Plain strings, the source classification's `{"id", "name", "scope",
    "prefix"}` objects, and the built registry's `{"name", "scope"}` all load,
    mirroring the concept-file loader's tolerance of hand-written files.
    """
    if isinstance(raw, str):
        name = raw.strip()
        return ClusterConcept(name) if name else None
    if isinstance(raw, dict):
        name = str(raw.get("name", "")).strip()
        if not name:
            return None
        return ClusterConcept(name,
                              str(raw.get("scope") or "").strip(),
                              str(raw.get("seed_id") or raw.get("id") or "").strip())
    return None


def load_clusters(path: Optional[Union[str, Path]] = None) -> List[DomainCluster]:
    """Load the domain-cluster registry.

    Accepts both the source-classification seed file and a built registry;
    they differ only in whether concepts carry ids and whether clusters carry
    a boundary, and both shapes are read by the same tolerant parser.
    """
    path = Path(path) if path else DEFAULT_CLUSTERS_PATH
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_clusters = payload.get("clusters")
    if not isinstance(raw_clusters, list) or not raw_clusters:
        raise ValueError(f"{path}: expected a non-empty 'clusters' list.")

    clusters: List[DomainCluster] = []
    for i, raw in enumerate(raw_clusters, 1):
        name = str(raw.get("cluster") or "").strip()
        if not name:
            logger.warning(f"{path}: cluster #{i} has no name; skipping.")
            continue
        concepts = [c for c in (_as_concept(item, j)
                                for j, item in enumerate(raw.get("concepts", []), 1))
                    if c is not None]
        clusters.append(DomainCluster(
            cluster_id=str(raw.get("cluster_id") or i),
            cluster=name,
            concepts=concepts,
            scope=str(raw.get("scope") or "").strip(),
            domain=str(raw.get("domain") or "").strip(),
            domain_id=str(raw.get("domain_id") or "").strip(),
            division=str(raw.get("division") or "").strip(),
            division_id=str(raw.get("division_id") or "").strip(),
            division_scope=str(raw.get("division_scope") or "").strip(),
            boundary=str(raw.get("boundary") or "").strip(),
        ))

    if not clusters:
        raise ValueError(f"{path}: no usable clusters found.")
    logger.info(f"Loaded {len(clusters)} clusters "
                f"({sum(len(c.concepts) for c in clusters)} concepts) from {path}")
    return clusters


def clusters_payload(clusters: Sequence[DomainCluster],
                     source: Optional[dict] = None,
                     note: str = "") -> dict:
    """Assemble the on-disk registry."""
    domains: Dict[str, int] = {}
    for cluster in clusters:
        domains[cluster.domain] = domains.get(cluster.domain, 0) + 1
    return {
        "schema": "domainsteer/domain-clusters/2",
        "note": note or (
            "Every steerable domain in one file. A cluster is one domain; its "
            "concepts are the sub-areas that partition it. Every cluster "
            f"carries exactly {CONCEPTS_PER_CLUSTER} concepts, and no concept "
            "appears under two clusters."
        ),
        "source": source or {},
        "counts": {
            "domains": len(domains),
            "clusters": len(clusters),
            "concepts": sum(len(c.concepts) for c in clusters),
        },
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "cluster": c.cluster,
                "scope": c.scope,
                "domain": c.domain,
                "domain_id": c.domain_id,
                "division": c.division,
                "division_id": c.division_id,
                "division_scope": c.division_scope,
                "boundary": c.boundary,
                "n_concepts": len(c.concepts),
                "concepts": [
                    {"name": k.name, "scope": k.scope, "seed_id": k.seed_id}
                    for k in c.concepts
                ],
            }
            for c in clusters
        ],
    }


def save_clusters(clusters: Sequence[DomainCluster],
                  path: Union[str, Path],
                  source: Optional[dict] = None, note: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clusters_payload(clusters, source, note), f,
                  indent=1, ensure_ascii=False)
    logger.info(f"Saved {len(clusters)} clusters to {path}")


# ---------------------------------------------------------------- selection

def find_cluster(clusters: Sequence[DomainCluster], query: str) -> DomainCluster:
    """Resolve a user-supplied domain name to exactly one cluster.

    Tries, in order: cluster id, exact name, case-insensitive name, then a
    unique substring. Ambiguous substrings raise rather than guessing, since
    silently steering toward the wrong domain is worse than a failed command.
    """
    text = str(query).strip()
    if not text:
        raise ValueError("Empty domain name.")

    for cluster in clusters:
        if cluster.cluster_id == text:
            return cluster
    for cluster in clusters:
        if cluster.cluster == text:
            return cluster

    lowered = text.lower()
    exact = [c for c in clusters if c.cluster.lower() == lowered]
    if len(exact) == 1:
        return exact[0]

    partial = [c for c in clusters if lowered in c.cluster.lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        names = ", ".join(f"{c.cluster_id} {c.cluster}" for c in partial[:8])
        raise ValueError(
            f"'{query}' matches {len(partial)} clusters ({names}"
            f"{', ...' if len(partial) > 8 else ''}). Use the full name or the "
            "cluster id."
        )

    near = [c for c in clusters
            if any(word in c.cluster.lower() for word in lowered.split() if len(word) > 3)]
    hint = ("Closest: " + ", ".join(f"{c.cluster_id} {c.cluster}" for c in near[:5])
            if near else
            "Run `domainsteer clusters --search <word>` to find one.")
    raise ValueError(f"No cluster matches '{query}'. {hint}")


def select_clusters(clusters: Sequence[DomainCluster],
                    ids: Optional[Iterable[str]] = None,
                    names: Optional[Iterable[str]] = None,
                    divisions: Optional[Iterable[str]] = None,
                    domains: Optional[Iterable[str]] = None,
                    search: Optional[str] = None) -> List[DomainCluster]:
    """Filter clusters by id prefix, name, division, domain, or free text.

    Filters combine as OR; with no filters every cluster is returned.
    """
    if not any((ids, names, divisions, domains, search)):
        return list(clusters)

    id_prefixes = [str(i).strip() for i in (ids or []) if str(i).strip()]
    name_set = {str(n).strip().lower() for n in (names or []) if str(n).strip()}
    division_set = {str(d).strip().lower() for d in (divisions or []) if str(d).strip()}
    domain_set = {str(d).strip().lower() for d in (domains or []) if str(d).strip()}
    needle = search.strip().lower() if search else None

    def matches(cluster: DomainCluster) -> bool:
        if cluster.cluster.lower() in name_set:
            return True
        if cluster.division.lower() in division_set:
            return True
        if cluster.domain.lower() in domain_set:
            return True
        if any(cluster.cluster_id.startswith(p) for p in id_prefixes):
            return True
        if needle and (needle in cluster.cluster.lower()
                       or any(needle in k.name.lower() for k in cluster.concepts)):
            return True
        return False

    selected = [c for c in clusters if matches(c)]
    if not selected:
        available = ", ".join(f"{c.cluster_id} {c.cluster}" for c in clusters[:8])
        raise ValueError(
            f"No clusters matched (ids={id_prefixes}, names={sorted(name_set)}, "
            f"divisions={sorted(division_set)}, domains={sorted(domain_set)}, "
            f"search={search!r}). Clusters begin: {available}, ... "
            f"({len(clusters)} total)."
        )
    return selected


# --------------------------------------------------------------- invariants

def _significant_tokens(text: str) -> set:
    """Lowercase content words, singularised — the unit name comparison uses."""
    words = re.findall(r"[a-z']+", text.lower())
    return {_singularize(w) for w in words if w not in _STOPWORDS}


def is_restatement(concept_name: str, cluster_name: str) -> bool:
    """True when a concept merely restates its cluster's name.

    The source classification's narrow clusters subdivide into nothing but
    their own name — "Nuclear and plasma physics" into "Nuclear physics" and
    "Plasma physics; fusion plasmas; electrical discharges". Such an entry is
    useless as a concept and actively harmful as a generation seed, because
    the model would then invent sub-areas underneath a heading that is the
    entire domain.

    The leading segment is what gets compared, since the classification writes
    compound entries as "Plasma physics; fusion plasmas; electrical
    discharges" where only the first part names the thing.
    """
    head = re.split(r"[;,]", concept_name)[0]
    concept_tokens = _significant_tokens(head)
    cluster_tokens = _significant_tokens(cluster_name)
    if not concept_tokens or not cluster_tokens:
        return False
    return concept_tokens <= cluster_tokens or cluster_tokens <= concept_tokens


def is_valid_concept_name(name: str) -> bool:
    """Deterministic gate on a concept name's form."""
    text = name.strip()
    if len(text) < 3:
        return False
    if not MIN_CONCEPT_WORDS <= len(text.split()) <= MAX_CONCEPT_WORDS:
        return False
    return bool(_CONCEPT_NAME_RE.match(text))


def concept_index(clusters: Sequence[DomainCluster]) -> Dict[str, List[str]]:
    """Concept key → the cluster id of every occurrence.

    Occurrences rather than distinct clusters, so the same list distinguishes a
    cross-cluster collision from one cluster listing a concept twice.
    """
    index: Dict[str, List[str]] = {}
    for cluster in clusters:
        for concept in cluster.concepts:
            index.setdefault(_dedup_key(concept.name.lower()), []).append(
                cluster.cluster_id
            )
    return index


def find_collisions(clusters: Sequence[DomainCluster]) -> Dict[str, List[str]]:
    """Concept keys claimed by more than one *distinct* cluster.

    Exact matching on `_dedup_key`, which already folds singular and plural, so
    "Reaction Mechanism" and "reaction mechanisms" collide as they should.

    Repeats inside a single cluster are deliberately excluded: they are a
    defect, but not one adjudication can resolve — there is no second claimant
    to award the concept to — so a resolver fed them would loop without making
    progress. `find_duplicates` reports those instead.
    """
    return {key: sorted(set(ids)) for key, ids in concept_index(clusters).items()
            if len(set(ids)) > 1}


def find_duplicates(clusters: Sequence[DomainCluster]) -> Dict[str, List[str]]:
    """Cluster id → concept keys that cluster lists more than once."""
    duplicates: Dict[str, List[str]] = {}
    for cluster in clusters:
        counts: Dict[str, int] = {}
        for concept in cluster.concepts:
            key = _dedup_key(concept.name.lower())
            counts[key] = counts.get(key, 0) + 1
        repeated = sorted(k for k, n in counts.items() if n > 1)
        if repeated:
            duplicates[cluster.cluster_id] = repeated
    return duplicates


def find_near_collisions(clusters: Sequence[DomainCluster]) -> List[tuple]:
    """Cross-cluster concept pairs where one name's tokens contain the other's.

    Reported rather than enforced. "Reaction Mechanisms" inside "Inorganic
    Reaction Mechanisms" is a genuine overlap worth a human's attention, but
    treating containment as a hard failure would reject legitimately distinct
    names and make the invariant unmaintainable.
    """
    entries = [(c.cluster_id, k.name, _significant_tokens(k.name))
               for c in clusters for k in c.concepts]
    near = []
    for i, (id_a, name_a, tokens_a) in enumerate(entries):
        for id_b, name_b, tokens_b in entries[i + 1:]:
            if id_a == id_b or not tokens_a or not tokens_b:
                continue
            if tokens_a == tokens_b:
                continue                       # an exact collision already
            if tokens_a <= tokens_b or tokens_b <= tokens_a:
                near.append((id_a, name_a, id_b, name_b))
    return near


def verify_clusters(clusters: Sequence[DomainCluster],
                    expected: int = CONCEPTS_PER_CLUSTER,
                    strict: bool = False) -> dict:
    """Check the registry's two invariants and return a report.

    With `strict`, a violation raises instead of being reported — which is how
    the build's final round and the test suite consume this.
    """
    wrong_count = {c.cluster_id: len(c.concepts) for c in clusters
                   if len(c.concepts) != expected}
    collisions = find_collisions(clusters)
    duplicates = find_duplicates(clusters)
    restatements = [(c.cluster_id, k.name) for c in clusters for k in c.concepts
                    if is_restatement(k.name, c.cluster)]
    malformed = [(c.cluster_id, k.name) for c in clusters for k in c.concepts
                 if not is_valid_concept_name(k.name)]

    report = {
        "clusters": len(clusters),
        "concepts": sum(len(c.concepts) for c in clusters),
        "wrong_count": wrong_count,
        "collisions": collisions,
        "duplicates": duplicates,
        "restatements": restatements,
        "malformed": malformed,
        "near_collisions": find_near_collisions(clusters),
        "ok": not (wrong_count or collisions or duplicates or restatements
                   or malformed),
    }

    if strict and not report["ok"]:
        problems = []
        if wrong_count:
            problems.append(f"{len(wrong_count)} clusters without exactly "
                            f"{expected} concepts")
        if collisions:
            problems.append(f"{len(collisions)} concepts claimed by "
                            "more than one cluster")
        if duplicates:
            problems.append(f"{len(duplicates)} clusters listing a concept twice")
        if restatements:
            problems.append(f"{len(restatements)} concepts restating their "
                            "cluster's name")
        if malformed:
            problems.append(f"{len(malformed)} malformed concept names")
        raise ValueError("Registry invariants violated: " + "; ".join(problems))
    return report


def format_verify_report(report: dict) -> str:
    """Human-readable summary of `verify_clusters`."""
    lines = [f"{report['clusters']} clusters, {report['concepts']} concepts",
             ""]
    if report["ok"]:
        lines.append("All invariants hold: exactly ten concepts per cluster, "
                     "no concept shared between clusters.")
    else:
        for label, key in (("clusters without exactly ten concepts", "wrong_count"),
                           ("concepts claimed by two or more clusters", "collisions"),
                           ("clusters listing a concept twice", "duplicates"),
                           ("concepts restating their cluster name", "restatements"),
                           ("malformed concept names", "malformed")):
            items = report[key]
            if items:
                lines.append(f"{len(items)} {label}")
    near = report.get("near_collisions") or []
    if near:
        lines += ["", f"{len(near)} near-collisions (reported, not enforced):"]
        lines += [f"  {a} {an!r} ~ {b} {bn!r}" for a, an, b, bn in near[:10]]
        if len(near) > 10:
            lines.append(f"  ... and {len(near) - 10} more")
    return "\n".join(lines)

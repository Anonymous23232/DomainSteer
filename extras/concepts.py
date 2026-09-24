"""Domain concepts: the terms a contrastive pair is generated for.

A concept is just a term plus optional provenance (`score`, `count`). Concepts
come from a curated file, or from the ten concept names on a registry cluster.
This module only defines the record and reads/writes the on-disk format both
routes share.

The file format is deliberately loose — `load_concepts` accepts either this
module's `{"concepts": [{"term": ...}, ...]}` or a hand-written
`{"concepts": ["term", ...]}` — so a generated file can be edited by hand
without ceremony. Unknown keys (`category`, notes) are ignored, not rejected.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    term: str
    score: float
    count: int


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


def save_concepts(concepts: Sequence[Concept], path: Union[str, Path],
                  domain: str, params: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "domain": domain,
        "params": params or {},
        "concepts": [asdict(c) for c in concepts],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(concepts)} concepts to {path}")


def load_concepts(path: Union[str, Path]) -> List[Concept]:
    """Load a concepts file — either this module's format or a hand-written
    {"concepts": ["term", ...]} list."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = payload.get("concepts", payload if isinstance(payload, list) else [])
    concepts: List[Concept] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            concepts.append(Concept(item.strip(), 0.0, 0))
        elif isinstance(item, dict) and isinstance(item.get("term"), str) \
                and item["term"].strip():
            concepts.append(Concept(item["term"].strip(),
                                    float(item.get("score", 0.0)),
                                    int(item.get("count", 0))))
    if not concepts:
        raise ValueError(f"No concepts found in {path}.")
    return concepts

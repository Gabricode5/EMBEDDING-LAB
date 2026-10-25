"""Paires d'unités pour la classification des relations.

Chaque paire est ordonnée chronologiquement : A est l'unité la plus ancienne, B la plus récente
(à date égale, ordre des identifiants). Le label décrit le lien entre les deux, le sens étant
fixé par la chronologie : « supersedes » signifie que B remplace A, « answers » que B répond à A, etc.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations

from sae.data import TYPE_LABELS_FR, gid
from sae.schemas import Scenario, Unit

RELATIONS = ("supersedes", "contradicts", "answers", "creates_action", "supports", "unrelated")
LABEL2ID = {r: i for i, r in enumerate(RELATIONS)}


@dataclass(frozen=True)
class UnitPair:
    a: str          # identifiant global de l'unité la plus ancienne
    b: str          # identifiant global de l'unité la plus récente
    label: str      # une des RELATIONS


def unit_label(u: Unit) -> str:
    return f"{u.source_date.isoformat()} · {TYPE_LABELS_FR[u.type]} : {u.statement}"


def _ordered(s: Scenario, x: str, y: str) -> tuple[str, str]:
    units = {u.unit_id: u for u in s.units}
    kx, ky = (units[x].source_date, x), (units[y].source_date, y)
    return (x, y) if kx <= ky else (y, x)


def all_units(scenarios: list[Scenario]) -> dict[str, Unit]:
    return {gid(s.project_id, u.unit_id): u for s in scenarios for u in s.units}


def candidate_pairs(scenarios: list[Scenario]) -> list[tuple[str, str]]:
    """Toutes les paires d'unités d'un même projet, ordonnées chronologiquement."""
    out = []
    for s in scenarios:
        for x, y in combinations([u.unit_id for u in s.units], 2):
            a, b = _ordered(s, x, y)
            out.append((gid(s.project_id, a), gid(s.project_id, b)))
    return out


def gold_relations(scenarios: list[Scenario]) -> dict[tuple[str, str], str]:
    gold: dict[tuple[str, str], str] = {}
    for s in scenarios:
        for r in s.relations:
            a, b = _ordered(s, r.from_id, r.to_id)
            gold[(gid(s.project_id, a), gid(s.project_id, b))] = r.relation
    return gold


def labelled_pairs(scenarios: list[Scenario], unrelated_ratio: float = 2.0, seed: int = 42) -> list[UnitPair]:
    """Relations annotées + paires non reliées tirées au hasard dans le même projet."""
    rng = random.Random(seed)
    gold = gold_relations(scenarios)
    linked = [UnitPair(a, b, lab) for (a, b), lab in gold.items()]
    pool = [p for p in candidate_pairs(scenarios) if p not in gold]
    n = min(len(pool), int(round(unrelated_ratio * sum(p.label != "unrelated" for p in linked))))
    return linked + [UnitPair(a, b, "unrelated") for a, b in rng.sample(pool, n)]


def all_pairs_with_labels(scenarios: list[Scenario]) -> list[UnitPair]:
    """Toutes les paires candidates, avec « unrelated » par défaut (évaluation réaliste)."""
    gold = gold_relations(scenarios)
    return [UnitPair(a, b, gold.get((a, b), "unrelated")) for a, b in candidate_pairs(scenarios)]

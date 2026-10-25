"""Construction des exemples d'entraînement contrastif à partir des scénarios.

Deux sources de paires (ancre, positif) :
- question de test -> unité qui y répond ;
- extrait oral (evidence) -> reformulation de l'unité (statement).

Négatifs difficiles :
- explicites : négatifs annotés dans les questions, et unités reliées par
  supersedes / contradicts (même sujet, validité différente) ;
- minés : unités du même projet les mieux classées par un modèle, hors positifs
  et hors unités qui soutiennent le positif (pour limiter les faux négatifs).
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from sae.data import Corpus, gid
from sae.schemas import Scenario

NEGATIVE_MODES = ("none", "explicit", "mined", "both")


@dataclass(frozen=True)
class Pair:
    anchor: str
    positive: str                     # identifiant global de l'unité positive
    source: str                       # "query" ou "evidence"
    explicit_negatives: tuple[str, ...]
    exclude: frozenset[str]           # jamais utilisés comme négatifs

    @property
    def project(self) -> str:
        return self.positive.split("/", 1)[0]


def build_pairs(scenarios: list[Scenario], use_queries: bool = True, use_evidence: bool = True) -> list[Pair]:
    pairs: list[Pair] = []
    for s in scenarios:
        p = s.project_id
        rivals: dict[str, set[str]] = defaultdict(set)    # même sujet, validité différente
        allies: dict[str, set[str]] = defaultdict(set)    # se soutiennent : à ne pas opposer
        for r in s.relations:
            a, b = gid(p, r.from_id), gid(p, r.to_id)
            if r.relation in ("supersedes", "contradicts"):
                rivals[a].add(b)
                rivals[b].add(a)
            elif r.relation in ("supports", "answers"):
                allies[a].add(b)
                allies[b].add(a)

        if use_queries:
            for q in s.queries:
                if not q.answerable:
                    continue
                relevant = {gid(p, u) for u in q.relevant_unit_ids}
                negatives = tuple(gid(p, u) for u in q.hard_negative_unit_ids)
                exclude = frozenset(relevant | {x for r in relevant for x in allies[r]})
                for r in sorted(relevant):
                    pairs.append(Pair(q.text, r, "query", negatives, exclude))

        if use_evidence:
            for u in s.units:
                g = gid(p, u.unit_id)
                pairs.append(Pair(u.evidence, g, "evidence", tuple(sorted(rivals[g])),
                                  frozenset({g} | allies[g])))
    return pairs


def mine_negatives(pairs: list[Pair], corpus: Corpus, score_fn: Callable[[list[str]], np.ndarray],
                   k: int = 2) -> list[list[str]]:
    """Pour chaque paire, les k unités du même projet les mieux classées (hors positifs)."""
    scores = score_fn([pr.anchor for pr in pairs])
    by_project: dict[str, np.ndarray] = {}
    for proj in {pr.project for pr in pairs}:
        by_project[proj] = np.array([i for i, d in enumerate(corpus.ids) if d.startswith(proj + "/")])
    mined: list[list[str]] = []
    for i, pr in enumerate(pairs):
        idx = by_project[pr.project]
        order = idx[np.argsort(-scores[i, idx], kind="stable")]
        picks = [corpus.ids[j] for j in order if corpus.ids[j] not in pr.exclude][:k]
        mined.append(picks)
    return mined


def to_rows(pairs: list[Pair], corpus: Corpus, mode: str, mined: list[list[str]] | None,
            query_prefix: str, passage_prefix: str, k: int = 2, seed: int = 42) -> dict[str, list[str]]:
    """Colonnes pour Sentence Transformers : anchor, positive (+ negative).

    mode="none"     : (anchor, positive), négatifs pris dans le batch uniquement
    mode="explicit" : négatifs annotés ; à défaut, une unité au hasard du même projet
    mode="mined"    : négatifs minés
    mode="both"     : négatifs annotés + minés
    Une ligne par négatif.
    """
    if mode not in NEGATIVE_MODES:
        raise ValueError(f"mode inconnu : {mode}")
    if mode in ("mined", "both") and mined is None:
        raise ValueError("négatifs minés requis pour ce mode")
    rng = random.Random(seed)
    text = dict(zip(corpus.ids, corpus.texts))
    project_units: dict[str, list[str]] = defaultdict(list)
    for d in corpus.ids:
        project_units[d.split("/", 1)[0]].append(d)

    rows: dict[str, list[str]] = {"anchor": [], "positive": []}
    if mode != "none":
        rows["negative"] = []
    for i, pr in enumerate(pairs):
        anchor, positive = query_prefix + pr.anchor, passage_prefix + text[pr.positive]
        if mode == "none":
            rows["anchor"].append(anchor)
            rows["positive"].append(positive)
            continue
        negs: list[str] = []
        if mode in ("explicit", "both"):
            negs += [n for n in pr.explicit_negatives if n not in pr.exclude][:k]
            if not negs and mode == "explicit":
                pool = [d for d in project_units[pr.project] if d not in pr.exclude]
                negs = rng.sample(pool, min(1, len(pool)))
        if mode in ("mined", "both"):
            negs += [n for n in mined[i] if n not in negs][:k]
        for n in negs:
            rows["anchor"].append(anchor)
            rows["positive"].append(positive)
            rows["negative"].append(passage_prefix + text[n])
    return rows

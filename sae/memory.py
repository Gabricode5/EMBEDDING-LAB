"""Mémoire sémantique : relations de remplacement entre unités et ajustement des scores de recherche.

Trois opérations, appliquées après la recherche (sur la matrice de scores) :
- redirection : une unité remplacée transmet son score à l'unité qui la remplace (en suivant
  toute la chaîne de remplacements) ;
- statut : les unités remplacées ou annulées passent sous toutes les autres, sauf pour les
  questions de comparaison, qui ont besoin de l'ancienne et de la nouvelle version ;
- projet : seules les unités du projet visé par la question restent en tête.

Les statuts et relations viennent ici de la vérité terrain des scénarios. Dans le système complet,
ils viendront du classifieur de relations : ces résultats sont donc une borne haute.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sae.data import Corpus, EvalQuery, build_corpus, gid
from sae.schemas import Scenario

INACTIVE = frozenset({"superseded", "cancelled"})


@dataclass
class Memory:
    corpus: Corpus
    successor: dict[str, str] = field(default_factory=dict)   # ancienne unité -> unité qui la remplace
    inactive_override: set[str] | None = None                 # unités inactives prédites (sinon : statuts annotés)

    @classmethod
    def from_scenarios(cls, scenarios: list[Scenario], with_type: bool = False) -> "Memory":
        mem = cls(build_corpus(scenarios, with_type=with_type))
        for s in scenarios:
            for r in s.relations:
                if r.relation == "supersedes":
                    mem.successor[gid(s.project_id, r.to_id)] = gid(s.project_id, r.from_id)
        return mem

    def current(self, uid: str) -> str:
        """Unité en vigueur au bout de la chaîne de remplacements."""
        seen = {uid}
        while uid in self.successor:
            uid = self.successor[uid]
            if uid in seen:          # cycle : données incohérentes, on s'arrête
                break
            seen.add(uid)
        return uid

    def inactive_mask(self) -> np.ndarray:
        if self.inactive_override is not None:
            return np.array([d in self.inactive_override for d in self.corpus.ids])
        return np.array([self.corpus.units[d].status in INACTIVE for d in self.corpus.ids])

    def project_array(self) -> np.ndarray:
        return np.array([d.split("/", 1)[0] for d in self.corpus.ids])


def _demote(row: np.ndarray, mask: np.ndarray) -> None:
    """Fait passer les éléments masqués sous tous les autres, en gardant leur ordre relatif."""
    if mask.any() and (~mask).any():
        row[mask] -= (row.max() - row.min()) + 1.0


def apply_memory(scores: np.ndarray, queries: list[EvalQuery], memory: Memory,
                 status: bool = True, project: bool = False) -> np.ndarray:
    out = scores.astype(np.float64, copy=True)
    index = memory.corpus.index()
    inactive = memory.inactive_mask()
    projects = memory.project_array()
    chains = [(index[old], index[memory.current(old)]) for old in memory.successor
              if old in index and memory.current(old) in index]

    for qi, q in enumerate(queries):
        row = out[qi]
        if status and q.intent != "compare":
            for old, new in chains:                       # redirection avant déclassement
                row[new] = max(row[new], row[old])
            _demote(row, inactive)
        if project:
            _demote(row, projects != q.project)
    return out


def rrf(*score_matrices: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion : somme des 1 / (k + rang) de chaque système."""
    fused = np.zeros(score_matrices[0].shape, dtype=np.float64)
    for scores in score_matrices:
        order = np.argsort(-scores, axis=1, kind="stable")
        ranks = np.empty_like(order)
        rows = np.arange(scores.shape[0])[:, None]
        ranks[rows, order] = np.arange(1, scores.shape[1] + 1)
        fused += 1.0 / (k + ranks)
    return fused

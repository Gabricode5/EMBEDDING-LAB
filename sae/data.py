"""Chargement des scénarios et construction du corpus / des requêtes d'évaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sae.schemas import Scenario, Unit


def gid(project_id: str, unit_id: str) -> str:
    """Identifiant global, unique sur tout le corpus."""
    return f"{project_id}/{unit_id}"


def load_scenarios(split_dir: str | Path) -> list[Scenario]:
    paths = sorted(Path(split_dir).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"aucun scénario dans {split_dir}")
    scenarios = [Scenario.model_validate_json(p.read_text(encoding="utf-8")) for p in paths]
    ids = [s.project_id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError(f"project_id dupliqué dans {split_dir}")
    return scenarios


@dataclass
class Corpus:
    ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    units: dict[str, Unit] = field(default_factory=dict)

    def index(self) -> dict[str, int]:
        return {d: i for i, d in enumerate(self.ids)}


TYPE_LABELS_FR = {
    "decision": "décision", "action": "action", "commitment": "engagement",
    "question": "question", "answer": "réponse", "fact": "fait",
    "risk": "risque", "idea": "idée",
}


def unit_text(unit: Unit, with_type: bool = False) -> str:
    return f"{TYPE_LABELS_FR[unit.type]} : {unit.statement}" if with_type else unit.statement


def build_corpus(scenarios: list[Scenario], with_type: bool = False) -> Corpus:
    corpus = Corpus()
    for s in scenarios:
        for u in s.units:
            g = gid(s.project_id, u.unit_id)
            corpus.ids.append(g)
            corpus.texts.append(unit_text(u, with_type))
            corpus.units[g] = u
    return corpus


@dataclass(frozen=True)
class EvalQuery:
    qid: str
    text: str
    relevant: tuple[str, ...]
    hard_negatives: tuple[str, ...]
    answerable: bool
    origin: str
    tags: tuple[str, ...]
    intent: str = "answer"

    @property
    def project(self) -> str:
        return self.qid.split("/", 1)[0]


def build_queries(scenarios: list[Scenario]) -> list[EvalQuery]:
    out: list[EvalQuery] = []
    for s in scenarios:
        for q in s.queries:
            out.append(EvalQuery(
                qid=gid(s.project_id, q.query_id),
                text=q.text,
                relevant=tuple(gid(s.project_id, u) for u in q.relevant_unit_ids),
                hard_negatives=tuple(gid(s.project_id, u) for u in q.hard_negative_unit_ids),
                answerable=q.answerable,
                origin=q.origin,
                tags=tuple(q.tags),
                intent=q.intent,
            ))
    return out

"""Contrats de données du Semantic Agent Engine.

Corpus : Scenario -> Meeting -> Unit, Relation, Query (vérité terrain).
Agent  : IntentPrediction, EmailDraft, TaskDraft, EventDraft, Clarification, Dossier.
"""
from __future__ import annotations

from datetime import date as Date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

UnitType = Literal["decision", "action", "commitment", "question", "answer", "fact", "risk", "idea"]
UnitStatus = Literal["active", "proposed", "hypothetical", "superseded", "cancelled"]
RelationType = Literal["supports", "contradicts", "supersedes", "creates_action", "answers", "unrelated"]
IntentType = Literal["answer", "compare", "build_dossier", "draft_email", "create_task", "schedule_event", "clarify"]
QueryOrigin = Literal["generated", "manual", "assisted"]

QUERY_TAGS: tuple[str, ...] = (
    "current_decision", "proposal_vs_decision", "hypothesis_vs_decision", "conditional_agreement",
    "date_change", "budget_change", "owner_change", "person_cited_vs_owner", "negation",
    "compare", "commitment", "multi_meeting", "unanswerable", "paraphrase", "informal", "polite",
)


# --------------------------------------------------------------------------- corpus

class Person(BaseModel):
    name: str
    role: str | None = None
    email: str | None = None

    @model_validator(mode="after")
    def _fictive_email(self) -> "Person":
        if self.email is not None and "@example." not in self.email:
            raise ValueError(f"email non fictif pour {self.name}: utiliser un domaine example.*")
        return self


class Unit(BaseModel):
    unit_id: str
    source_id: str
    source_date: Date
    type: UnitType
    statement: str = Field(min_length=5)
    speaker: str | None = None
    owner: str | None = None
    due_date: Date | None = None
    evidence: str = Field(min_length=3)
    status: UnitStatus = "active"


class Relation(BaseModel):
    from_id: str
    to_id: str
    relation: RelationType
    evidence_source_id: str


class Meeting(BaseModel):
    meeting_id: str
    date: Date
    title: str
    participants: list[str]
    units: list[Unit]


class Query(BaseModel):
    query_id: str
    text: str = Field(min_length=3)
    intent: IntentType = "answer"
    relevant_unit_ids: list[str] = Field(default_factory=list)
    hard_negative_unit_ids: list[str] = Field(default_factory=list)
    answerable: bool = True
    origin: QueryOrigin = "generated"
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> "Query":
        if self.answerable and not self.relevant_unit_ids:
            raise ValueError(f"{self.query_id}: requête répondable sans unité pertinente")
        if not self.answerable and self.relevant_unit_ids:
            raise ValueError(f"{self.query_id}: requête non répondable avec des unités pertinentes")
        if set(self.relevant_unit_ids) & set(self.hard_negative_unit_ids):
            raise ValueError(f"{self.query_id}: une unité est à la fois pertinente et négatif difficile")
        unknown = set(self.tags) - set(QUERY_TAGS)
        if unknown:
            raise ValueError(f"{self.query_id}: tags inconnus {sorted(unknown)}")
        return self


class Scenario(BaseModel):
    project_id: str = Field(pattern=r"^[a-z0-9-]+$")
    project_name: str
    description: str
    people: list[Person]
    meetings: list[Meeting] = Field(min_length=1)
    relations: list[Relation] = Field(default_factory=list)
    queries: list[Query] = Field(default_factory=list)

    @property
    def units(self) -> list[Unit]:
        return [u for m in self.meetings for u in m.units]

    @model_validator(mode="after")
    def _integrity(self) -> "Scenario":
        errors: list[str] = []
        people = {p.name for p in self.people}
        meetings = {m.meeting_id: m for m in self.meetings}
        units: dict[str, Unit] = {}

        dates = [m.date for m in self.meetings]
        if dates != sorted(dates) or len(set(dates)) != len(dates):
            errors.append("réunions non triées par date strictement croissante")

        for m in self.meetings:
            for p in m.participants:
                if p not in people:
                    errors.append(f"{m.meeting_id}: participant inconnu {p!r}")
            for u in m.units:
                if u.unit_id in units:
                    errors.append(f"unit_id dupliqué {u.unit_id}")
                units[u.unit_id] = u
                if u.source_id != m.meeting_id or u.source_date != m.date:
                    errors.append(f"{u.unit_id}: source_id/source_date incohérents avec {m.meeting_id}")
                for who in (u.speaker, u.owner):
                    if who is not None and who not in people:
                        errors.append(f"{u.unit_id}: personne inconnue {who!r}")

        superseded_targets: set[str] = set()
        for r in self.relations:
            if r.from_id not in units or r.to_id not in units:
                errors.append(f"relation {r.from_id}->{r.to_id}: unité inconnue")
                continue
            if r.evidence_source_id not in meetings:
                errors.append(f"relation {r.from_id}->{r.to_id}: source inconnue {r.evidence_source_id}")
            if r.relation == "supersedes":
                new, old = units[r.from_id], units[r.to_id]
                if new.source_date < old.source_date:
                    errors.append(f"{r.from_id} supersedes {r.to_id} mais est plus ancien")
                if old.status not in ("superseded", "cancelled"):
                    errors.append(f"{r.to_id} est remplacé mais son statut est {old.status!r}")
                superseded_targets.add(r.to_id)

        for uid, u in units.items():
            if u.status == "superseded" and uid not in superseded_targets:
                errors.append(f"{uid}: statut superseded sans relation supersedes entrante")

        qids: set[str] = set()
        for q in self.queries:
            if q.query_id in qids:
                errors.append(f"query_id dupliqué {q.query_id}")
            qids.add(q.query_id)
            for uid in q.relevant_unit_ids + q.hard_negative_unit_ids:
                if uid not in units:
                    errors.append(f"{q.query_id}: unité inconnue {uid}")

        if errors:
            raise ValueError("Scénario invalide :\n- " + "\n- ".join(errors))
        return self


# --------------------------------------------------------------------------- agent

class Slots(BaseModel):
    person: str | None = None
    topic: str | None = None
    deadline: str | None = None
    action: str | None = None


class IntentPrediction(BaseModel):
    intent: IntentType
    slots: Slots = Field(default_factory=Slots)
    confidence: float = Field(ge=0.0, le=1.0)


class Recipient(BaseModel):
    name: str | None
    email: str | None = None  # jamais inventé : None si absent des sources autorisées
    evidence_unit_ids: list[str] = Field(default_factory=list)
    resolution_status: Literal["resolved", "email_missing", "ambiguous", "not_found"]


class _ProposedAction(BaseModel):
    source_unit_ids: list[str] = Field(min_length=1)
    needs_confirmation: Literal[True] = True
    executed: Literal[False] = False


class EmailDraft(_ProposedAction):
    intent: Literal["draft_email"] = "draft_email"
    tool_candidate: Literal["email.create_draft"] = "email.create_draft"
    recipient: Recipient
    subject: str
    body: str


class TaskDraft(_ProposedAction):
    intent: Literal["create_task"] = "create_task"
    tool_candidate: Literal["tasks.create"] = "tasks.create"
    title: str
    assignee: Recipient
    due_date: Date | None = None


class EventDraft(_ProposedAction):
    intent: Literal["schedule_event"] = "schedule_event"
    tool_candidate: Literal["calendar.propose_event"] = "calendar.propose_event"
    title: str
    when: Date | None = None
    participants: list[Recipient] = Field(default_factory=list)


class Clarification(BaseModel):
    intent: Literal["clarify"] = "clarify"
    question: str
    reason: str
    candidates: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    text: str
    unit_ids: list[str] = Field(min_length=1)  # toute affirmation est sourcée


class DossierSection(BaseModel):
    title: str
    claims: list[Claim] = Field(default_factory=list)


class Dossier(BaseModel):
    project_id: str
    title: str
    sections: list[DossierSection]
    missing_information: list[str] = Field(default_factory=list)

"""Compréhension de la demande : intention, garde-fou sur les hypothèses, paramètres (slots).

Approche par règles, explicables et testables. Chaque décision est accompagnée de sa raison.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from sae.schemas import Scenario


def norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn").replace("’", "'")


# L'ordre compte : la première règle qui correspond l'emporte.
INTENT_RULES: list[tuple[str, re.Pattern]] = [
    ("draft_email", re.compile(r"\b(e-?mail|mail|courriel|ecris(-moi)? (a|au|un)|redige (un|une) (message|mot)|envoie(r)? un message)\b")),
    ("create_task", re.compile(r"\b(tache|to-?do|a faire pour|assigne(r)?|cree(r)? une action)\b")),
    ("schedule_event", re.compile(r"\b(planifie|programme(r)? (un|une)|cale(r)? un|organise(r)? (un|une) (point|reunion|rdv)|prepare(r)? (un|une) (point|reunion)|rendez-vous|rdv)\b")),
    ("build_dossier", re.compile(r"\b(dossier|synthese|recap|recapitulatif|fais (le|un) point complet|bilan complet)\b")),
    ("compare", re.compile(r"(qu'est-ce qui a (change|bouge)|ce qui a (change|bouge)|a change|evolue|evolution|difference|contredi|par rapport a)")),
]
ACTION_INTENTS = {"draft_email", "create_task", "schedule_event"}

# Phrase hypothétique sans demande explicite : aucune action ne doit être créée.
HYPOTHESIS = re.compile(r"\b(on pourrait|il faudrait peut-etre|peut-etre|eventuellement|et si on|ce serait bien de|j'imagine)\b")
REQUEST = re.compile(r"(\?|\b(peux-tu|pourrais-tu|est-ce que|redige|ecris|prepare|fais|envoie|cree|planifie|donne|dis-moi|qui|quand|quel|quelle|combien|comment|pourquoi)\b)")

WEEKDAYS = "lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche"
MONTHS = "janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre"
DEADLINE = re.compile(rf"\b((avant|d'ici|pour|le) )?((ce |cette )?({WEEKDAYS})( prochain)?|\d{{1,2}}(er)? ({MONTHS})|demain|la semaine prochaine|fin de (la )?semaine|fin du mois)\b")


@dataclass
class Understanding:
    intent: str
    reason: str
    hypothetical: bool = False
    person: str | None = None
    project: str | None = None
    deadline: str | None = None
    candidates_projects: list[str] = field(default_factory=list)


def detect_intent(message: str) -> tuple[str, str]:
    m = norm(message)
    for intent, pattern in INTENT_RULES:
        hit = pattern.search(m)
        if hit:
            return intent, f"mot-clé « {hit.group(0)} »"
    return "answer", "aucune action demandée : question d'information"


def is_hypothetical(message: str) -> bool:
    m = norm(message)
    return bool(HYPOTHESIS.search(m)) and not REQUEST.search(m)


def find_person(message: str, scenarios: list[Scenario]) -> str | None:
    """Personne citée explicitement (nom complet ou prénom), si elle est unique."""
    m = norm(message)
    hits = set()
    for s in scenarios:
        for p in s.people:
            first = norm(p.name.split()[0])
            if norm(p.name) in m or re.search(rf"\b{re.escape(first)}\b", m):
                hits.add(p.name)
    return hits.pop() if len(hits) == 1 else None


def find_projects(message: str, scenarios: list[Scenario]) -> list[str]:
    m = norm(message)
    found = []
    for s in scenarios:
        names = {norm(s.project_name), norm(s.project_id.replace("-", " "))}
        words = [w for w in re.findall(r"\w+", norm(s.project_name)) if len(w) > 4]
        if any(n in m for n in names) or any(re.search(rf"\b{re.escape(w)}\b", m) for w in words):
            found.append(s.project_id)
    return found


def understand(message: str, scenarios: list[Scenario], default_project: str | None = None) -> Understanding:
    intent, reason = detect_intent(message)
    projects = find_projects(message, scenarios)
    project = projects[0] if len(projects) == 1 else None
    if project is None and default_project:
        project = default_project
    if project is None and len(scenarios) == 1:
        project = scenarios[0].project_id
    deadline = DEADLINE.search(norm(message))
    u = Understanding(intent=intent, reason=reason, person=find_person(message, scenarios), project=project,
                      deadline=deadline.group(0) if deadline else None, candidates_projects=projects)
    if is_hypothetical(message):
        u.hypothetical = True
        u.reason = "formulation hypothétique sans demande explicite : aucune action"
    return u

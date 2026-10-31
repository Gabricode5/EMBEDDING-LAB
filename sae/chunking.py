"""Représentations du corpus pour la recherche : unités sémantiques, tours de parole, découpage fixe.

Toutes partent des mêmes réunions. La transcription est *reconstituée* : on enchaîne, dans l'ordre, les
phrases dites à l'oral (`evidence`) des unités de la réunion, sous la forme « Locuteur : phrase ». Ce n'est
pas une vraie transcription : il n'y a ni digressions, ni bavardage, ni phrases sans unité. Le découpage
fixe et les tours y sont donc avantagés par rapport à une transcription réelle.

Chaque document garde les unités dont il contient l'evidence, pour être évalué avec les mêmes questions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sae.data import EvalQuery, gid, unit_text
from sae.schemas import Meeting, Scenario

# Découpage fixe retenu sur val (+ train pour bm25 et e5-base) par scripts/eval_chunking.py --select,
# figé avant toute mesure sur test et test2.
FIXED_SIZE = 60
FIXED_OVERLAP = 0.5
SELECTION_GRID = [(n, o) for n in (15, 30, 60, 120) for o in (0.0, 0.25, 0.5)]


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    unit_ids: frozenset[str]   # unités dont l'evidence figure dans le document

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def turn_line(speaker: str | None, evidence: str) -> str:
    return f"{speaker} : {evidence}" if speaker else evidence


def transcript_lines(project: str, meeting: Meeting) -> list[tuple[str, str]]:
    """(unité, « Locuteur : phrase ») dans l'ordre de la réunion."""
    return [(gid(project, u.unit_id), turn_line(u.speaker, u.evidence)) for u in meeting.units]


def transcript(project: str, meeting: Meeting) -> str:
    return "\n".join(line for _, line in transcript_lines(project, meeting))


def build_units(scenarios: list[Scenario], with_type: bool = False) -> list[Document]:
    """Fonctionnement actuel : un document par unité, texte = statement (type en préfixe en option)."""
    return [Document(gid(s.project_id, u.unit_id), unit_text(u, with_type), frozenset({gid(s.project_id, u.unit_id)}))
            for s in scenarios for u in s.units]


def build_turns(scenarios: list[Scenario]) -> list[Document]:
    """Un document par phrase de la transcription (locuteur + evidence)."""
    return [Document(f"{g}#tour", line, frozenset({g}))
            for s in scenarios for m in s.meetings for g, line in transcript_lines(s.project_id, m)]


def windows(n_words: int, size: int, overlap: float) -> list[tuple[int, int]]:
    """Fenêtres [début, fin) de `size` mots, avec chevauchement, jusqu'à couvrir tout le texte."""
    step = max(1, round(size * (1 - overlap)))
    out, start = [], 0
    while True:
        out.append((start, min(start + size, n_words)))
        if start + size >= n_words:
            return out
        start += step


def build_fixed(scenarios: list[Scenario], size: int, overlap: float) -> list[Document]:
    """Transcription de chaque réunion découpée en fenêtres de `size` mots, sans dépasser une réunion.

    Une unité est rattachée à une fenêtre si celle-ci contient au moins la moitié des mots de sa ligne :
    chaque unité appartient ainsi à au moins une fenêtre, sans compter les simples fragments.
    """
    docs = []
    for s in scenarios:
        for m in s.meetings:
            words, owner = [], []
            for g, line in transcript_lines(s.project_id, m):
                w = line.split()
                words += w
                owner += [g] * len(w)
            line_len = {g: owner.count(g) for g in set(owner)}
            for k, (a, b) in enumerate(windows(len(words), size, overlap)):
                inside = owner[a:b]
                units = frozenset(g for g in set(inside) if 2 * inside.count(g) >= line_len[g])
                docs.append(Document(f"{gid(s.project_id, m.meeting_id)}#fixe{k}", " ".join(words[a:b]), units))
    return docs


def build_documents(scenarios: list[Scenario], representation: str) -> list[Document]:
    """« unites », « unites-type », « tours » ou « fixe-<N>-<chevauchement en %> » (ex. fixe-60-25)."""
    if representation in ("unites", "unites-type"):
        return build_units(scenarios, with_type=representation == "unites-type")
    if representation == "tours":
        return build_turns(scenarios)
    m = re.fullmatch(r"fixe-(\d+)-(\d+)", representation)
    if m:
        return build_fixed(scenarios, int(m.group(1)), int(m.group(2)) / 100)
    raise ValueError(f"représentation inconnue : {representation}")


def fixed_name(size: int, overlap: float) -> str:
    return f"fixe-{size}-{round(overlap * 100)}"


def mixed_share(docs: list[Document], queries: list[EvalQuery]) -> float:
    """Part des documents qui contiennent, pour une même question, une unité pertinente et un négatif difficile.

    Un tel document (« fixée au 15 » et « reportée au 22 ») ne permet pas de savoir quelle version est en vigueur.
    """
    pairs = [(set(q.relevant), set(q.hard_negatives)) for q in queries if q.answerable and q.hard_negatives]
    mixed = sum(any(d.unit_ids & rel and d.unit_ids & hn for rel, hn in pairs) for d in docs)
    return mixed / len(docs) if docs else 0.0

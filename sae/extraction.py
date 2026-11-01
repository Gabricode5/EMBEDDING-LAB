"""Extraction automatique des unités de connaissance à partir de la transcription d'une réunion.

Entrée : la transcription (« Locuteur : phrase »), les participants, la date et le titre de la réunion.
Sortie : des unités au format de sae.schemas.Unit. Le statut « superseded » n'est pas extrait : il dépend des
réunions suivantes et vient ensuite du classifieur de relations (sae/relations/predict.py).

Le modèle (Ollama, JSON contraint par schéma) propose les unités ; le code les ancre ensuite dans la
transcription : l'evidence doit être une phrase de la transcription, les noms ceux des participants,
l'échéance une date valide. Ce qui ne peut pas être ancré est écarté plutôt qu'inventé.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from datetime import date as Date
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from sae.agent.understanding import norm
from sae.chunking import turn_line
from sae.schemas import Meeting, Person, Query, Unit, UnitType

ExtractedStatus = Literal["active", "proposed", "hypothetical", "cancelled"]
OWNER_TYPES = {"action", "commitment", "fact"}   # types qui peuvent désigner un responsable
DUE_TYPES = {"action", "commitment"}            # types qui peuvent avoir une échéance
GROUNDING_MIN = 0.6   # recouvrement minimal pour rattacher une evidence reformulée à une phrase de la transcription
ALIGN_MIN = 0.5       # recouvrement minimal des evidence pour aligner une unité extraite et une unité annotée


class ExtractedUnit(BaseModel):
    """Ce que le modèle doit produire pour chaque unité (schéma imposé à Ollama)."""
    type: UnitType
    statement: str = Field(min_length=5)
    evidence: str = Field(min_length=3)
    speaker: str | None = None
    owner: str | None = None
    due_date: str | None = None
    status: ExtractedStatus = "active"


class Extraction(BaseModel):
    units: list[ExtractedUnit]


SYSTEM = """Tu extrais des unités de connaissance à partir de transcriptions de réunions de projet.
Tu réponds UNIQUEMENT avec un objet JSON valide, sans texte autour."""

PROMPT = """Réunion « {title} » du {date} (date au format AAAA-MM-JJ).
Participants : {participants}.

Transcription (une ligne = une prise de parole, « Locuteur : phrase ») :
{transcript}

Extrais chaque information utile de la réunion sous forme d'unité. Pour chaque unité :
- type :
  decision = choix acté ou confirmé (« on confirme… », « c'est validé », « le budget a été augmenté à… », « on laisse tomber… ») ;
  action = tâche confiée à quelqu'un (« X, tu finalises… », « X, tu peux envoyer… ») ;
  commitment = la personne qui parle s'engage elle-même (« je livre… avant le… », « je m'en occupe ») ;
  question = question posée ; answer = réponse à une question posée juste avant (souvent « oui, … » ou « non, … ») ;
  fact = constat, objectif ou rappel (« l'objectif, c'est… », « pour rappel, c'est X qui pilote… ») ;
  risk = crainte ou problème possible (« … me fait peur, on peut déraper… ») ;
  idea = proposition non encore tranchée (« je propose… », « je mettrais… ») ou hypothèse (« on pourrait (peut-être)… »,
  y compris « si ça dérape, on pourrait peut-être… » : c'est une idea, pas un risk).
- statement : reformulation neutre et complète, à la 3e personne, qui nomme explicitement le sujet (objet, montant, date,
  personne) sans pronom ambigu.
- evidence : copie EXACTE de la phrase de la transcription, sans le nom du locuteur.
- speaker : nom exact du locuteur de cette phrase, tel qu'écrit dans la liste des participants.
- owner : seulement pour une action, un engagement, ou un fait qui désigne un responsable (« c'est X qui pilote… ») :
  nom exact de la personne désignée ou qui s'engage. null pour une décision, une idée, un risque, une question ou une
  réponse. Une personne simplement citée n'est pas responsable, et le locuteur n'est pas responsable de ce qu'il annonce.
- due_date : seulement pour une action ou un engagement, la date limite à laquelle la tâche doit être faite, au format
  AAAA-MM-JJ ; elle est postérieure à la date de la réunion (« le 20 » = le prochain 20 du mois). null sinon, et en
  particulier pour une date décidée (« on confirme le lancement au 15 novembre » n'a pas d'échéance).
- status : hypothetical pour toute hypothèse (« on pourrait (peut-être)… », « éventuellement ») ;
  cancelled seulement pour une unité dont la phrase annonce l'annulation d'une décision prise avant (une décision
  d'abandonner quelque chose, « on laisse tomber… », reste une decision active) ; proposed seulement si la phrase dit explicitement qu'une
  validation est attendue (« à faire valider », « à confirmer ») ; active dans tous les autres cas, y compris une
  proposition, une question, un risque ou un engagement (le type suffit à les distinguer).
En général, une ligne de la transcription donne une unité. N'invente aucune information absente de la transcription."""


# ---------------------------------------------------------------------- entrée
def meeting_turns(meeting: Meeting) -> list[tuple[str | None, str]]:
    """Transcription reconstituée : (locuteur, phrase) dans l'ordre de la réunion."""
    return [(u.speaker, u.evidence) for u in meeting.units]


def build_prompt(meeting: Meeting) -> str:
    transcript = "\n".join(turn_line(s, e) for s, e in meeting_turns(meeting))
    return PROMPT.format(title=meeting.title, date=meeting.date.isoformat(),
                         participants=", ".join(meeting.participants), transcript=transcript)


def prompt_hash(model: str) -> str:
    """Empreinte du prompt, du schéma et du modèle : le cache est invalidé si l'un d'eux change."""
    blob = SYSTEM + PROMPT + json.dumps(Extraction.model_json_schema(), sort_keys=True) + model
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------- appel au modèle
def ollama_chat(prompt: str, model: str, host: str, temperature: float = 0.0, seed: int = 0) -> str:
    """Appel à Ollama avec sortie JSON contrainte par le schéma Extraction (même principe que generate_scenarios)."""
    payload = {"model": model, "stream": False, "format": Extraction.model_json_schema(),
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
               "options": {"temperature": temperature, "seed": seed, "num_ctx": 8192}}
    if model.startswith("qwen3"):
        payload["think"] = False
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["message"]["content"]
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama injoignable sur {host} : lance `ollama serve` ou vérifie --ollama-host") from e


def parse(raw: str) -> Extraction:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return Extraction.model_validate_json(raw)


def extract_raw(meeting: Meeting, model: str, host: str, retries: int = 3) -> str:
    """Réponse brute validée ; nouvelle tentative (graine différente) si le JSON ou le schéma est invalide."""
    error: Exception | None = None
    for attempt in range(retries):
        raw = ollama_chat(build_prompt(meeting), model, host, seed=attempt)
        try:
            parse(raw)
            return raw
        except (ValidationError, ValueError) as e:
            error = e
    raise RuntimeError(f"{meeting.meeting_id} : extraction invalide après {retries} essais ({error})")


# ---------------------------------------------------------------------- ancrage dans la transcription
def words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", norm(text)) if len(w) > 1}


def overlap(a: str, b: str) -> float:
    """Coefficient de recouvrement : mots communs / taille du plus court des deux textes."""
    wa, wb = words(a), words(b)
    return len(wa & wb) / min(len(wa), len(wb)) if wa and wb else 0.0


def strip_speaker(evidence: str, names: list[str]) -> str:
    """Retire un préfixe « Locuteur : » recopié par le modèle (seulement si c'est bien un participant)."""
    head, sep, tail = evidence.partition(":")
    return tail.strip() if sep and match_person(head, names) else evidence.strip()


def ground_evidence(evidence: str, turns: list[tuple[str | None, str]],
                    names: list[str]) -> tuple[str | None, int | None]:
    """Phrase exacte de la transcription correspondant à l'evidence proposée, et son indice ; (None, None) sinon.

    Un extrait recopié tel quel est gardé ; une evidence reformulée est remplacée par la phrase la plus proche.
    """
    ev = strip_speaker(evidence, names)
    for i, (_, phrase) in enumerate(turns):
        if len(words(ev)) >= 3 and norm(ev) in norm(phrase):
            return ev if ev in phrase else phrase, i
    best = max(range(len(turns)), key=lambda i: overlap(ev, turns[i][1]), default=None)
    if best is not None and overlap(ev, turns[best][1]) >= GROUNDING_MIN:
        return turns[best][1], best
    return None, None


def match_person(name: str | None, people: list[str]) -> str | None:
    """Nom exact d'un participant : nom complet, ou prénom s'il est unique ; None sinon (jamais inventé)."""
    if not name:
        return None
    n = norm(name).strip()
    for p in people:
        if norm(p) == n:
            return p
    hits = [p for p in people if n in norm(p) or norm(p.split()[0]) == n.split()[0]]
    return hits[0] if len(hits) == 1 else None


def parse_date(value: str | None) -> Date | None:
    try:
        return Date.fromisoformat(value) if value else None
    except ValueError:
        return None


MONTHS = "janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre"
BARE_DAY = re.compile(rf"\ble (\d{{1,2}})(?:er)?\b(?!\s*(?:{MONTHS}))")


def next_day(day: int, after: Date) -> Date | None:
    """Prochaine date portant ce numéro de jour, à partir de la date de la réunion."""
    year, month = after.year, after.month
    for _ in range(13):
        try:
            d = Date(year, month, day)
            if d >= after:
                return d
        except ValueError:
            pass
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return None


def resolve_due(value: str | None, evidence: str, meeting_date: Date) -> Date | None:
    """Échéance proposée par le modèle ; un jour sans mois (« avant le 25 ») est ramené à sa prochaine occurrence."""
    due = parse_date(value)
    m = BARE_DAY.search(norm(evidence))
    if due is not None and m and int(m.group(1)) == due.day:
        return next_day(due.day, meeting_date)
    return due


def to_units(extraction: Extraction, meeting: Meeting, people: list[Person], start: int) -> tuple[list[Unit], dict]:
    """Unités ancrées dans la transcription, numérotées à partir de `start` ; statistiques d'ancrage."""
    turns = meeting_turns(meeting)
    names = [p.name for p in people]
    units, stats = [], {"proposed": len(extraction.units), "regrounded": 0, "dropped": 0}
    for x in extraction.units:
        evidence, line = ground_evidence(x.evidence, turns, names)
        if evidence is None:
            stats["dropped"] += 1
            continue
        stats["regrounded"] += evidence != x.evidence.strip()
        speaker = match_person(x.speaker, names) or turns[line][0]
        units.append(Unit(unit_id=f"x-{start + len(units):03d}", source_id=meeting.meeting_id,
                          source_date=meeting.date, type=x.type, statement=x.statement.strip(), speaker=speaker,
                          owner=match_person(x.owner, names) if x.type in OWNER_TYPES else None,
                          due_date=resolve_due(x.due_date, evidence, meeting.date) if x.type in DUE_TYPES else None,
                          evidence=evidence, status=x.status))
    return units, stats


# ---------------------------------------------------------------------- alignement avec les unités annotées
def jaccard(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def align_meeting(annotated: list[Unit], extracted: list[Unit]) -> dict[str, str]:
    """Appariement un pour un dans une réunion : unité annotée -> unité extraite.

    Score principal : recouvrement des evidence (la même phrase dite) ; départage : similarité des statements.
    Appariement glouton par score décroissant, sous le seuil ALIGN_MIN une unité reste seule.
    """
    pairs = [(overlap(a.evidence, e.evidence), jaccard(a.statement, e.statement), a.unit_id, e.unit_id)
             for a in annotated for e in extracted]
    pairs = sorted((p for p in pairs if p[0] >= ALIGN_MIN), reverse=True)
    out, used = {}, set()
    for _, _, a, e in pairs:
        if a not in out and e not in used:
            out[a] = e
            used.add(e)
    return out


def align(annotated: list[Unit], extracted: list[Unit]) -> dict[str, str]:
    """Alignement de tout un projet, réunion par réunion."""
    out = {}
    for meeting in {u.source_id for u in annotated}:
        out |= align_meeting([u for u in annotated if u.source_id == meeting],
                             [u for u in extracted if u.source_id == meeting])
    return out


def rewrite_queries(queries: list[Query], mapping: dict[str, str]) -> tuple[list[Query], list[str]]:
    """Questions réécrites vers les unités extraites ; renvoie aussi les questions perdues (aucune unité pertinente)."""
    kept, lost = [], []
    for q in queries:
        rel = sorted({mapping[u] for u in q.relevant_unit_ids if u in mapping})
        if q.answerable and not rel:
            lost.append(q.query_id)
            continue
        hn = sorted({mapping[u] for u in q.hard_negative_unit_ids if u in mapping} - set(rel))
        kept.append(q.model_copy(update={"relevant_unit_ids": rel, "hard_negative_unit_ids": hn}))
    return kept, lost

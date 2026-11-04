"""Rédaction des réponses (intentions answer et compare) par un modèle local, à partir des unités trouvées.

Le modèle ne décide de rien : l'agent garde ses règles (abstention, destinataires, brouillons, précisions,
hypothèses). Il reçoit la question et les K premières unités après la mémoire, et rédige 1 à 3 phrases en
citant les identifiants utilisés. Le code vérifie que tout identifiant cité fait partie des unités fournies ;
sinon la réponse est rejetée et l'agent revient à son affichage habituel (première unité trouvée).

Sortie JSON contrainte par le schéma WriterOutput (format d'Ollama), comme pour l'extraction.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

WRITER_MODELS = ("qwen3:8b", "qwen3.5:4b")
DEFAULT_WRITER_MODEL = "qwen3:8b"
DEFAULT_HOST = "http://localhost:11434"
# Avec qwen3.5:4b, Ollama ignore le schéma (format) quand la réflexion est désactivée, et la réflexion prend
# plusieurs minutes : le JSON est alors demandé dans le prompt, puis validé par le même schéma.
JSON_IN_PROMPT = {"qwen3.5:4b"}
JSON_INSTRUCTION = ('\n\nRéponds uniquement par un objet JSON de la forme {"answer": "...", "cited_ids": ["u-..."], '
                    '"found": true, "disagreement": false}, sans aucun autre texte.')
DEFAULT_K = 6
# mémoire occupée dans Ollama avec num_ctx 4096 (Mio, /api/ps, mesurée ici) : qwen3.5:4b, multimodal, n'en prend
# pas moins que qwen3:8b. Ollama déborde sur le CPU si le GPU manque de place (plus lent, mais fonctionne).
VRAM_MIB = {"qwen3:8b": 5700, "qwen3.5:4b": 6200}

SYSTEM = """Tu rédiges la réponse d'un assistant de suivi de projet. Tu ne connais du projet que les extraits de
réunions fournis (les « unités »). Tu réponds en français, en 1 à 3 phrases courtes, uniquement à partir de ces
unités, sans rien ajouter qui n'y figure pas."""

PROMPT = """Projet : {project}

Question : {question}

Unités trouvées dans les réunions (de la plus pertinente à la moins pertinente selon la recherche) :
{units}

Consignes :
- D'abord, vérifie si une unité répond exactement à la question posée, sur le même sujet précis. Une unité sur un
  sujet voisin ne suffit pas : le responsable du projet n'est pas le responsable d'une tâche précise (la
  formation, par exemple) ; le budget du projet n'est pas le budget d'un autre événement ou d'un autre projet.
- Si aucune unité ne répond exactement : found = false, cited_ids vide, et answer = « Je n'ai pas trouvé cette
  information dans les réunions. » Rien d'autre.
- Sinon (found = true) : réponds en 1 à 3 phrases, uniquement à partir des unités. N'ajoute aucun nom, chiffre,
  date ou fait qui n'y figure pas.
- La première unité n'est pas forcément la bonne. Une question posée en réunion n'est pas une réponse : cherche
  l'unité qui y répond.
- Privilégie les informations « en vigueur ». Si tu mentionnes une unité « remplacée », présente-la comme une
  ancienne version (« initialement », « avant d'être reporté »), jamais comme l'état actuel. Une unité
  « proposée » ou « hypothétique » n'est pas une décision : dis-le si tu l'utilises.
- Si des unités en vigueur se contredisent sur le point demandé (deux chiffres, deux dates, deux responsables),
  dis-le explicitement et mets disagreement à true.
- Pour une question « qu'est-ce qui a changé », décris l'évolution dans l'ordre des dates.
- cited_ids : les identifiants exacts (par exemple u-012) des seules unités sur lesquelles repose ta réponse.
  N'écris pas les identifiants dans answer."""


class WriterOutput(BaseModel):
    answer: str
    cited_ids: list[str] = Field(default_factory=list)
    found: bool
    disagreement: bool


@dataclass
class WriterUnit:
    uid: str            # identifiant complet (projet/u-xxx) ; le modèle voit la partie après « / »
    date: str
    meeting: str
    speaker: str | None
    type: str
    status: str         # en vigueur, remplacée, proposée, hypothétique, annulée
    statement: str
    evidence: str

    @property
    def short(self) -> str:
        return self.uid.split("/", 1)[-1]


@dataclass
class WriterResult:
    ok: bool                         # réponse acceptée par la vérification
    status: str                      # accepted, rejected, error
    message: str = ""                # raison du rejet ou de l'erreur
    output: WriterOutput | None = None
    cited: list[str] = field(default_factory=list)   # identifiants complets des unités citées
    seconds: float = 0.0
    raw: str = ""


TYPE_FR = {"decision": "décision", "commitment": "engagement", "action": "action", "fact": "fait",
           "risk": "risque", "idea": "idée", "question": "question", "answer": "réponse"}


def format_units(units: list[WriterUnit]) -> str:
    lines = []
    for x in units:
        who = x.speaker or "locuteur inconnu"
        lines.append(f"[{x.short}] {x.date} · réunion « {x.meeting} » · {who} · {TYPE_FR.get(x.type, x.type)} · "
                     f"{x.status}\n  résumé : {x.statement}\n  phrase exacte : « {x.evidence} »")
    return "\n".join(lines)


def build_prompt(question: str, units: list[WriterUnit], project: str = "") -> str:
    return PROMPT.format(project=project or "non précisé", question=question, units=format_units(units))


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn").replace("’", "'")


def claimed(answer: str) -> set[str]:
    """Nombres et noms propres (majuscule hors début de phrase, ou après une élision : « d'Atlas ») de la réponse."""
    answer = re.sub(r"\[?\bu-\d+\]?", " ", answer)          # identifiants recopiés dans le texte : ignorés
    out = {re.sub(r"\D", "", n) for n in re.findall(r"\d[\d\u202f\u00a0 ]*\d|\d", answer)}
    for sent in re.split(r"(?<=[.!?:;])\s+", answer):
        words = re.findall(r"[\w'’-]+", sent)
        out |= {_norm(re.split(r"['’]", w)[-1]) for w in words[1:]
                if w[:1].isupper() or re.match(r"\w+['’][A-ZÀ-Ý]", w)}
    return {x for x in out if x}


def ungrounded(answer: str, units: list[WriterUnit], project: str = "") -> list[str]:
    """Nombres et noms propres de la réponse absents des unités citées (texte, locuteur, réunion) et du nom du projet."""
    support = _norm(project + " " + " ".join(f"{x.statement} {x.evidence} {x.speaker or ''} {x.meeting}" for x in units))
    digits = re.sub(r"(?<=\d)[\u202f\u00a0 ](?=\d)", "", support)
    return sorted(c for c in claimed(answer)
                  if not (re.search(rf"\b{re.escape(c)}\b", support) or (c.isdigit() and c in digits)))


def check(output: WriterOutput, units: list[WriterUnit], project: str = "") -> tuple[bool, str, list[str]]:
    """Vérification par le code : identifiants cités parmi les unités fournies (au moins un si found), et noms
    propres et nombres de la réponse présents dans les unités citées."""
    by_short = {x.short: x.uid for x in units} | {x.uid: x.uid for x in units}
    cited = [re.sub(r"^\[|\]$", "", c.strip()) for c in output.cited_ids]
    unknown = [c for c in cited if c not in by_short]
    if unknown:
        return False, f"identifiants cités hors des unités fournies : {', '.join(unknown)}", []
    if output.found and not cited:
        return False, "réponse sans identifiant cité", []
    if not output.answer.strip():
        return False, "réponse vide", []
    ids = list(dict.fromkeys(by_short[c] for c in cited))
    if output.found:
        missing = ungrounded(output.answer, [x for x in units if x.uid in ids], project)
        if missing:
            return False, f"noms ou nombres absents des unités citées : {', '.join(missing)}", []
    return True, "", ids


class Writer:
    def __init__(self, model: str = DEFAULT_WRITER_MODEL, host: str = DEFAULT_HOST, k: int = DEFAULT_K,
                 timeout: float = 120, num_ctx: int = 4096) -> None:
        self.model, self.host, self.k, self.timeout, self.num_ctx = model, host.rstrip("/"), k, timeout, num_ctx

    def unavailable(self) -> str | None:
        """None si Ollama répond et connaît le modèle, sinon un message clair."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as r:
                names = {m["name"] for m in json.load(r)["models"]}
        except (urllib.error.URLError, OSError):
            return f"Ollama n'est pas lancé sur {self.host} (lance `ollama serve`)"
        if self.model not in names:
            return f"modèle {self.model} absent d'Ollama (`ollama pull {self.model}`)"
        return None

    def _chat(self, prompt: str) -> str:
        payload = {"model": self.model, "stream": False,
                   "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                   "options": {"temperature": 0.0, "seed": 0, "num_ctx": self.num_ctx}, "keep_alive": "15m"}
        if self.model in JSON_IN_PROMPT:
            payload["messages"][1]["content"] += JSON_INSTRUCTION
        else:
            payload["format"] = WriterOutput.model_json_schema()
        if self.model.startswith("qwen3"):
            payload["think"] = False
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["message"]["content"]

    def write(self, question: str, units: list[WriterUnit], project: str = "") -> WriterResult:
        t0 = time.perf_counter()
        try:
            raw = self._chat(build_prompt(question, units, project))
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as e:
            return WriterResult(False, "error", f"appel à Ollama impossible ({e})", seconds=time.perf_counter() - t0)
        dt = time.perf_counter() - t0
        try:
            text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
            out = WriterOutput.model_validate_json(text)
        except ValidationError:
            return WriterResult(False, "rejected", "sortie hors du schéma JSON", seconds=dt, raw=raw)
        ok, why, cited = check(out, units, project)
        return WriterResult(ok, "accepted" if ok else "rejected", why, out, cited, dt, raw)

    def unload(self) -> None:
        """Libère la mémoire GPU occupée par le modèle dans Ollama."""
        try:
            req = urllib.request.Request(f"{self.host}/api/generate", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"model": self.model, "keep_alive": 0}).encode("utf-8"))
            urllib.request.urlopen(req, timeout=30).read()
        except (urllib.error.URLError, OSError):
            pass

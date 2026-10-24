"""Génère des scénarios de projets fictifs validés par schéma.

La vérité terrain (unités, statuts, relations, requêtes) est produite en même temps
que le scénario : aucune annotation a posteriori n'est nécessaire.

Fournisseurs :
    ollama     modèle local, gratuit (JSON contraint par le schéma)
    manual     gratuit via Claude.ai : le script écrit le prompt, tu colles la réponse
    anthropic  API payante (ANTHROPIC_API_KEY)
    openai     API payante (OPENAI_API_KEY)

Exemples :
    python scripts/generate_scenarios.py --n 1 --out data/scenarios/train --provider ollama --model qwen3:8b
    python scripts/generate_scenarios.py --n 3 --out data/scenarios/test --provider ollama --model mistral-nemo \
        --theme-offset 9 --style "ton très oral, phrases courtes, hésitations"
    python scripts/generate_scenarios.py --n 1 --out data/scenarios/train --provider manual
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from pydantic import ValidationError

from sae.data import load_scenarios
from sae.schemas import QUERY_TAGS, Scenario

THEMES = [
    "migration du CRM d'une agence immobilière",
    "ouverture d'une deuxième boutique à Lyon",
    "refonte de l'application RH interne",
    "lancement d'une offre d'abonnement pour une salle de sport",
    "déménagement des bureaux d'une agence de communication",
    "mise en conformité RGPD d'une start-up de santé",
    "salon professionnel pour un fabricant de mobilier",
    "nouvelle gamme de produits bio d'une PME alimentaire",
    "refonte du site e-commerce d'une marque de vêtements",
    "déploiement d'un logiciel de planning dans une clinique",
    "organisation d'un séminaire annuel d'entreprise",
    "partenariat logistique d'une entreprise de livraison",
    "application mobile pour un réseau de bibliothèques",
    "recrutement d'une équipe support client",
]

SYSTEM = """Tu génères des jeux de données d'évaluation pour un moteur de recherche sémantique sur des réunions d'équipe.
Tu réponds UNIQUEMENT avec un objet JSON valide, sans texte autour ni balises Markdown."""

RULES = """Contraintes :
- Projet fictif, en français, 4 à 6 réunions à dates strictement croissantes entre septembre et novembre 2026.
- 4 à 6 personnes fictives aux noms variés ; emails en domaine example.org ou null (au moins une personne sans email).
- 15 à 25 unités au total. Chaque unité : statement (reformulation neutre, à la 3e personne) ET evidence (phrase orale exacte prononcée en réunion, style naturel, différent du statement).
- source_id et source_date de chaque unité = meeting_id et date de sa réunion. speaker, owner et participants = noms exacts de la liste people.
- Cas difficiles OBLIGATOIRES : une proposition plus tard validée avec un autre montant ou une autre date ; un accord conditionnel ("si…, alors on valide") ; un changement de date ou de responsable ; une hypothèse ("on pourrait peut-être…") sans suite ; une personne citée qui n'est PAS la responsable ; une information devenue obsolète.
- Statuts : "proposed" pour une proposition non tranchée, "hypothetical" pour une simple hypothèse, "superseded" pour toute unité remplacée. Toute unité "superseded" DOIT être la cible (to_id) d'une relation "supersedes" dont la source (from_id) est plus récente.
- Relations : au moins 6, couvrant supersedes, answers, supports, creates_action et au moins un contradicts ou unrelated. Sens des relations : supersedes va de la nouvelle unité vers l'ancienne ; answers va de la réponse vers la question ; creates_action va de la cause (décision, risque, engagement) vers l'action qu'elle déclenche.
- 10 à 14 requêtes utilisateur variées (dont 1 à 2 avec answerable=false et relevant_unit_ids=[]). Pour chaque requête répondable : relevant_unit_ids, et si possible hard_negative_unit_ids = unités du même sujet mais fausses par le statut, la personne, la date ou la négation.
- Tags autorisés uniquement : {tags}.
- intent parmi : answer, compare, build_dossier, draft_email, create_task, schedule_event, clarify.
- Ne réutilise aucun de ces noms de personnes ni de projets : {forbidden}.
{style}{schema}
Exemple complet d'un scénario valide (ne pas copier le contenu) :
{example}

Thème du nouveau projet : {theme}"""


def _extract_json(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("aucun objet JSON dans la réponse")
    return text[start:end + 1]


class LLM:
    def __init__(self, args: argparse.Namespace) -> None:
        self.provider = args.provider
        self.model = args.model
        self.args = args
        self._manual_dir = Path(args.manual_dir)
        self._rounds: dict[str, int] = {}

    def __call__(self, messages: list[dict], tag: str) -> str:
        return getattr(self, f"_{self.provider}")(messages, tag)

    def _anthropic(self, messages: list[dict], tag: str) -> str:
        import anthropic
        resp = anthropic.Anthropic().messages.create(
            model=self.model, max_tokens=16000, system=SYSTEM, messages=messages)
        return "".join(b.text for b in resp.content if b.type == "text")

    def _openai(self, messages: list[dict], tag: str) -> str:
        from openai import OpenAI
        resp = OpenAI().chat.completions.create(
            model=self.model, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM}, *messages])
        return resp.choices[0].message.content or ""

    def _ollama(self, messages: list[dict], tag: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, *messages],
            "stream": False,
            "format": "json" if self.args.no_schema else Scenario.model_json_schema(),
            "options": {"num_ctx": self.args.num_ctx, "temperature": 0.7},
        }
        if self.model.startswith("qwen3"):
            payload["think"] = False
        req = urllib.request.Request(
            f"{self.args.ollama_host}/api/chat", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=7200) as r:
                return json.load(r)["message"]["content"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama a répondu {e.code} : {e.read().decode()[:500]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError("Ollama injoignable : lance `ollama serve` ou vérifie --ollama-host") from e

    def _manual(self, messages: list[dict], tag: str) -> str:
        self._manual_dir.mkdir(parents=True, exist_ok=True)
        round_ = self._rounds[tag] = self._rounds.get(tag, 0) + 1
        prompt_path = self._manual_dir / f"{tag}-prompt-{round_}.txt"
        resp_path = self._manual_dir / f"{tag}-response-{round_}.json"
        prompt_path.write_text(f"{SYSTEM}\n\n{messages[-1]['content']}", encoding="utf-8")
        print(f"  1. Copie le contenu de {prompt_path} et colle-le dans une NOUVELLE conversation Claude.")
        print(f"  2. Colle la réponse complète dans {resp_path}")
        while True:
            input("  3. Appuie sur Entrée quand c'est fait… ")
            if resp_path.exists() and resp_path.stat().st_size > 0:
                return resp_path.read_text(encoding="utf-8")
            print(f"  {resp_path} est vide ou absent.")


def normalize(data: dict) -> dict:
    """Répare les oublis de forme dont la bonne valeur se déduit sans ambiguïté."""
    for m in data.get("meetings") or []:
        for u in m.get("units") or []:
            u["source_id"] = m.get("meeting_id")
            u["source_date"] = m.get("date")

    units = {u["unit_id"]: u for m in data.get("meetings") or [] for u in m.get("units") or []
             if isinstance(u, dict) and "unit_id" in u}
    for r in data.get("relations") or []:
        if r.get("relation") == "supersedes" and r.get("to_id") in units:
            target = units[r["to_id"]]
            if target.get("status") not in ("superseded", "cancelled"):
                target["status"] = "superseded"

    for q in data.get("queries") or []:
        q["origin"] = "generated"
        q["tags"] = [t for t in q.get("tags") or [] if t in QUERY_TAGS]
        relevant = q.get("relevant_unit_ids") or []
        if not relevant or "unanswerable" in q["tags"]:
            q["answerable"] = False
            q["relevant_unit_ids"] = []
        else:
            q["answerable"] = True
            q["relevant_unit_ids"] = relevant
        q["hard_negative_unit_ids"] = [u for u in q.get("hard_negative_unit_ids") or []
                                       if u not in q["relevant_unit_ids"]]
    return data


def generate_one(llm: LLM, prompt: str, tag: str, max_attempts: int) -> Scenario:
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            content = prompt
        else:
            # Prompt d'origine + erreurs précédentes, sans l'ancien JSON : le contexte reste constant.
            content = (f"{prompt}\n\nUne tentative précédente était invalide pour ces raisons :\n"
                       f"{last_error}\nÉvite ces erreurs. Réponds avec le scénario COMPLET en JSON.")
        raw = llm([{"role": "user", "content": content}], tag)
        try:
            data = normalize(json.loads(_extract_json(raw)))
            return Scenario.model_validate(data)
        except (ValueError, ValidationError, TypeError, AttributeError) as e:
            last_error = str(e)[:2000]
            print(f"  tentative {attempt} invalide :")
            for line in last_error.splitlines()[:8]:
                print(f"    {line}")
    raise RuntimeError(f"échec après {max_attempts} tentatives :\n{last_error}")


def known_names(root: Path) -> tuple[set[str], set[str]]:
    people, projects = set(), set()
    for split in ("train", "val", "test"):
        d = root / split
        if any(d.glob("*.json")):
            for s in load_scenarios(d):
                projects |= {s.project_id, s.project_name}
                people |= {p.name for p in s.people}
    return people, projects


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", choices=("ollama", "manual", "anthropic", "openai"), default="ollama")
    ap.add_argument("--model", default=os.environ.get("SAE_LLM_MODEL", "qwen3:8b"))
    ap.add_argument("--example", default="data/scenarios/train/atlas.json")
    ap.add_argument("--theme-offset", type=int, default=0, help="décale la liste de thèmes entre splits")
    ap.add_argument("--style", default="", help="consigne de style supplémentaire (utile pour le test)")
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--num-ctx", type=int, default=24576, help="fenêtre de contexte Ollama")
    ap.add_argument("--no-schema", action="store_true", help="Ollama : JSON libre au lieu du schéma imposé")
    ap.add_argument("--manual-dir", default="data/manual_io")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    example = Path(args.example).read_text(encoding="utf-8")
    # Avec Ollama, le schéma est imposé par le décodage : inutile de l'envoyer dans le prompt.
    schema_block = "" if args.provider == "ollama" and not args.no_schema else \
        "\nSchéma JSON cible :\n" + json.dumps(Scenario.model_json_schema(), ensure_ascii=False) + "\n"
    llm = LLM(args)
    label = "manuel (Claude.ai)" if args.provider == "manual" else f"{args.provider}:{args.model}"

    for i in range(args.n):
        theme = THEMES[(args.theme_offset + i) % len(THEMES)]
        people, projects = known_names(out.parent)
        prompt = RULES.format(
            tags=", ".join(QUERY_TAGS), forbidden=", ".join(sorted(people | projects)) or "aucun",
            style=f"- Style des répliques : {args.style}\n" if args.style else "",
            schema=schema_block, example=example, theme=theme,
        )
        print(f"[{i + 1}/{args.n}] {theme} ({label})")
        tag = f"{out.name}-{args.theme_offset + i:02d}"
        scenario = generate_one(llm, prompt, tag, args.max_attempts)
        path = out / f"{scenario.project_id}.json"
        if path.exists():
            path = out / f"{scenario.project_id}-{i}.json"
        path.write_text(scenario.model_dump_json(indent=2), encoding="utf-8")
        print(f"  -> {path} ({len(scenario.units)} unités, {len(scenario.queries)} requêtes)")


if __name__ == "__main__":
    main()

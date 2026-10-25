"""Agent en ligne de commande : démonstration des 7 cas du sujet, mode conversation, évaluation des intentions.

    python scripts/agent.py --project clinique-tilleuls --demo
    python scripts/agent.py --project clinique-tilleuls --chat
    python scripts/agent.py --eval-intents
    python scripts/agent.py --relations gold --demo     # mémoire annotée (borne haute)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from sae.agent.engine import Agent
from sae.agent.understanding import detect_intent, is_hypothetical
from sae.data import load_scenarios
from sae.memory import Memory

DEMO = {
    "clinique-tilleuls": [
        ("1. Question sur une décision", "Quelle date a finalement été retenue pour le démarrage au bloc ?"),
        ("2. Proposition ou décision", "Le budget de 45 000 euros a-t-il été validé ?"),
        ("3. Comparaison", "Qu'est-ce qui a changé pour la maintenance de l'ancien logiciel ?"),
        ("4. Dossier multi-sources", "Prépare un dossier complet sur le projet."),
        ("5. Mail à la bonne personne", "Rédige un mail à la personne responsable de la formation au bloc "
                                        "pour lui demander où en est la formation avant vendredi."),
        ("6. Demande trop vague", "Envoie un mail à la bonne personne."),
        ("7. Phrase hypothétique", "On pourrait peut-être revoir le budget."),
    ],
    "atlas": [
        ("1. Question sur une décision", "Quelle date de lancement a finalement été retenue ?"),
        ("2. Proposition ou décision", "Le budget de 120 000 euros a-t-il été validé ?"),
        ("3. Comparaison", "Qu'est-ce qui a changé pour la date de lancement ?"),
        ("4. Dossier multi-sources", "Prépare un dossier complet sur le projet Atlas."),
        ("5. Mail à la bonne personne", "Rédige un mail à la personne responsable du budget pour lui demander "
                                        "la dernière version validée avant vendredi."),
        ("6. Demande trop vague", "Envoie un mail à la bonne personne."),
        ("7. Phrase hypothétique", "On pourrait peut-être revoir le budget."),
    ],
}


def make_retriever(name: str):
    if name == "bm25":
        from sae.retrieval.bm25 import BM25
        return BM25()
    from sae.retrieval.dense import DenseRetriever, resolve_spec
    return DenseRetriever(resolve_spec(name))


def build_agent(args) -> Agent:
    scenarios = load_scenarios(args.split)
    memory = Memory.from_scenarios(scenarios, with_type=not args.no_type)
    if args.relations == "predicted":
        from sae.relations.predict import CrossEncoderRelations
        memory = CrossEncoderRelations(args.relations_model, args.threshold).build_memory(scenarios, memory)
    elif args.relations == "none":
        memory.successor, memory.inactive_override = {}, set()
    print(f"Mémoire : {len(memory.corpus.ids)} unités, {len(memory.successor)} remplacements ({args.relations})")
    return Agent(scenarios, memory, make_retriever(args.retriever), with_type=not args.no_type)


def show(label: str, message: str, agent: Agent, project: str | None, as_json: bool) -> None:
    r = agent.handle(message, project=project)
    print(f"\n{'=' * 80}\n{label}\n> {message}\n[intention : {r.understanding.intent} · {r.understanding.reason}"
          f" · résultat : {r.kind}]\n")
    print(r.text)
    if as_json and r.payload:
        print("\n" + json.dumps(r.payload, ensure_ascii=False, indent=2))


def eval_intents(split: str) -> None:
    scenarios = load_scenarios(split)
    gold, pred = [], []
    for s in scenarios:
        for q in s.queries:
            gold.append(q.intent)
            pred.append(detect_intent(q.text)[0])
    ok = sum(g == p for g, p in zip(gold, pred))
    print(f"Intentions sur {split} : {ok}/{len(gold)} correctes ({ok / len(gold):.1%})")
    print("Répartition attendue :", dict(Counter(gold)))
    for g, p, q in zip(gold, pred, [q for s in scenarios for q in s.queries]):
        if g != p:
            print(f"  attendu {g:<14} prédit {p:<14} « {q.text} »")
    tests = ["On pourrait peut-être revoir le budget.", "Il faudrait peut-être décaler la réunion",
             "Peut-être qu'on devrait changer de prestataire ?", "Est-ce qu'on pourrait avoir le budget validé ?"]
    print("\nGarde-fou hypothèses :")
    for t in tests:
        print(f"  {'aucune action' if is_hypothetical(t) else 'demande traitée':<16} « {t} »")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/scenarios/test")
    ap.add_argument("--project", default=None, help="projet par défaut (contexte de la conversation)")
    ap.add_argument("--retriever", default="models/e5b-both-type-s1")
    ap.add_argument("--no-type", action="store_true")
    ap.add_argument("--relations", choices=("predicted", "gold", "none"), default="predicted")
    ap.add_argument("--relations-model", default="models/relations-cross")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--eval-intents", action="store_true")
    ap.add_argument("--json", action="store_true", help="affiche aussi la sortie structurée")
    args = ap.parse_args()

    if args.eval_intents:
        eval_intents(args.split)
        return
    agent = build_agent(args)
    if args.demo:
        for label, msg in DEMO.get(args.project, DEMO["clinique-tilleuls"]):
            show(label, msg, agent, args.project, args.json)
    if args.chat:
        print("\nPose tes questions (ligne vide pour quitter).")
        while (msg := input("\n> ").strip()):
            show("", msg, agent, args.project, args.json)


if __name__ == "__main__":
    main()

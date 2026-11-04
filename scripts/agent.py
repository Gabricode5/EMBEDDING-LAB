"""Agent en ligne de commande : démonstration des 7 cas du sujet, mode conversation, évaluation des intentions.

    python scripts/agent.py --project clinique-tilleuls --demo
    python scripts/agent.py --project clinique-tilleuls --chat
    python scripts/agent.py --eval-intents
    python scripts/agent.py --relations gold --demo     # mémoire annotée (borne haute)
    python scripts/agent.py --abstention score --chat   # abstention si l'information est absente
    python scripts/agent.py --writer qwen3:8b --chat    # réponses rédigées par un modèle local (Ollama)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from sae.agent.engine import ABSTENTION_STRATEGIES, DEFAULT_ABSTENTION, Agent
from sae.agent.writer import DEFAULT_HOST, DEFAULT_K, VRAM_MIB, WRITER_MODELS, Writer
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


def make_writer(args) -> Writer | None:
    """Rédacteur local, ou None (option none, ou Ollama injoignable : l'agent garde l'affichage en règles)."""
    name = getattr(args, "writer", "none")
    if name == "none":
        return None
    writer = Writer(name, getattr(args, "ollama_host", DEFAULT_HOST), getattr(args, "writer_k", DEFAULT_K))
    problem = writer.unavailable()
    if problem:
        print(f"Rédaction désactivée : {problem}. Réponses affichées sans rédaction (première unité trouvée).")
        return None
    import torch
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / 2**20
        need = VRAM_MIB.get(name, 0)
        if free < need:
            other = [m for m in WRITER_MODELS if m != name and VRAM_MIB.get(m, need) < need]
            print(f"Attention : {free:.0f} Mio libres sur le GPU, {name} en demande environ {need} ; Ollama en "
                  "mettra une partie sur le CPU (plus lent). Ferme les autres programmes qui utilisent le GPU"
                  + (f", ou essaie --writer {other[0]}." if other else " (qwen3.5:4b n'en demande pas moins)."))
    return writer


def build_agent(args) -> Agent:
    scenarios = load_scenarios(args.split)
    memory = Memory.from_scenarios(scenarios, with_type=not args.no_type)
    if args.relations == "predicted":
        from sae.relations.predict import CrossEncoderRelations
        memory = CrossEncoderRelations(args.relations_model, args.threshold).build_memory(scenarios, memory)
        free_gpu()   # le classifieur de relations ne sert plus : sa mémoire GPU revient au modèle de rédaction
    elif args.relations == "none":
        memory.successor, memory.inactive_override = {}, set()
    print(f"Mémoire : {len(memory.corpus.ids)} unités, {len(memory.successor)} remplacements ({args.relations})")
    retriever = make_retriever(args.retriever)
    return Agent(scenarios, memory, retriever, with_type=not args.no_type,
                 abstention=getattr(args, "abstention", DEFAULT_ABSTENTION), writer=make_writer(args))


def free_gpu() -> None:
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def add_writer_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--writer", choices=("none", *WRITER_MODELS), default="none",
                    help="rédaction des réponses answer et compare par un modèle local (none : affichage en règles)")
    ap.add_argument("--writer-k", type=int, default=DEFAULT_K, help="nombre d'unités données au rédacteur (5 à 8)")
    ap.add_argument("--ollama-host", default=DEFAULT_HOST)


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
    ap.add_argument("--abstention", choices=ABSTENTION_STRATEGIES, default=DEFAULT_ABSTENTION,
                    help="stratégie d'abstention pour les questions sans réponse (none : toujours répondre)")
    add_writer_args(ap)
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

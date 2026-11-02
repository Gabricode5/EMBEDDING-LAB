"""Mesure l'apport du reranking (cross-encodeur BAAI/bge-reranker-v2-m3, non fine-tuné) avant la mémoire.

    python scripts/eval_rerank.py --split data/scenarios/val --ks 10 20 50 --texts type statement   # choix de K
    python scripts/eval_rerank.py --split data/scenarios/test --ks 20 --texts type --relations gold predicted

Chaîne : recherche dense (models/e5b-both-type-s1, type dans le texte) -> reranking des K premiers candidats
-> mémoire (statut, projet), exactement comme scripts/eval_memory.py. Quatre variantes : dense, dense + rerank,
dense + statut + projet, dense + rerank + statut + projet. Le texte donné au reranker est celui du corpus
(« type : statement », comme pour le bi-encodeur) ou le statement seul.

K et le texte se choisissent sur val uniquement ; test et test2 ne sont mesurés qu'une fois, avec la valeur figée.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from sae.data import build_queries, load_scenarios
from sae.eval.metrics import aggregate, grouped, per_query
from sae.memory import Memory, apply_memory
from sae.retrieval.rerank import RERANKER, CrossEncoderReranker, place_on_top

COLS = ("recall@1", "recall@5", "mrr@10", "ndcg@10", "hn_acc")
FOCUS = ("origin=generated", "origin=assisted", "tag=current_decision", "tag=compare")


def memory_for(scenarios, relations: str, args) -> Memory:
    memory = Memory.from_scenarios(scenarios, with_type=True)
    if relations == "predicted":
        from sae.relations.predict import CrossEncoderRelations
        memory = CrossEncoderRelations(args.relations_model, args.threshold).build_memory(scenarios, memory)
    return memory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/scenarios/val")
    ap.add_argument("--model", default="models/e5b-both-type-s1")
    ap.add_argument("--reranker", default=RERANKER)
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50], help="nombre de candidats reclassés")
    ap.add_argument("--texts", nargs="+", choices=("type", "statement"), default=["type"],
                    help="texte de l'unité donné au reranker")
    ap.add_argument("--relations", nargs="+", choices=("gold", "predicted"), default=["gold"])
    ap.add_argument("--relations-model", default="models/relations-cross")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--out-dir", default="experiments")
    args = ap.parse_args()

    import numpy as np
    from sae.retrieval.dense import DenseRetriever, resolve_spec

    scenarios = load_scenarios(args.split)
    base = Memory.from_scenarios(scenarios, with_type=True)
    queries = build_queries(scenarios)
    texts = [q.text for q in queries]
    corpus = base.corpus
    index = corpus.index()
    print(f"{args.split} : {len(scenarios)} projets · {len(corpus.ids)} unités · {len(queries)} questions")

    dense = DenseRetriever(resolve_spec(args.model)).fit(corpus.texts).score(texts)
    reranker = CrossEncoderReranker(args.reranker)
    kmax = min(max(args.ks), len(corpus.ids))
    top = np.argsort(-dense, axis=1, kind="stable")[:, :kmax]
    docs = {"type": corpus.texts, "statement": [corpus.units[g].statement for g in corpus.ids]}
    # scores du reranker calculés une fois pour les kmax premiers ; les K plus petits en sont des préfixes
    ce = {}
    for t in args.texts:
        pairs = [(qi, int(d)) for qi in range(len(queries)) for d in top[qi]]
        ce[t] = reranker.pair_scores([texts[qi] for qi, _ in pairs],
                                     [docs[t][d] for _, d in pairs]).reshape(len(queries), kmax)

    reranked = {f"rerank K={k} · {t}": place_on_top(dense, top[:, :k], ce[t][:, :k])
                for t in args.texts for k in args.ks}
    results: dict = {}
    for relations in args.relations:
        memory = memory_for(scenarios, relations, args)
        mem = lambda s: apply_memory(s, queries, memory, status=True, project=True)  # noqa: E731
        variants = {"dense": dense, "dense + statut + projet": mem(dense)}
        for name, s in reranked.items():
            variants[f"dense + {name}"] = s
            variants[f"dense + {name} + statut + projet"] = mem(s)
        results[relations] = {}
        for name, scores in variants.items():
            rows = per_query(scores, queries, index)
            results[relations][name] = {"overall": aggregate(rows), "groups": grouped(rows)}

        print(f"\nRelations {'annotées' if relations == 'gold' else 'prédites'} "
              f"({len(memory.successor)} remplacements)\n")
        print("| variante | " + " | ".join(COLS) + " |\n|" + "---|" * (len(COLS) + 1))
        for name, r in results[relations].items():
            print(f"| {name} | " + " | ".join(f"{r['overall'][c]:.3f}" for c in COLS) + " |")
        print("\nMRR@10 par sous-groupe :\n\n| variante | " + " | ".join(FOCUS) + " |\n|" + "---|" * (len(FOCUS) + 1))
        for name, r in results[relations].items():
            vals = [r["groups"].get(g, {}).get("mrr@10") for g in FOCUS]
            print(f"| {name} | " + " | ".join("-" if v is None else f"{v:.3f}" for v in vals) + " |")

    out = Path(args.out_dir) / f"rerank-{Path(args.split).name}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"config": vars(args), "results": results}, ensure_ascii=False,
                                                 indent=2), encoding="utf-8")
    print(f"\nRésultats : {out}/results.json")


if __name__ == "__main__":
    main()

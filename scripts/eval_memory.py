"""Mesure l'apport de la mémoire (statut, projet) et de la recherche hybride.

    python scripts/eval_memory.py --split data/scenarios/test --model models/e5b-both-type-s1 --with-type

Les statuts et relations utilisés sont ceux de la vérité terrain (borne haute) ; le filtre
par projet suppose que le projet visé est connu (contexte de la conversation).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from sae.data import build_queries, load_scenarios
from sae.eval.metrics import aggregate, grouped, per_query
from sae.memory import Memory, apply_memory, rrf
from sae.retrieval.bm25 import BM25

COLS = ("recall@1", "recall@5", "mrr@10", "ndcg@10", "hn_acc")
FOCUS = ("origin=generated", "origin=assisted", "tag=current_decision", "tag=compare")


def make_retriever(name: str):
    if name == "bm25":
        return BM25()
    from sae.retrieval.dense import DenseRetriever, resolve_spec
    return DenseRetriever(resolve_spec(name))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/scenarios/test")
    ap.add_argument("--model", default="models/e5b-both-type-s1")
    ap.add_argument("--with-type", action="store_true")
    ap.add_argument("--out-dir", default="experiments")
    args = ap.parse_args()

    scenarios = load_scenarios(args.split)
    memory = Memory.from_scenarios(scenarios, with_type=args.with_type)
    queries = build_queries(scenarios)
    texts = [q.text for q in queries]
    corpus = memory.corpus
    print(f"{len(scenarios)} projets · {len(corpus.ids)} unités · {len(queries)} questions · "
          f"{len(memory.successor)} remplacements")

    dense = make_retriever(args.model).fit(corpus.texts).score(texts)
    lexical = BM25().fit(corpus.texts).score(texts)
    hybrid = rrf(dense, lexical)

    variants = {
        "dense": dense,
        "dense + statut": apply_memory(dense, queries, memory, status=True),
        "dense + projet": apply_memory(dense, queries, memory, status=False, project=True),
        "dense + statut + projet": apply_memory(dense, queries, memory, status=True, project=True),
        "hybride (RRF)": hybrid,
        "hybride + statut + projet": apply_memory(hybrid, queries, memory, status=True, project=True),
    }

    results = {}
    for name, scores in variants.items():
        rows = per_query(scores, queries, corpus.index())
        results[name] = {"overall": aggregate(rows), "groups": grouped(rows)}

    print(f"\nModèle dense : {args.model}{' (avec type)' if args.with_type else ''}\n")
    print("| variante | " + " | ".join(COLS) + " |")
    print("|" + "---|" * (len(COLS) + 1))
    for name, r in results.items():
        print(f"| {name} | " + " | ".join(f"{r['overall'][c]:.3f}" for c in COLS) + " |")

    print("\nMRR@10 par sous-groupe :\n")
    print("| variante | " + " | ".join(FOCUS) + " |")
    print("|" + "---|" * (len(FOCUS) + 1))
    for name, r in results.items():
        vals = [r["groups"].get(g, {}).get("mrr@10") for g in FOCUS]
        print(f"| {name} | " + " | ".join("-" if v is None else f"{v:.3f}" for v in vals) + " |")

    out = Path(args.out_dir) / f"memory-{Path(args.split).name}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({
        "split": args.split, "model": args.model, "with_type": args.with_type, "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats : {out}/results.json")


if __name__ == "__main__":
    main()

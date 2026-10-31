"""Compare trois représentations du corpus pour la recherche : unités sémantiques, tours de parole, découpage fixe.

Deux temps, à ne pas mélanger :
    python scripts/eval_chunking.py --select                 # val + train : choix de N et du chevauchement
    python scripts/eval_chunking.py --splits test test2      # mesure finale, découpage figé dans sae/chunking.py

Mêmes questions et mêmes règles de pertinence pour toutes les représentations (voir per_query_docs).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from sae.chunking import FIXED_OVERLAP, FIXED_SIZE, SELECTION_GRID, build_documents, fixed_name, mixed_share
from sae.data import build_queries, load_scenarios
from sae.eval.metrics import aggregate, per_query_docs

BASE_REPRESENTATIONS = ["unites", "unites-type", "tours"]
SELECTION_SPLITS = ("train", "val")
DEFAULT_MODELS = "bm25,e5-base,models/e5b-both-type-s1"


def trained_on_train(model: str) -> bool:
    """Les modèles locaux ont été fine-tunés sur train (questions et paires evidence -> statement)."""
    return model.startswith("models/")


def evaluate(scores: np.ndarray, queries, docs) -> dict:
    rows = per_query_docs(scores, queries, [d.unit_ids for d in docs])
    out = aggregate(rows)
    hn_total = sum(r["hn_total"] for r in rows)
    with_hn = [r for r in rows if r["top1_mixed"] is not None]
    out["hn_non_separes"] = sum(r["hn_mixed"] for r in rows) / hn_total if hn_total else float("nan")
    out["top1_melange"] = sum(r["top1_mixed"] for r in with_hn) / len(with_hn) if with_hn else float("nan")
    return out


def stats(docs, queries) -> dict:
    return {"n_docs": len(docs), "mots_moyens": float(np.mean([d.n_words for d in docs])),
            "docs_melanges": mixed_share(docs, queries)}


def select(results: dict, doc_stats: dict, models: list[str]) -> tuple[str, list[dict]]:
    """Découpage fixe au meilleur MRR@10 moyen (puis nDCG@10, puis moins de documents mélangés).

    Moyenne sur les cases (modèle, split) non contaminées : le modèle fine-tuné n'est compté que sur val.
    """
    ranking = []
    for n, o in SELECTION_GRID:
        rep = fixed_name(n, o)
        cells = [results[sp][rep][m] for sp in results for m in models if not (sp == "train" and trained_on_train(m))]
        mixed = float(np.mean([doc_stats[sp][rep]["docs_melanges"] for sp in results]))
        ranking.append({"representation": rep, "mrr@10": float(np.mean([c["mrr@10"] for c in cells])),
                        "ndcg@10": float(np.mean([c["ndcg@10"] for c in cells])), "docs_melanges": mixed,
                        "n_cases": len(cells)})
    ranking.sort(key=lambda r: (-r["mrr@10"], -r["ndcg@10"], r["docs_melanges"]))
    return ranking[0]["representation"], ranking


def print_tables(results: dict, doc_stats: dict, n_queries: dict) -> None:
    cols = ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10", "hn_acc")
    for split, by_rep in results.items():
        print(f"\n### {split} · {n_queries[split]} questions avec réponse\n")
        print("| représentation | modèle | " + " | ".join(cols) + " | hn non séparés | top-1 mélangé | "
              "docs | mots/doc | docs mélangés |")
        print("|" + "---|" * (len(cols) + 7))
        for rep, by_model in by_rep.items():
            st = doc_stats[split][rep]
            for model, m in by_model.items():
                vals = " | ".join(f"{m[c]:.3f}" for c in cols)
                print(f"| {rep} | {model} | {vals} | {m['hn_non_separes']:.3f} | {m['top1_melange']:.3f} | "
                      f"{st['n_docs']} | {st['mots_moyens']:.1f} | {st['docs_melanges']:.3f} |")


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from run_baselines import make_retriever  # mêmes modèles que les baselines

    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=list(SELECTION_SPLITS))
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--select", action="store_true", help="choisit N et le chevauchement du découpage fixe")
    args = ap.parse_args()

    if args.select and set(args.splits) - set(SELECTION_SPLITS):
        sys.exit("Le choix du découpage fixe se fait sur train et val uniquement.")
    if not args.select and not FIXED_SIZE:
        sys.exit("Aucun découpage fixe figé dans sae/chunking.py : lancer d'abord --select sur val + train.")
    fixed = [fixed_name(n, o) for n, o in SELECTION_GRID] if args.select else [fixed_name(FIXED_SIZE, FIXED_OVERLAP)]
    representations = BASE_REPRESENTATIONS + fixed
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    data = {}
    for sp in args.splits:
        scenarios = load_scenarios(f"data/scenarios/{sp}")
        queries = build_queries(scenarios)
        data[sp] = (queries, {rep: build_documents(scenarios, rep) for rep in representations})
    doc_stats = {sp: {rep: stats(docs, q) for rep, docs in reps.items()} for sp, (q, reps) in data.items()}
    n_queries = {sp: sum(x.answerable for x in q) for sp, (q, _) in data.items()}

    results: dict[str, dict[str, dict[str, dict]]] = {sp: {rep: {} for rep in representations} for sp in data}
    for model in models:                         # un seul chargement par modèle
        retriever = make_retriever(model)
        for sp, (queries, reps) in data.items():
            texts = [q.text for q in queries]
            for rep, docs in reps.items():
                scores = retriever.fit([d.text for d in docs]).score(texts)
                results[sp][rep][model] = evaluate(scores, queries, docs)
        del retriever

    print_tables(results, doc_stats, n_queries)
    out = {"mode": "selection" if args.select else "final", "splits": args.splits, "models": models,
           "representations": representations, "n_queries": n_queries, "doc_stats": doc_stats, "results": results}
    if args.select:
        chosen, ranking = select(results, doc_stats, models)
        out["chosen"], out["selection"] = chosen, ranking
        print("\n| découpage fixe | MRR@10 moyen | nDCG@10 moyen | docs mélangés | cases |\n|---|---|---|---|---|")
        for r in ranking:
            print(f"| {r['representation']} | {r['mrr@10']:.3f} | {r['ndcg@10']:.3f} | {r['docs_melanges']:.3f} | "
                  f"{r['n_cases']} |")
        print(f"\nDécoupage retenu : {chosen} (à figer dans FIXED_SIZE / FIXED_OVERLAP de sae/chunking.py)")
    small = [sp for sp, n in n_queries.items() if n < 30]
    if small:
        print(f"Attention : effectifs faibles pour {', '.join(small)} ({', '.join(str(n_queries[s]) for s in small)} "
              "questions) ; un écart de quelques centièmes n'y est pas significatif.")
    if any(trained_on_train(m) for m in models):
        print("Rappel : l'encodeur fine-tuné a été entraîné sur les unités de train et sur des paires "
              "evidence -> statement ; ses scores sur train ne sont pas une mesure.")

    path = Path("experiments") / f"chunking-{'+'.join(args.splits)}-{datetime.now():%Y%m%d-%H%M%S}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDétails : {path}/results.json")


if __name__ == "__main__":
    main()

"""Évalue BM25 et des encodeurs denses sur un split.

Exemples :
    python scripts/run_baselines.py --split data/scenarios/test
    python scripts/run_baselines.py --split data/scenarios/test --models bm25,e5-base,bge-m3 --with-type
    python scripts/run_baselines.py --split data/scenarios/test --models bm25,models/e5-base-hn
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from sae.data import build_corpus, build_queries, load_scenarios
from sae.eval.metrics import aggregate, grouped, per_query
from sae.retrieval.bm25 import BM25

MAIN_COLS = ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10", "hn_acc")


def make_retriever(name: str):
    if name == "bm25":
        return BM25()
    from sae.retrieval.dense import DenseRetriever, resolve_spec
    return DenseRetriever(resolve_spec(name))


def hardware() -> dict:
    info = {"host": platform.node(), "python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["gpu"] = None
    return info


def error_cases(scores: np.ndarray, queries, corpus, rows, top: int = 3, threshold: int = 3) -> list[dict]:
    """Requêtes dont la première unité pertinente est au-delà du rang `threshold`."""
    by_qid = {r["qid"]: r for r in rows}
    out = []
    for qi, q in enumerate(queries):
        r = by_qid.get(q.qid)
        if r is None or r["best_rank"] <= threshold:
            continue
        top_idx = np.argsort(-scores[qi], kind="stable")[:top]
        out.append({
            "qid": q.qid, "query": q.text, "tags": list(q.tags), "best_rank": r["best_rank"],
            "expected": [{"id": d, "type": corpus.units[d].type, "status": corpus.units[d].status,
                          "text": corpus.units[d].statement} for d in q.relevant],
            "retrieved": [{"id": corpus.ids[i], "type": corpus.units[corpus.ids[i]].type,
                           "status": corpus.units[corpus.ids[i]].status, "text": corpus.texts[i],
                           "score": round(float(scores[qi, i]), 4)} for i in top_idx],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="dossier de scénarios JSON")
    ap.add_argument("--models", default="bm25,e5-base,bge-m3")
    ap.add_argument("--with-type", action="store_true", help="préfixe le type d'unité au texte indexé")
    ap.add_argument("--out-dir", default="experiments")
    args = ap.parse_args()

    scenarios = load_scenarios(args.split)
    corpus = build_corpus(scenarios, with_type=args.with_type)
    queries = build_queries(scenarios)
    doc_index = corpus.index()
    texts = [q.text for q in queries]
    print(f"{len(scenarios)} projets · {len(corpus.ids)} unités · {len(queries)} requêtes "
          f"({sum(not q.answerable for q in queries)} sans réponse)")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    variant = "typed" if args.with_type else "plain"
    out_dir = Path(args.out_dir) / f"baselines-{Path(args.split).name}-{variant}-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        t0 = time.perf_counter()
        retriever = make_retriever(name).fit(corpus.texts)
        t_index = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = retriever.score(texts)
        latency_ms = 1000 * (time.perf_counter() - t0) / max(len(texts), 1)

        rows = per_query(scores, queries, doc_index)
        results[name] = {"overall": aggregate(rows), "groups": grouped(rows),
                         "index_seconds": round(t_index, 2), "query_latency_ms": round(latency_ms, 2)}
        safe = name.replace("/", "_")
        (out_dir / f"errors-{safe}.json").write_text(
            json.dumps(error_cases(scores, queries, corpus, rows), ensure_ascii=False, indent=2), encoding="utf-8")
        del retriever

    header = "| modèle | " + " | ".join(MAIN_COLS) + " | latence (ms/req) |"
    lines = [header, "|" + "---|" * (len(MAIN_COLS) + 2)]
    for name, r in results.items():
        vals = " | ".join(f"{r['overall'].get(c, float('nan')):.3f}" for c in MAIN_COLS)
        lines.append(f"| {name} | {vals} | {r['query_latency_ms']:.1f} |")
    table = "\n".join(lines)
    print("\n" + table)

    (out_dir / "results.json").write_text(json.dumps({
        "run_id": run_id, "split": args.split, "with_type": args.with_type,
        "hardware": hardware(), "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "results.md").write_text(table + "\n", encoding="utf-8")
    print(f"\nRésultats et erreurs : {out_dir}/")


if __name__ == "__main__":
    main()

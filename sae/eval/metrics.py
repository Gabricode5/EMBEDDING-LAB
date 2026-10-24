"""Métriques de retrieval : Recall@K, MRR@10, nDCG@10, hard-negative accuracy."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from sae.data import EvalQuery

KS = (1, 5, 10)


def _ranks(row: np.ndarray) -> np.ndarray:
    """Rang (1 = meilleur) de chaque document pour une requête."""
    order = np.argsort(-row, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(row) + 1)
    return ranks


def per_query(scores: np.ndarray, queries: list[EvalQuery], doc_index: dict[str, int]) -> list[dict]:
    rows: list[dict] = []
    for qi, q in enumerate(queries):
        if not q.answerable:
            continue
        row = scores[qi]
        ranks = _ranks(row)
        rel = np.array([doc_index[d] for d in q.relevant])
        rel_ranks = ranks[rel]
        best = int(rel_ranks.min())

        metrics = {f"recall@{k}": float((rel_ranks <= k).mean()) for k in KS}
        metrics["mrr@10"] = 1.0 / best if best <= 10 else 0.0
        dcg = sum(1.0 / math.log2(r + 1) for r in rel_ranks if r <= 10)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), 10) + 1))
        metrics["ndcg@10"] = dcg / idcg

        hn_total = hn_ok = 0
        if q.hard_negatives:
            best_rel_score = float(row[rel].max())
            for d in q.hard_negatives:
                hn_total += 1
                hn_ok += int(best_rel_score > float(row[doc_index[d]]))
        rows.append({"qid": q.qid, "origin": q.origin, "tags": list(q.tags), "best_rank": best,
                     "hn_total": hn_total, "hn_ok": hn_ok, **metrics})
    return rows


def aggregate(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k.startswith(("recall@", "mrr@", "ndcg@"))]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    hn_total = sum(r["hn_total"] for r in rows)
    out["hn_acc"] = sum(r["hn_ok"] for r in rows) / hn_total if hn_total else float("nan")
    out["n_queries"] = len(rows)
    return out


def grouped(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Agrégats par origine (generated/manual) et par tag de difficulté."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[f"origin={r['origin']}"].append(r)
        for t in r["tags"]:
            groups[f"tag={t}"].append(r)
    return {g: aggregate(rs) for g, rs in sorted(groups.items())}

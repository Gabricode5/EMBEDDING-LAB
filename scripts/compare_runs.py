"""Regroupe les résultats de toutes les évaluations d'un split dans un seul tableau.

    python scripts/compare_runs.py --split test
    python scripts/compare_runs.py --split test --by-origin
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10", "hn_acc")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--by-origin", action="store_true", help="une ligne par origine des questions")
    args = ap.parse_args()

    latest: dict[tuple[str, bool, str], dict] = {}
    for f in sorted(Path("experiments").glob(f"baselines-{args.split}-*/results.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for model, r in data["results"].items():
            groups = {"toutes": r["overall"]}
            if args.by_origin:
                groups = {g.split("=", 1)[1]: v for g, v in r["groups"].items() if g.startswith("origin=")}
                groups["toutes"] = r["overall"]
            for origin, m in groups.items():
                latest[(model, data["with_type"], origin)] = m   # le run le plus récent l'emporte

    if not latest:
        print(f"Aucun résultat pour le split {args.split}")
        return
    head = "| modèle | type | questions | n | " + " | ".join(COLS) + " |"
    print(head)
    print("|" + "---|" * (len(COLS) + 4))
    for (model, typed, origin), m in sorted(latest.items(), key=lambda x: (x[0][2], -x[1].get("mrr@10", 0))):
        vals = " | ".join(f"{m.get(c, float('nan')):.3f}" for c in COLS)
        print(f"| {Path(model).name} | {'oui' if typed else 'non'} | {origin} | {m.get('n_queries', 0)} | {vals} |")


if __name__ == "__main__":
    main()

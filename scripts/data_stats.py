"""Statistiques des données, calculées à partir des fichiers (tableaux Markdown de docs/DATA.md).

    python scripts/data_stats.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sae.data import load_scenarios

SPLITS = ("train", "val", "test", "test2")
TYPES = ("decision", "commitment", "action", "fact", "risk", "idea", "question", "answer")
STATUSES = ("active", "proposed", "hypothetical", "superseded", "cancelled")


def row(name: str, values: list) -> str:
    return f"| {name} | " + " | ".join(str(v) for v in values) + " |"


def table(title: str, keys: tuple | list, counts: dict[str, Counter]) -> None:
    print(f"\n{title}\n\n| | " + " | ".join(SPLITS) + " | total |\n|" + "---|" * (len(SPLITS) + 2))
    for k in keys:
        vals = [counts[s][k] for s in SPLITS]
        print(row(k, vals + [sum(vals)]))


def main() -> None:
    argparse.ArgumentParser(description="Statistiques des données (tableaux de docs/DATA.md).").parse_args()
    data = {s: load_scenarios(f"data/scenarios/{s}") for s in SPLITS}
    print("| | " + " | ".join(SPLITS) + " | total |\n|" + "---|" * (len(SPLITS) + 2))
    lines = {
        "projets": lambda sc: len(sc),
        "réunions": lambda sc: sum(len(s.meetings) for s in sc),
        "personnes": lambda sc: sum(len(s.people) for s in sc),
        "unités": lambda sc: sum(len(s.units) for s in sc),
        "relations": lambda sc: sum(len(s.relations) for s in sc),
        "questions": lambda sc: sum(len(s.queries) for s in sc),
        "questions sans réponse": lambda sc: sum(not q.answerable for s in sc for q in s.queries),
    }
    for name, f in lines.items():
        vals = [f(data[s]) for s in SPLITS]
        print(row(name, vals + [sum(vals)]))
    print("\nProjets : " + " ; ".join(f"{s} : {', '.join(x.project_id for x in data[s])}" for s in SPLITS))

    table("Unités par type", TYPES, {s: Counter(u.type for x in data[s] for u in x.units) for s in SPLITS})
    table("Unités par statut", STATUSES, {s: Counter(u.status for x in data[s] for u in x.units) for s in SPLITS})
    rel = {s: Counter(r.relation for x in data[s] for r in x.relations) for s in SPLITS}
    table("Relations par type", sorted({k for c in rel.values() for k in c}), rel)
    orig = {s: Counter(q.origin for x in data[s] for q in x.queries) for s in SPLITS}
    table("Questions par origine", sorted({k for c in orig.values() for k in c}), orig)
    intent = {s: Counter(q.intent for x in data[s] for q in x.queries) for s in SPLITS}
    table("Questions par intention", sorted({k for c in intent.values() for k in c}), intent)
    tags = {s: Counter(t for x in data[s] for q in x.queries for t in q.tags) for s in SPLITS}
    table("Questions par tag (une question peut en avoir plusieurs)", sorted({k for c in tags.values() for k in c}),
          tags)

    print("\nDemandes d'évaluation de l'agent (data/agent_eval)\n\n| fichier | lignes | projets | type attendu |\n"
          "|---|---|---|---|")
    for f in sorted(Path("data/agent_eval").glob("*.jsonl")):
        cases = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        projects = sorted({c["project"] for c in cases if c.get("project")})
        if "file" in cases[0]:   # slots.jsonl : demandes reprises d'autres fichiers
            projects = [f"repris de {', '.join(sorted({c['file'] for c in cases}))}"]
        kinds = Counter(c.get("expected_kind", "slots") for c in cases)
        print(f"| {f.name} | {len(cases)} | {', '.join(projects)} | "
              + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) + " |")

    print("\nUnités extraites (data/extracted)\n\n| split | projets | unités extraites | unités annotées alignées "
          "sur une unité extraite |\n|---|---|---|---|")
    for s in SPLITS:
        folder = Path("data/extracted") / s
        if not folder.exists():
            continue
        scen = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]
        n_units = sum(len(m["units"]) for x in scen for m in x["meetings"])
        aligned = sum(v is not None for p in sorted((folder / "alignment").glob("*.json"))
                      for v in json.loads(p.read_text(encoding="utf-8"))["units"].values())
        annotated = sum(len(x.units) for x in data[s])
        print(f"| {s} | {len(scen)} | {n_units} | {aligned} / {annotated} |")


if __name__ == "__main__":
    main()

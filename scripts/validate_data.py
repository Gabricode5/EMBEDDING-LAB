"""Valide tous les scénarios et contrôle les fuites entre splits.

    python scripts/validate_data.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from sae.data import load_scenarios
from sae.schemas import QUERY_TAGS

ROOT = Path("data/scenarios")
SPLITS = ("train", "val", "test", "test2")
REQUIRED_TAGS = ("proposal_vs_decision", "hypothesis_vs_decision", "date_change",
                 "person_cited_vs_owner", "unanswerable", "compare")


def main() -> int:
    ok = True
    loaded = {}
    for split in SPLITS:
        d = ROOT / split
        if not any(d.glob("*.json")):
            print(f"[{split}] vide")
            continue
        try:
            loaded[split] = load_scenarios(d)
        except Exception as e:  # noqa: BLE001
            print(f"[{split}] ERREUR\n{e}")
            ok = False
            continue
        sc = loaded[split]
        units = [u for s in sc for u in s.units]
        queries = [q for s in sc for q in s.queries]
        tags = Counter(t for q in queries for t in q.tags)
        print(f"\n[{split}] {len(sc)} projets · {sum(len(s.meetings) for s in sc)} réunions · "
              f"{len(units)} unités · {len(queries)} requêtes")
        print("  types   :", dict(Counter(u.type for u in units)))
        print("  statuts :", dict(Counter(u.status for u in units)))
        print("  origine :", dict(Counter(q.origin for q in queries)))
        print("  tags    :", {t: tags.get(t, 0) for t in QUERY_TAGS})
        missing = [t for t in REQUIRED_TAGS if tags.get(t, 0) == 0]
        if missing:
            print(f"  ATTENTION cas difficiles absents : {missing}")

    def projects(s):
        return {x.project_id for x in loaded.get(s, [])}

    def people(s):
        return {p.name for x in loaded.get(s, []) for p in x.people}

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        leak = projects(a) & projects(b)
        if leak:
            print(f"FUITE projets {a}/{b} : {sorted(leak)}")
            ok = False
    shared = people("train") & people("test")
    if shared:
        print(f"ATTENTION personnes présentes en train et en test : {sorted(shared)}")

    print("\nOK" if ok else "\nÉCHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

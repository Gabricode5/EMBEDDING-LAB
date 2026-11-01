"""Exactitude des paramètres (slots) extraits des demandes par sae/agent/understanding.py.

    python scripts/eval_slots.py --file data/agent_eval/slots.jsonl

Fichier d'entrée (JSON Lines), vérité terrain validée à la main :
{"id": "c-07", "file": "test.jsonl", "message": "...", "person": "Baptiste N'Diaye", "project": null,
 "deadline": "avant vendredi"}

    personne    nom complet de la personne nommée dans le message (un rôle ou un pronom : null)
    projet      projet nommé dans le message (null sinon : ni le sujet ni la personne ne suffisent)
    échéance    texte tel qu'écrit, préposition comprise ; comparé en minuscules et sans accents

`understand` reçoit tous les projets du split de la demande et aucun projet par défaut : le projet prédit
vient donc du seul message. Exactitude sur toutes les demandes, puis sur celles où le paramètre est renseigné.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from sae.agent.understanding import norm, understand
from sae.data import load_scenarios

SLOTS = ("person", "project", "deadline")
LABELS = {"person": "personne", "project": "projet", "deadline": "échéance"}
# fichier de demandes -> split dont il interroge les projets
SPLITS = {"test.jsonl": "test", "holdout.jsonl": "test", "test2.jsonl": "test2"}


def same(slot: str, gold: str | None, pred: str | None) -> bool:
    if slot == "deadline" and gold is not None and pred is not None:
        return norm(gold).strip() == norm(pred).strip()
    return gold == pred


def predict(cases: list[dict]) -> list[dict]:
    scenarios = {sp: load_scenarios(f"data/scenarios/{sp}") for sp in set(SPLITS.values())}
    rows = []
    for c in cases:
        u = understand(c["message"], scenarios[SPLITS[c["file"]]], default_project=None)
        pred = {"person": u.person, "project": u.project, "deadline": u.deadline}
        rows.append({**c, "pred": pred, "ok": {s: same(s, c[s], pred[s]) for s in SLOTS}})
    return rows


def accuracy(rows: list[dict], slot: str) -> dict:
    filled = [r for r in rows if r[slot] is not None]
    rate = lambda xs: sum(r["ok"][slot] for r in xs) / len(xs) if xs else float("nan")  # noqa: E731
    return {"exactitude": rate(rows), "n": len(rows), "si_renseigne": rate(filled), "n_renseigne": len(filled),
            "faux_positifs": sum(r[slot] is None and r["pred"][slot] is not None for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/agent_eval/slots.jsonl")
    args = ap.parse_args()

    cases = [json.loads(l) for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = predict(cases)
    results = {s: accuracy(rows, s) for s in SLOTS}

    print(f"\n{len(rows)} demandes\n")
    print("| paramètre | exactitude | si renseigné | prédit à tort (attendu null) |\n|---|---|---|---|")
    for s, m in results.items():
        print(f"| {LABELS[s]} | {m['exactitude']:.3f} ({m['n']}) | {m['si_renseigne']:.3f} ({m['n_renseigne']}) | "
              f"{m['faux_positifs']} |")
    errors = [(r["id"], s, r[s], r["pred"][s], r["message"]) for r in rows for s in SLOTS if not r["ok"][s]]
    print(f"\nErreurs ({len(errors)}) :")
    for rid, s, gold, pred, msg in errors:
        print(f"  [{rid}] {LABELS[s]} : attendu {gold!r}, prédit {pred!r} · « {msg} »")

    out = Path("experiments") / f"slots-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"config": vars(args), "results": results, "rows": rows},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDétails : {out}/results.json")


if __name__ == "__main__":
    main()

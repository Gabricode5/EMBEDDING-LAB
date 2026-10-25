"""Évaluation de bout en bout de l'agent sur des demandes annotées.

Fichier d'entrée (JSON Lines), une demande par ligne :
{"id": "c-01", "project": "clinique-tilleuls", "message": "...",
 "expected_kind": "email_draft", "expected_recipient": "Agnès Rousseau-Lin",
 "expected_units": ["u-014"], "origin": "assisted", "category": "changement de responsable"}

    python scripts/eval_agent.py --file data/agent_eval/test.jsonl
    python scripts/eval_agent.py --file data/agent_eval/test.jsonl --relations gold
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from sklearn.metrics import classification_report

from sae.data import gid

ACTION_KINDS = {"email_draft", "task_draft", "event_draft"}


def recipient_of(result) -> str | None:
    p = result.payload or {}
    if result.kind == "email_draft":
        return (p.get("recipient") or {}).get("name")
    if result.kind == "task_draft":
        return (p.get("assignee") or {}).get("name")
    return None


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from agent import build_agent  # réutilise la construction de l'agent de scripts/agent.py

    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/agent_eval/test.jsonl")
    ap.add_argument("--split", default="data/scenarios/test")
    ap.add_argument("--retriever", default="models/e5b-both-type-s1")
    ap.add_argument("--no-type", action="store_true")
    ap.add_argument("--relations", choices=("predicted", "gold", "none"), default="predicted")
    ap.add_argument("--relations-model", default="models/relations-cross")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--verbose", action="store_true", help="affiche chaque demande ratée")
    args = ap.parse_args()

    cases = [json.loads(l) for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    agent = build_agent(args)

    rows = []
    for c in cases:
        r = agent.handle(c["message"], project=c.get("project"))
        expected_units = {gid(c["project"], u) for u in c.get("expected_units", [])}
        top_source = r.sources[0].unit_id if r.sources else None
        rows.append({**c, "kind": r.kind, "recipient": recipient_of(r), "top_source": top_source,
                     "source_ok": (top_source in expected_units) if expected_units else None, "text": r.text})

    y_true = [x["expected_kind"] for x in rows]
    y_pred = [x["kind"] for x in rows]
    labels = sorted(set(y_true) | set(y_pred))
    kind_acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(rows)

    with_rec = [x for x in rows if x.get("expected_recipient")]
    p_at_1 = sum(x["recipient"] == x["expected_recipient"] for x in with_rec) / len(with_rec) if with_rec else None
    drafted = [x for x in rows if x["recipient"] is not None]
    wrong = [x for x in drafted if x["recipient"] != x.get("expected_recipient")]
    false_rec = len(wrong) / len(drafted) if drafted else None
    should_clarify = [x for x in rows if x["expected_kind"] == "clarify"]
    abst = sum(x["kind"] == "clarify" for x in should_clarify) / len(should_clarify) if should_clarify else None
    clarified = [x for x in rows if x["kind"] == "clarify"]
    useless = sum(x["expected_kind"] != "clarify" for x in clarified) / len(clarified) if clarified else None
    hypo = [x for x in rows if x["expected_kind"] == "no_action"]
    hypo_ok = sum(x["kind"] not in ACTION_KINDS for x in hypo) / len(hypo) if hypo else None
    with_src = [x for x in rows if x["source_ok"] is not None]
    src_acc = sum(x["source_ok"] for x in with_src) / len(with_src) if with_src else None

    fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    print(f"\n{len(rows)} demandes · mémoire : {args.relations}\n")
    print("| métrique | valeur | sur |\n|---|---|---|")
    print(f"| bon type de sortie | {fmt(kind_acc)} | {len(rows)} |")
    print(f"| destinataire correct (Precision@1) | {fmt(p_at_1)} | {len(with_rec)} |")
    print(f"| taux de faux destinataires | {fmt(false_rec)} | {len(drafted)} brouillons adressés |")
    print(f"| abstention justifiée | {fmt(abst)} | {len(should_clarify)} |")
    print(f"| demandes de précision inutiles | {fmt(useless)} | {len(clarified)} |")
    print(f"| hypothèses sans action créée | {fmt(hypo_ok)} | {len(hypo)} |")
    print(f"| bonne unité en première source | {fmt(src_acc)} | {len(with_src)} |")
    print("\nType de sortie par classe :")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=3))

    by_cat = defaultdict(list)
    for x in rows:
        ok = x["kind"] == x["expected_kind"] and (not x.get("expected_recipient") or
                                                  x["recipient"] == x["expected_recipient"]) and x["source_ok"] is not False
        by_cat[x.get("category", "-")].append(ok)
    print("Réussite complète par catégorie :")
    for cat, oks in sorted(by_cat.items()):
        print(f"  {cat:<40} {sum(oks)}/{len(oks)}")

    failures = [x for x in rows if x["kind"] != x["expected_kind"]
                or (x.get("expected_recipient") and x["recipient"] != x["expected_recipient"]) or x["source_ok"] is False]
    if args.verbose:
        for x in failures:
            print(f"\n[{x['id']}] « {x['message']} »\n  attendu : {x['expected_kind']} "
                  f"{x.get('expected_recipient') or ''} {x.get('expected_units') or ''}\n"
                  f"  obtenu  : {x['kind']} {x['recipient'] or ''} source={x['top_source']}")

    out = Path("experiments") / f"agent-{Path(args.file).stem}-{args.relations}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({
        "config": vars(args), "kind_accuracy": kind_acc, "recipient_precision_at_1": p_at_1,
        "false_recipient_rate": false_rec, "justified_abstention": abst, "useless_clarification": useless,
        "hypothesis_no_action": hypo_ok, "source_accuracy": src_acc, "categories": dict(Counter(y_true)),
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(failures)} demandes ratées · détails : {out}/results.json (ou --verbose)")


if __name__ == "__main__":
    main()

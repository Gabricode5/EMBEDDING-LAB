"""Évaluation de l'abstention de l'agent sur les questions d'information (intentions answer et compare).

Deux temps, à ne pas mélanger :
    python scripts/eval_abstention.py --calibrate             # train + val : seuils et choix de la stratégie
    python scripts/eval_abstention.py --splits test test2     # mesure finale, avec les seuils figés dans engine.py

Mesures, par stratégie et par split :
- abstention justifiée : part des questions sans réponse (answerable = false) où l'agent répond no_answer ;
- abstention à tort : part des questions avec réponse où l'agent répond no_answer ;
- 1re source correcte : parmi les questions avec réponse non abstenues, part dont la 1re source est attendue.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from sae.agent.engine import ABSTENTION_STRATEGIES, ABSTENTION_THRESHOLDS, Agent
from sae.data import gid, load_scenarios

INFO_INTENTS = ("answer", "compare")
CALIBRATION_SPLITS = ("train", "val")
INFO_KINDS = {"answer", "compare", "no_answer"}


def make_agent(split: str, args) -> Agent:
    from agent import build_agent  # réutilise la construction de l'agent de scripts/agent.py
    ns = argparse.Namespace(**{**vars(args), "split": f"data/scenarios/{split}", "abstention": "none"})
    return build_agent(ns)


def run(agent: Agent, split: str, strategy: str, thresholds: dict[str, float]) -> list[dict]:
    """Fait passer toutes les questions d'information du split à l'agent, avec la stratégie donnée."""
    agent.abstention, agent.thresholds = strategy, thresholds
    rows = []
    for s in agent.scenarios.values():
        for q in s.queries:
            if q.intent not in INFO_INTENTS:
                continue
            r = agent.handle(q.text, project=s.project_id)
            relevant = [gid(s.project_id, u) for u in q.relevant_unit_ids]
            top = r.sources[0].unit_id if r.sources else None
            rows.append({"split": split, "qid": gid(s.project_id, q.query_id), "text": q.text,
                         "gold_intent": q.intent, "intent": r.understanding.intent, "answerable": q.answerable,
                         "relevant": relevant, "kind": r.kind, "top_source": top,
                         "source_ok": top in relevant if relevant else None, "signals": r.signals})
    return rows


def metrics(rows: list[dict]) -> dict:
    unans = [r for r in rows if not r["answerable"]]
    ans = [r for r in rows if r["answerable"]]
    answered = [r for r in ans if r["kind"] != "no_answer"]
    return {"n_unanswerable": len(unans), "justified": sum(r["kind"] == "no_answer" for r in unans),
            "n_answerable": len(ans), "wrong": sum(r["kind"] == "no_answer" for r in ans),
            "n_answered": len(answered), "source_ok": sum(bool(r["source_ok"]) for r in answered),
            "other_kind": sum(r["kind"] not in INFO_KINDS for r in rows)}


# ---------------------------------------------------------------------- calibrage (train + val uniquement)
def depends_on_threshold(strategy: str, sig: dict) -> bool:
    """Questions dont la décision dépend du seuil (les autres ne contraignent pas son choix)."""
    return {"score": True, "gap": True, "both": not sig["lexical"], "either": sig["lexical"]}[strategy]


def best_threshold(rows: list[dict], strategy: str) -> tuple[float, float, float] | None:
    """Seuil qui minimise les abstentions à tort, puis maximise les justifiées, au milieu de l'intervalle le plus large.

    Renvoie (seuil, borne basse, borne haute), ou None si aucun seuil n'apporte d'abstention justifiée.
    """
    key = "gap" if strategy == "gap" else "score"
    active = [r for r in rows if depends_on_threshold(strategy, r["signals"])]
    if all(r["answerable"] for r in active):
        return None
    values = sorted({r["signals"][key] for r in active})
    cuts = list(zip(values, values[1:])) + [(values[-1], values[-1] + 0.01)]
    best, best_key = None, None
    for lo, hi in cuts:
        t = (lo + hi) / 2
        wrong = sum(r["answerable"] and r["signals"][key] < t for r in active)
        justified = sum(not r["answerable"] and r["signals"][key] < t for r in active)
        k = (wrong, -justified, -(hi - lo))
        if best_key is None or k < best_key:
            best, best_key = (round(t, 4), lo, hi), k
    return best if best_key[1] < 0 else None


def calibrate(rows: list[dict], strategy: str) -> tuple[dict[str, float], dict]:
    """Un seuil par intention détectée ; pas de seuil si l'intention n'a aucune question sans réponse."""
    thresholds, detail = {}, {}
    if strategy in ("none", "lexical"):
        return thresholds, detail
    for intent in INFO_INTENTS:
        sub = [r for r in rows if r["signals"] and r["intent"] == intent]
        found = best_threshold(sub, strategy) if sub else None
        if found:
            thresholds[intent] = found[0]
            detail[intent] = {"threshold": found[0], "interval": [round(found[1], 4), round(found[2], 4)]}
    return thresholds, detail


def choose(pooled: dict[str, dict]) -> str:
    """Moins d'abstentions à tort, puis plus d'abstentions justifiées, puis meilleure 1re source."""
    useful = {s: m for s, m in pooled.items() if m["justified"] > 0} or pooled
    return min(useful, key=lambda s: (useful[s]["wrong"], -useful[s]["justified"], -useful[s]["source_ok"]))


# ---------------------------------------------------------------------- affichage
def frac(a: int, n: int) -> str:
    return f"{a}/{n} ({a / n:.2f})" if n else "-"


def print_table(table: dict[str, dict[str, dict]], thresholds: dict[str, dict]) -> None:
    print("\n| stratégie | seuils | split | abstention justifiée (sans réponse) | abstention à tort (avec réponse)"
          " | 1re source correcte (répondues) | autre sortie |")
    print("|---|---|---|---|---|---|---|")
    for strategy, by_split in table.items():
        th = ", ".join(f"{k} < {v}" for k, v in thresholds.get(strategy, {}).items()) or "-"
        for split, m in by_split.items():
            print(f"| {strategy} | {th} | {split} | {frac(m['justified'], m['n_unanswerable'])} | "
                  f"{frac(m['wrong'], m['n_answerable'])} | {frac(m['source_ok'], m['n_answered'])} | "
                  f"{m['other_kind']} |")


def warn_small(table: dict[str, dict[str, dict]]) -> None:
    first = next(iter(table.values()))
    for split, m in first.items():
        if m["n_unanswerable"] < 5:
            print(f"Attention : {split} ne contient que {m['n_unanswerable']} question(s) sans réponse ; "
                  "les taux d'abstention justifiée y sont très instables.")


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=list(CALIBRATION_SPLITS))
    ap.add_argument("--calibrate", action="store_true", help="calcule les seuils et choisit la stratégie")
    ap.add_argument("--retriever", default="models/e5b-both-type-s1")
    ap.add_argument("--no-type", action="store_true")
    ap.add_argument("--relations", choices=("predicted", "gold", "none"), default="predicted")
    ap.add_argument("--relations-model", default="models/relations-cross")
    ap.add_argument("--threshold", type=float, default=0.45)
    args = ap.parse_args()

    if args.calibrate and set(args.splits) - set(CALIBRATION_SPLITS):
        sys.exit("Le calibrage se fait sur train et val uniquement.")
    if not args.calibrate and not ABSTENTION_THRESHOLDS:
        sys.exit("Aucun seuil figé dans sae/agent/engine.py : lancer d'abord --calibrate sur train + val.")

    agents = {sp: make_agent(sp, args) for sp in args.splits}
    base = {sp: run(a, sp, "none", {}) for sp, a in agents.items()}

    if args.calibrate:
        pooled_rows = [r for rows in base.values() for r in rows]
        calib = {s: calibrate(pooled_rows, s) for s in ABSTENTION_STRATEGIES}
        thresholds = {s: c[0] for s, c in calib.items()}
        detail = {s: c[1] for s, c in calib.items()}
    else:
        thresholds = {s: ABSTENTION_THRESHOLDS.get(s, {}) for s in ABSTENTION_STRATEGIES}
        detail = {}

    table, rows_by = {}, {}
    pooled_name = "+".join(args.splits)
    for strategy in ABSTENTION_STRATEGIES:
        runs = {sp: run(a, sp, strategy, thresholds[strategy]) for sp, a in agents.items()}
        rows_by[strategy] = [r for rows in runs.values() for r in rows]
        table[strategy] = {sp: metrics(rows) for sp, rows in runs.items()}
        if len(runs) > 1:
            table[strategy][pooled_name] = metrics(rows_by[strategy])

    print_table(table, thresholds)
    warn_small(table)
    result = {"config": vars(args), "mode": "calibration" if args.calibrate else "final",
              "thresholds": thresholds, "calibration": detail, "table": table, "rows": rows_by}
    if args.calibrate:
        chosen = choose({s: by[pooled_name] if pooled_name in by else next(iter(by.values()))
                         for s, by in table.items()})
        result["chosen"] = chosen
        print(f"\nStratégie retenue sur {pooled_name} : {chosen} · seuils {thresholds[chosen]}")
        for s, d in detail.items():
            for intent, x in d.items():
                print(f"  {s:<7} {intent:<8} seuil {x['threshold']} (intervalle sans changement : {x['interval']})")
        print("À recopier dans ABSTENTION_THRESHOLDS (sae/agent/engine.py) :")
        print("  " + json.dumps({s: t for s, t in thresholds.items() if t}))

    out = Path("experiments") / f"abstention-{pooled_name}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDétails : {out}/results.json")


if __name__ == "__main__":
    main()

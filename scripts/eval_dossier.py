"""Évalue les dossiers produits par l'agent contre les faits attendus, construits à partir des annotations.

    python scripts/eval_dossier.py --splits test test2

Faits attendus par projet (annotations de data/scenarios) :
    Décisions en vigueur            décisions au statut active
    Engagements et actions          engagements et actions au statut active, avec leur responsable annoté
    Risques                         risques au statut active
    Propositions et idées           idées non remplacées ni annulées (active, proposed, hypothetical)
    Historique des changements      relations supersedes annotées (ancienne -> nouvelle)

Mesures :
    couverture            part des faits attendus cités dans la bonne section
    citations exactes     part des affirmations dont l'unité citée appartient bien aux faits attendus de la section
    obsolètes             part des affirmations des sections « actuelles » qui citent une unité remplacée ou annulée
    non étayées           part des affirmations sans unité citée, ou citant une unité absente de la mémoire
    responsable correct   part des engagements couverts dont le responsable affiché est le responsable annoté

Trois conditions : unités et relations annotées ; unités annotées et relations prédites ; unités extraites
(data/extracted/<split>) et relations prédites. Les unités extraites sont ramenées aux unités annotées par
l'alignement déjà calculé (data/extracted/<split>/alignment) ; une unité extraite non alignée n'est pas
vérifiable et compte comme citation inexacte.

Le dossier ne dépend pas de la recherche : l'agent est construit avec BM25 par défaut, sans GPU.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from sae.data import gid, load_scenarios
from sae.memory import INACTIVE
from sae.schemas import Scenario

MESSAGE = "Prépare un dossier complet sur le projet."
# titre de section de l'agent -> clé des faits attendus
SECTIONS = {"Décisions en vigueur": "decisions", "Engagements et actions": "engagements", "Risques": "risques",
            "Propositions et idées non tranchées": "idees", "Historique des changements": "historique"}
CURRENT = ("decisions", "engagements", "risques", "idees")
OWNER = re.compile(r"\(responsable : ([^,)]+)")


# ---------------------------------------------------------------------- faits attendus
def expected_facts(s: Scenario) -> dict[str, set]:
    """Faits attendus par section : identifiants globaux, ou couples (ancienne, nouvelle) pour l'historique."""
    p = s.project_id
    by_rule = {"decisions": lambda u: u.type == "decision" and u.status == "active",
               "engagements": lambda u: u.type in ("commitment", "action") and u.status == "active",
               "risques": lambda u: u.type == "risk" and u.status == "active",
               "idees": lambda u: u.type == "idea" and u.status not in INACTIVE}
    out = {k: {gid(p, u.unit_id) for u in s.units if rule(u)} for k, rule in by_rule.items()}
    out["historique"] = {(gid(p, r.to_id), gid(p, r.from_id)) for r in s.relations if r.relation == "supersedes"}
    return out


# ---------------------------------------------------------------------- notation d'un dossier
def score_dossier(dossier: dict, expected: dict[str, set], annotated: dict, to_annot: dict[str, str] | None,
                  known: set[str]) -> dict:
    """Compte, pour un dossier, les faits couverts et les affirmations exactes, obsolètes ou non étayées."""
    back = (lambda g: to_annot.get(g)) if to_annot is not None else (lambda g: g)
    cited = {k: set() for k in SECTIONS.values()}
    owner_shown: dict[str, str | None] = {}
    c = {"affirmations": 0, "exactes": 0, "actuelles": 0, "obsoletes": 0, "non_etayees": 0, "non_alignees": 0}
    errors = {"citation inexacte": [], "obsolète présentée comme actuelle": []}
    for sec in dossier["sections"]:
        key = SECTIONS[sec["title"]]
        for claim in sec["claims"]:
            c["affirmations"] += 1
            ids = claim["unit_ids"]
            if not ids or any(g not in known for g in ids):
                c["non_etayees"] += 1
                continue
            mapped = [back(g) for g in ids]
            if any(m is None for m in mapped):
                c["non_alignees"] += 1
                errors["citation inexacte"].append(f"[{key}] {claim['text']} (unité extraite non alignée)")
                continue
            if key == "historique":
                fact = (mapped[0], mapped[-1])
            else:
                fact = mapped[0]
                c["actuelles"] += 1
                m = OWNER.search(claim["text"])
                owner_shown[fact] = m.group(1).strip() if m else None
                if annotated[fact].status in INACTIVE:
                    c["obsoletes"] += 1
                    errors["obsolète présentée comme actuelle"].append(
                        f"[{key}] {fact} ({annotated[fact].status}) : {claim['text']}")
            cited[key].add(fact)
            if fact in expected[key]:
                c["exactes"] += 1
            else:
                u = annotated.get(fact if key != "historique" else fact[0])
                what = f"{u.type}/{u.status}" if key != "historique" else "remplacement non annoté"
                errors["citation inexacte"].append(f"[{key}] {fact} ({what}) : {claim['text']}")

    per_section = {k: (len(expected[k] & cited[k]), len(expected[k])) for k in SECTIONS.values()}
    with_owner = [g for g in expected["engagements"] & cited["engagements"] if annotated[g].owner]
    c["responsable_ok"] = sum(owner_shown.get(g) == annotated[g].owner for g in with_owner)
    c["responsable_n"] = len(with_owner)
    errors["fait manquant"] = [f"[{k}] {f} : {describe(f, annotated)}" for k in SECTIONS.values()
                               for f in sorted(expected[k] - cited[k])]
    errors["mauvais responsable"] = [f"{g} : annoté {annotated[g].owner}, affiché {owner_shown.get(g)}"
                                     for g in with_owner if owner_shown.get(g) != annotated[g].owner]
    return {"compte": c, "sections": per_section, "erreurs": errors}


def describe(fact, annotated: dict) -> str:
    if isinstance(fact, tuple):
        return f"{annotated[fact[0]].statement} → {annotated[fact[1]].statement}"
    return annotated[fact].statement


def merge(results: list[dict]) -> dict:
    """Agrège les comptes de plusieurs dossiers et calcule les taux."""
    c = {k: sum(r["compte"][k] for r in results) for k in results[0]["compte"]}
    sec = {k: tuple(sum(r["sections"][k][i] for r in results) for i in (0, 1)) for k in SECTIONS.values()}
    covered, total = sum(v[0] for v in sec.values()), sum(v[1] for v in sec.values())
    rate = lambda a, b: a / b if b else float("nan")  # noqa: E731
    return {"couverture": rate(covered, total), "n_faits": total,
            "citations_exactes": rate(c["exactes"], c["affirmations"]), "n_affirmations": c["affirmations"],
            "obsoletes": rate(c["obsoletes"], c["actuelles"]), "n_actuelles": c["actuelles"],
            "non_etayees": rate(c["non_etayees"], c["affirmations"]), "non_alignees": c["non_alignees"],
            "responsable_correct": rate(c["responsable_ok"], c["responsable_n"]), "n_responsable": c["responsable_n"],
            "couverture_par_section": {k: f"{a}/{b}" for k, (a, b) in sec.items()}}


# ---------------------------------------------------------------------- conditions
def run_condition(split: str, name: str, relations: str, extracted: bool, args) -> tuple[dict, dict]:
    sys.path.insert(0, str(Path(__file__).parent))
    from agent import build_agent  # même construction que scripts/agent.py et scripts/eval_agent.py

    annotated_sc = load_scenarios(f"data/scenarios/{split}")
    folder = f"{args.extracted_root}/{split}" if extracted else f"data/scenarios/{split}"
    agent = build_agent(Namespace(split=folder, no_type=False, relations=relations, retriever=args.retriever,
                                  relations_model=args.relations_model, threshold=args.threshold))
    to_annot = None
    if extracted:
        to_annot = {}
        for p in Path(folder, "alignment").glob("*.json"):
            to_annot |= {e: a for a, e in json.loads(p.read_text(encoding="utf-8"))["units"].items()}
    known = set(agent.memory.corpus.ids)
    per_project = {}
    for s in annotated_sc:
        r = agent.handle(MESSAGE, project=s.project_id)
        if r.kind != "dossier":
            raise RuntimeError(f"{s.project_id} : l'agent a répondu {r.kind} au lieu d'un dossier")
        annotated = {gid(s.project_id, u.unit_id): u for u in s.units}
        per_project[s.project_id] = score_dossier(r.payload, expected_facts(s), annotated, to_annot, known)
    return merge(list(per_project.values())), per_project


COLS = (("couverture", "n_faits"), ("citations_exactes", "n_affirmations"), ("obsoletes", "n_actuelles"),
        ("non_etayees", "n_affirmations"), ("responsable_correct", "n_responsable"))


def print_table(split: str, rows: dict[str, dict]) -> None:
    print(f"\n## {split}\n")
    print("| condition | couverture | citations exactes | obsolètes présentées comme actuelles | non étayées | "
          "responsable correct |\n|" + "---|" * 6)
    for name, m in rows.items():
        print(f"| {name} | " + " | ".join(f"{m[k]:.3f} ({m[n]})" for k, n in COLS) + " |")
    print("\nCouverture par section :\n\n| condition | " + " | ".join(SECTIONS) + " |\n|" + "---|" * 6)
    for name, m in rows.items():
        print(f"| {name} | " + " | ".join(m["couverture_par_section"].values()) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "test2"])
    ap.add_argument("--extracted-root", default="data/extracted")
    ap.add_argument("--retriever", default="bm25", help="sans effet sur le dossier, qui n'utilise pas la recherche")
    ap.add_argument("--relations-model", default="models/relations-cross")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()

    conditions = {"annotées · relations annotées": ("gold", False),
                  "annotées · relations prédites": ("predicted", False),
                  "extraites · relations prédites": ("predicted", True)}
    results: dict = {}
    for sp in args.splits:
        results[sp] = {}
        for name, (relations, extracted) in conditions.items():
            total, per_project = run_condition(sp, name, relations, extracted, args)
            results[sp][name] = {"total": total, "projets": per_project}
        print_table(sp, {n: r["total"] for n, r in results[sp].items()})
        for name, r in results[sp].items():
            print(f"\nErreurs · {name}")
            for kind in ("fait manquant", "citation inexacte", "obsolète présentée comme actuelle",
                         "mauvais responsable"):
                lines = [x for p in r["projets"].values() for x in p["erreurs"][kind]]
                for x in lines[:args.examples]:
                    print(f"  [{kind}] {x}")
                if len(lines) > args.examples:
                    print(f"  [{kind}] … {len(lines) - args.examples} de plus")

    out = Path("experiments") / f"dossier-{'+'.join(args.splits)}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"config": vars(args), "results": results}, ensure_ascii=False,
                                                 indent=2, default=str), encoding="utf-8")
    print(f"\nDétails : {out}/results.json")


if __name__ == "__main__":
    main()

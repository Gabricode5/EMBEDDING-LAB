"""Évalue l'extraction automatique des unités contre les unités annotées, et son coût pour la recherche.

    python scripts/eval_extraction.py --splits val train
    python scripts/eval_extraction.py --splits test test2 --retrieval     # mesure finale, avec la recherche

Alignement (sae/extraction.py, align_meeting) : dans chaque réunion, une unité extraite et une unité annotée sont
appariées si leurs evidence se recouvrent (mots communs / taille de la plus courte >= ALIGN_MIN), un pour un, par
score décroissant ; la similarité des statements départage deux candidats sur la même phrase.

Coût en aval (--retrieval) : même protocole que scripts/eval_memory.py (dense, puis dense + statut + projet),
avec les relations prédites par models/relations-cross pour les deux corpus. Les unités attendues des questions
sont réécrites vers les unités extraites alignées ; une question sans unité pertinente extraite compte comme un échec.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from sae.data import EvalQuery, build_queries, gid, load_scenarios
from sae.eval.metrics import KS, aggregate, per_query
from sae.extraction import align
from sae.schemas import Scenario, Unit

TYPES = ("decision", "action", "commitment", "question", "answer", "fact", "risk", "idea")
FIELDS = ("type", "status", "speaker", "owner", "due_date")
SEARCH_COLS = ("recall@1", "mrr@10", "hn_acc")


# ---------------------------------------------------------------------- qualité de l'extraction
def pairs_of(annotated: list[Scenario], extracted: list[Scenario]) -> tuple[list[tuple[Unit, Unit]], list, list]:
    """Unités alignées (annotée, extraite), annotées manquées, extraites en trop."""
    by_id = {s.project_id: s for s in extracted}
    pairs, missed, spurious = [], [], []
    for s in annotated:
        x = by_id[s.project_id]
        mapping = align(s.units, x.units)
        ex = {u.unit_id: u for u in x.units}
        pairs += [(a, ex[mapping[a.unit_id]]) for a in s.units if a.unit_id in mapping]
        missed += [(s.project_id, a) for a in s.units if a.unit_id not in mapping]
        used = set(mapping.values())
        spurious += [(s.project_id, e) for e in x.units if e.unit_id not in used]
    return pairs, missed, spurious


def field_ok(a: Unit, e: Unit, f: str) -> bool:
    return getattr(a, f) == getattr(e, f)


def quality(pairs, missed, spurious) -> dict:
    n_a, n_e, n = len(pairs) + len(missed), len(pairs) + len(spurious), len(pairs)
    p, r = n / n_e if n_e else 0.0, n / n_a if n_a else 0.0
    out = {"annotees": n_a, "extraites": n_e, "alignees": n, "precision": p, "rappel": r,
           "f1": 2 * p * r / (p + r) if p + r else 0.0}
    for f in FIELDS:
        sub = [(a, e) for a, e in pairs if f != "status" or a.status != "superseded"]
        out[f] = sum(field_ok(a, e, f) for a, e in sub) / len(sub) if sub else float("nan")
        out[f"n_{f}"] = len(sub)
    for f in ("owner", "due_date"):          # sur les seules unités où l'annotation renseigne le champ
        sub = [(a, e) for a, e in pairs if getattr(a, f) is not None]
        out[f"{f}_si_annote"] = sum(field_ok(a, e, f) for a, e in sub) / len(sub) if sub else float("nan")
        out[f"n_{f}_si_annote"] = len(sub)
    return out


def confusion(pairs) -> dict[str, Counter]:
    return {t: Counter(e.type for a, e in pairs if a.type == t) for t in TYPES}


def print_confusion(conf: dict[str, Counter]) -> None:
    print("\nMatrice de confusion des types (lignes : annoté, colonnes : extrait)\n")
    print("| annoté \\ extrait | " + " | ".join(TYPES) + " |\n|" + "---|" * (len(TYPES) + 1))
    for t in TYPES:
        if sum(conf[t].values()):
            print(f"| {t} | " + " | ".join(str(conf[t][x] or ".") for x in TYPES) + " |")


def examples(pairs, missed, spurious, k: int = 3) -> dict[str, list[str]]:
    """Quelques erreurs typiques, pour l'analyse."""
    out: dict[str, list[str]] = {}
    for f in FIELDS:
        bad = [(a, e) for a, e in pairs if not field_ok(a, e, f) and (f != "status" or a.status != "superseded")]
        out[f] = [f"« {a.evidence} » · annoté {getattr(a, f)} / extrait {getattr(e, f)} · extrait : {e.statement}"
                  for a, e in bad[:k]]
    out["manquées"] = [f"{p}/{a.unit_id} « {a.evidence} » ({a.type})" for p, a in missed[:k]]
    out["en trop"] = [f"{p}/{e.unit_id} « {e.evidence} » ({e.type}) : {e.statement}" for p, e in spurious[:k]]
    return out


def print_quality(name: str, q: dict) -> None:
    print(f"| {name} | {q['annotees']} | {q['extraites']} | {q['alignees']} | {q['precision']:.3f} | "
          f"{q['rappel']:.3f} | {q['f1']:.3f} | " + " | ".join(f"{q[f]:.3f}" for f in FIELDS) +
          f" | {q['owner_si_annote']:.3f} ({q['n_owner_si_annote']}) | "
          f"{q['due_date_si_annote']:.3f} ({q['n_due_date_si_annote']}) |")


# ---------------------------------------------------------------------- coût pour la recherche
def rewritten_queries(annotated: list[Scenario], extracted: list[Scenario]) -> tuple[list[EvalQuery], list[str]]:
    """Questions annotées, unités réécrites vers l'extraction ; les questions sans unité extraite sont perdues."""
    by_id = {s.project_id: s for s in extracted}
    out, lost = [], []
    for s in annotated:
        mapping = align(s.units, by_id[s.project_id].units)
        for q in build_queries([s]):
            rel = tuple(sorted({gid(s.project_id, mapping[r.split("/", 1)[1]]) for r in q.relevant
                                if r.split("/", 1)[1] in mapping}))
            hn = tuple(sorted({gid(s.project_id, mapping[h.split("/", 1)[1]]) for h in q.hard_negatives
                               if h.split("/", 1)[1] in mapping} - set(rel)))
            if q.answerable and not rel:
                lost.append(q.qid)
            out.append(EvalQuery(q.qid, q.text, rel, hn, q.answerable and bool(rel), q.origin, q.tags, q.intent))
    return out, lost


def failure_row(qid: str) -> dict:
    return {"qid": qid, "origin": "-", "tags": [], "best_rank": 0, "hn_total": 0, "hn_ok": 0,
            **{f"recall@{k}": 0.0 for k in KS}, "mrr@10": 0.0, "ndcg@10": 0.0}


def search_variants(scenarios, queries, relations, retriever, gold: bool) -> dict[str, dict]:
    from sae.memory import Memory, apply_memory
    memory = Memory.from_scenarios(scenarios, with_type=True)
    if not gold:
        memory = relations.build_memory(scenarios, memory)
    dense = retriever.fit(memory.corpus.texts).score([q.text for q in queries])
    full = apply_memory(dense, queries, memory, status=True, project=True)
    return {"dense": dense, "dense + statut + projet": full}, memory


def search_cost(annotated, extracted, retriever, relations) -> dict:
    gold_q = build_queries(annotated)
    ext_q, lost = rewritten_queries(annotated, extracted)
    out = {"questions_perdues": lost, "n_questions": sum(q.answerable for q in gold_q)}
    setups = {"annotées (relations annotées)": (annotated, gold_q, True),
              "annotées (relations prédites)": (annotated, gold_q, False),
              "extraites (relations prédites)": (extracted, ext_q, False)}
    for name, (scen, queries, gold) in setups.items():
        variants, memory = search_variants(scen, queries, relations, retriever, gold)
        for v, scores in variants.items():
            rows = per_query(scores, queries, memory.corpus.index())
            if scen is extracted:
                rows += [failure_row(q) for q in lost]
            out[f"{name} · {v}"] = aggregate(rows)
    return out


def print_search(split: str, res: dict) -> None:
    print(f"\nRecherche · {split} · {res['n_questions']} questions avec réponse · "
          f"{len(res['questions_perdues'])} perdues à l'extraction (comptées comme échecs)\n")
    print("| corpus · variante | " + " | ".join(SEARCH_COLS) + " |\n|" + "---|" * (len(SEARCH_COLS) + 1))
    for k, m in res.items():
        if isinstance(m, dict):
            print(f"| {k} | " + " | ".join(f"{m[c]:.3f}" for c in SEARCH_COLS) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "train"])
    ap.add_argument("--extracted-root", default="data/extracted")
    ap.add_argument("--retrieval", action="store_true", help="mesure aussi le coût pour la recherche")
    ap.add_argument("--model", default="models/e5b-both-type-s1")
    ap.add_argument("--relations-model", default="models/relations-cross")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    retriever = relations = None
    if args.retrieval:
        from sae.relations.predict import CrossEncoderRelations
        from sae.retrieval.dense import DenseRetriever, resolve_spec
        retriever = DenseRetriever(resolve_spec(args.model))
        relations = CrossEncoderRelations(args.relations_model, args.threshold)

    results = {}
    print("| split | annotées | extraites | alignées | précision | rappel | F1 | type | statut | locuteur | "
          "responsable | échéance | responsable (si annoté) | échéance (si annotée) |\n|" + "---|" * 14)
    data = {}
    for sp in args.splits:
        annotated = load_scenarios(f"data/scenarios/{sp}")
        extracted = load_scenarios(f"{args.extracted_root}/{sp}")
        pairs, missed, spurious = pairs_of(annotated, extracted)
        data[sp] = (annotated, extracted, pairs, missed, spurious)
        results[sp] = {"qualite": quality(pairs, missed, spurious)}
        print_quality(sp, results[sp]["qualite"])

    for sp, (annotated, extracted, pairs, missed, spurious) in data.items():
        conf = confusion(pairs)
        results[sp]["confusion"] = {t: dict(c) for t, c in conf.items()}
        results[sp]["exemples"] = examples(pairs, missed, spurious, args.examples)
        print(f"\n## {sp}")
        print_confusion(conf)
        print("\nExemples d'erreurs :")
        for f, ex in results[sp]["exemples"].items():
            for line in ex:
                print(f"  [{f}] {line}")
        if args.retrieval:
            results[sp]["recherche"] = search_cost(annotated, extracted, retriever, relations)
            print_search(sp, results[sp]["recherche"])

    out = Path("experiments") / f"extraction-{'+'.join(args.splits)}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"config": vars(args), "results": results}, ensure_ascii=False,
                                                 indent=2, default=str), encoding="utf-8")
    print(f"\nDétails : {out}/results.json")


if __name__ == "__main__":
    main()

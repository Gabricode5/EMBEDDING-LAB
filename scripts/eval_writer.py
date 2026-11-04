"""Évaluation de la rédaction des réponses (sae/agent/writer.py) sur les questions answer et compare des scénarios.

    python scripts/eval_writer.py --splits train val          # mise au point des consignes
    python scripts/eval_writer.py --splits test test2         # mesure unique, consignes figées

Quatre systèmes, sur les mêmes questions et la même recherche (models/e5b-both-type-s1, mémoire prédite) :
- première unité : l'affichage actuel de l'agent, sans abstention ;
- première unité + abstention : l'agent actuel (abstention en règles, stratégie score) ;
- rédaction : le modèle lit les K premières unités ; s'il dit ne rien trouver, c'est une abstention ;
  une réponse rejetée par la vérification revient à la première unité ;
- abstention + rédaction : l'abstention en règles d'abord, la rédaction ensuite (ce que fait l'agent avec --writer).

Mesures (unités attendues et remplacements annotés, jamais montrés au modèle) :
- cite une unité pertinente : au moins une unité attendue parmi les sources (questions avec réponse) ;
- remplacée présentée comme en vigueur : cite une unité annotée comme remplacée sans citer la version qui la
  remplace (questions avec réponse) ;
- abstention justifiée (questions sans réponse) et abstention à tort (questions avec réponse) ;
- rejets de la vérification et temps moyen par réponse (rédaction).
La recherche est la même pour tous : ces mesures portent sur la rédaction, pas sur la recherche.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from sae.agent.engine import ABSTENTION_THRESHOLDS, should_abstain
from sae.agent.writer import DEFAULT_HOST, DEFAULT_K, WRITER_MODELS, Writer
from sae.data import build_queries, load_scenarios
from sae.memory import Memory

SYSTEMS = ("première unité", "première unité + abstention", "rédaction", "abstention + rédaction")


def gold_successor(split: str) -> dict[str, str]:
    """Remplacements annotés : unité remplacée -> unité qui la remplace directement."""
    return Memory.from_scenarios(load_scenarios(split)).successor


def as_current(cited: list[str], successor: dict[str, str]) -> bool:
    """Une unité remplacée est citée sans aucune de ses versions ultérieures."""
    for c in cited:
        later, x, seen = set(), c, {c}
        while x in successor and successor[x] not in seen:
            x = successor[x]
            seen.add(x)
            later.add(x)
        if later and not later & set(cited):
            return True
    return False


def gpu_state(host: str, model: str) -> dict:
    import torch
    out = {"free_mib_before_writer": round(torch.cuda.mem_get_info()[0] / 2**20) if torch.cuda.is_available() else None}
    try:
        with urllib.request.urlopen(f"{host}/api/ps", timeout=5) as r:
            for m in json.load(r)["models"]:
                if m["name"] == model:
                    out |= {"ollama_size_mib": m["size"] // 2**20, "ollama_vram_mib": m["size_vram"] // 2**20}
    except OSError:
        pass
    return out


def evaluate(split: str, writer: Writer, args) -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    from agent import build_agent, free_gpu

    writer.unload()   # le classifieur de relations et le bi-encodeur passent avant le modèle de rédaction

    agent = build_agent(Namespace(split=split, retriever=args.retriever, no_type=False, relations=args.relations,
                                  relations_model="models/relations-cross", threshold=0.45, abstention="none"))
    successor = gold_successor(split)
    queries = [q for q in build_queries(load_scenarios(split)) if q.intent in ("answer", "compare")]
    if args.limit:
        queries = queries[:args.limit]
    free = gpu_state(writer.host, writer.model)
    rows = []
    for i, q in enumerate(queries, 1):
        base = agent.handle(q.text, project=q.project)
        hits3 = agent.search(q.text, q.project, q.intent, k=3)
        rule = should_abstain("score", ABSTENTION_THRESHOLDS["score"], q.intent, agent.signals(q.text, hits3))
        hits = agent.search(q.text, q.project, q.intent, k=writer.k)
        given = [agent.writer_unit(g) for g, _ in hits]
        res = writer.write(q.text, given, agent.scenarios[q.project].project_name)
        out = res.output
        rows.append({
            "qid": q.qid, "question": q.text, "intent": q.intent, "answerable": q.answerable,
            "relevant": list(q.relevant), "tags": list(q.tags), "rule_abstains": rule,
            "baseline_kind": base.kind, "baseline_cited": [s.unit_id for s in base.sources],
            "baseline_text": base.text, "given": [(x.uid, x.status, x.statement) for x in given],
            "writer_status": res.status, "writer_message": res.message, "seconds": round(res.seconds, 2),
            "answer": out.answer if out else None, "cited": res.cited, "found": out.found if out else None,
            "disagreement": out.disagreement if out else None, "raw": res.raw if not res.ok else None,
        })
        print(f"  [{i}/{len(queries)}] {res.status:<8} {res.seconds:5.1f} s · {q.text[:70]}", flush=True)
    free |= gpu_state(writer.host, writer.model)
    del agent
    free_gpu()
    return {"rows": rows, "metrics": metrics(rows, successor), "gpu": free}


def decide(x: dict, system: str) -> tuple[bool, list[str]]:
    """(abstention ?, unités citées) pour un système donné."""
    if system == "première unité":
        return False, x["baseline_cited"]
    if system == "première unité + abstention":
        return (True, []) if x["rule_abstains"] else (False, x["baseline_cited"])
    if system == "abstention + rédaction" and x["rule_abstains"]:
        return True, []
    if x["writer_status"] != "accepted":
        return False, x["baseline_cited"]         # rejet : retour à la première unité
    return (True, []) if not x["found"] else (False, x["cited"])


def metrics(rows: list[dict], successor: dict[str, str]) -> dict:
    ans = [x for x in rows if x["answerable"]]
    unans = [x for x in rows if not x["answerable"]]
    out = {"n": len(rows), "n_answerable": len(ans), "n_unanswerable": len(unans), "systems": {}}
    for s in SYSTEMS:
        d = {x["qid"]: decide(x, s) for x in rows}
        out["systems"][s] = {
            "cites_relevant": sum(bool(set(d[x["qid"]][1]) & set(x["relevant"])) for x in ans) / max(len(ans), 1),
            "superseded_as_current": sum(as_current(d[x["qid"]][1], successor) for x in ans) / max(len(ans), 1),
            "abstain_justified": sum(d[x["qid"]][0] for x in unans) / len(unans) if unans else None,
            "abstain_wrong": sum(d[x["qid"]][0] for x in ans) / max(len(ans), 1),
        }
    out["rejected"] = sum(x["writer_status"] == "rejected" for x in rows) / len(rows)
    out["errors"] = sum(x["writer_status"] == "error" for x in rows) / len(rows)
    out["mean_seconds"] = sum(x["seconds"] for x in rows) / len(rows)
    out["disagreement_flagged"] = sum(bool(x["disagreement"]) for x in rows)
    for intent in ("answer", "compare"):
        sub = [x for x in ans if x["intent"] == intent]
        out[f"cites_relevant_{intent}"] = {
            s: sum(bool(set(decide(x, s)[1]) & set(x["relevant"])) for x in sub) / len(sub) if sub else None
            for s in SYSTEMS}
    return out


def fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--model", choices=WRITER_MODELS, default=WRITER_MODELS[0])
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--retriever", default="models/e5b-both-type-s1")
    ap.add_argument("--relations", choices=("predicted", "gold", "none"), default="predicted")
    ap.add_argument("--ollama-host", default=DEFAULT_HOST)
    ap.add_argument("--limit", type=int, default=0, help="nombre maximal de questions par split (mise au point)")
    args = ap.parse_args()

    writer = Writer(args.model, args.ollama_host, args.k)
    if (problem := writer.unavailable()):
        sys.exit(f"Rédaction impossible : {problem}")
    out = Path("experiments") / f"writer-{'-'.join(args.splits)}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    t0 = time.perf_counter()
    for name in args.splits:
        split = f"data/scenarios/{name}"
        print(f"\n{split}")
        results[name] = evaluate(split, writer, args)
        m = results[name]["metrics"]
        print(f"\n{name} : {m['n']} questions ({m['n_answerable']} avec réponse, {m['n_unanswerable']} sans) · "
              f"rejets {m['rejected']:.3f} · erreurs {m['errors']:.3f} · {m['mean_seconds']:.1f} s par réponse · "
              f"désaccords signalés {m['disagreement_flagged']}\n")
        print("| système | cite une unité pertinente | remplacée présentée comme en vigueur | abstention justifiée "
              "| abstention à tort |\n|---|---|---|---|---|")
        for s, v in m["systems"].items():
            print(f"| {s} | {fmt(v['cites_relevant'])} | {fmt(v['superseded_as_current'])} | "
                  f"{fmt(v['abstain_justified'])} | {fmt(v['abstain_wrong'])} |")
        print("\ncite une unité pertinente, par intention :", {k: {s: fmt(v) for s, v in m[k].items()}
                                                              for k in ("cites_relevant_answer", "cites_relevant_compare")})
        (out / "results.json").write_text(json.dumps({"config": vars(args), "results": results}, ensure_ascii=False,
                                                     indent=2), encoding="utf-8")
    writer.unload()
    print(f"\n{time.perf_counter() - t0:.0f} s · résultats : {out}/results.json")


if __name__ == "__main__":
    main()

"""Évaluation de la recherche sur de vraies réunions en anglais : QMSum (Zhong et al., NAACL 2021). Aucun entraînement.

    python scripts/eval_qmsum.py --download                     # val et test dans data/external/qmsum/
    python scripts/eval_qmsum.py --split val --windows 1 3 5 8  # choix de la taille des fenêtres (val uniquement)
    python scripts/eval_qmsum.py --split test --frozen experiments/qmsum-val-.../results.json   # mesure unique

Corpus : pour chaque réunion, ses tours de parole (« Locuteur: texte ») et, pour une taille de fenêtre w > 1, des
fenêtres de w tours consécutifs (pas de w/2, arrondi au-dessus). La recherche se fait dans la réunion de la question.
Questions : les questions spécifiques (specific_query_list), avec leurs passages pertinents (relevant_text_span,
plages de tours inclusives). Un document est pertinent s'il recoupe un passage de la question.

Métriques : celles de sae/eval/metrics.py par per_query_docs, avec les passages comme unités : Recall@k = part des
passages de la question recoupés par au moins un des k premiers documents ; MRR@10 = rang du premier document
pertinent ; nDCG@10 = gain des passages nouveaux. Le texte reste en anglais ; BM25 en racinisation anglaise ;
encodeurs avec leurs préfixes query/passage, sans préfixe de type d'unité.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from sae.data import EvalQuery
from sae.eval.metrics import aggregate, grouped, per_query_docs

COMMIT = "83d7768c1f2b4dfeb091385d3dc7e239b8e5bb7e"   # dépôt Yale-LILY/QMSum, 29 août 2023
RAW = f"https://raw.githubusercontent.com/Yale-LILY/QMSum/{COMMIT}/data"
ROOT = Path("data/external/qmsum")
# domaine du dépôt -> source des transcriptions
DOMAINS = {"Product": "AMI", "Academic": "ICSI", "Committee": "commissions"}
MODELS = ("bm25", "e5-small", "e5-base", "bge-m3", "models/e5b-both-type-s1")
COLS = ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10")
NOISE = re.compile(r"\{\w+\}")    # marques de transcription : {disfmarker}, {vocalsound}, {gap}…


def download(splits: list[str]) -> None:
    """Fichiers ALL (questions) et fichiers par domaine (pour connaître la source de chaque réunion)."""
    for folder in ("ALL", *DOMAINS):
        for split in splits:
            dest = ROOT / folder / f"{split}.jsonl"
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = f"{RAW}/{folder}/jsonl/{split}.jsonl"
            print(f"  {url}")
            with urllib.request.urlopen(url, timeout=120) as r:
                dest.write_bytes(r.read())
    (ROOT / "SOURCE.txt").write_text(f"https://github.com/Yale-LILY/QMSum, commit {COMMIT} (licence MIT)\n"
                                     "AMI et ICSI : CC BY 4.0 ; commissions : Parlement gallois et Parlement du "
                                     "Canada, réutilisation non commerciale.\n", encoding="utf-8")


def load(split: str) -> list[dict]:
    """Réunions du split, chacune avec sa source (AMI, ICSI, commissions)."""
    def key(m: dict) -> str:
        return "".join(t["content"] for t in m["meeting_transcripts"][:20])

    source = {}
    for folder, name in DOMAINS.items():
        for line in (ROOT / folder / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            source[key(json.loads(line))] = name
    meetings = [json.loads(l) for l in (ROOT / "ALL" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    for i, m in enumerate(meetings):
        m["id"], m["source"] = f"{split}-{i:03d}", source.get(key(m), "inconnue")
    return meetings


def turn_text(t: dict) -> str:
    return f"{t['speaker']}: {' '.join(NOISE.sub(' ', t['content']).split())}"


def documents(m: dict, w: int) -> tuple[list[str], list[tuple[int, int]]]:
    """Tours, plus fenêtres de w tours (pas de ceil(w/2)) ; chaque document avec sa plage de tours [début, fin]."""
    turns = [turn_text(t) for t in m["meeting_transcripts"]]
    texts, spans = list(turns), [(i, i) for i in range(len(turns))]
    if w > 1:
        step = (w + 1) // 2
        for s in range(0, max(len(turns) - w, 0) + 1, step):
            texts.append("\n".join(turns[s:s + w]))
            spans.append((s, min(s + w, len(turns)) - 1))
    return texts, spans


def queries_of(m: dict) -> tuple[list[EvalQuery], dict[str, tuple[int, int]]]:
    """Questions spécifiques ; chaque passage pertinent devient une « unité » (identifiant qid#k)."""
    out, passages = [], {}
    for qi, q in enumerate(m["specific_query_list"]):
        qid = f"{m['id']}/q{qi:02d}"
        rel = []
        for k, (a, b) in enumerate(q["relevant_text_span"]):
            passages[f"{qid}#{k}"] = (int(a), int(b))
            rel.append(f"{qid}#{k}")
        if rel:
            out.append(EvalQuery(qid=qid, text=q["query"], relevant=tuple(rel), hard_negatives=(), answerable=True,
                                 origin=m["source"], tags=()))
    return out, passages


def doc_units(spans: list[tuple[int, int]], passages: dict[str, tuple[int, int]]) -> list[frozenset[str]]:
    return [frozenset(p for p, (a, b) in passages.items() if s <= b and a <= e) for s, e in spans]


def make_retriever(name: str):
    if name == "bm25":
        from sae.retrieval.bm25 import BM25
        return BM25(language="english")
    from sae.retrieval.dense import DenseRetriever, resolve_spec
    return DenseRetriever(resolve_spec(name))


def free_gpu() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate(meetings: list[dict], model: str, windows: list[int]) -> dict[int, list[dict]]:
    """Lignes par question (per_query_docs) pour chaque taille de fenêtre ; un seul modèle chargé à la fois."""
    retriever = make_retriever(model)
    rows: dict[int, list[dict]] = {w: [] for w in windows}
    t0 = time.perf_counter()
    for i, m in enumerate(meetings, 1):
        queries, passages = queries_of(m)
        if not queries:
            continue
        # tous les documents de la réunion encodés une fois : tours, puis fenêtres de chaque taille
        texts, spans, size = documents(m, 1) + ([1] * len(m["meeting_transcripts"]),)
        for w in windows:
            if w > 1:
                t_w, s_w = documents(m, w)
                n = len(m["meeting_transcripts"])
                texts, spans, size = texts + t_w[n:], spans + s_w[n:], size + [w] * (len(t_w) - n)
        scores = retriever.fit(texts).score([q.text for q in queries])
        units = doc_units(spans, passages)
        for w in windows:
            keep = [j for j, s in enumerate(size) if s in (1, w)]
            rows[w] += per_query_docs(scores[:, keep], queries, [units[j] for j in keep])
        print(f"    [{i}/{len(meetings)}] {m['source']:<11} {len(m['meeting_transcripts']):>5} tours · "
              f"{len(queries):>2} questions · {time.perf_counter() - t0:6.0f} s", flush=True)
    del retriever
    free_gpu()
    return rows


def table(results: dict, key: str) -> None:
    print("| modèle | fenêtre | " + " | ".join(COLS) + " |\n|" + "---|" * (len(COLS) + 2))
    for model, r in results.items():
        for w, agg in r[key].items():
            print(f"| {model} | {w} | " + " | ".join(f"{agg[c]:.3f}" for c in COLS) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="télécharge val et test dans data/external/qmsum/")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 3, 5, 8],
                    help="tailles de fenêtre essayées (1 : tours seuls) ; choix sur val uniquement")
    ap.add_argument("--frozen", default=None,
                    help="results.json de val : fenêtre choisie par modèle (mesure unique sur test)")
    ap.add_argument("--limit", type=int, default=0, help="nombre maximal de réunions (essai)")
    args = ap.parse_args()

    if args.download:
        download(["val", "test"])
        return
    if not (ROOT / "ALL" / f"{args.split}.jsonl").exists():
        sys.exit(f"{ROOT} introuvable : lance d'abord --download")
    if args.split == "test" and not args.frozen:
        sys.exit("test : passe --frozen avec le results.json de val (fenêtres choisies sur val)")

    meetings = load(args.split)[:args.limit or None]
    n_q = sum(len(queries_of(m)[0]) for m in meetings)
    print(f"QMSum {args.split} : {len(meetings)} réunions, {n_q} questions spécifiques, "
          f"sources {sorted({m['source'] for m in meetings})}")
    chosen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))["chosen"] if args.frozen else None

    results, chosen_out = {}, {}
    for model in args.models:
        # test : tours seuls (référence) et fenêtre figée sur val ; val : toutes les tailles essayées
        windows = sorted({1, chosen[model]}) if chosen else args.windows
        print(f"\n  {model} · fenêtres {windows}")
        rows = evaluate(meetings, model, windows)
        results[model] = {"overall": {w: aggregate(r) for w, r in rows.items()},
                          "groups": {w: grouped(r) for w, r in rows.items()},
                          "rows": {w: r for w, r in rows.items()}}
        # choix sur val : meilleure MRR@10, la plus petite fenêtre en cas d'égalité
        chosen_out[model] = chosen[model] if chosen else max(windows, key=lambda w: (round(
            results[model]["overall"][w]["mrr@10"], 4), -w))

    print(f"\nQMSum {args.split} · toutes sources\n")
    table(results, "overall")
    print(f"\nFenêtre {'figée sur val' if chosen else 'choisie (meilleure MRR@10)'} : {chosen_out}")
    sources = sorted({m["source"] for m in meetings})
    print("\nMRR@10 par source (fenêtre choisie) :\n\n| modèle | " + " | ".join(sources) + " |\n|"
          + "---|" * (len(sources) + 1))
    for model, r in results.items():
        g = r["groups"][chosen_out[model]]
        print(f"| {model} | " + " | ".join(f"{g[f'origin={s}']['mrr@10']:.3f} ({g[f'origin={s}']['n_queries']})"
                                            for s in sources) + " |")

    out = Path("experiments") / f"qmsum-{args.split}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"config": vars(args), "commit": COMMIT, "chosen": chosen_out,
                                                  "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nRésultats : {out}/results.json")


if __name__ == "__main__":
    main()

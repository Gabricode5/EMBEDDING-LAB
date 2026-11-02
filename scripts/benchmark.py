"""Mesures système : latence par étape, construction de la mémoire, extraction, mémoire GPU, taille des modèles.

    python scripts/benchmark.py --split data/scenarios/test
    python scripts/benchmark.py --split data/scenarios/test --parts latence relations     # sans Ollama

Chaque mesure de latence se fait question par question (lot de 1, comme dans l'agent), après une phase de chauffe,
sur plusieurs répétitions ; on donne la moyenne, la médiane et le 95e centile. La mémoire GPU est lue avec
torch.cuda.max_memory_allocated (modèles PyTorch) ; qwen3:8b tourne dans Ollama, hors de PyTorch : sa mémoire est
celle que rapporte Ollama (/api/ps). L'extraction n'écrit rien dans data/extracted : le cache n'a pas de durées,
on relance donc le modèle sur quelques réunions seulement.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import time
import urllib.request
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

from sae.data import EvalQuery, build_queries, load_scenarios
from sae.memory import Memory, apply_memory
from sae.relations.pairs import UnitPair, all_units

MiB = 1024 ** 2
PARTS = ("extraction", "latence", "relations")


# ---------------------------------------------------------------------- outils de mesure
def timed(fn, items: list, warmup: int, reps: int) -> dict:
    """Durées (ms) de fn(item) pour chaque item, répétées `reps` fois après `warmup` appels de chauffe."""
    for x in items[:warmup]:
        fn(x)
    times = []
    for _ in range(reps):
        for x in items:
            t0 = time.perf_counter()
            fn(x)
            times.append((time.perf_counter() - t0) * 1000)
    return summary(times)


def summary(times: list[float]) -> dict:
    return {"moyenne_ms": statistics.fmean(times), "mediane_ms": statistics.median(times),
            "p95_ms": float(np.percentile(times, 95)), "n": len(times)}


class GpuPeak:
    """Pic de mémoire GPU (PyTorch) pendant un bloc, au-delà de ce qui était déjà alloué."""

    def __enter__(self) -> "GpuPeak":
        import torch
        self.on = torch.cuda.is_available()
        if self.on:
            torch.cuda.synchronize()
            self.base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc) -> None:
        import torch
        self.peak = self.extra = 0.0
        if self.on:
            torch.cuda.synchronize()
            self.peak = torch.cuda.max_memory_allocated() / MiB
            self.extra = self.peak - self.base / MiB


def dir_size(path: Path) -> float:
    """Taille sur disque (Mio), en suivant les liens (cache Hugging Face)."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / MiB if path.exists() else float("nan")


def machine() -> dict:
    import torch
    cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                if l.startswith("model name")), platform.processor())
    ram = next((int(l.split()[1]) / 1024 ** 2 for l in Path("/proc/meminfo").read_text().splitlines()
                if l.startswith("MemTotal")), float("nan"))
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3 if gpu else None
    return {"cpu": cpu, "coeurs_logiques": os.cpu_count(), "ram_gio": round(ram, 1), "gpu": gpu,
            "vram_gio": round(vram, 1) if vram else None, "os": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__}


# ---------------------------------------------------------------------- recherche et agent
def bench_search(scenarios, args) -> tuple[dict, dict, object, Memory, float]:
    from sae.relations.predict import CrossEncoderRelations
    from sae.retrieval.dense import DenseRetriever, resolve_spec
    from sae.retrieval.rerank import CrossEncoderReranker

    gpu: dict = {}
    with GpuPeak() as g:
        retriever = DenseRetriever(resolve_spec(args.retriever))
        memory = Memory.from_scenarios(scenarios, with_type=True)
        retriever.fit(memory.corpus.texts)
    gpu["bi-encodeur (chargement + corpus)"] = g.extra
    queries = build_queries(scenarios)
    spec = retriever.spec

    with GpuPeak() as g:
        relations = CrossEncoderRelations(args.relations_model, args.threshold)
        t0 = time.perf_counter()
        memory = relations.build_memory(scenarios, memory)
        build_s = time.perf_counter() - t0
    gpu["classifieur de relations (chargement + paires du split)"] = g.extra
    with GpuPeak() as g:
        reranker = CrossEncoderReranker(args.reranker)
        reranker.pair_scores([queries[0].text] * args.k, memory.corpus.texts[:args.k])
    gpu[f"reranker (chargement + {args.k} paires)"] = g.extra

    doc_emb = retriever._doc_emb
    lat: dict = {}
    w, r = args.warmup, args.reps
    lat["encodage de la question"] = timed(lambda q: retriever._encode([spec.query_prefix + q.text]), queries, w, r)
    q_emb = {q.qid: retriever._encode([spec.query_prefix + q.text]) for q in queries}

    def search(q: EvalQuery) -> np.ndarray:
        s = q_emb[q.qid] @ doc_emb.T
        np.argsort(-s[0], kind="stable")
        return s
    lat["recherche (produit scalaire + tri)"] = timed(search, queries, w, r)
    dense = {q.qid: search(q) for q in queries}
    lat[f"reranking (K={args.k})"] = timed(
        lambda q: reranker.rerank(dense[q.qid], [q.text], memory.corpus.texts, args.k), queries, w, r)
    lat["mémoire (statut + projet)"] = timed(
        lambda q: apply_memory(dense[q.qid], [q], memory, status=True, project=True), queries, w, r)
    lat["chaîne sans agent (encodage + recherche + mémoire)"] = timed(
        lambda q: apply_memory(retriever.score([q.text]), [q], memory, status=True, project=True), queries, w, r)
    lat[f"chaîne avec reranking K={args.k}"] = timed(
        lambda q: apply_memory(reranker.rerank(retriever.score([q.text]), [q.text], memory.corpus.texts, args.k),
                               [q], memory, status=True, project=True), queries, w, r)
    return lat, gpu, retriever, memory, build_s


def bench_agent(scenarios, retriever, memory: Memory, args) -> dict:
    from sae.agent.engine import Agent
    agent = Agent(scenarios, memory, retriever)
    questions = [(q.text, q.project) for q in build_queries(scenarios) if q.intent == "answer"]
    dossiers = [("Prépare un dossier complet sur le projet.", s.project_id) for s in scenarios]
    projects = {s.project_id for s in scenarios}
    mails = [(c["message"], c["project"]) for f in ("test.jsonl", "holdout.jsonl", "test2.jsonl")
             for c in map(json.loads, Path("data/agent_eval", f).read_text(encoding="utf-8").splitlines())
             if c["expected_kind"] == "email_draft" and c["project"] in projects]
    handle = lambda m: agent.handle(m[0], project=m[1])  # noqa: E731
    return {f"agent · réponse ({len(questions)} questions)": timed(handle, questions, args.warmup, args.reps),
            f"agent · dossier ({len(dossiers)} projets)": timed(handle, dossiers, 1, args.reps * 5),
            f"agent · brouillon de mail ({len(mails)} demandes)": timed(handle, mails, 2, args.reps)}


# ---------------------------------------------------------------------- construction de la mémoire
def bench_relations(args) -> list[dict]:
    """Temps de prédiction des relations sur toutes les paires d'un projet fictif de n unités."""
    from sae.relations.predict import CrossEncoderRelations
    pool = all_units([s for sp in ("train", "val", "test", "test2") for s in load_scenarios(f"data/scenarios/{sp}")])
    ids = sorted(pool, key=lambda g: (pool[g].source_date, g))
    relations = CrossEncoderRelations(args.relations_model, args.threshold)
    relations.predict_proba([UnitPair(ids[0], ids[1], "unrelated")] * 64, pool)        # chauffe
    out = []
    for n in args.sizes:
        if n > len(ids):
            continue
        pairs = [UnitPair(a, b, "unrelated") for a, b in combinations(ids[:n], 2)]
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            relations.predict_proba(pairs, pool)
            times.append(time.perf_counter() - t0)
        out.append({"unites": n, "paires": len(pairs), "mediane_s": statistics.median(times),
                    "ms_par_paire": statistics.median(times) * 1000 / len(pairs)})
    return out


def extrapolate(rows: list[dict], sizes: tuple[int, ...] = (500, 1000, 5000)) -> list[dict]:
    """Temps prévu sur toutes les paires, avec le coût par paire mesuré sur le plus grand projet."""
    per_pair = rows[-1]["ms_par_paire"] / 1000
    return [{"unites": n, "paires": n * (n - 1) // 2, "prevu_s": per_pair * n * (n - 1) / 2} for n in sizes]


# ---------------------------------------------------------------------- extraction (Ollama)
def ollama(path: str, payload: dict | None, host: str) -> dict:
    req = urllib.request.Request(f"{host}{path}", data=json.dumps(payload).encode() if payload else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def bench_extraction(split: str, args) -> dict:
    """Durée d'extraction par réunion, avec les mêmes prompt et options que sae/extraction.py."""
    from sae.extraction import SYSTEM, Extraction, build_prompt
    meetings = [m for s in load_scenarios(split) for m in s.meetings][:args.meetings]

    def call(m) -> dict:
        payload = {"model": args.llm, "stream": False, "format": Extraction.model_json_schema(), "think": False,
                   "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(m)}],
                   "options": {"temperature": 0.0, "seed": 0, "num_ctx": 8192}}
        t0 = time.perf_counter()
        r = ollama("/api/chat", payload, args.ollama_host)
        return {"wall_s": time.perf_counter() - t0, "load_s": r.get("load_duration", 0) / 1e9,
                "prompt_tokens": r.get("prompt_eval_count"), "sortie_tokens": r.get("eval_count"),
                "generation_s": r.get("eval_duration", 0) / 1e9}

    cold = call(meetings[0])                        # chauffe : chargement du modèle en mémoire GPU
    runs = [dict(call(m), reunion=m.meeting_id, unites_annotees=len(m.units))
            for _ in range(args.reps) for m in meetings]
    ps = ollama("/api/ps", None, args.ollama_host).get("models", [])
    vram = next((x.get("size_vram", 0) / MiB for x in ps if x.get("name", "").startswith(args.llm)), float("nan"))
    ollama("/api/generate", {"model": args.llm, "keep_alive": 0}, args.ollama_host)   # libère la mémoire GPU
    return {"chargement_a_froid_s": cold["wall_s"], "par_reunion": summary([x["wall_s"] * 1000 for x in runs]),
            "tokens_par_s": statistics.fmean(x["sortie_tokens"] / x["generation_s"] for x in runs if x["generation_s"]),
            "vram_ollama_mio": vram, "details": runs}


def model_sizes(args) -> dict:
    from huggingface_hub import scan_cache_dir
    hf = {r.repo_id: r.size_on_disk / MiB for r in scan_cache_dir().repos}
    out = {args.retriever: dir_size(Path(args.retriever)), args.relations_model: dir_size(Path(args.relations_model)),
           args.reranker: hf.get(args.reranker, float("nan"))}
    try:
        tags = ollama("/api/tags", None, args.ollama_host)["models"]
        out[args.llm] = next(m["size"] / MiB for m in tags if m["name"] == args.llm)
    except Exception:                          # Ollama arrêté : taille inconnue
        out[args.llm] = float("nan")
    return out


# ---------------------------------------------------------------------- affichage
def print_latency(title: str, lat: dict) -> None:
    print(f"\n{title}\n\n| étape | moyenne (ms) | médiane (ms) | p95 (ms) | mesures |\n|---|---|---|---|---|")
    for k, v in lat.items():
        print(f"| {k} | {v['moyenne_ms']:.2f} | {v['mediane_ms']:.2f} | {v['p95_ms']:.2f} | {v['n']} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/scenarios/test")
    ap.add_argument("--retriever", default="models/e5b-both-type-s1")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--k", type=int, default=10, help="candidats reclassés (choisi sur val)")
    ap.add_argument("--relations-model", default="models/relations-cross")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 40, 80, 160, 240])
    ap.add_argument("--llm", default="qwen3:8b")
    ap.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--meetings", type=int, default=3, help="réunions extraites pour la mesure")
    ap.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS), help="parties à mesurer")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    import torch
    info = machine()
    print(json.dumps(info, ensure_ascii=False, indent=1))
    res: dict = {"machine": info, "taille_disque_mio": model_sizes(args)}
    out = Path("experiments") / f"benchmark-{Path(args.split).name}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    def save() -> None:          # après chaque partie : une interruption ne perd pas les mesures déjà faites
        (out / "results.json").write_text(json.dumps({"config": vars(args), **res}, ensure_ascii=False, indent=2,
                                                     default=str), encoding="utf-8")
    save()
    print("\nTaille des modèles sur disque :")
    for k, v in res["taille_disque_mio"].items():
        print(f"  {k} : {v:.0f} Mio")

    # l'extraction d'abord : qwen3:8b (Ollama) occupe 6,1 Gio de VRAM et ne doit pas partager le GPU avec PyTorch
    if "extraction" in args.parts:
        res["extraction"] = e = bench_extraction(args.split, args)
        save()
        print(f"\nExtraction ({args.llm}, {args.meetings} réunions × {args.reps}) : chargement à froid "
              f"{e['chargement_a_froid_s']:.1f} s, médiane {e['par_reunion']['mediane_ms'] / 1000:.1f} s par réunion, "
              f"{e['tokens_par_s']:.0f} tokens/s, VRAM Ollama {e['vram_ollama_mio']:.0f} Mio")

    if "latence" in args.parts:
        scenarios = load_scenarios(args.split)
        lat, gpu, retriever, memory, build_s = bench_search(scenarios, args)
        lat |= bench_agent(scenarios, retriever, memory, args)
        res["latence"], res["gpu_mio"], res["memoire_split_s"] = lat, gpu, build_s
        save()
        print_latency(f"Latence par question ({args.split}, lot de 1, {args.reps} répétitions après chauffe)", lat)
        print(f"\nConstruction de la mémoire du split (relations prédites, {len(memory.corpus.ids)} unités, "
              f"premier appel) : {build_s:.2f} s")
        print("\nMémoire GPU (PyTorch, au-delà de l'existant) :")
        for k, v in gpu.items():
            print(f"  {k} : {v:.0f} Mio")
        del retriever
        gc.collect()
        torch.cuda.empty_cache()

    if "relations" in args.parts:
        res["relations"] = bench_relations(args)
        res["relations_extrapolees"] = extrapolate(res["relations"])
        save()
        print("\nPrédiction des relations sur toutes les paires d'un projet\n\n| unités | paires | temps (s) | "
              "ms par paire |\n|---|---|---|---|")
        for x in res["relations"]:
            print(f"| {x['unites']} | {x['paires']} | {x['mediane_s']:.2f} | {x['ms_par_paire']:.3f} |")
        for x in res["relations_extrapolees"]:
            print(f"| {x['unites']} (prévu) | {x['paires']} | {x['prevu_s']:.0f} | - |")

    print(f"\nDétails : {out}/results.json")

if __name__ == "__main__":
    main()

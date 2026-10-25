"""Fine-tuning contrastif d'un encodeur sur les scénarios d'entraînement.

Exemples :
    python scripts/train_embedder.py --negatives none     --name e5b-none
    python scripts/train_embedder.py --negatives explicit --name e5b-explicit
    python scripts/train_embedder.py --negatives both     --name e5b-both
    python scripts/train_embedder.py --negatives both --with-type --name e5b-both-type
    python scripts/train_embedder.py --model e5-small --negatives both --name e5s-both

Le modèle est enregistré dans models/<name>/ ; la configuration et les scores sur la
validation (avant / après) dans experiments/train-<name>/metrics.json.
Évaluation finale sur le test : scripts/run_baselines.py --models models/<name>
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
try:  # Sentence Transformers 5+
    from sentence_transformers.sentence_transformer.losses import (
        CachedMultipleNegativesRankingLoss, MultipleNegativesRankingLoss, TripletDistanceMetric, TripletLoss)
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers
except ImportError:  # versions précédentes
    from sentence_transformers.losses import (
        CachedMultipleNegativesRankingLoss, MultipleNegativesRankingLoss, TripletDistanceMetric, TripletLoss)
    from sentence_transformers.training_args import BatchSamplers
from transformers import set_seed

from sae.data import build_corpus, build_queries, load_scenarios
from sae.eval.metrics import aggregate, per_query
from sae.retrieval.bm25 import BM25
from sae.retrieval.dense import resolve_spec
from sae.training.pairs import NEGATIVE_MODES, build_pairs, mine_negatives, to_rows


def encode(model: SentenceTransformer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def evaluate(model: SentenceTransformer, spec, split_dir: str, with_type: bool) -> dict:
    scenarios = load_scenarios(split_dir)
    corpus = build_corpus(scenarios, with_type=with_type)
    queries = build_queries(scenarios)
    docs = encode(model, [spec.passage_prefix + t for t in corpus.texts])
    qs = encode(model, [spec.query_prefix + q.text for q in queries])
    return aggregate(per_query(qs @ docs.T, queries, corpus.index()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="nom du run (dossier models/<name>)")
    ap.add_argument("--model", default="e5-base", help="preset (e5-base, e5-small) ou chemin")
    ap.add_argument("--train", default="data/scenarios/train")
    ap.add_argument("--val", default="data/scenarios/val")
    ap.add_argument("--negatives", choices=NEGATIVE_MODES, default="both")
    ap.add_argument("--miner", choices=("model", "bm25"), default="model",
                    help="modèle servant au minage des négatifs (le modèle de départ, ou BM25)")
    ap.add_argument("--k-neg", type=int, default=2, help="négatifs par paire et par source")
    ap.add_argument("--no-evidence", action="store_true", help="n'utiliser que les paires question/unité")
    ap.add_argument("--with-type", action="store_true", help="préfixe le type d'unité au texte")
    ap.add_argument("--loss", choices=("mnrl", "cached-mnrl", "triplet"), default="cached-mnrl")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--mini-batch", type=int, default=16, help="cached-mnrl : taille des sous-lots en mémoire")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seq-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.loss == "triplet" and args.negatives == "none":
        ap.error("--loss triplet nécessite des négatifs (--negatives explicit, mined ou both)")

    set_seed(args.seed)
    spec = resolve_spec(args.model)
    out_model = Path("models") / args.name
    out_exp = Path("experiments") / f"train-{args.name}"
    out_exp.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(spec.path)
    model.max_seq_length = args.max_seq_length

    # --- données
    train_sc = load_scenarios(args.train)
    corpus = build_corpus(train_sc, with_type=args.with_type)
    pairs = build_pairs(train_sc, use_evidence=not args.no_evidence)

    mined = None
    if args.negatives in ("mined", "both"):
        if args.miner == "bm25":
            score_fn = BM25().fit(corpus.texts).score
        else:
            docs = encode(model, [spec.passage_prefix + t for t in corpus.texts])
            score_fn = lambda texts: encode(model, [spec.query_prefix + t for t in texts]) @ docs.T  # noqa: E731
        mined = mine_negatives(pairs, corpus, score_fn, k=args.k_neg)

    rows = to_rows(pairs, corpus, args.negatives, mined, spec.query_prefix, spec.passage_prefix,
                   k=args.k_neg, seed=args.seed)
    dataset = Dataset.from_dict(rows).shuffle(seed=args.seed)
    n_pairs = {s: sum(p.source == s for p in pairs) for s in ("query", "evidence")}
    print(f"{len(pairs)} paires ({n_pairs}) -> {len(dataset)} exemples, colonnes {dataset.column_names}")

    # --- évaluation avant entraînement
    val_before = evaluate(model, spec, args.val, args.with_type)
    print("validation avant :", {k: round(v, 3) for k, v in val_before.items() if k != "n_queries"})

    # --- perte
    if args.loss == "mnrl":
        loss = MultipleNegativesRankingLoss(model)
    elif args.loss == "cached-mnrl":
        loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch)
    else:
        loss = TripletLoss(model, distance_metric=TripletDistanceMetric.COSINE, triplet_margin=0.2)

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    steps = math.ceil(len(dataset) / args.batch_size) * args.epochs
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(out_exp / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=max(1, int(0.1 * steps)),
        bf16=bf16,
        fp16=cuda and not bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=max(1, steps // 10),
        seed=args.seed,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(model=model, args=training_args, train_dataset=dataset, loss=loss)

    t0 = time.perf_counter()
    trainer.train()
    duration = time.perf_counter() - t0

    val_after = evaluate(model, spec, args.val, args.with_type)
    print("validation après :", {k: round(v, 3) for k, v in val_after.items() if k != "n_queries"})

    model.save(str(out_model))
    (out_exp / "metrics.json").write_text(json.dumps({
        "name": args.name, "date": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args), "base_model": spec.path,
        "train": {"pairs": len(pairs), "pairs_by_source": n_pairs, "examples": len(dataset),
                  "steps": steps, "seconds": round(duration, 1)},
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "precision": "bf16" if bf16 else "fp16" if cuda else "fp32",
        "val_before": val_before, "val_after": val_after,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nModèle : {out_model}/  ·  métriques : {out_exp}/metrics.json  ·  {duration:.0f} s")


if __name__ == "__main__":
    main()

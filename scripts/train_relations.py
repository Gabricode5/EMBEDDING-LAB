"""Classification des relations entre unités, puis mémoire construite à partir des relations prédites.

    # régression logistique sur les embeddings du modèle fine-tuné
    python scripts/train_relations.py --method logreg --encoder models/e5b-both-type-s1
    # cross-encodeur NLI multilingue fine-tuné sur les 6 relations
    python scripts/train_relations.py --method cross

Étapes :
1. entraînement sur les relations du split train (+ paires non reliées tirées au hasard) ;
2. choix du seuil de « supersedes » sur la validation (meilleur F1 sur toutes les paires) ;
3. évaluation sur le test : classification (macro-F1, matrice de confusion) puis recherche
   avec une mémoire construite à partir des relations prédites.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, precision_recall_fscore_support

from sae.data import build_queries, load_scenarios
from sae.eval.metrics import aggregate, per_query
from sae.memory import Memory, apply_memory
from sae.relations.pairs import (LABEL2ID, RELATIONS, UnitPair, all_pairs_with_labels, all_units,
                                 labelled_pairs, unit_label)
from sae.schemas import UnitType

TYPES = UnitType.__args__  # type: ignore[attr-defined]
SUP = LABEL2ID["supersedes"]


# ----------------------------------------------------------------------------- régression logistique

class LogRegClassifier:
    def __init__(self, encoder: str, seed: int) -> None:
        self.encoder, self.seed = encoder, seed
        self._cache: dict[str, np.ndarray] = {}

    def _embed(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            if self.encoder == "tfidf":
                vecs = self._tfidf.transform(missing).toarray()
            else:
                vecs = self._st.encode(["passage: " + t for t in missing], normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=False)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            self._cache.update(zip(missing, vecs / np.maximum(norms, 1e-9)))
        return np.stack([self._cache[t] for t in texts])

    def _features(self, pairs: list[UnitPair], units) -> np.ndarray:
        ua, ub = [units[p.a] for p in pairs], [units[p.b] for p in pairs]
        ea = self._embed([unit_label(u) for u in ua])
        eb = self._embed([unit_label(u) for u in ub])
        extra = np.array([[
            float((ea[i] * eb[i]).sum()),
            float(x.type == y.type),
            float(x.source_id == y.source_id),
            (y.source_date - x.source_date).days / 30.0,
            float(x.speaker is not None and x.speaker == y.speaker),
            *[float(x.type == t) for t in TYPES], *[float(y.type == t) for t in TYPES],
        ] for i, (x, y) in enumerate(zip(ua, ub))])
        return np.hstack([ea, eb, np.abs(ea - eb), ea * eb, extra])

    def fit(self, pairs: list[UnitPair], units) -> "LogRegClassifier":
        from sklearn.linear_model import LogisticRegression
        if self.encoder == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit([unit_label(u) for u in units.values()])
        else:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self.encoder)
        X = self._features(pairs, units)
        y = np.array([LABEL2ID[p.label] for p in pairs])
        self.clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, random_state=self.seed)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, pairs: list[UnitPair], units) -> np.ndarray:
        probs = np.zeros((len(pairs), len(RELATIONS)))
        probs[:, self.clf.classes_] = self.clf.predict_proba(self._features(pairs, units))
        return probs


# ----------------------------------------------------------------------------- cross-encodeur

class CrossEncoderClassifier:
    def __init__(self, model_name: str, epochs: int, lr: float, batch_size: int, seed: int, out_dir: Path) -> None:
        self.model_name, self.epochs, self.lr, self.bs, self.seed, self.out_dir = \
            model_name, epochs, lr, batch_size, seed, out_dir

    def fit(self, pairs: list[UnitPair], units) -> "CrossEncoderClassifier":
        import torch
        from datasets import Dataset
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                                  Trainer, TrainingArguments, set_seed)
        set_seed(self.seed)
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(RELATIONS), id2label=dict(enumerate(RELATIONS)),
            label2id=LABEL2ID, ignore_mismatched_sizes=True)
        self.model = self.model.float()  # poids publiés en fp16 : conversion en fp32 pour un entraînement stable
        ds = Dataset.from_dict({
            "a": [unit_label(units[p.a]) for p in pairs], "b": [unit_label(units[p.b]) for p in pairs],
            "label": [LABEL2ID[p.label] for p in pairs],
        }).shuffle(seed=self.seed)
        ds = ds.map(lambda x: self.tok(x["a"], x["b"], truncation=True, max_length=256), batched=True,
                    remove_columns=["a", "b"])
        # DeBERTa-v3 diverge en précision réduite (bf16/fp16) : entraînement en fp32
        steps = -(-len(ds) // self.bs) * self.epochs
        args = TrainingArguments(
            output_dir=str(self.out_dir / "checkpoints"), num_train_epochs=self.epochs,
            per_device_train_batch_size=self.bs, learning_rate=self.lr, warmup_steps=max(1, steps // 10),
            weight_decay=0.01, bf16=False, fp16=False, max_grad_norm=1.0, save_strategy="no", eval_strategy="no",
            logging_steps=max(1, steps // 10), report_to="none", seed=self.seed)
        Trainer(model=self.model, args=args, train_dataset=ds,
                data_collator=DataCollatorWithPadding(self.tok)).train()
        return self

    def predict_proba(self, pairs: list[UnitPair], units, batch_size: int = 64) -> np.ndarray:
        import torch
        self.model.eval()
        device = next(self.model.parameters()).device
        out = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                chunk = pairs[i:i + batch_size]
                enc = self.tok([unit_label(units[p.a]) for p in chunk], [unit_label(units[p.b]) for p in chunk],
                               truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
                out.append(torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy())
        return np.vstack(out)

    def save(self, path: Path) -> None:
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


# ----------------------------------------------------------------------------- mémoire prédite

NEVER_REPLACED = {"question", "answer"}          # une question ou une réponse ne devient pas obsolète
CAN_REPLACE = {"decision", "fact", "action", "commitment", "risk"}


def plausible_replacement(units, p: UnitPair) -> bool:
    a, b = units[p.a], units[p.b]
    if a.type in NEVER_REPLACED or b.type not in CAN_REPLACE:
        return False
    return not (a.type == "decision" and b.type == "idea")


def predicted_memory(scenarios, pairs: list[UnitPair], probs: np.ndarray, threshold: float, with_type: bool,
                     units=None) -> Memory:
    """Pour chaque unité A, l'unité B la plus probable qui la remplace (si au-dessus du seuil)."""
    mem = Memory.from_scenarios(scenarios, with_type=with_type)
    best: dict[str, tuple[float, str]] = {}
    for p, pr in zip(pairs, probs[:, SUP]):
        if units is not None and not plausible_replacement(units, p):
            continue
        if pr >= threshold and pr > best.get(p.a, (0.0, ""))[0]:
            best[p.a] = (float(pr), p.b)
    mem.successor = {a: b for a, (_, b) in best.items()}
    mem.inactive_override = set(mem.successor)
    return mem


def constrain(probs: np.ndarray, pairs: list[UnitPair], units) -> np.ndarray:
    """Met à zéro la probabilité de remplacement des paires de types incompatibles."""
    out = probs.copy()
    mask = np.array([not plausible_replacement(units, p) for p in pairs])
    out[mask, SUP] = 0.0
    return out


def best_threshold(pairs: list[UnitPair], probs: np.ndarray, beta: float) -> tuple[float, float]:
    y = np.array([p.label == "supersedes" for p in pairs])
    best = (0.5, -1.0)
    for t in np.arange(0.05, 0.96, 0.05):
        f = fbeta_score(y, probs[:, SUP] >= t, beta=beta, zero_division=0)
        if f > best[1]:
            best = (float(round(t, 2)), float(f))
    return best


def retrieval(scores, queries, mem, status: bool, project: bool) -> dict:
    s = apply_memory(scores, queries, mem, status=status, project=project) if (status or project) else scores
    return aggregate(per_query(s, queries, mem.corpus.index()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=("logreg", "cross"), required=True)
    ap.add_argument("--encoder", default="models/e5b-both-type-s1",
                    help="logreg : encodeur pour les embeddings (ou 'tfidf')")
    ap.add_argument("--cross-model", default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    ap.add_argument("--retriever", default="models/e5b-both-type-s1", help="modèle de recherche pour l'évaluation finale")
    ap.add_argument("--with-type", action="store_true", default=True)
    ap.add_argument("--train", default="data/scenarios/train")
    ap.add_argument("--val", default="data/scenarios/val")
    ap.add_argument("--test", default="data/scenarios/test")
    ap.add_argument("--unrelated-ratio", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beta", type=float, default=0.5,
                    help="seuil choisi sur la validation avec le F-beta (beta < 1 favorise la précision)")
    ap.add_argument("--no-type-rules", action="store_true", help="désactive la contrainte de types")
    args = ap.parse_args()

    name = f"relations-{args.method}" + (f"-{Path(args.encoder).name}" if args.method == "logreg" else "")
    out = Path("experiments") / f"{name}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    train_sc, val_sc, test_sc = (load_scenarios(p) for p in (args.train, args.val, args.test))
    units = {**all_units(train_sc), **all_units(val_sc), **all_units(test_sc)}
    train_pairs = labelled_pairs(train_sc, args.unrelated_ratio, args.seed)
    counts = {r: sum(p.label == r for p in train_pairs) for r in RELATIONS}
    print(f"entraînement : {len(train_pairs)} paires {counts}")

    clf = (LogRegClassifier(args.encoder, args.seed) if args.method == "logreg" else
           CrossEncoderClassifier(args.cross_model, args.epochs, args.lr, args.batch_size, args.seed, out))
    clf.fit(train_pairs, units)

    # seuil de « supersedes » choisi sur la validation, sur toutes les paires candidates
    rules = not args.no_type_rules
    val_all = all_pairs_with_labels(val_sc)
    p_val = clf.predict_proba(val_all, units)
    if rules:
        p_val = constrain(p_val, val_all, units)
    threshold, val_f1 = best_threshold(val_all, p_val, args.beta)
    print(f"seuil supersedes (validation, F{args.beta:g}) : {threshold} (score {val_f1:.3f})")

    # 1) classification sur le test : paires annotées + non reliées (même ratio qu'à l'entraînement)
    test_bal = labelled_pairs(test_sc, args.unrelated_ratio, args.seed + 1)
    p_bal = clf.predict_proba(test_bal, units)
    y_true = [LABEL2ID[p.label] for p in test_bal]
    y_pred = p_bal.argmax(1).tolist()
    present = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=present, target_names=[RELATIONS[i] for i in present],
                                   zero_division=0, output_dict=True)
    print("\nClassification sur le test (relations annotées + paires non reliées) :")
    print(classification_report(y_true, y_pred, labels=present, target_names=[RELATIONS[i] for i in present],
                                zero_division=0, digits=3))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(RELATIONS))))
    print("Matrice de confusion (lignes = vrai, colonnes = prédit) :")
    print("                 " + " ".join(f"{r[:10]:>10}" for r in RELATIONS))
    for r, row in zip(RELATIONS, cm):
        print(f"{r:>16} " + " ".join(f"{v:>10}" for v in row))

    # 2) détection des remplacements sur toutes les paires du test (cas réaliste)
    test_all = all_pairs_with_labels(test_sc)
    p_all = clf.predict_proba(test_all, units)
    if rules:
        p_all = constrain(p_all, test_all, units)
    y_sup = np.array([p.label == "supersedes" for p in test_all])
    pr, rc, f1, _ = precision_recall_fscore_support(y_sup, p_all[:, SUP] >= threshold, average="binary", zero_division=0)
    print(f"\nRemplacements sur toutes les paires du test ({len(test_all)} paires, {int(y_sup.sum())} vrais) : "
          f"précision {pr:.3f} · rappel {rc:.3f} · F1 {f1:.3f}")

    # 3) recherche avec la mémoire prédite
    gold_mem = Memory.from_scenarios(test_sc, with_type=args.with_type)
    pred_mem = predicted_memory(test_sc, test_all, p_all, threshold, args.with_type,
                                units=units if rules else None)
    print(f"remplacements prédits : {len(pred_mem.successor)}")
    queries = build_queries(test_sc)
    if args.retriever == "bm25":
        from sae.retrieval.bm25 import BM25
        retriever = BM25()
    else:
        from sae.retrieval.dense import DenseRetriever, resolve_spec
        retriever = DenseRetriever(resolve_spec(args.retriever))
    scores = retriever.fit(gold_mem.corpus.texts).score([q.text for q in queries])
    table = {
        "sans mémoire": retrieval(scores, queries, gold_mem, False, False),
        "mémoire prédite + projet": retrieval(scores, queries, pred_mem, True, True),
        "mémoire annotée + projet (borne haute)": retrieval(scores, queries, gold_mem, True, True),
    }
    cols = ("recall@1", "mrr@10", "ndcg@10", "hn_acc")
    print(f"\nRecherche sur le test ({args.retriever}) :\n")
    print("| mémoire | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))
    for k, v in table.items():
        print(f"| {k} | " + " | ".join(f"{v[c]:.3f}" for c in cols) + " |")

    if args.method == "cross":
        clf.save(Path("models") / "relations-cross")
    (out / "results.json").write_text(json.dumps({
        "config": vars(args), "train_counts": counts, "threshold": threshold, "val_supersedes_fbeta": val_f1,
        "test_classification": report, "test_confusion": cm.tolist(), "labels": list(RELATIONS),
        "test_supersedes_all_pairs": {"precision": pr, "recall": rc, "f1": f1, "n_pairs": len(test_all)},
        "retrieval": table,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nRésultats : {out}/results.json")


if __name__ == "__main__":
    main()

"""Reranking par cross-encodeur (BAAI/bge-reranker-v2-m3, sans fine-tuning) des K premiers candidats denses.

Le cross-encodeur lit la question et le texte de l'unité ensemble et donne un score de pertinence. Seuls les K
premiers candidats de la recherche dense sont reclassés : ils passent devant toutes les autres unités, qui gardent
leur ordre dense. La matrice obtenue se donne ensuite telle quelle à apply_memory (statut, projet).
"""
from __future__ import annotations

import numpy as np

RERANKER = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    def __init__(self, path: str = RERANKER, device: str | None = None, batch_size: int = 32,
                 max_length: int = 256) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size, self.max_length = batch_size, max_length
        self.tok = AutoTokenizer.from_pretrained(path)
        # fp32 imposé : un modèle stocké en fp16 et utilisé tel quel donnait des scores dégradés dans le projet
        self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32)
        self.model = self.model.float().eval().to(self.device)
        dtypes = {p.dtype for p in self.model.parameters()}
        if dtypes != {torch.float32}:
            raise RuntimeError(f"poids du reranker hors fp32 : {dtypes}")

    def pair_scores(self, queries: list[str], docs: list[str]) -> np.ndarray:
        """Score (logit) de chaque couple (question, texte), dans l'ordre donné."""
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(queries), self.batch_size):
                enc = self.tok(queries[i:i + self.batch_size], docs[i:i + self.batch_size], truncation=True,
                               max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                out.append(self.model(**enc).logits.float().view(-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def rerank(self, scores: np.ndarray, queries: list[str], docs: list[str], k: int) -> np.ndarray:
        """Reclasse les k premiers candidats de chaque ligne ; les autres restent dessous, dans l'ordre dense."""
        k = min(k, scores.shape[1])
        top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        pairs = [(qi, int(d)) for qi in range(len(queries)) for d in top[qi]]
        ce = self.pair_scores([queries[qi] for qi, _ in pairs], [docs[d] for _, d in pairs]).reshape(len(queries), k)
        return place_on_top(scores, top, ce)


def place_on_top(scores: np.ndarray, top: np.ndarray, ce: np.ndarray) -> np.ndarray:
    """Matrice où les candidats reclassés (scores ce) dépassent toutes les autres unités (scores denses)."""
    out = scores.astype(np.float64, copy=True)
    rows = np.arange(len(scores))[:, None]
    floor = out.max(axis=1, keepdims=True) + 1.0
    out[rows, top] = floor + (ce - ce.min(axis=1, keepdims=True))
    return out

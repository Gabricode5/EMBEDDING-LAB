"""Retriever dense (bi-encodeur Sentence Transformers), cosinus sur embeddings normalisés."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    path: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 256


PRESETS: dict[str, EncoderSpec] = {
    "e5-small": EncoderSpec("e5-small", "intfloat/multilingual-e5-small", "query: ", "passage: "),
    "e5-base": EncoderSpec("e5-base", "intfloat/multilingual-e5-base", "query: ", "passage: "),
    "bge-m3": EncoderSpec("bge-m3", "BAAI/bge-m3", "", "", 512),
}


def resolve_spec(name_or_path: str) -> EncoderSpec:
    """Preset connu, sinon chemin local d'un modèle fine-tuné (préfixes e5 par défaut)."""
    if name_or_path in PRESETS:
        return PRESETS[name_or_path]
    return EncoderSpec(name_or_path, name_or_path, "query: ", "passage: ")


class DenseRetriever:
    def __init__(self, spec: EncoderSpec, device: str | None = None, batch_size: int = 64) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(spec.path, device=self.device)
        self.model.max_seq_length = spec.max_seq_length
        if self.device == "cuda":
            # bf16 sur RTX 50xx, fp16 sur T4 (Colab) qui ne gère pas le bf16.
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.model.to(dtype)
        self._doc_emb: np.ndarray | None = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return emb.astype(np.float32)

    def fit(self, docs: list[str]) -> "DenseRetriever":
        self._doc_emb = self._encode([self.spec.passage_prefix + d for d in docs])
        return self

    def score(self, queries: list[str]) -> np.ndarray:
        if self._doc_emb is None:
            raise RuntimeError("appeler fit() avant score()")
        q = self._encode([self.spec.query_prefix + t for t in queries])
        return q @ self._doc_emb.T

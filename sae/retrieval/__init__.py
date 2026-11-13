"""Méthodes de recherche : BM25 (bm25.py), encodeurs denses (dense.py), reranking (rerank.py)."""
from __future__ import annotations


def make_retriever(name: str, language: str = "french"):
    """BM25 (« bm25 », racinisation dans la langue donnée) ou encodeur dense (preset ou chemin d'un modèle)."""
    if name == "bm25":
        from sae.retrieval.bm25 import BM25
        return BM25(language=language)
    from sae.retrieval.dense import DenseRetriever, resolve_spec
    return DenseRetriever(resolve_spec(name))

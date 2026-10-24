"""BM25 (Okapi) en numpy, avec racinisation française.

Implémentation maison pour disposer de la matrice de scores complète
(nécessaire à la hard-negative accuracy) sans dépendre d'une API externe.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import Stemmer

_TOKEN = re.compile(r"\w+", re.UNICODE)

# Négations ("ne", "pas", "non", "aucun", "jamais", "sans") volontairement conservées.
FRENCH_STOPWORDS = frozenset("""
a à au aux avec ce ces c cette d dans de des du elle en et eux il ils je j l la le les leur
leurs lui m ma mais me même mes moi mon n nos notre nous on ou par pour qu que qui s sa se ses
son sur t ta te tes toi ton tu un une vos votre vous y été être est sont était ont avoir as ai
""".split())


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75, language: str = "french") -> None:
        self.k1, self.b = k1, b
        self._stem = Stemmer.Stemmer(language).stemWords
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}
        self._norm: np.ndarray | None = None
        self._n_docs = 0

    def tokenize(self, text: str) -> list[str]:
        tokens = [t for t in _TOKEN.findall(text.lower()) if t not in FRENCH_STOPWORDS]
        return self._stem(tokens)

    def fit(self, docs: list[str]) -> "BM25":
        tokenized = [Counter(self.tokenize(d)) for d in docs]
        self._n_docs = len(docs)
        lengths = np.array([sum(c.values()) for c in tokenized], dtype=np.float32)
        avgdl = float(lengths.mean()) if self._n_docs else 1.0
        self._norm = self.k1 * (1.0 - self.b + self.b * lengths / max(avgdl, 1e-9))

        buckets: dict[str, tuple[list[int], list[int]]] = {}
        for i, counts in enumerate(tokenized):
            for term, tf in counts.items():
                docs_i, tfs = buckets.setdefault(term, ([], []))
                docs_i.append(i)
                tfs.append(tf)
        self._postings = {t: (np.array(d), np.array(f, dtype=np.float32)) for t, (d, f) in buckets.items()}
        self._idf = {
            t: math.log(1.0 + (self._n_docs - len(d) + 0.5) / (len(d) + 0.5))
            for t, (d, _) in self._postings.items()
        }
        return self

    def score(self, queries: list[str]) -> np.ndarray:
        if self._norm is None:
            raise RuntimeError("appeler fit() avant score()")
        scores = np.zeros((len(queries), self._n_docs), dtype=np.float32)
        for qi, q in enumerate(queries):
            for term in set(self.tokenize(q)):
                post = self._postings.get(term)
                if post is None:
                    continue
                docs, tf = post
                scores[qi, docs] += self._idf[term] * tf * (self.k1 + 1.0) / (tf + self._norm[docs])
        return scores

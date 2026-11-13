"""BM25 et métriques sur de petits exemples dont le résultat se calcule à la main."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sae.data import EvalQuery
from sae.eval.metrics import aggregate, per_query, per_query_docs
from sae.retrieval import make_retriever
from sae.retrieval.bm25 import BM25

DOCS = ["Le budget du projet est validé.", "La réunion est reportée à lundi.", "Il ne faut pas oublier les captures."]


def test_bm25_tokenisation_francaise():
    # mots vides retirés (« il », « les »), racinisation, négations « ne » et « pas » conservées
    assert BM25().tokenize("Il ne faut pas oublier les captures") == ["ne", "faut", "pas", "oubli", "captur"]


def test_bm25_classement():
    scores = BM25().fit(DOCS).score(["budget validé", "réunion lundi", "football"])
    assert scores.shape == (3, 3)
    assert np.argmax(scores[0]) == 0 and np.argmax(scores[1]) == 1
    assert scores[0, 1] == scores[0, 2] == 0.0          # aucun mot en commun
    assert not scores[2].any()                          # mot absent du corpus


def test_bm25_anglais():
    bm25 = make_retriever("bm25", language="english")
    assert bm25.tokenize("We're not going to use the buttons") == ["re", "not", "go", "use", "button"]


def query(relevant, hard=()):
    return EvalQuery(qid="p/q", text="", relevant=tuple(relevant), hard_negatives=tuple(hard), answerable=True,
                     origin="manual", tags=())


def test_metriques_rang_2():
    # la bonne unité (u1) arrive deuxième, derrière le négatif difficile (u0)
    scores = np.array([[0.9, 0.8, 0.1, 0.0]])
    index = {f"u{i}": i for i in range(4)}
    row = per_query(scores, [query(["u1"], hard=["u0"])], index)[0]
    assert row["recall@1"] == 0.0 and row["recall@5"] == 1.0
    assert row["mrr@10"] == 0.5
    assert row["ndcg@10"] == pytest.approx(1 / math.log2(3))
    assert aggregate([row])["hn_acc"] == 0.0


def test_per_query_docs_equivaut_a_per_query():
    # avec un document par unité, per_query_docs redonne exactement per_query
    rng = np.random.default_rng(0)
    scores = rng.random((1, 6))
    q = query(["u2", "u4"], hard=["u1"])
    a = per_query(scores, [q], {f"u{i}": i for i in range(6)})[0]
    b = per_query_docs(scores, [q], [frozenset({f"u{i}"}) for i in range(6)])[0]
    for k in ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10", "hn_ok"):
        assert a[k] == pytest.approx(b[k])

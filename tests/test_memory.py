"""Mémoire : chaîne de remplacements, redirection vers la version en vigueur, déclassement, filtre projet."""
from __future__ import annotations

import numpy as np

from sae.data import EvalQuery
from sae.memory import Memory, apply_memory


def query(intent="answer", project="atlas"):
    return EvalQuery(qid=f"{project}/q", text="", relevant=(), hard_negatives=(), answerable=True, origin="manual",
                     tags=(), intent=intent)


def test_chaine_de_remplacements(atlas):
    m = Memory.from_scenarios([atlas])
    # proposition du 15 novembre -> date fixée au 15 novembre -> report au 22 novembre
    assert m.successor["atlas/u-002"] == "atlas/u-007"
    assert m.current("atlas/u-002") == "atlas/u-011"
    assert m.current("atlas/u-011") == "atlas/u-011"


def test_redirection_et_declassement(atlas):
    m = Memory.from_scenarios([atlas])
    idx = m.corpus.index()
    scores = np.zeros((1, len(m.corpus.ids)))
    scores[0, idx["atlas/u-002"]] = 1.0          # l'ancienne proposition sort en premier
    out = apply_memory(scores, [query()], m, status=True)[0]
    assert m.corpus.ids[int(np.argmax(out))] == "atlas/u-011"
    inactive = m.inactive_mask()
    assert out[inactive].max() < out[~inactive].min()   # les unités remplacées passent sous toutes les autres


def test_comparaison_garde_les_anciennes_versions(atlas):
    m = Memory.from_scenarios([atlas])
    scores = np.random.default_rng(0).random((1, len(m.corpus.ids)))
    out = apply_memory(scores, [query(intent="compare")], m, status=True)
    assert np.array_equal(out, scores)


def test_filtre_projet(atlas):
    other = atlas.model_copy(update={"project_id": "autre"})
    m = Memory.from_scenarios([atlas, other])
    scores = np.random.default_rng(1).random((1, len(m.corpus.ids)))
    out = apply_memory(scores, [query(project="atlas")], m, status=False, project=True)[0]
    projects = m.project_array()
    assert out[projects == "autre"].max() < out[projects == "atlas"].min()

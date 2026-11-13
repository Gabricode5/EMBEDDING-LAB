"""Agent de bout en bout sur Atlas, avec BM25 et les relations annotées (sans GPU ni Ollama)."""
from __future__ import annotations

import pytest

from sae.agent.engine import Agent
from sae.memory import Memory
from sae.retrieval.bm25 import BM25
from scripts.agent import DEMO

EXPECTED = ["answer", "answer", "compare", "dossier", "email_draft", "clarify", "no_action"]


@pytest.fixture(scope="module")
def agent():
    from sae.data import load_scenarios
    scenarios = load_scenarios("data/scenarios/train")
    return Agent(scenarios, Memory.from_scenarios(scenarios, with_type=True), BM25(), with_type=True)


@pytest.mark.parametrize("case, kind", list(zip(DEMO["atlas"], EXPECTED)), ids=[label for label, _ in DEMO["atlas"]])
def test_demo_atlas(agent, case, kind):
    _, message = case
    assert agent.handle(message, project="atlas").kind == kind


def test_decision_en_vigueur(agent):
    r = agent.handle(DEMO["atlas"][0][1], project="atlas")
    assert r.sources[0].unit_id == "atlas/u-011"          # report au 22 novembre, pas la date du 15
    assert r.sources[0].status == "active"


def test_mail_au_responsable(agent):
    r = agent.handle(DEMO["atlas"][4][1], project="atlas")
    rec = r.payload["recipient"]
    assert rec["name"] == "Karim Benali"                   # responsable annoté du suivi du budget
    assert rec["email"] and rec["email"].endswith("@example.org")
    assert "non envoyé" in r.text

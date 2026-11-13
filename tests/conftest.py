"""Fixtures communes : données lues dans data/scenarios (jamais modifiées), sans GPU ni Ollama."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # pour importer scripts.agent (DEMO)

ATLAS = ROOT / "data/scenarios/train/atlas.json"


@pytest.fixture
def atlas_dict() -> dict:
    """Le projet Atlas (écrit à la main) sous forme de dictionnaire, à modifier librement dans un test."""
    return json.loads(ATLAS.read_text(encoding="utf-8"))


@pytest.fixture
def atlas(atlas_dict):
    from sae.schemas import Scenario
    return Scenario.model_validate(atlas_dict)

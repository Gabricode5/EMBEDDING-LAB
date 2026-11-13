"""Validation des scénarios : un projet valide passe, une incohérence est refusée avec un message clair."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from sae.schemas import Scenario


def test_scenario_valide(atlas_dict):
    s = Scenario.model_validate(atlas_dict)
    assert s.project_id == "atlas"
    assert len(s.units) == sum(len(m.units) for m in s.meetings)


def test_unite_remplacee_encore_active(atlas_dict):
    # u-003 (budget proposé) est remplacée par u-006 : la remettre « active » doit être refusé
    for m in atlas_dict["meetings"]:
        for u in m["units"]:
            if u["unit_id"] == "u-003":
                u["status"] = "active"
    with pytest.raises(ValidationError, match="u-003 est remplacé mais son statut est 'active'"):
        Scenario.model_validate(atlas_dict)


def test_relation_vers_unite_inconnue(atlas_dict):
    atlas_dict["relations"][0]["to_id"] = "u-999"
    with pytest.raises(ValidationError, match="unité inconnue"):
        Scenario.model_validate(atlas_dict)

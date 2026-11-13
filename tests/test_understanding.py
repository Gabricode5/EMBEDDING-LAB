"""Règles d'intention, salutations et garde-fou des hypothèses."""
from __future__ import annotations

import pytest

from sae.agent.understanding import detect_intent, is_hypothetical


@pytest.mark.parametrize("message, intent", [
    ("Rédige un mail à Karim pour lui demander le budget.", "draft_email"),
    ("Crée une tâche pour Marie : envoyer le planning.", "create_task"),
    ("Planifie une réunion lundi avec l'équipe.", "schedule_event"),
    ("Prépare un dossier complet sur le projet.", "build_dossier"),
    ("Qu'est-ce qui a changé pour la date de lancement ?", "compare"),
    ("Quelle date de lancement a finalement été retenue ?", "answer"),
    ("bonjour", "help"),
    ("Merci beaucoup !", "help"),
    ("Bonjour, prépare un dossier complet.", "build_dossier"),
])
def test_intentions(message, intent):
    assert detect_intent(message)[0] == intent


@pytest.mark.parametrize("message, hypothetique", [
    ("On pourrait peut-être revoir le budget.", True),
    ("Il faudrait peut-être décaler la réunion", True),
    ("Est-ce qu'on pourrait avoir le budget validé ?", False),
    ("Le budget a-t-il été validé ?", False),
])
def test_garde_fou_hypotheses(message, hypothetique):
    assert is_hypothetical(message) is hypothetique

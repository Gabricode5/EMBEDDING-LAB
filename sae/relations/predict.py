"""Prédiction des remplacements avec un classifieur de relations entraîné, et mémoire associée."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sae.memory import Memory
from sae.relations.pairs import LABEL2ID, UnitPair, all_pairs_with_labels, all_units, unit_label

SUP = LABEL2ID["supersedes"]
NEVER_REPLACED = {"question", "answer"}          # une question ou une réponse ne devient pas obsolète
CAN_REPLACE = {"decision", "fact", "action", "commitment", "risk"}


def plausible_replacement(units, p: UnitPair) -> bool:
    a, b = units[p.a], units[p.b]
    if a.type in NEVER_REPLACED or b.type not in CAN_REPLACE:
        return False
    return not (a.type == "decision" and b.type == "idea")


def predicted_memory(memory: Memory, pairs: list[UnitPair], probs: np.ndarray, threshold: float,
                     units=None) -> Memory:
    """Remplit la mémoire avec, pour chaque unité, celle qui la remplace le plus probablement."""
    best: dict[str, tuple[float, str]] = {}
    for p, pr in zip(pairs, probs[:, SUP]):
        if units is not None and not plausible_replacement(units, p):
            continue
        if pr >= threshold and pr > best.get(p.a, (0.0, ""))[0]:
            best[p.a] = (float(pr), p.b)
    memory.successor = {a: b for a, (_, b) in best.items()}
    memory.inactive_override = set(memory.successor)
    return memory


class CrossEncoderRelations:
    """Charge le cross-encodeur entraîné par scripts/train_relations.py (models/relations-cross)."""

    def __init__(self, path: str | Path = "models/relations-cross", threshold: float = 0.45) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.threshold = threshold
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).float().eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict_proba(self, pairs: list[UnitPair], units, batch_size: int = 64) -> np.ndarray:
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                chunk = pairs[i:i + batch_size]
                enc = self.tok([unit_label(units[p.a]) for p in chunk], [unit_label(units[p.b]) for p in chunk],
                               truncation=True, max_length=256, padding=True, return_tensors="pt").to(self.device)
                out.append(torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, len(LABEL2ID)))

    def build_memory(self, scenarios, memory: Memory) -> Memory:
        units = all_units(scenarios)
        pairs = all_pairs_with_labels(scenarios)
        return predicted_memory(memory, pairs, self.predict_proba(pairs, units), self.threshold, units)

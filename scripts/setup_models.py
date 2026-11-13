"""Recrée les deux modèles entraînés dont l'agent a besoin, s'ils sont absents de models/.

    python scripts/setup_models.py            # entraîne seulement ce qui manque
    python scripts/setup_models.py --force    # réentraîne les deux

Les modèles ne sont pas versionnés (trop lourds). Ce script relance les deux entraînements avec exactement les
réglages des modèles retenus :
- models/e5b-both-type-s1 : e5-base fine-tuné, négatifs explicites et minés, type dans le texte, graine 1 ;
- models/relations-cross : cross-encodeur mDeBERTa, réglages par défaut de train_relations.py (graine 42).
Il faut un GPU (quelques minutes en tout). Les modèles de base sont téléchargés depuis Hugging Face au premier
lancement.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("models/e5b-both-type-s1", ["scripts/train_embedder.py", "--negatives", "both", "--with-type",
                                 "--name", "e5b-both-type-s1", "--seed", "1"]),
    # après l'encodeur : train_relations.py l'utilise pour son évaluation finale
    ("models/relations-cross", ["scripts/train_relations.py", "--method", "cross"]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Recrée models/e5b-both-type-s1 et models/relations-cross s'ils manquent.")
    ap.add_argument("--force", action="store_true", help="réentraîne même si le modèle existe")
    args = ap.parse_args()

    for target, command in STEPS:
        path = ROOT / target
        if (path / "config.json").exists() and not args.force:
            print(f"{target} : déjà présent")
            continue
        if path.exists():
            shutil.rmtree(path)
        print(f"{target} : entraînement ({' '.join(command)})", flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
        if not (path / "config.json").exists():
            sys.exit(f"{target} n'a pas été créé")
        print(f"{target} : prêt", flush=True)


if __name__ == "__main__":
    main()

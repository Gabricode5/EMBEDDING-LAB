# Embedding Lab : moteur sémantique pour un agent

Projet réalisé dans le cadre de la piscine Intelligence Lab (ECE, du 16 octobre au 16 novembre 2026), sur le sujet « De la mémoire sémantique à l'agent intelligent ».

L'idée est d'entraîner un modèle d'embeddings qui ne se contente pas de retrouver des phrases sur le même sujet, mais qui fait la différence entre une proposition, une décision validée, une décision remplacée ou une simple hypothèse. Ce modèle sert ensuite de base à un petit agent qui répond à des questions sur des réunions en citant ses sources, et qui peut préparer des actions (brouillon de mail, tâche) sans jamais les exécuter.

Le fonctionnement détaillé du système est expliqué dans [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Avancement (25 octobre)

Fait :
- format des données (unités de connaissance, relations, questions de test), validé avec Pydantic
- corpus de 12 projets fictifs : 7 pour l'entraînement, 2 pour la validation, 3 pour le test
- script d'évaluation avec BM25 et e5-base (Recall@K, MRR, nDCG, précision sur les négatifs difficiles)
- premiers résultats sur un projet, et analyse des erreurs

En cours :
- écriture de questions de test à la main
- mesure des baselines (BM25, e5-base, bge-m3) sur le jeu de test

À venir :
- semaine 2 : fine-tuning de e5-base par apprentissage contrastif
- semaine 3 : classification des relations, mémoire et agent
- semaine 4 : évaluation complète, rapport et démo

## Organisation du dépôt

```
EMBEDDING-LAB/
├── sae/                        code principal (package Python « Semantic Agent Engine »)
│   ├── schemas.py              format des données et des sorties de l'agent
│   ├── data.py                 lecture des projets, préparation du corpus et des questions
│   ├── retrieval/              méthodes de recherche
│   │   ├── bm25.py             recherche par mots-clés
│   │   └── dense.py            recherche par embeddings (e5, bge-m3, modèles fine-tunés)
│   └── eval/
│       └── metrics.py          calcul des métriques
├── scripts/                    commandes à lancer dans le terminal
│   ├── generate_scenarios.py   génère des projets fictifs
│   ├── validate_data.py        vérifie la forme des données
│   ├── review_scenarios.py     signale les incohérences à relire
│   └── run_baselines.py        évalue les modèles sur un split
├── data/scenarios/             les données : un fichier JSON par projet
│   ├── train/                  projets pour l'entraînement
│   ├── val/                    projets pour régler l'entraînement
│   └── test/                   projets pour l'évaluation finale
├── docs/                       documentation (architecture)
├── experiments/                résultats des évaluations (non versionnés)
├── .env.example                modèle du fichier de clés d'API
├── requirements.txt            bibliothèques à installer
└── pyproject.toml              permet d'importer `sae` depuis les scripts
```

Le code réutilisable est dans `sae/`, et les scripts de `scripts/` l'utilisent. Par exemple, `run_baselines.py` charge les données avec `sae/data.py`, lance la recherche avec `sae/retrieval/` et calcule les scores avec `sae/eval/metrics.py`.

Les dossiers suivants seront ajoutés au fil du projet :
- `sae/training/` : fine-tuning de l'encodeur (semaine 2)
- `sae/relations/` et `sae/agent/` : relations entre unités et agent (semaine 3)
- `app/` : interface de démonstration (semaine 3)
- `tests/` : tests automatiques (semaine 4)

## Installation

Il faut Python 3.10 ou plus. Sur une carte NVIDIA récente (série RTX 50), PyTorch doit être installé dans sa version CUDA 12.8 :

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

Pour vérifier que le GPU est bien détecté :

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Sur Google Colab, PyTorch est déjà installé : seules les deux dernières lignes de `pip` sont nécessaires.

Les clés d'API (facultatives) se mettent dans un fichier `.env`, à créer à partir de `.env.example`. Ce fichier n'est jamais commité.

## Données

Je n'ai pas trouvé de jeu de données public annoté avec des décisions, leur statut et les responsables. Les corpus de réunions existants (AMI, QMSum, MeetingBank) ne contiennent pas ces informations. J'ai donc créé des projets fictifs.

Chaque fichier `data/scenarios/<split>/<projet>.json` décrit un projet complet :
- les personnes (noms et adresses fictifs, domaine example.org)
- les réunions, avec leur date
- les unités de connaissance : une décision, un engagement, un risque… avec son statut (en vigueur, proposée, hypothétique, remplacée), la personne qui l'a dite, le responsable et la phrase exacte prononcée
- les relations entre unités (une décision en remplace une autre, une réponse répond à une question…)
- des questions de test avec les réponses attendues et des « négatifs difficiles », c'est-à-dire des unités sur le même sujet qui ne sont pas la bonne réponse

Le projet Atlas a été écrit à la main et sert d'exemple. Les projets d'entraînement et de validation ont été générés en local avec qwen3:8b (via Ollama), et ceux de test avec Claude, pour que le test ne vienne pas du même modèle que l'entraînement. Tous les projets ont été relus et corrigés à la main.

Les splits sont faits par projet : un projet est soit en entraînement, soit en validation, soit en test. Le modèle n'est donc jamais évalué sur un projet qu'il a vu pendant l'entraînement.

Commandes utilisées pour générer les données :

```bash
# entraînement et validation : modèle local avec Ollama
python scripts/generate_scenarios.py --n 5 --out data/scenarios/train --model qwen3:8b --theme-offset 1
python scripts/generate_scenarios.py --n 2 --out data/scenarios/val --model qwen3:8b --theme-offset 7

# test : copier-coller avec Claude.ai (le script prépare le prompt et vérifie la réponse)
python scripts/generate_scenarios.py --n 3 --out data/scenarios/test --provider manual \
    --theme-offset 9 --style "ton très oral, phrases courtes, hésitations"
```

Pour vérifier les données :

```bash
python scripts/validate_data.py        # forme : schéma, dates, identifiants, fuites entre splits
python scripts/review_scenarios.py     # fond : incohérences probables à relire
```

## Évaluation

```bash
python scripts/run_baselines.py --split data/scenarios/test --models bm25,e5-base,bge-m3
python scripts/run_baselines.py --split data/scenarios/test --models bm25,e5-base --with-type
```

L'option `--with-type` ajoute le type de l'unité devant son texte (par exemple « risque : … ») pour voir si ça aide la recherche.

Métriques calculées :
- Recall@1, @5, @10 : la bonne réponse est-elle dans les premiers résultats ?
- MRR@10 et nDCG@10 : à quel rang arrive la bonne réponse ?
- hn_acc : la bonne réponse est-elle classée devant les négatifs difficiles ?

Les résultats sont aussi séparés selon l'origine des questions (générées ou écrites à la main) et le type de difficulté. Chaque lancement crée un dossier dans `experiments/` avec les scores et la liste des questions ratées.

## Premiers résultats

Sur le seul projet Atlas (18 unités, 9 questions), BM25 fait mieux que e5-base sans entraînement :

| Modèle | Recall@1 | Recall@5 | MRR@10 | hn_acc |
| --- | --- | --- | --- | --- |
| BM25 | 0,500 | 1,000 | 0,744 | 0,800 |
| e5-base | 0,333 | 0,889 | 0,581 | 0,600 |

C'est beaucoup trop peu de questions pour conclure : ces chiffres servaient surtout à tester le code. Les résultats sur le jeu de test complet viendront remplacer ce tableau.

## Matériel utilisé

- portable avec une RTX 5070 Laptop (8 Go de VRAM) et 32 Go de RAM
- Google Colab gratuit en complément

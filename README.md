# Semantic Agent Engine : de la mémoire sémantique à l'agent

Projet réalisé pendant la piscine Intelligence Lab de l'ECE, du 16 octobre au 16 novembre 2026, sur le sujet
« De la mémoire sémantique à l'agent intelligent ».

Je voulais un moteur de recherche qui ne retrouve pas seulement des phrases sur le même sujet, mais qui fait
la différence entre une proposition, une décision validée, une décision remplacée et une simple hypothèse.
Par-dessus, j'ai construit un agent qui répond à des questions sur des réunions de projet en citant ses
sources. Il prépare aussi des dossiers et des brouillons de mail, de tâche ou d'événement, qu'il n'envoie
jamais.

Le projet est terminé. Tout ce qui est décrit ici est codé et évalué, sauf le reranking, que j'ai mesuré mais
pas intégré à l'agent. Le fonctionnement est expliqué dans [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) et les
données dans [docs/DATA.md](docs/DATA.md). Le rapport complet, avec la démarche détaillée et tous les
résultats, est dans [docs/RAPPORT.pdf](docs/RAPPORT.pdf).

## Démarche

Le détail de chaque étape est dans le [rapport](docs/RAPPORT.pdf). Je raconte ici les étapes dans l'ordre où
je les ai faites, avec ce qui n'a pas marché. Les dates viennent de l'historique git. Les chiffres viennent de
`docs/` et `experiments/`. Sauf mention contraire, la métrique est le MRR@10 : il vaut 1 quand la bonne unité
est classée première, 0,5 quand elle est deuxième, etc.

### Données (16 au 25 octobre)

Je n'ai pas trouvé de corpus public de réunions où les décisions sont annotées avec leur statut et leur
responsable. AMI ou QMSum ont des transcriptions et des résumés, mais pas cette information. J'ai donc fabriqué
des projets fictifs. Le premier, Atlas, a été écrit par un assistant d'IA (Claude) comme scénario exemple,
puis relu et corrigé par moi pour fixer le format. Il sert ensuite d'exemple dans le prompt de génération.
Les projets d'entraînement et de validation ont été générés en local avec qwen3:8b.

Pour le test, je voulais un autre générateur que pour l'entraînement, sinon le modèle aurait pu apprendre le
style de qwen plutôt que la tâche. J'ai d'abord essayé llama3.1:8b. Les trois projets obtenus passaient le
schéma, mais je les ai trouvés trop pauvres en les relisant : 12 à 20 unités par projet, et un oral qui
ressemblait à de l'écrit. Je les ai mis de côté (`data/rejected/`, hors du dépôt). J'ai refait le test avec
Claude, par copier-coller, ce qui a donné 3 projets de 23 unités en moyenne, avec un oral plus naturel. J'ai
aussi ajouté 33 questions dont les idées venaient en partie de suggestions d'une IA, que j'ai choisies ou
adaptées, puis reformulées par Claude dans un style plus oral ou plus poli. J'ai vérifié chaque réponse
attendue.

La validation par le schéma ne voyait pas les erreurs de sens. J'ai donc écrit `review_scenarios.py`, qui
signale les cas suspects, et j'ai relu chaque projet. Deux erreurs revenaient tout le temps dans les projets
générés : des engagements sans responsable, et des relations `creates_action` écrites dans le mauvais sens.

### Premières mesures de recherche

Sur Atlas seul (9 questions), BM25 faisait mieux qu'e5-base sans entraînement : 0,744 contre 0,581. Avec
aussi peu de questions, ça servait surtout à tester le code. Sur le vrai test (63 questions avec réponse),
j'ai essayé d'ajouter le type de l'unité devant son texte (« Décision : … »). e5-base passe de 0,717 à 0,741,
et BM25 ne bouge pas (0,778). bge-m3, plus gros, fait 0,807. Au départ, BM25 était donc à peine derrière
bge-m3 et devant e5-base.

### Fine-tuning de l'encodeur

J'ai entraîné e5-base de façon contrastive, avec deux sortes de paires : question → unité pertinente, et
phrase dite à l'oral → résumé de l'unité. J'ai comparé plusieurs façons de choisir les négatifs, avec 5
graines par configuration, parce qu'avec si peu de données je me méfiais du hasard. Un entraînement ne prend
que 11 secondes (195 paires), donc je pouvais me le permettre.

| configuration (moyenne sur 5 graines) | val | test |
|---|---|---|
| e5-base, sans négatifs | 0,761 | 0,779 |
| e5-base, négatifs annotés | 0,770 | 0,782 |
| e5-base, négatifs annotés et minés | 0,843 | 0,794 |
| e5-base, négatifs annotés et minés, type dans le texte | 0,854 | 0,834 |
| e5-small, négatifs annotés et minés | 0,825 | 0,807 |
| e5-small, négatifs annotés et minés, type dans le texte | 0,818 | 0,829 |

La meilleure configuration est la même sur val et sur test, mais val ne contient que 20 questions et elle
s'est trompée sur un point. Pour e5-small, elle disait que le type dans le texte faisait baisser le score
(0,825 → 0,818), alors que sur test il le fait monter (0,807 → 0,829). C'est la première fois que j'ai vu que
ma validation était trop petite pour trancher des écarts de 0,01. J'ai gardé la graine 1 sans regarder le
test : ce n'est d'ailleurs pas la meilleure (0,826 contre 0,850 pour la graine 2), mais je voulais un seul
modèle par configuration.

### Mémoire et relations

L'encodeur ne peut pas savoir qu'une décision a été remplacée : l'ancienne et la nouvelle parlent du même
sujet. J'ai donc ajouté une mémoire qui connaît les remplacements. Une unité remplacée transmet son score à la
version en vigueur et passe sous toutes les autres. Avec les relations annotées, ça fait passer le test de
0,826 à 0,933. J'ai aussi essayé une recherche hybride (BM25 et dense fusionnés). Elle donnait 0,940 sur test
mais 0,910 sur test2, contre 0,931 sans elle, donc je ne l'ai pas gardée.

Dans la réalité, les relations ne sont pas annotées, il faut les prédire. J'ai commencé par une régression
logistique sur les embeddings des deux unités. Avec un seuil réglé sur le F1, elle prédisait beaucoup de
fausses relations (précision 0,35), et le résultat m'a surpris : la recherche devenait moins bonne qu'avec
aucune mémoire (0,766 contre 0,826). Une fausse relation fait disparaître une unité valide, ce qui coûte plus
cher qu'un remplacement raté. J'ai donc choisi le seuil avec un F0,5, qui favorise la précision. Avec ce seuil
plus prudent, la régression logistique remonte à 0,875.

Ensuite j'ai fine-tuné un cross-encodeur mDeBERTa pour classer la relation entre deux unités. Les deux premiers
entraînements n'apprenaient rien : le modèle ne prédisait jamais de remplacement (précision et rappel à 0).
Après avoir cherché, j'ai compris que les poids publiés étaient en fp16 et que DeBERTa diverge en précision
réduite. En convertissant le modèle en fp32 avant l'entraînement, j'obtiens une précision de 0,750 et un rappel
de 0,429 sur les remplacements de test, et 0,883 pour la recherche avec la mémoire prédite. Les graines 1 et 2
donnent 0,875 et 0,859. Le rappel reste faible : plus de la moitié des remplacements sont ratés, et c'est
le point faible du système.

### Un deuxième jeu de test

Comme tous mes projets se ressemblent (un budget proposé puis validé, une date reportée), j'avais peur que les
résultats ne tiennent qu'à ce style. J'ai créé test2 : deux autres projets, jamais vus, écrits dans un autre
style (une boutique et un CRM d'agence immobilière), générés avec Claude comme pour le test, avec la consigne
« ton familier, phrases courtes, beaucoup de tutoiement ». L'encodeur fine-tuné y fait 0,791, à peine devant
bge-m3 (0,788). Avec la mémoire, il monte à 0,847 avec les relations prédites et à 0,931 avec les relations
annotées. Le gain vient donc surtout de la mémoire, et beaucoup moins du fine-tuning, quand on change de
projets et de style.

### L'agent

L'agent est fait de règles. Il détecte l'intention par mots-clés, refuse d'agir sur une phrase hypothétique
(« on pourrait peut-être revoir le budget »), cherche les unités, applique la mémoire puis choisit
l'opération. Pour un mail, le destinataire doit être désigné comme responsable dans une réunion. Une personne
seulement citée ne suffit pas, et aucune adresse n'est inventée.

Les demandes utilisées pour l'évaluer viennent d'un assistant d'IA, qui les a proposées à partir des unités de
chaque projet. J'ai vérifié chaque réaction attendue, en particulier les destinataires.

Pour ne pas régler l'agent sur les données qui le mesurent, j'ai séparé ses demandes en trois jeux. Le
premier, 6 demandes sur Atlas, servait à la mise au point. Au début, 5 sur 6 passaient : « Crée une tâche pour
Sophie » donnait une demande de précision au lieu d'une tâche. Le deuxième jeu servait à vérifier : 12
demandes sur la clinique et 24 sur deux autres projets de test. J'y ai trouvé trois erreurs que je n'avais pas
vues sur Atlas. Un mail partait à Inès alors que la formation avait été réattribuée à Agnès. « Envoie-lui un
message » recevait une réponse au lieu d'une demande de précision. Et un mail partait à Grégoire pour un sujet
dont personne n'était responsable. Après correction, j'avais 12/12 et 24/24. Ensuite je n'ai plus rien changé,
et test2 a été mesuré une seule fois : 24 demandes réussies sur 25, sans aucun mail adressé à la mauvaise
personne. La demande ratée est prudente plutôt que fausse : l'agent demande à qui écrire au lieu de trouver
Corentin, chargé du dédoublonnage.

### Abstention (30 octobre)

En testant l'agent, je me suis rendu compte qu'il répondait toujours, même à une question dont la réponse
n'était dans aucune réunion. Il donnait alors l'unité la plus proche comme si c'était la réponse. J'ai essayé
six stratégies d'abstention (mots communs, score, écart entre les deux premiers scores, et des combinaisons),
avec des seuils calibrés sur train et val. La stratégie retenue s'abstient quand le score de la première unité
est trop bas. Sur train et val, elle trouve 6 questions sans réponse sur 9 sans aucune abstention à tort. Sur
test et test2, elle n'en trouve plus que 6 sur 14, avec 2 abstentions à tort sur 79.

En regardant ces deux erreurs, j'ai trouvé un autre problème. Une question « Comment le budget a-t-il
évolué… ? » était classée comme une question simple et pas comme une comparaison. Or j'avais corrigé cette
règle le 25 octobre (commit 85b6a08). En remontant l'historique, j'ai vu qu'un commit suivant (b7a3d9b) avait
remis l'ancienne version sans que je m'en rende compte. J'ai restauré la règle. Depuis, je relance la démo et
l'évaluation de l'agent après chaque modification pour vérifier que rien n'a bougé.

### Unités, tours de parole ou découpage fixe (31 octobre)

Le sujet demandait de comparer la représentation en unités à des représentations plus simples. J'ai comparé
les unités, les tours de parole et des fenêtres de mots de taille fixe, avec la taille choisie sur val et
train. Sur val, la fenêtre de 60 mots dépassait les unités avec l'encodeur fine-tuné (0,907 contre 0,852).
Mais il n'y avait que 12 documents et 18 questions, et sur test c'est l'inverse : 0,826 pour les unités,
0,649 pour les tours et 0,636 pour les fenêtres. C'est la deuxième fois que val prédisait mal le test. Je n'ai
rien changé après avoir vu test, mais je l'ai noté dans [docs/chunking.md](docs/chunking.md).

### Extraction automatique des unités (1er novembre)

Jusque-là, les unités venaient des annotations. Pour que le système marche sur de nouvelles réunions, il faut
les extraire de la transcription. J'ai utilisé qwen3:8b, puis du code qui vérifie chaque unité : la phrase
citée doit exister dans la réunion, les noms doivent être ceux des participants. Les unités sont bien
retrouvées (F1 de 0,993 sur test), mais leur type est souvent faux : 0,754 sur test et 0,622 sur test2, contre
0,91 sur train et val, où j'avais réglé le prompt. Sur toute la chaîne, ça coûte cher. La recherche passe de
0,883 à 0,777 sur test, et l'agent de 24/25 à 21/25 sur test2. L'extraction est le maillon faible du système
complet, et je ne l'utilise pas dans la démonstration.

Le même jour, j'ai mesuré les dossiers et les paramètres des demandes. Avec les relations prédites, un dossier
couvre 78 % des faits attendus sur test. Avec les unités extraites, ce n'est plus que 65 %. Pour les paramètres,
la seule erreur sur 35 demandes est une échéance relative (« avant la bascule ») que mes règles ne savent pas
lire.

### Reranking (2 novembre)

J'ai essayé de reclasser les 10 premiers résultats avec bge-reranker-v2-m3, sans fine-tuning. J'ai choisi K
sur val, et sur val le reranking n'apportait rien avec la mémoire (0,917 sans, 0,903 avec). Sur test, il aide
(0,883 → 0,925 avec les relations prédites), mais sur test2 presque pas (0,847 → 0,854), et les questions sur
la décision en vigueur reculent (0,833 → 0,667). Le reranker préfère souvent l'ancienne version d'une
décision, parce que rien dans le texte ne dit laquelle est en vigueur. Il ajoute aussi 60 ms par question,
alors que la recherche avec la mémoire en prend 7. Je ne l'ai pas intégré à l'agent.

### Interface et rédaction par un modèle local (2 au 4 novembre)

J'ai fait une interface Streamlit pour la démonstration. En la testant, j'ai vu une limite que les
métriques ne montraient pas. À « On a le droit de mettre notre logo sur la vitrine ? », l'agent répondait par
la question posée en réunion par Lou, au lieu de la réponse d'Inaya, parce qu'il affichait simplement la
première unité trouvée. J'ai ajouté une étape de rédaction : qwen3:8b lit les 6 premières unités et rédige la
réponse en citant leurs identifiants. Le code rejette la réponse si elle cite une unité qui n'a pas été
fournie, ou si elle contient un nom ou un nombre absent des unités citées. J'ai ajouté cette dernière
vérification pendant la mise au point, après une réponse qui attribuait le budget d'Atlas à un « salon
professionnel de Lyon » qui n'existe pas. Sur test, la part des réponses qui citent une unité pertinente
passe de 0,810 à 0,948. Sur test2, elle passe de 0,810 à 0,905. Mais chaque réponse prend environ 6 secondes
au lieu de 8 millisecondes.

### De vraies réunions en anglais (4 novembre)

Toutes mes données sont synthétiques et en français. Je voulais savoir si le fine-tuning avait fait perdre
quelque chose sur de vraies réunions. J'ai évalué la recherche seule sur QMSum (réunions AMI, ICSI et de
commissions parlementaires, en anglais). e5-base d'origine fait 0,681 et mon modèle fine-tuné 0,644. La perte
vient surtout des réunions de recherche ICSI (0,607 contre 0,509). Sur les réunions de projet AMI, les plus
proches de mes données, les deux modèles sont presque à égalité (0,663 contre 0,657). Le fine-tuning a donc
spécialisé le modèle sur mon domaine, au prix d'un peu de généralité.

### Ce que je n'ai pas eu le temps de faire

Je n'ai pas réussi à améliorer le rappel des relations prédites, alors que c'est ce qui limite le plus le
système. Je n'ai pas non plus corrigé les erreurs de type de l'extraction, ni testé une mise à jour
incrémentale des relations. Aujourd'hui, toutes les paires d'un projet sont recalculées, ce qui prendrait
environ 29 minutes pour 1 000 unités.

### Méthode d'évaluation

La règle est la même partout : je règle sur train et val, puis je mesure test et test2 une seule fois, sans
rien changer après. Les seules exceptions sont les corrections de l'agent faites sur ses deux premiers jeux de
demandes, avant la mesure finale sur test2.

| composant | réglé sur | mesuré sur | métriques |
|---|---|---|---|
| recherche et fine-tuning | train (entraînement), val | test (63 questions), test2 (24) | Recall@1, @5, @10, MRR@10, nDCG@10, hn_acc |
| relations | train, val (seuil) | test | précision, rappel, F1 des remplacements, MRR@10 avec mémoire |
| mémoire | aucun réglage | test, test2 | MRR@10 avec et sans filtres |
| agent | exemple (6), clinique (12), holdout (24) | test2 (25 demandes) | type de sortie, destinataire, première source |
| abstention | train et val (seuils) | test, test2 | abstentions justifiées et à tort |
| découpages | train et val (taille) | test, test2 | mêmes métriques que la recherche |
| extraction | train et val (prompt) | test, test2 | précision, rappel, F1, exactitude des champs, coût sur la recherche et l'agent |
| dossiers et paramètres | aucun réglage | test, test2, 35 demandes | couverture, citations exactes, exactitude |
| reranking | val (K) | test, test2 | MRR@10, par type de question |
| rédaction | train et val (consignes) | test, test2 | cite une unité pertinente, abstentions, rejets, temps |
| QMSum | val (taille de fenêtre) | test (244 questions) | Recall@k, MRR@10, nDCG@10 |

## Installation

Il faut Python 3.10 ou plus (j'ai utilisé 3.12) et un GPU NVIDIA pour entraîner les modèles. Sur Ubuntu, la
commande s'appelle `python3` en dehors du venv, puis `python` une fois le venv activé. Sur une carte de
la série RTX 50, PyTorch doit être installé dans sa version CUDA 12.8 :

```bash
git clone <adresse du dépôt> EMBEDDING-LAB
cd EMBEDDING-LAB
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

Pour vérifier que le GPU est détecté :

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Les modèles entraînés ne sont pas dans le dépôt (trop lourds). Cette commande les recrée avec les mêmes
réglages que ceux que j'ai utilisés, s'ils ne sont pas déjà dans `models/` :

```bash
python scripts/setup_models.py
```

Elle entraîne l'encodeur `models/e5b-both-type-s1` puis le classifieur de relations `models/relations-cross`.
Au premier lancement, elle télécharge e5-base et mDeBERTa depuis Hugging Face (environ 2 Go). Sur ma machine,
les deux entraînements prennent 53 secondes en tout, sans compter le téléchargement. Pendant l'entraînement des
relations, un avertissement « MISMATCH » sur `classifier.weight` est normal : mDeBERTa a été publié avec 3
classes, et je le réentraîne avec 6. J'ai vérifié depuis un clone neuf que les modèles recréés redonnent
exactement mes résultats (même démo, 24/25 pour l'agent sur test2).

## Lancer la démonstration

```bash
streamlit run app/streamlit_app.py
```

L'interface s'ouvre sur http://localhost:8501. Le déroulé que je montre est dans [docs/demo.md](docs/demo.md).
On peut aussi lancer l'agent en ligne de commande :

```bash
python scripts/agent.py --project clinique-tilleuls --demo     # les 7 cas du sujet
python scripts/agent.py --project clinique-tilleuls --chat     # conversation
```

## Ce qui demande Ollama

Ollama sert à faire tourner qwen3:8b en local. Il est facultatif :

```bash
ollama pull qwen3:8b
ollama serve
```

Sans Ollama, l'agent, l'interface, les tests et presque toutes les évaluations fonctionnent. Dans l'interface,
la rédaction des réponses est activée par défaut. Si Ollama ne répond pas, un message l'indique et l'agent
affiche la première unité trouvée, comme en ligne de commande. On peut aussi choisir « aucune » dans les
options avancées.

Ollama est nécessaire pour la rédaction des réponses (`--writer qwen3:8b`, `eval_writer.py`), pour
l'extraction des unités (`extract_units.py`) et pour la génération de nouveaux projets
(`generate_scenarios.py`). Il l'est aussi pour la partie extraction de `benchmark.py`, que l'on peut sauter
avec `--parts latence relations`. Les unités déjà extraites sont dans le dépôt (`data/extracted/`), donc
`eval_extraction.py` marche sans Ollama.

## Reproduire les résultats

Toutes les commandes se lancent depuis la racine, après `setup_models.py`. Chaque évaluation écrit ses
résultats dans un nouveau dossier `experiments/<nom>-<date>/`. Les modèles de base (e5, bge-m3, le reranker)
sont téléchargés au premier usage.

Données. La génération demande Ollama (ou un copier-coller pour le test). Les projets sont déjà dans le dépôt,
et ces commandes servent seulement à refaire ou vérifier les données :

```bash
python scripts/generate_scenarios.py --n 5 --out data/scenarios/train --model qwen3:8b --theme-offset 1
python scripts/generate_scenarios.py --n 3 --out data/scenarios/test --provider manual --theme-offset 9 \
    --style "ton très oral, phrases courtes, hésitations"
python scripts/validate_data.py          # schéma et fuites entre splits
python scripts/review_scenarios.py       # cas suspects à relire
python scripts/data_stats.py             # statistiques de docs/DATA.md
```

Recherche, fine-tuning, relations et mémoire. `setup_models.py` a déjà lancé les deux entraînements, ces
commandes donnent les mesures :

```bash
python scripts/run_baselines.py --split data/scenarios/test --models bm25,e5-base,bge-m3 --with-type
python scripts/run_baselines.py --split data/scenarios/test --models models/e5b-both-type-s1 --with-type
python scripts/compare_runs.py --split test          # tableau de tous les runs de run_baselines
python scripts/eval_memory.py --split data/scenarios/test --model models/e5b-both-type-s1 --with-type
```

Pour refaire la comparaison des configurations de fine-tuning, `train_embedder.py` prend `--negatives`
(none, explicit, mined, both), `--with-type`, `--model e5-small` et `--seed`. `train_relations.py --method
logreg` entraîne la régression logistique.

Agent, abstention et intentions :

```bash
python scripts/eval_agent.py --file data/agent_eval/test2.jsonl --split data/scenarios/test2
python scripts/eval_abstention.py --calibrate              # seuils sur train et val
python scripts/eval_abstention.py --splits test test2      # mesure finale
python scripts/agent.py --eval-intents --split data/scenarios/test
```

Découpages, extraction, dossiers et paramètres :

```bash
python scripts/eval_chunking.py --select                   # taille de fenêtre sur train et val
python scripts/eval_chunking.py --splits test test2
python scripts/extract_units.py --split test               # Ollama, avec un cache
python scripts/eval_extraction.py --splits test test2 --retrieval
python scripts/eval_dossier.py --splits test test2
python scripts/eval_slots.py --file data/agent_eval/slots.jsonl
```

Reranking, rédaction, QMSum et mesures système :

```bash
python scripts/eval_rerank.py --split data/scenarios/val --ks 5 10 20 50 --texts type statement
python scripts/eval_rerank.py --split data/scenarios/test --ks 10 --texts type --relations gold predicted
python scripts/eval_writer.py --splits train val           # Ollama
python scripts/eval_writer.py --splits test test2
python scripts/eval_qmsum.py --download
python scripts/eval_qmsum.py --split val
python scripts/eval_qmsum.py --split test --frozen experiments/qmsum-val-<date>/results.json
python scripts/benchmark.py --split data/scenarios/test
```

Tests :

```bash
pytest                                       # 34 tests, sans GPU ni Ollama
python -m pyflakes app sae scripts tests
```

## Principaux résultats

Les réglages ont tous été choisis sur train et val, et test et test2 ont été mesurés une seule fois.

| quoi | mesure | test | test2 | détail |
|---|---|---|---|---|
| recherche, BM25 | MRR@10 | 0,778 | 0,748 | `experiments/baselines-*-typed` |
| recherche, e5-base sans entraînement | MRR@10 | 0,741 | 0,748 | idem |
| recherche, bge-m3 sans entraînement | MRR@10 | 0,807 | 0,788 | idem |
| recherche, e5-base fine-tuné (graine 1) | MRR@10 | 0,826 | 0,791 | idem, de 0,826 à 0,850 sur test selon la graine |
| fine-tuné + mémoire prédite | MRR@10 | 0,883 | 0,847 | [extraction.md](docs/extraction.md) |
| fine-tuné + mémoire annotée (borne haute) | MRR@10 | 0,933 | 0,931 | idem |
| relations, remplacements sur toutes les paires | précision / rappel / F1 | 0,750 / 0,429 / 0,545 | | `experiments/relations-cross-20261025-174306` |
| agent, demandes réussies | | | 24/25 | [extraction.md](docs/extraction.md) |
| abstention | justifiées / à tort | 5/10 et 2/58 | 1/4 et 0/21 | [abstention.md](docs/abstention.md) |
| extraction des unités | F1 / type correct | 0,993 / 0,754 | 0,978 / 0,622 | [extraction.md](docs/extraction.md) |
| dossier, relations prédites | couverture des faits attendus | 0,783 | 0,839 | [dossiers_et_slots.md](docs/dossiers_et_slots.md) |
| paramètres des demandes (35 demandes) | exactitude personne / projet / échéance | 1,000 / 1,000 / 0,971 | | idem |
| reranking + mémoire prédite | MRR@10 | 0,883 → 0,925 | 0,847 → 0,854 | [reranking_et_systeme.md](docs/reranking_et_systeme.md) |
| rédaction par qwen3:8b | cite une unité pertinente | 0,810 → 0,948 | 0,810 → 0,905 | [redaction.md](docs/redaction.md) |
| QMSum, e5-base d'origine / fine-tuné | MRR@10 | 0,681 / 0,644 | | [qmsum.md](docs/qmsum.md) |
| temps | par question / par réunion extraite | 7,4 ms | 12 s | [reranking_et_systeme.md](docs/reranking_et_systeme.md) |

## Limites

Les données sont synthétiques et en français, et les projets se ressemblent beaucoup. Avec 73 questions sur
test et 28 sur test2, une question pèse 1,4 point sur test et 3,6 points sur test2. Plusieurs écarts de mes
tableaux sont donc de l'ordre du bruit.

Le classifieur de relations rate plus de la moitié des remplacements (rappel 0,429). Chaque remplacement raté
laisse une information périmée dans une réponse ou un dossier. Son coût grandit comme le carré du nombre
d'unités, puisque toutes les paires d'un projet sont comparées.

L'abstention ne trouve que 6 questions sans réponse sur 14. La compréhension des demandes repose sur des
règles qui marchent sur mes demandes, mais qui ne comprennent pas une échéance comme « avant la bascule ». Sur
de vraies réunions en anglais, mon fine-tuning fait un peu moins bien que le modèle d'origine (0,644 contre
0,681).

## Organisation du dépôt

```
EMBEDDING-LAB/
├── app/streamlit_app.py        interface de démonstration
├── sae/                        code principal (package « Semantic Agent Engine »)
│   ├── schemas.py              format des données et des sorties de l'agent
│   ├── data.py                 chargement des projets, corpus et questions
│   ├── chunking.py             tours de parole et fenêtres (comparaison des découpages)
│   ├── memory.py               mémoire : remplacements, déclassement, filtre projet
│   ├── extraction.py           extraction des unités avec un modèle local
│   ├── retrieval/              BM25, encodeurs denses, reranking
│   ├── eval/metrics.py         métriques de recherche
│   ├── training/pairs.py       paires et négatifs pour le fine-tuning
│   ├── relations/              classifieur de relations et mémoire prédite
│   └── agent/                  compréhension, agent, rédaction
├── scripts/                    une commande par étape (setup_models.py pour les modèles)
├── tests/                      tests pytest, sans GPU ni Ollama
├── data/
│   ├── scenarios/              projets annotés : train, val, test, test2
│   ├── agent_eval/             demandes pour évaluer l'agent et ses paramètres
│   ├── extracted/              unités extraites automatiquement et alignement
│   └── external/               QMSum, retéléchargé (non versionné)
├── docs/                       architecture, données, résultats détaillés
│   └── RAPPORT.pdf             rapport final du projet
├── experiments/                résultats des évaluations (non versionnés)
├── models/                     modèles entraînés (non versionnés, voir setup_models.py)
├── requirements.txt
└── pyproject.toml
```

Les résultats détaillés sont dans [abstention](docs/abstention.md), [découpages](docs/chunking.md),
[extraction](docs/extraction.md), [dossiers et paramètres](docs/dossiers_et_slots.md),
[reranking et mesures système](docs/reranking_et_systeme.md), [rédaction](docs/redaction.md) et
[QMSum](docs/qmsum.md).

## Matériel

J'ai tout fait sur un portable avec une RTX 5070 Laptop (8 Go de mémoire GPU), un Intel Core i7-13650HX et
32 Go de RAM. Tout tourne en local, sans API payante.

## Outils utilisés

J'ai utilisé des assistants d'IA (Claude et Claude Code) pour écrire une partie du code et de la
documentation. J'ai conçu, lancé et analysé les expériences moi-même.

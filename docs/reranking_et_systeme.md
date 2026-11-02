# Reranking et mesures système

Deux éléments techniques du sujet : un reranking par cross-encodeur dans la comparaison des stratégies de
recherche, et les mesures système (latence, mémoire, coût). Le reranking n'est **pas** branché dans l'agent :
la décision se prendra sur ces résultats.

```bash
python scripts/eval_rerank.py --split data/scenarios/val --ks 5 10 20 50 --texts type statement --relations gold predicted
python scripts/eval_rerank.py --split data/scenarios/test  --ks 10 --texts type --relations gold predicted
python scripts/eval_rerank.py --split data/scenarios/test2 --ks 10 --texts type --relations gold predicted
python scripts/benchmark.py --split data/scenarios/test
```

## 1. Reranking

### Où il s'insère

Chaîne actuelle : `DenseRetriever.score` (cosinus question × unités, bi-encodeur `models/e5b-both-type-s1`,
type dans le texte) → `apply_memory` (redirection des remplacements, déclassement des unités remplacées, filtre
par projet) → tri. Le reranking se place entre les deux : les K premiers candidats denses sont relus par le
cross-encodeur `BAAI/bge-reranker-v2-m3` (non fine-tuné), qui lit la question et l'unité ensemble ; ils passent
devant toutes les autres unités, dans l'ordre du reranker, et les autres gardent leur ordre dense
(`sae/retrieval/rerank.py`, `place_on_top`). La mémoire s'applique ensuite sans changement. Dans l'agent, ce
serait au même endroit (`Agent.search`, entre `retriever.score` et `apply_memory`).

Le reranker est chargé explicitement en fp32 (`dtype=torch.float32`, puis vérification que tous les paramètres
sont en float32, sinon erreur). Contrôle : « Paris est la capitale de la France » obtient +8,9 contre −11,0
pour une phrase hors sujet.

Variantes (`scripts/eval_rerank.py`, nouveau script : `eval_memory.py` n'a ni choix de K ni relations prédites,
et ses variantes restent inchangées) : dense ; dense + rerank ; dense + statut + projet ;
dense + rerank + statut + projet.

### Choix de K sur val

Val seul (2 projets, 34 unités, 20 questions), relations annotées ; les relations prédites donnent ici les mêmes
chiffres (même nombre de remplacements, mêmes effets). Texte donné au reranker : « type : statement » (comme le
bi-encodeur) ou statement seul.

| variante | R@1 | MRR@10 | hn_acc |
|---|---|---|---|
| dense | 0,639 | 0,852 | 1,000 |
| dense + statut + projet | 0,750 | **0,917** | 1,000 |
| dense + rerank K=5 · type | 0,389 | 0,694 | 0,800 |
| dense + rerank K=10 · type | 0,389 | 0,686 | 0,800 |
| dense + rerank K=5 · type + statut + projet | **0,778** | 0,903 | 1,000 |
| dense + rerank K=10 · type + statut + projet | **0,778** | 0,903 | 1,000 |
| dense + rerank K=20 · type + statut + projet | 0,778 | 0,886 | 1,000 |
| dense + rerank K=50 · type + statut + projet | 0,778 | 0,886 | 1,000 |
| dense + rerank K=10 · statement + statut + projet | 0,722 | 0,866 | 1,000 |
| dense + rerank K=20 · statement + statut + projet | 0,667 | 0,821 | 1,000 |

Choix figé : **K = 10, texte avec type**. K=5 et K=10 sont à égalité sur toute la chaîne ; K ≥ 20 laisse
entrer davantage d'anciennes versions et de voisins dans la tête du classement ; le texte avec type est meilleur
partout. Sur val, le reranking n'améliore pas le MRR@10 de la chaîne complète (0,903 contre 0,917), seulement
le R@1 : la décision de le mesurer sur test n'était donc pas gagnée d'avance.

Diagnostic sur val : le reranker seul fait nettement moins bien que le bi-encodeur fine-tuné (MRR 0,69 contre
0,85). Ses erreurs sont presque toutes du même genre : pour « quelle date a **finalement** été retenue ? », il
préfère l'ancienne version (« fixé au 20 décembre », remplacée) à la nouvelle (« reporté au 5 janvier ») ; rien
dans le texte ne dit laquelle est en vigueur, et le bi-encodeur, lui, a appris cette préférence au fine-tuning.
La mémoire corrige précisément ce défaut (déclassement des unités remplacées) : reranker et mémoire sont
complémentaires, le reranker seul ne l'est pas.

### Mesure unique sur test et test2 (K = 10, texte avec type)

**test** (3 projets, 70 unités, 73 questions)

| variante | relations | R@1 | R@5 | MRR@10 | nDCG@10 | hn_acc |
|---|---|---|---|---|---|---|
| dense | - | 0,579 | 0,944 | 0,826 | 0,853 | 0,758 |
| dense + rerank | - | 0,611 | 0,950 | 0,853 | 0,865 | 0,774 |
| dense + statut + projet | annotées | 0,746 | 0,976 | 0,933 | 0,940 | 0,952 |
| dense + rerank + statut + projet | annotées | 0,743 | 0,968 | **0,956** | 0,941 | 0,919 |
| dense + statut + projet | prédites | 0,659 | 0,966 | 0,883 | 0,895 | 0,839 |
| dense + rerank + statut + projet | prédites | 0,688 | 0,958 | **0,925** | 0,909 | 0,855 |

**test2** (2 projets, 47 unités, 28 questions)

| variante | relations | R@1 | R@5 | MRR@10 | nDCG@10 | hn_acc |
|---|---|---|---|---|---|---|
| dense | - | 0,542 | 0,896 | 0,791 | 0,817 | 0,667 |
| dense + rerank | - | 0,542 | 0,938 | 0,812 | 0,830 | 0,619 |
| dense + statut + projet | annotées | 0,688 | 0,938 | 0,931 | 0,913 | 0,952 |
| dense + rerank + statut + projet | annotées | 0,729 | 0,958 | **0,958** | 0,931 | 0,952 |
| dense + statut + projet | prédites | 0,583 | 0,896 | 0,847 | 0,835 | 0,857 |
| dense + rerank + statut + projet | prédites | 0,583 | 0,917 | **0,854** | 0,837 | 0,810 |

MRR@10 par sous-groupe (dense + statut + projet → avec rerank) :

| | current_decision | compare |
|---|---|---|
| test · annotées | 0,902 → 0,940 | 0,806 → 1,000 |
| test · prédites | 0,775 → 0,869 | 0,806 → 1,000 |
| test2 · annotées | 1,000 → 0,917 | 1,000 → 1,000 |
| test2 · prédites | 0,833 → 0,667 | 1,000 → 1,000 |

### Interprétation

- Avec la mémoire, le reranking gagne en MRR@10 dans les quatre cas : +0,023 et +0,042 sur test (annotées,
  prédites), +0,027 et +0,007 sur test2. Le gain le plus net est sur test avec relations prédites, là où la
  mémoire est imparfaite.
- Les gains viennent surtout des comparaisons (test : 0,806 → 1,000) : le reranker remonte ensemble l'ancienne
  et la nouvelle version, ce qu'une question « qu'est-ce qui a changé » demande.
- Il reste fragile sur les décisions en vigueur sur test2 : current_decision baisse de 1,000 à 0,917 avec les
  relations annotées et de 0,833 à 0,667 avec les relations prédites (hn_acc de 0,857 à 0,810), alors qu'il
  monte sur test. C'est le
  défaut vu sur val : sans remplacement connu, le reranker préfère volontiers l'ancienne décision.
- Sur val, le reranking ne gagnait rien en MRR ; sur test et test2, il gagne un peu. Avec 20, 73 et 28
  questions, une question vaut 1 à 5 points de MRR : le gain est cohérent en signe, mais du même ordre que la
  variance d'échantillonnage.
- Coût : environ 60 ms de plus par question (voir plus bas), environ 9 fois la chaîne actuelle.

Recommandation : le reranking ne vaut la peine dans l'agent qu'avec la mémoire, et plutôt pour les questions de
comparaison, où le gain est net ; pour les questions sur la décision en vigueur, il ajoute un risque quand les
relations sont prédites. Une option raisonnable serait de ne l'activer que pour l'intention `compare`, à valider
sur des données qui n'ont pas servi ici.

## 2. Mesures système

### Machine et protocole

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU, 8 Gio (7,5 Gio utilisables) |
| CPU | Intel Core i7-13650HX, 20 cœurs logiques |
| RAM | 31 Gio |
| logiciel | Linux 7.0, Python 3.12, PyTorch 2.11 (CUDA 12.8), transformers 5.20, Ollama (qwen3:8b Q4_K_M) |

`scripts/benchmark.py`, split test (70 unités, 73 questions). Latences mesurées question par question (lot de 1,
comme dans l'agent), après 5 appels de chauffe, sur 3 répétitions ; moyenne, médiane, 95e centile. Bi-encodeur
en bf16 (comme dans le projet), reranker et classifieur de relations en fp32. Mémoire GPU des modèles PyTorch :
`torch.cuda.max_memory_allocated` pendant le chargement et l'inférence, au-delà de ce qui était déjà alloué.
qwen3:8b tourne dans Ollama, hors de PyTorch : sa mémoire est celle que rapporte Ollama (`/api/ps`).
L'extraction est mesurée en premier, GPU libre, car qwen3:8b et les trois modèles PyTorch ne tiennent pas
ensemble dans 8 Gio.

### Latence par question

| étape | moyenne (ms) | médiane (ms) | p95 (ms) |
|---|---|---|---|
| encodage de la question (bi-encodeur) | 7,2 | 7,2 | 7,9 |
| recherche (produit scalaire + tri, 70 unités) | 0,01 | 0,01 | 0,01 |
| reranking (K = 10 paires) | 60,6 | 58,9 | 74,2 |
| mémoire (statut + projet) | 0,05 | 0,05 | 0,06 |
| chaîne actuelle (encodage + recherche + mémoire) | 7,4 | 7,4 | 7,7 |
| chaîne avec reranking | 65,8 | 64,2 | 80,5 |
| agent complet · réponse (62 questions) | 8,0 | 7,9 | 9,6 |
| agent complet · dossier (3 projets) | 0,4 | 0,4 | 0,4 |
| agent complet · brouillon de mail (6 demandes) | 27,1 | 25,5 | 33,0 |

Une première exécution, la veille, donnait 4,3 ms pour l'encodage et 4,7 ms pour une réponse, avec le même
reranking (~60 ms) : les petites latences varient du simple au double selon l'état d'alimentation du
portable ; les ordres de grandeur, eux, ne changent pas.

- Une réponse de l'agent coûte à peine plus que l'encodage de la question : la recherche et la mémoire sont
  négligeables à cette taille de corpus.
- Le brouillon de mail fait plusieurs recherches (sujet, réattribution, contexte) : environ 3 fois une réponse.
- Le dossier ne fait aucune recherche ni appel de modèle : moins d'une milliseconde.
- Le reranking multiplie la latence de la recherche par environ 9 (~60 ms pour 10 paires avec un modèle de
  568 M de paramètres en fp32). Cela reste interactif, mais ce n'est plus négligeable ; en bf16, il serait
  probablement 2 à 3 fois plus rapide (non mesuré, et à vérifier, vu les problèmes de précision déjà rencontrés).

### Construction de la mémoire : prédiction des relations

Le classifieur de relations (`models/relations-cross`) lit toutes les paires d'unités d'un même projet :
n(n−1)/2 paires pour n unités. Mesuré sur des projets fictifs de n unités réelles (lot de 64, médiane de 3) :

| unités | paires | temps |
|---|---|---|
| 10 | 45 | 0,15 s |
| 20 | 190 | 0,75 s |
| 40 | 780 | 2,7 s |
| 80 | 3 160 | 11,4 s |
| 160 | 12 720 | 44 s |
| 240 | 28 680 | 100 s |
| 500 (prévu) | 124 750 | ≈ 7 min |
| 1 000 (prévu) | 499 500 | ≈ 29 min |
| 5 000 (prévu) | 12,5 M | ≈ 12 h |

Le coût par paire est constant (3,5 ms ± 0,3), donc le temps croît exactement comme le carré du nombre
d'unités : t ≈ 1,75 ms × n². Les projets actuels (17 à 24 unités) se traitent en moins d'une seconde ; tout le
split test (3 projets, 782 paires) en 3,7 s, chargement compris. Au-delà de quelques centaines d'unités par
projet, il faudrait :

- **mettre à jour incrémentalement** : une nouvelle réunion de m unités dans un projet de n unités n'ajoute que
  m × n paires (10 nouvelles unités dans un projet de 1 000 : 10 000 paires, ≈ 35 s) ;
- **présélectionner les paires** par la similarité du bi-encodeur (les k plus proches voisins de chaque unité),
  ce qui rend le coût linéaire en n : un remplacement porte presque toujours sur le même sujet ;
- éventuellement passer le classifieur en bf16 (à vérifier, comme pour le reranker).

### Extraction des unités (qwen3:8b, Ollama)

Le cache de `data/extracted` ne contient pas de durées : mesure sur 3 réunions de test, 3 fois chacune, avec
exactement le prompt et les options de `sae/extraction.py` (rien n'est écrit dans `data/extracted`).

| | valeur |
|---|---|
| chargement à froid (premier appel) | 18,7 s |
| durée par réunion (médiane, 9 appels) | 12,0 s (11,9 à 14,3) |
| tokens en entrée / en sortie | ≈ 1 050 / 525 à 630 |
| débit de génération | 48 tokens/s |
| mémoire GPU (Ollama) | 6,1 Gio (37 couches sur 37 sur le GPU, contexte 8 192) |

La durée est presque entièrement de la génération (≈ 11 s sur 12) : elle est proportionnelle au nombre
d'unités produites, environ 2 s par unité. Un projet de 4 réunions s'extrait en moins d'une minute ; une heure
de réunion réelle (bien plus de texte et d'unités) prendrait plutôt plusieurs minutes, et dépasserait le
contexte de 8 192 tokens. À température 0 et graine fixe, une même réunion a donné deux sorties de longueur
différente (604 et 631 tokens) : l'extraction n'est pas strictement reproductible, d'où l'intérêt du cache.

### Mémoire GPU et disque

| modèle | rôle | précision | GPU (pic, Mio) | disque (Mio) |
|---|---|---|---|---|
| models/e5b-both-type-s1 | recherche dense | bf16 | 1 427 | 1 077 |
| models/relations-cross | relations | fp32 | 1 747 | 1 079 |
| BAAI/bge-reranker-v2-m3 | reranking | fp32 | 2 189 | 2 187 |
| qwen3:8b (Q4_K_M) | extraction | 4 bits | 6 262 (Ollama) | 4 983 |

Les trois modèles PyTorch tiennent ensemble (≈ 5,4 Gio de pic), mais pas avec qwen3:8b : sur 8 Gio,
l'extraction doit tourner seule, avant ou après, puis décharger le modèle (`keep_alive: 0`). Ajouter le
reranker à l'agent ferait passer l'agent de ≈ 3,2 à ≈ 5,4 Gio de mémoire GPU.

### Coût

Tout tourne en local sur le portable décrit plus haut : **aucun coût d'API**, aucune donnée envoyée à un
service extérieur. Le seul coût est le matériel (GPU grand public 8 Gio) et l'électricité, non mesurée. Le
poste le plus lourd est l'extraction (12 s de GPU par réunion), puis la construction des relations, quadratique ;
les requêtes elles-mêmes coûtent quelques millisecondes.

## 3. Limites

- **Petits effectifs** : 20 questions sur val pour choisir K (deux valeurs à égalité), 73 et 28 sur test et
  test2 ; les écarts entre variantes avec mémoire sont de l'ordre d'une ou deux questions.
- **Reranker non fine-tuné**, sur des statements courts et propres : son intérêt pourrait être plus grand sur
  des textes longs (fenêtres de transcription, docs/chunking.md) et plus faible sur des unités extraites.
- **Corpus minuscules** (34 à 70 unités) : avec K=10, le reranker relit jusqu'à un tiers du corpus ; sur un vrai
  corpus de milliers d'unités, le rappel de la recherche dense avant reranking compterait beaucoup plus.
- **Latences mesurées sur une seule machine**, lot de 1, sans concurrence ; la latence du reranker dépend
  fortement de la longueur des textes et du lot.
- **Extrapolation des relations** linéaire en nombre de paires, à coût par paire constant, mesurée jusqu'à
  240 unités (projet fictif fait d'unités réelles de plusieurs projets).
- **Extraction** mesurée sur 3 réunions seulement, le cache ne contenant pas de durées.

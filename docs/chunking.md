# Représentation du corpus : découpage fixe, tours de parole, unités sémantiques

Le sujet demande de comparer trois façons de découper les réunions avant la recherche. Jusqu'ici, seules les unités sémantiques avaient été évaluées. Ce document compare les trois avec les mêmes questions, les mêmes règles de pertinence et les mêmes modèles.

## Méthode

**Transcription reconstituée.** Les scénarios ne contiennent pas de transcription. Pour chaque réunion, on enchaîne donc dans l'ordre les phrases dites à l'oral (`evidence`) de ses unités, sous la forme « Locuteur : phrase ». Ce n'est **pas** une vraie transcription : il n'y a ni digressions, ni bavardage, ni phrases sans information. Chaque phrase correspond exactement à une unité. Les deux représentations qui partent de la transcription sont donc avantagées par rapport à une situation réelle.

Représentations (`sae/chunking.py`) :

| représentation | document | texte |
|---|---|---|
| `unites` / `unites-type` | une unité (fonctionnement actuel) | `statement`, type en préfixe en option |
| `tours` | une phrase de la transcription | « Locuteur : evidence » |
| `fixe-N-C` | une fenêtre de N mots, chevauchement C %, sans dépasser une réunion | extrait de la transcription |

Chaque document garde la liste des unités dont il contient l'evidence. Pour une fenêtre fixe, une unité compte si la fenêtre contient au moins la moitié des mots de sa ligne. Ainsi, chaque unité est dans au moins une fenêtre, sans qu'un simple fragment ne compte.

**Règles de pertinence**, identiques pour toutes les représentations (`per_query_docs` dans `sae/eval/metrics.py`) :

- un document est pertinent s'il contient au moins une unité pertinente de la question ;
- un document est un piège s'il contient un négatif difficile et aucune unité pertinente.

Les métriques existantes sont adaptées ainsi. Avec un document par unité, elles redonnent **exactement** celles de `per_query`, ce qui a été vérifié sur train avec BM25.

- **Recall@k** : part des unités pertinentes présentes dans au moins un des k premiers documents.
- **MRR@10** : rang du premier document pertinent.
- **nDCG@10** : le gain d'un document est le nombre d'unités pertinentes *nouvelles* qu'il apporte. Une fenêtre qui chevauche la précédente ne rapporte donc rien de plus. L'idéal est celui de `per_query`, et le score est plafonné à 1 (une fenêtre peut apporter plusieurs unités pertinentes au rang 1).
- **hn_acc** : pour chaque négatif difficile, le meilleur document pertinent doit dépasser tous les pièges qui le contiennent. Un négatif qui n'apparaît que mélangé à une unité pertinente compte comme un échec.

Mesures propres aux découpages :

- nombre de documents et longueur moyenne en mots ;
- **docs mélangés** : part des documents qui contiennent, pour une même question, une unité pertinente et un négatif difficile (« fixée au 15 » et « reportée au 22 » dans la même fenêtre) ;
- **hn non séparés** : part des négatifs qui ne sont jamais dans un document sans unité pertinente ;
- **top-1 mélangé** : part des questions dont le 1er document est mélangé.

**Choix du découpage fixe** (`python scripts/eval_chunking.py --select`, refusé hors train/val). Grille : N ∈ {15, 30, 60, 120} × chevauchement ∈ {0, 25, 50 %}. 15 a été ajouté parce que les réunions sont courtes (53 à 76 mots en moyenne, 109 au maximum). Avec 18 questions, val est trop petite : on ajoute train pour bm25 et e5-base. L'encodeur fine-tuné n'est compté que sur val, puisqu'il a été entraîné sur train. Critère : meilleur MRR@10 moyen sur ces 5 cases (modèle, split), puis nDCG@10, puis moins de documents mélangés. Le découpage retenu, **fixe-60-50**, a été figé dans `FIXED_SIZE` / `FIXED_OVERLAP` avant toute mesure sur test et test2.

## Choix sur train + val

| découpage | MRR@10 moyen | nDCG@10 moyen |
|---|---|---|
| **fixe-60-50** | **0,648** | **0,696** |
| fixe-120-* (= une réunion entière) | 0,640 | 0,695 |
| fixe-60-0 | 0,562 | 0,630 |
| fixe-30-25 | 0,536 | 0,597 |
| fixe-15-25 | 0,461 | 0,484 |

60-50 et 120 sont à égalité dans le bruit : avec ces réunions courtes, les deux reviennent presque à « une réunion par document ». Les petites fenêtres (15, 30) sont nettement moins bonnes. Sur val, MRR@10 avec l'encodeur fine-tuné : unites-type 0,852, tours 0,759, fixe-60-50 0,907. Ce dernier chiffre porte sur 12 documents et 18 questions (voir les limites). Les résultats complets de train et val sont dans `experiments/chunking-train+val-*/results.json`.

## Résultats finaux (test et test2, mesurés une seule fois)

**test**, 63 questions avec réponse :

| représentation | modèle | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | hn_acc | docs | mots/doc | docs mélangés | top-1 mélangé |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unites-type | bm25 | 0,524 | 0,937 | 0,976 | 0,778 | 0,818 | 0,694 | 70 | 16,4 | 0 | 0 |
| unites-type | e5-base | 0,484 | 0,921 | 0,981 | 0,741 | 0,796 | 0,677 | 70 | 16,4 | 0 | 0 |
| unites-type | e5b fine-tuné | **0,579** | **0,944** | 0,966 | **0,826** | **0,853** | **0,758** | 70 | 16,4 | 0 | 0 |
| unites | e5b fine-tuné | 0,524 | 0,913 | 0,968 | 0,784 | 0,824 | 0,710 | 70 | 14,4 | 0 | 0 |
| tours | bm25 | 0,357 | 0,661 | 0,738 | 0,557 | 0,584 | 0,484 | 70 | 16,2 | 0 | 0 |
| tours | e5-base | 0,365 | 0,717 | 0,828 | 0,608 | 0,633 | 0,613 | 70 | 16,2 | 0 | 0 |
| tours | e5b fine-tuné | 0,429 | 0,772 | 0,836 | 0,649 | 0,669 | 0,677 | 70 | 16,2 | 0 | 0 |
| fixe-60-50 | bm25 | 0,410 | 0,714 | 0,786 | 0,605 | 0,642 | 0,435 | 29 | 53,6 | 0,138 | 0,085 |
| fixe-60-50 | e5-base | 0,402 | 0,757 | 0,854 | 0,612 | 0,654 | 0,484 | 29 | 53,6 | 0,138 | 0,128 |
| fixe-60-50 | e5b fine-tuné | 0,376 | 0,844 | 0,915 | 0,636 | 0,690 | 0,516 | 29 | 53,6 | 0,138 | 0,128 |

**test2**, 24 questions, autre domaine et autre générateur :

| représentation | modèle | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | hn_acc | docs | mots/doc | docs mélangés | top-1 mélangé |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unites-type | bm25 | 0,479 | 0,896 | 0,917 | 0,748 | 0,762 | 0,619 | 47 | 15,5 | 0 | 0 |
| unites-type | e5-base | 0,458 | 0,875 | 0,917 | 0,748 | 0,779 | 0,571 | 47 | 15,5 | 0 | 0 |
| unites-type | e5b fine-tuné | **0,542** | **0,896** | 0,958 | **0,791** | **0,817** | **0,667** | 47 | 15,5 | 0 | 0 |
| tours | bm25 | 0,208 | 0,667 | 0,688 | 0,419 | 0,465 | 0,333 | 47 | 16,0 | 0 | 0 |
| tours | e5-base | 0,396 | 0,604 | 0,792 | 0,576 | 0,596 | 0,571 | 47 | 16,0 | 0 | 0 |
| tours | e5b fine-tuné | 0,375 | 0,792 | 0,875 | 0,637 | 0,665 | 0,571 | 47 | 16,0 | 0 | 0 |
| fixe-60-50 | bm25 | 0,271 | 0,667 | 0,792 | 0,524 | 0,554 | 0,333 | 17 | 56,6 | 0,118 | 0,056 |
| fixe-60-50 | e5-base | 0,312 | 0,792 | 0,979 | 0,612 | 0,677 | 0,667 | 17 | 56,6 | 0,118 | 0,056 |
| fixe-60-50 | e5b fine-tuné | 0,375 | 0,812 | **1,000** | 0,675 | 0,725 | 0,571 | 17 | 56,6 | 0,118 | 0,056 |

## Interprétation

- **Les unités sémantiques l'emportent sur les deux splits, quel que soit le modèle.** Avec l'encodeur fine-tuné, le MRR@10 est de 0,826 contre 0,649 (tours) et 0,636 (fixe) sur test, et de 0,791 contre 0,637 et 0,675 sur test2. L'écart existe aussi avec BM25 (0,778 contre 0,557 et 0,605) et avec e5-base non entraîné : il ne vient donc pas seulement du fine-tuning.
- **Le gain est surtout en tête de liste.** C'est ce qui compte pour l'agent, qui répond avec la 1re source. Recall@1 vaut 0,58 pour les unités contre 0,38–0,43 pour les autres, et hn_acc 0,76 contre 0,52–0,68.
- **Le Recall@10 élevé du découpage fixe (0,915 ; 1,000 sur test2) est un artefact.** Il n'y a que 29 et 17 documents, de 55 mots environ. Dix documents représentent donc 35 à 60 % du corpus, soit 10 fois plus de texte que dix unités. Ce n'est pas une meilleure recherche, et MRR@10 et Recall@1 le montrent.
- **Les tours sont les moins bons, alors qu'ils contiennent la même information que les unités.** La phrase orale (« Bon, c'est acté pour le local… ») est elliptique, informelle et pleine de pronoms. Le `statement` est explicite et nomme le sujet, la personne et le statut (« propose », « décide »). C'est l'apport de la structuration.
- **Mélange pertinent / piège.** Sur test et test2, 12 à 14 % des fenêtres contiennent à la fois la bonne information et un piège de la même question. Le 1er document est dans ce cas pour 6 à 13 % des questions. Le lecteur ne peut alors pas savoir quelle version est en vigueur. Les unités et les tours, eux, ne mélangent jamais par construction. Sur train et val, cette mesure vaut 0 : dans ces scénarios, une unité pertinente et un négatif difficile ne sont jamais dans la même réunion (0 paire sur 96), et les fenêtres ne dépassent pas une réunion. Les scénarios de test ont, eux, des changements au sein d'une même réunion.
- **Le choix de N n'a pas tenu sur test.** Sur val, fixe-60-50 dépassait les unités avec le modèle fine-tuné (0,907 contre 0,852), mais sur seulement 12 documents et 18 questions. Ce n'était pas reproductible : sur test, il est loin derrière.

## Biais et limites

- **Transcription reconstituée.** Sans bavardage ni digressions, chaque fenêtre et chaque tour ne contiennent que de l'information utile. Sur une vraie transcription, avec du bruit, des phrases coupées et des répétitions, le découpage fixe et les tours seraient vraisemblablement encore moins bons. Les écarts mesurés sont donc plutôt une borne basse de l'avantage des unités.
- **Encodeur fine-tuné.** `e5b-both-type-s1` a été entraîné sur les unités de train (avec le type en préfixe) et sur des paires evidence → statement. Côté passage, il n'a vu que des `statement` typés, ce qui avantage `unites-type`. Les paires evidence → statement lui ont appris à rapprocher une phrase orale d'une unité, et non une question d'une phrase orale. L'avantage est donc indirect pour `tours` et `fixe`. Les conclusions restent les mêmes avec BM25 et e5-base, qui n'ont pas ce biais.
- **Vocabulaire commun.** Les questions et les `statement` viennent du même générateur et partagent un vocabulaire neutre, ce qui favorise les unités. test2 (autre domaine, autre générateur) atténue ce biais, et le classement y est le même.
- **Les unités sont une annotation.** Ici, elles sont données par les scénarios. En production, il faudrait les extraire, avec des erreurs, et ce coût n'apparaît pas dans la comparaison.
- **Effectifs faibles et réunions courtes.** val a 18 questions et test2 24 : quelques centièmes d'écart n'y sont pas significatifs. Avec des réunions de 50 à 110 mots, N=60 ou N=120 revient presque à indexer la réunion entière, et la grille ne teste pas de vraies longues transcriptions.
- **Règle d'appartenance.** Une unité compte dans une fenêtre si celle-ci contient au moins la moitié de sa phrase. Une règle plus stricte ou plus lâche changerait légèrement les chiffres du découpage fixe.

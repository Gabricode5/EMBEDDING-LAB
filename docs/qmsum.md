# Recherche sur de vraies réunions : QMSum

Évaluation uniquement, sans aucun entraînement. La question principale : le fine-tuning de l'encodeur sur des
réunions synthétiques en français (`models/e5b-both-type-s1`) a-t-il dégradé, conservé ou amélioré la recherche
sur de vraies réunions en anglais ?

```bash
python scripts/eval_qmsum.py --download                     # val et test dans data/external/qmsum/
python scripts/eval_qmsum.py --split val                    # choix de la fenêtre (val uniquement)
python scripts/eval_qmsum.py --split test --frozen experiments/qmsum-val-20261104-185912/results.json
```

## 1. Licence

Vérifiée avant tout téléchargement.

- **Dépôt QMSum** ([Yale-LILY/QMSum](https://github.com/Yale-LILY/QMSum)) : licence MIT (« Copyright (c) 2021
  Yale-LILY »), usage, copie et modification libres avec mention de la licence. Article : Zhong et al.,
  *QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization*, NAACL 2021.
- **Sources des transcriptions** (citées dans la section éthique de l'article, qui les dit « public and open
  to use ») :
  - AMI Meeting Corpus (réunions de conception de produit) : CC BY 4.0, attribution requise
    ([licence](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)) ;
  - ICSI Meeting Corpus (réunions de recherche) : CC BY 4.0 dans la version distribuée par l'université
    d'Édimbourg, celle que cite l'article ([licence](https://groups.inf.ed.ac.uk/ami/icsi/license.shtml)) ;
    la version LDC (LDC2004T04) a d'autres conditions ;
  - commissions du Parlement gallois : réutilisation sous la licence ouverte du Senedd (licence
    gouvernementale non commerciale, d'après sa page
    [Re-using Public Sector Information](https://senedd.wales/commission/access-to-information/re-using-public-sector-information)) ;
  - commissions de la Chambre des communes du Canada : reproduction permise si elle est exacte, non présentée
    comme officielle et non commerciale ([Important notices](https://www.ourcommons.ca/en/important-notices)).

Usage ici : évaluation académique non commerciale, sans redistribution. Les données brutes sont dans
`data/external/qmsum/`, ignoré par git (`.gitignore`), et le script les retélécharge à une version figée du dépôt
(commit 83d7768, 29 août 2023). Une diffusion des transcriptions, ou un usage commercial, demanderait de
revérifier les conditions des commissions.

## 2. Données

Splits val et test seulement (fichiers `data/ALL/jsonl`, plus les fichiers par domaine `Product`, `Academic`,
`Committee` pour connaître la source de chaque réunion). Seules les questions spécifiques
(`specific_query_list`) sont utilisées : les questions générales (une par réunion, « résume la réunion ») n'ont
pas de passage pertinent.

| split | réunions (AMI / ICSI / commissions) | questions spécifiques (AMI / ICSI / commissions) | tours par réunion (médiane) |
|---|---|---|---|
| val | 35 (20 / 9 / 6) | 237 | 541 |
| test | 35 (20 / 9 / 6) | 244 (129 / 49 / 66) | 607 |

Les passages pertinents (`relevant_text_span`) sont des plages de tours inclusives, longues : 23 tours en
médiane (val), 1,05 passage par question en moyenne. En moyenne, **10 % des tours d'une réunion sont
pertinents** pour une question ; un classement au hasard des tours obtient déjà MRR@10 ≈ 0,20 (AMI 0,25,
ICSI et commissions 0,13). Les scores ci-dessous sont à lire par rapport à ce niveau.

## 3. Méthode

- **Corpus** : la recherche se fait dans la réunion de la question. Documents : les tours de parole
  (« Locuteur: texte », marques de transcription comme `{disfmarker}` retirées, texte anglais non traduit) et,
  pour une fenêtre w > 1, des fenêtres de w tours consécutifs, avec un pas de w/2 arrondi au-dessus, ajoutées aux
  tours.
- **Pertinence** : un document est pertinent s'il recoupe un passage de la question. Métriques de
  `sae/eval/metrics.py` (`per_query_docs`), avec les passages comme unités :
  - Recall@k : part des passages de la question recoupés dans les k premiers documents (≈ réussite@k, puisqu'il
    y a presque toujours un seul passage) ;
  - MRR@10 : rang du premier document pertinent ;
  - nDCG@10 : gain par passage nouveau.
- **Modèles** :
  - BM25, avec un nouveau paramètre de langue qui donne la racinisation et des mots vides anglais. Par défaut,
    BM25 reste en français et le reste du projet est inchangé : la démo de l'agent et les intentions sont
    identiques ;
  - e5-small et e5-base d'origine (intfloat/multilingual-e5-*) ;
  - bge-m3 ;
  - l'encodeur fine-tuné `models/e5b-both-type-s1`.

  Les encodeurs e5 utilisent les préfixes « query: » et « passage: », et aucun préfixe de type d'unité (il
  n'y a pas d'unités ici). Longueur maximale : 256 jetons pour les e5, 512 pour bge-m3.
- **Choix sur val uniquement** : taille de fenêtre parmi 1 (tours seuls), 3, 5, 8, par la meilleure MRR@10.
  Les cinq modèles choisissent 8. La grille s'arrête à 8 (« quelques tours », et une fenêtre plus longue
  recoupe mécaniquement plus facilement un long passage). Test est mesuré une seule fois, avec la fenêtre figée
  et les tours seuls comme référence.
- **Mémoire GPU** : un modèle chargé à la fois, libéré avant le suivant ; chaque réunion est encodée une fois
  pour toutes les tailles de fenêtre.

## 4. Résultats

### Val (choix de la fenêtre)

MRR@10 :

| modèle | tours seuls | w = 3 | w = 5 | w = 8 |
|---|---|---|---|---|
| BM25 | 0,547 | 0,522 | 0,537 | **0,554** |
| e5-small | 0,526 | 0,557 | 0,569 | **0,591** |
| e5-base d'origine | 0,606 | 0,628 | 0,641 | **0,646** |
| bge-m3 | 0,570 | 0,553 | 0,585 | **0,599** |
| e5-base fine-tuné | 0,570 | 0,564 | 0,583 | **0,612** |

### Test (mesure unique, 244 questions)

| modèle | corpus | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| BM25 | tours seuls | 0,402 | 0,729 | 0,845 | 0,584 | 0,634 |
| BM25 | tours + fenêtres de 8 | 0,400 | 0,740 | 0,808 | 0,580 | 0,620 |
| e5-small | tours seuls | 0,481 | 0,784 | 0,867 | 0,648 | 0,686 |
| e5-small | tours + fenêtres de 8 | 0,501 | 0,741 | 0,857 | 0,651 | 0,683 |
| e5-base d'origine | tours seuls | 0,496 | 0,756 | 0,876 | 0,654 | 0,693 |
| e5-base d'origine | tours + fenêtres de 8 | **0,528** | 0,768 | **0,893** | **0,681** | **0,716** |
| bge-m3 | tours seuls | 0,421 | 0,730 | 0,881 | 0,598 | 0,651 |
| bge-m3 | tours + fenêtres de 8 | 0,454 | 0,748 | 0,873 | 0,623 | 0,666 |
| e5-base fine-tuné | tours seuls | 0,435 | 0,788 | 0,849 | 0,610 | 0,658 |
| e5-base fine-tuné | tours + fenêtres de 8 | 0,468 | **0,790** | 0,880 | 0,644 | 0,688 |

MRR@10 par source (test, tours + fenêtres de 8 ; nombre de questions entre parenthèses) :

| modèle | AMI (129) | ICSI (49) | commissions (66) |
|---|---|---|---|
| BM25 | 0,544 | 0,429 | 0,762 |
| e5-small | 0,613 | 0,611 | 0,754 |
| e5-base d'origine | **0,663** | **0,607** | **0,772** |
| bge-m3 | 0,620 | 0,480 | 0,735 |
| e5-base fine-tuné | 0,657 | 0,509 | 0,718 |

**e5-base fine-tuné contre e5-base d'origine** (différence de MRR@10 par question, test, intervalle de confiance à
95 % par bootstrap apparié, 5 000 tirages) :

| corpus | différence | IC 95 % | questions meilleures / moins bonnes / égales |
|---|---|---|---|
| tours seuls | −0,044 | [−0,081 ; −0,005] | 49 / 66 / 129 |
| tours + fenêtres de 8 | −0,037 | [−0,072 ; −0,001] | 44 / 64 / 136 |

## 5. Interprétation

- **Le fine-tuning a légèrement dégradé la recherche sur ces vraies réunions anglaises.** La MRR@10 baisse
  d'environ 0,04 et le Recall@1 de 0,06. L'écart est petit : son intervalle de confiance exclut zéro de peu,
  et sur val il était du même ordre (0,612 contre 0,646). Les deux tiers des questions ne changent pas. Le
  Recall@5 est en revanche un peu plus haut (0,790 contre 0,768). Le modèle fine-tuné trouve donc le bon passage
  presque aussi souvent dans les 5 premiers, mais le met moins souvent en tête.
- **La perte est concentrée sur ICSI** (0,509 contre 0,607) et, dans une moindre mesure, sur les commissions
  (0,718 contre 0,772). Sur AMI, les deux modèles sont à égalité (0,657 contre 0,663). AMI est le domaine le
  plus proche des données d'entraînement : des réunions de projet avec des décisions, des responsables et des
  échéances. ICSI (réunions de recherche, vocabulaire technique, discussions longues) et les commissions (débats
  formels) s'en éloignent.
- **Lecture** : le fine-tuning a spécialisé l'encodeur sur une forme de texte (unités courtes, en français,
  préfixées par leur type) et un style de questions (décision en vigueur, responsable, date). Hors de ce
  domaine, une partie de la généralité d'e5-base est perdue, sans effondrement. Le modèle fine-tuné reste
  au-dessus de BM25 et de bge-m3, et au niveau d'e5-small. Pour le projet, cela confirme que le fine-tuning est
  adapté à son domaine. Pour un usage sur de vraies réunions anglaises, e5-base d'origine serait le meilleur
  point de départ.
- **Autres modèles** :
  - e5-base d'origine est le meilleur partout ;
  - BM25 est très bon sur les commissions (0,762), où les questions reprennent les termes exacts du débat, et
    faible sur ICSI ;
  - bge-m3 fait moins bien qu'e5-base, malgré une longueur maximale double.
- **Fenêtres** : elles aident les encodeurs (+0,03 de MRR@10 pour e5-base), pas BM25. Sur test, BM25 fait un
  peu mieux avec les tours seuls : son choix de fenêtre sur val ne se transfère pas.

## 6. Limites

- **Langue** : l'encodeur a été fine-tuné en français et il est évalué en anglais. La baisse mesurée mélange
  l'effet du domaine (réunions synthétiques contre réelles) et celui de la langue, sans pouvoir les séparer ;
  une version française de QMSum (traduction) le permettrait, mais elle n'est pas faite ici.
- **Pas d'annotations de statut** : QMSum n'indique ni remplacement, ni proposition contre décision. La
  mémoire, l'abstention et les pièges de statut, qui sont l'apport principal du projet, ne sont donc pas
  évalués. Seule la recherche brute l'est.
- **Questions orientées résumé** : les questions spécifiques demandent souvent un résumé d'une discussion
  (« What did the group discuss about… ») ; les passages sont longs (23 tours en médiane) et un classement au
  hasard atteint déjà MRR@10 ≈ 0,20. La mesure est grossière : elle récompense de tomber dans la bonne zone de
  la réunion, pas de trouver une phrase précise.
- **Troncature** : avec les fenêtres de 8 tours, 13 % des documents dépassent les 256 jetons des encodeurs e5
  (5 % dépassent 512) et sont tronqués ; les tours seuls ne le sont presque jamais (0,6 %).
- **Effectifs** : 244 questions sur test, dont 49 pour ICSI, sur seulement 9 réunions. Les écarts par source
  reposent sur peu de réunions.
- **Fenêtre au bord de la grille** : tous les modèles choisissent la plus grande taille essayée (8).
  Une taille plus grande aurait peut-être été choisie, mais avec une part mécanique croissante.

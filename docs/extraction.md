# Extraction automatique des unités

Jusqu'ici, toutes les mesures (recherche, mémoire, agent) supposaient des unités parfaites, annotées en même temps que les scénarios. Ce travail ajoute l'étape manquante : passer du texte d'une réunion à des unités de connaissance. Il l'évalue contre les unités annotées, puis mesure ce qu'elle coûte à la recherche et à l'agent.

## Méthode

**Entrée.** La transcription reconstituée de la réunion (« Locuteur : phrase », une ligne par evidence annotée, voir `sae/chunking.py`), la liste des participants, la date et le titre. La description du projet n'est pas fournie.

**Extraction** (`sae/extraction.py`, `scripts/extract_units.py`). qwen3:8b via Ollama, sortie JSON contrainte par le schéma Pydantic `Extraction`, à température 0. En cas de JSON ou de schéma invalide, on fait jusqu'à 3 tentatives avec une graine différente. Le modèle produit, pour chaque unité :
- type ;
- statement reformulé à la 3e personne ;
- evidence ;
- speaker, owner et due_date ;
- status parmi active, proposed, hypothetical et cancelled.

Le statut `superseded` n'est pas extrait : il vient ensuite du classifieur de relations (`models/relations-cross`).

**Ancrage.** Le code ne fait pas confiance au modèle :
- l'evidence doit être un extrait exact de la transcription, sinon elle est remplacée par la phrase la plus proche (recouvrement ≥ 0,6), sinon l'unité est écartée ;
- les noms sont ramenés aux participants (nom complet, ou prénom unique), sinon `null` ;
- le responsable n'est gardé que pour une action, un engagement ou un fait (« c'est X qui pilote ») ;
- l'échéance n'est gardée que pour une action ou un engagement, et un jour sans mois (« avant le 25 ») est ramené à sa prochaine occurrence après la réunion.

**Sorties.**
- `data/extracted/<split>/<projet>.json` : même format que les scénarios (mêmes réunions et personnes, unités `x-NNN`, relations vides, questions réécrites).
- `data/extracted/<split>/alignment/` : correspondance unités annotées → extraites, et questions perdues.
- Cache des réponses brutes dans `.cache/`, invalidé si le prompt, le schéma ou le modèle changent.

`data/scenarios` n'est jamais modifié.

**Alignement** (`align_meeting`). Dans chaque réunion, une unité extraite et une unité annotée sont appariées si leurs evidence se recouvrent : mots communs / taille du texte le plus court ≥ 0,5. L'appariement est un pour un, glouton par score décroissant, et la similarité des statements départage deux candidats sur la même phrase. L'encodeur n'a pas été nécessaire : grâce à l'ancrage, chaque evidence extraite est une phrase de la transcription.

**Métriques** (`scripts/eval_extraction.py`) :
- précision, rappel et F1 des unités ;
- sur les unités alignées, exactitude du type, du statut (hors unités annotées `superseded`), du locuteur, du responsable et de l'échéance ;
- le responsable et l'échéance sont aussi mesurés « si annotés », parce que l'annotation les renseigne rarement et de façon irrégulière : jamais dans val, 5 et 3 fois sur 119 unités dans train.

**Réglage, sur train et val uniquement.** Trois itérations :
1. Définitions des types alignées sur les conventions observées : « je propose » → idea ; « on pourrait peut-être » → idea hypothétique et non risk ; « on confirme » → decision ; « pour rappel, c'est X qui pilote » → fact. Le statut `proposed` est réservé à une validation explicitement attendue.
2. Responsable et échéance limités aux types qui en portent (le modèle désignait le locuteur comme responsable des décisions, et prenait une date décidée pour une échéance).
3. Résolution des jours sans mois.

Le prompt a ensuite été figé (empreinte `14f09679bc81`), puis test et test2 ont été extraits et mesurés une seule fois.

## Qualité de l'extraction

| split | annotées | extraites | P | R | F1 | type | statut | locuteur | responsable | échéance | resp. (si annoté) | éch. (si annotée) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val (réglage) | 34 | 34 | 1,000 | 1,000 | 1,000 | 0,912 | 0,929 | 1,000 | 0,735 | 0,853 | - (0) | - (0) |
| train (réglage) | 119 | 119 | 1,000 | 1,000 | 1,000 | 0,908 | 0,979 | 1,000 | 0,765 | 0,849 | 0,800 (5) | 0,667 (3) |
| **test** | 70 | 69 | 1,000 | 0,986 | 0,993 | **0,754** | 0,929 | 1,000 | 0,913 | 0,928 | 0,800 (15) | 0,636 (11) |
| **test2** | 47 | 45 | 1,000 | 0,957 | 0,978 | **0,622** | 0,971 | 1,000 | 0,911 | 0,933 | 1,000 (10) | 1,000 (7) |

Avant réglage, val était à 0,588 pour le type et 0,786 pour le statut.

- **Segmentation presque parfaite**, mais c'est une conséquence de la transcription reconstituée, qui contient une phrase par unité. Il ne manque que 3 unités sur 117 en test et test2.
- **Le type ne généralise pas.** Il passe de 0,91 sur train/val à 0,75 sur test et 0,62 sur test2. Les scénarios de test ont un oral beaucoup plus naturel (« euh… », « voilà, c'est calé »), et test2 vient d'un autre générateur. Les règles écrites à partir des tournures de train/val (« on confirme… ») ne couvrent pas « Ok, on garde le 14 novembre » ni « Le bloc, on démarre le 3 novembre. Voilà. C'est calé. ».
- **Le statut tient bien** (0,93 à 0,97).
- **Le locuteur est toujours juste**, mais il est écrit dans la transcription.
- **Le responsable** est juste dans 80 à 100 % des cas où l'annotation le renseigne.

Confusion des types sur test + test2 (lignes : annoté) : les décisions partent en `fact` (5) ou en `action` (4). Les réponses (« Non non, la formation c'est pas dans les 38 000 ») deviennent toutes des `fact` (5 sur 5). Les engagements deviennent des `action` (4 sur 10).

## Coût en aval

**Recherche** (protocole de `scripts/eval_memory.py`, `models/e5b-both-type-s1`, type en préfixe). Les unités attendues sont réécrites vers les unités extraites, et une question sans unité extraite compte comme un échec ; il n'y en a eu aucune.

| split | corpus | variante | Recall@1 | MRR@10 | hn_acc |
|---|---|---|---|---|---|
| test | annotées, relations annotées (borne haute) | dense + statut + projet | 0,746 | 0,933 | 0,952 |
| test | annotées, relations prédites | dense | 0,579 | 0,826 | 0,758 |
| test | annotées, relations prédites | dense + statut + projet | 0,659 | 0,883 | 0,839 |
| test | **extraites, relations prédites** | dense | 0,421 | 0,667 | 0,650 |
| test | **extraites, relations prédites** | dense + statut + projet | **0,529** | **0,777** | **0,750** |
| test2 | annotées, relations annotées (borne haute) | dense + statut + projet | 0,688 | 0,931 | 0,952 |
| test2 | annotées, relations prédites | dense | 0,542 | 0,791 | 0,667 |
| test2 | annotées, relations prédites | dense + statut + projet | 0,583 | 0,847 | 0,857 |
| test2 | **extraites, relations prédites** | dense | 0,375 | 0,654 | 0,619 |
| test2 | **extraites, relations prédites** | dense + statut + projet | **0,417** | **0,718** | **0,667** |

Dans le système complet (dense + statut + projet), l'extraction coûte environ **0,11 à 0,13 de MRR@10**, et 0,13 à 0,17 de Recall@1, par rapport aux unités annotées avec les mêmes relations prédites. Deux causes :

- **Les statements extraits sont moins bons pour la recherche**, même en dense seul (−0,14 à −0,16 de MRR). Ils reprennent souvent l'oral, oublient le sujet ou inventent un contexte (« … pour le bail »). L'encodeur, fine-tuné sur les statements annotés, est aussi moins à l'aise avec ce style.
- **La mémoire perd ses remplacements.** Le classifieur de relations ne retrouve que 6 remplacements sur test (contre 8 sur les unités annotées, et 14 annotés) et **1 seul sur test2** (contre 8, et 10 annotés). Sur test2, le statut et le projet n'ajoutent donc presque rien : +0,06 de MRR, contre +0,06 à +0,14 ailleurs.

Sur val, l'écart est faible (MRR 0,944 extraites contre 0,917 annotées), mais val a servi au réglage du prompt. Sur train, l'encodeur a été entraîné sur les statements annotés, ce qui fausse la comparaison.

**Agent** (`scripts/eval_agent.py --file data/agent_eval/test2.jsonl --split data/extracted/test2 --unit-map data/extracted/test2/alignment`) :

| métrique | unités annotées | unités extraites |
|---|---|---|
| réussite complète | **24/25** | **21/25** |
| bon type de sortie | 0,960 | 0,840 |
| destinataire correct (Precision@1) | 0,875 (8) | 0,875 (8) |
| taux de faux destinataires | 0,000 (7) | **0,125 (8)** |
| bonne unité en première source | 1,000 (6) | 0,667 (6) |

Les 3 nouveaux échecs :

- **g-11, faux destinataire.** « Ok, on garde le 14 novembre. Tu bloques ça dans ton agenda. » est extrait comme une `action` dont la locutrice est responsable, avec un statement inventé (« Romane Aubertin a confirmé la date du 14 novembre pour le bail »). L'agent écrit alors à Romane au lieu de demander une précision. C'est l'erreur la plus grave : avant, le taux de faux destinataires était nul.
- **p-01 et p-02, abstentions à tort.** « Le nouveau CRM, il passe en production quand ? » et « Les 45 000 euros… validés ? » restent sous le seuil d'abstention. Ce seuil (0,5121) a été calibré sur des statements annotés, et les statements extraits donnent des scores plus bas.

L'échec commun aux deux versions (p-07, demande de précision au lieu d'un mail) ne change pas.

## Exemples d'erreurs (test, test2)

- Décision lue comme un fait : « Le bloc, on démarre le 3 novembre. Voilà. C'est calé. » devient « Le lancement du logiciel est prévu le 3 novembre. » (fact).
- Proposition lue comme un engagement : « Je verrais bien l'inauguration le 14 novembre. Ça te va, Lou ? » devient « Romane Aubertin s'engage à organiser l'inauguration… », avec Romane comme responsable et une échéance au 14 novembre.
- Accord conditionnel lu comme une hypothèse : « Si le proprio nous lâche trois mois de franchise, on signe. » devient idea / hypothetical, alors que l'annotation est decision / active.
- Changement de responsable manqué : « Valentina, t'es sous l'eau avec les ventes, c'est Jonas qui reprend la formation. » n'est pas extrait du tout.
- Changement de responsable mal compris : dans « Euh, Agnès, ce serait mieux que ce soit toi pour le bloc, non ? … Moi je lâche », le responsable n'est pas extrait et le statut devient `proposed`.
- Dates relatives : « d'ici jeudi » devient le 24 au lieu du 22, et « je vous l'envoie le 1er » donne le 1er novembre au lieu du 1er décembre (la règle du jour seul ne s'applique pas à « le 1er »).

## Limites

- **Transcription reconstituée.** Elle est faite des seules evidence, avec une phrase par unité, sans bavardage, sans digressions ni tours coupés, et avec des locuteurs parfaits. La segmentation (F1 ≈ 0,98–1,00) et le locuteur (1,00) sont donc artificiellement faciles. Sur une vraie réunion, il faudrait aussi trouver les phrases utiles, regrouper les répliques et gérer la diarisation.
- **Effectifs petits.** 117 unités et 87 questions en test + test2, 25 demandes pour l'agent. Le responsable et l'échéance « si annotés » reposent sur 7 à 15 unités.
- **Annotation incomplète.** Le responsable et l'échéance sont souvent absents de l'annotation, même quand ils sont explicites (« Je livre le planning avant le 30 septembre »). Les scores « toutes unités » pénalisent donc des extractions correctes. Les scores « si annoté » sont plus fiables, mais très petits.
- **Réglage sur un oral différent.** Train et val ont des tournures très régulières. L'écart avec test montre que le prompt a appris ces tournures plus que les catégories.
- **Composants calibrés sur les annotations.** L'encodeur, le classifieur de relations et le seuil d'abstention de l'agent ont été entraînés ou réglés sur des unités annotées. Une partie du coût mesuré vient de ce décalage, et pourrait se réduire en les recalibrant sur des unités extraites de train/val.
- **Un seul modèle et une seule passe** (qwen3:8b, température 0). La variance entre exécutions n'est pas mesurée.

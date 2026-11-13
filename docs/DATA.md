# Données

Ce document décrit les données du projet : d'où elles viennent, comment je les ai préparées et annotées, comment
elles sont découpées, et sous quelles licences. Tous les chiffres de la section 4 sont calculés à partir des
fichiers par `python scripts/data_stats.py`.

## 1. Provenance

Je n'ai pas trouvé de corpus public de réunions annoté avec les décisions, leur statut (proposée, validée,
remplacée) et les responsables. J'ai donc créé des projets fictifs, en français.

| split | projets | comment ils ont été produits |
|---|---|---|
| train | atlas, communiquer, immobilis, lyon-second-store, rh-refresh, salle-sport, santech | Atlas écrit par Claude comme scénario exemple, puis relu et corrigé par moi (il sert d'exemple dans le prompt) ; les 6 autres générés en local avec qwen3:8b via Ollama |
| val | bio-gamme, e-com-site | générés avec qwen3:8b via Ollama |
| test | clinique-tilleuls, horizons-2026, nordstock-relais | générés avec Claude par copier-coller (mode `manual` de `generate_scenarios.py`) |
| test2 | grain-ocre-croix-rousse, pierre-angle-crm | générés avec Claude par copier-coller, comme le test, mais avec une autre consigne de style (« ton familier, phrases courtes, beaucoup de tutoiement ») et sur des projets jamais vus |

Un premier jeu de test généré avec llama3.1:8b a été rejeté pour sa mauvaise qualité. Trois projets rejetés
sont gardés hors du dépôt, dans `data/rejected/` (ignoré par git).

Le test vient d'un autre générateur (Claude) que le reste de l'entraînement (qwen3:8b), à part le projet
exemple Atlas, écrit lui aussi par Claude, pour que le modèle ne soit pas évalué sur le style de son propre
générateur. test2 vient du même générateur que le test, mais sur deux
projets jamais vus (une boutique, une agence immobilière), écrits avec une autre consigne de style.

QMSum (vraies réunions en anglais) sert uniquement à l'évaluation de la recherche : il n'est jamais utilisé pour
l'entraînement. Il est décrit dans [qmsum.md](qmsum.md).

## 2. Préparation et annotation

**Format.** Un fichier JSON par projet, décrit par la classe `Scenario` de [sae/schemas.py](../sae/schemas.py) :
personnes, réunions (date, titre, participants), unités de connaissance, relations entre unités et questions
de test. Une unité porte un type (decision, commitment, action, fact, risk, idea, question, answer), un statut
(active, proposed, hypothetical, superseded, cancelled), son locuteur, son responsable, son échéance, un résumé
(`statement`) et la phrase exacte prononcée (`evidence`).

Il n'y a pas de transcription complète : la « transcription » d'une réunion est reconstituée en enchaînant
les phrases exactes (`evidence`) de ses unités, sous la forme « Locuteur : phrase ». Elle ne contient donc ni
digressions ni phrases sans unité.

**Génération.** `scripts/generate_scenarios.py` produit en même temps les réunions, leurs phrases et la vérité
terrain (unités, statuts, relations, questions). Avec Ollama, le JSON est contraint par le schéma. En mode `manual`, le
script écrit le prompt dans `data/manual_io/` (ignoré par git) et vérifie la réponse collée.

**Normalisation.** Avant la validation, `normalize` (dans `generate_scenarios.py`) répare les oublis de forme
qui se déduisent sans ambiguïté : réunion et date de chaque unité, statut `superseded` pour une unité remplacée,
origine des questions générées, tags inconnus retirés, réponse attendue vide pour une question sans réponse.

**Validation automatique.** Le schéma refuse un scénario incohérent : identifiant dupliqué, personne inconnue,
relation vers une unité inconnue, remplacement d'une unité plus récente, unité remplacée encore active, réunions
mal datées, adresse e-mail qui n'est pas en example.org. `python scripts/validate_data.py` valide tout le
corpus, vérifie qu'aucun projet n'est partagé entre train, val et test, et signale les personnes communes à
train et test.

**Relecture manuelle.** J'ai relu chaque projet, aidé par `python scripts/review_scenarios.py`, qui signale les
cas suspects que le schéma ne voit pas : un engagement sans responsable, une hypothèse marquée comme décision
active, une relation entre deux sujets sans mot commun, une cause postérieure à l'action, une réponse attendue
remplacée. Deux corrections étaient systématiques : ajouter les responsables manquants des engagements et des
actions, et remettre dans le bon sens les relations `creates_action` (de la cause vers l'action).

**Origine des questions.**

- `generated` : écrites par le générateur en même temps que le projet ;
- `manual` : les 10 questions d'Atlas, écrites par Claude avec le scénario exemple, puis relues par moi ;
- `assisted` : questions du test dont les idées venaient en partie de suggestions d'une IA, que j'ai choisies
  ou adaptées, puis reformulées par Claude dans un style plus oral ou plus poli ; j'ai vérifié chaque réponse
  attendue.

**Origine des demandes.** Les demandes de `data/agent_eval/test.jsonl`, `holdout.jsonl` et `test2.jsonl`
(origin `suggested`) ont été proposées par un assistant d'IA, puis vérifiées par moi, en particulier les
destinataires attendus (`expected_recipient`).

**Paramètres annotés à la main.** `data/agent_eval/slots.jsonl` contient la personne, le projet et l'échéance
attendus pour 35 demandes d'action. Je les ai validés à la main avant toute mesure
([dossiers_et_slots.md](dossiers_et_slots.md)).

## 3. Découpage et fichiers

Les splits sont faits par projet : un projet est dans un seul split. Aucune personne n'est commune à test ou
test2 et à un autre split. En revanche, 7 noms sont communs à train et val (par exemple Sophie Laurent,
Thomas Girard) : qwen3:8b réutilise souvent les mêmes prénoms et noms. train sert à l'entraînement, val aux réglages (taille de fenêtre,
seuils, K du reranking, consignes de rédaction), test et test2 à une mesure unique, avec les réglages figés.

| dossier | contenu |
|---|---|
| `data/scenarios/<split>/` | les projets annotés (vérité terrain) |
| `data/agent_eval/exemple.jsonl` | 6 demandes sur Atlas, pour mettre au point l'évaluation de l'agent |
| `data/agent_eval/test.jsonl` | 12 demandes sur clinique-tilleuls (type de sortie, destinataire et unités attendus) |
| `data/agent_eval/holdout.jsonl` | 24 demandes sur horizons-2026 et nordstock-relais, gardées pour la fin |
| `data/agent_eval/test2.jsonl` | 25 demandes sur les deux projets de test2 (validation finale de l'agent) |
| `data/agent_eval/slots.jsonl` | personne, projet et échéance attendus pour 35 demandes reprises des fichiers ci-dessus |
| `data/extracted/<split>/` | unités extraites automatiquement de la transcription par qwen3:8b ([extraction.md](extraction.md)) |
| `data/extracted/<split>/alignment/` | correspondance entre unités annotées et unités extraites, pour l'évaluation |
| `data/external/qmsum/` | QMSum val et test, retéléchargés par `eval_qmsum.py --download` (ignoré par git) |

## 4. Statistiques

Calculées par `python scripts/data_stats.py`.

| | train | val | test | test2 | total |
|---|---|---|---|---|---|
| projets | 7 | 2 | 3 | 2 | 14 |
| réunions | 28 | 8 | 15 | 10 | 61 |
| personnes | 36 | 10 | 15 | 10 | 71 |
| unités | 119 | 34 | 70 | 47 | 270 |
| relations | 45 | 10 | 37 | 25 | 117 |
| questions | 70 | 20 | 73 | 28 | 191 |
| questions sans réponse | 7 | 2 | 10 | 4 | 23 |

Unités par type :

| | train | val | test | test2 | total |
|---|---|---|---|---|---|
| decision | 29 | 8 | 17 | 12 | 66 |
| commitment | 19 | 4 | 6 | 4 | 33 |
| action | 8 | 4 | 8 | 4 | 24 |
| fact | 20 | 6 | 21 | 14 | 61 |
| risk | 6 | 2 | 4 | 3 | 15 |
| idea | 27 | 8 | 8 | 6 | 49 |
| question | 6 | 2 | 3 | 2 | 13 |
| answer | 4 | 0 | 3 | 2 | 9 |

Unités par statut (aucune unité `cancelled`) :

| | train | val | test | test2 | total |
|---|---|---|---|---|---|
| active | 80 | 22 | 51 | 35 | 188 |
| proposed | 0 | 0 | 2 | 0 | 2 |
| hypothetical | 14 | 6 | 3 | 2 | 25 |
| superseded | 25 | 6 | 14 | 10 | 55 |

Relations par type :

| | train | val | test | test2 | total |
|---|---|---|---|---|---|
| supersedes | 25 | 6 | 14 | 10 | 55 |
| supports | 8 | 2 | 10 | 4 | 24 |
| answers | 6 | 2 | 3 | 2 | 13 |
| creates_action | 3 | 0 | 5 | 5 | 13 |
| unrelated | 3 | 0 | 3 | 2 | 8 |
| contradicts | 0 | 0 | 2 | 2 | 4 |

Questions par origine et par intention :

| | train | val | test | test2 | total |
|---|---|---|---|---|---|
| generated | 60 | 20 | 40 | 28 | 148 |
| manual | 10 | 0 | 0 | 0 | 10 |
| assisted | 0 | 0 | 33 | 0 | 33 |
| intention answer | 63 | 18 | 62 | 23 | 166 |
| intention compare | 7 | 2 | 6 | 2 | 17 |
| intention draft_email | 0 | 0 | 3 | 2 | 5 |
| intention create_task | 0 | 0 | 2 | 1 | 3 |

Questions par tag (une question peut en avoir plusieurs) :

| | train | val | test | test2 | total |
|---|---|---|---|---|---|
| multi_meeting | 13 | 4 | 19 | 9 | 45 |
| budget_change | 19 | 6 | 9 | 4 | 38 |
| current_decision | 7 | 2 | 21 | 6 | 36 |
| date_change | 14 | 4 | 8 | 4 | 30 |
| commitment | 12 | 4 | 7 | 3 | 26 |
| unanswerable | 7 | 2 | 10 | 4 | 23 |
| person_cited_vs_owner | 9 | 2 | 7 | 3 | 21 |
| proposal_vs_decision | 7 | 2 | 10 | 2 | 21 |
| polite | 0 | 0 | 16 | 2 | 18 |
| compare | 7 | 2 | 6 | 2 | 17 |
| paraphrase | 6 | 2 | 6 | 3 | 17 |
| informal | 6 | 2 | 4 | 3 | 15 |
| negation | 4 | 0 | 7 | 4 | 15 |
| hypothesis_vs_decision | 5 | 0 | 6 | 2 | 13 |
| conditional_agreement | 1 | 0 | 6 | 2 | 9 |
| owner_change | 0 | 0 | 6 | 2 | 8 |

Demandes d'évaluation de l'agent :

| fichier | demandes | projets | types de sortie attendus |
|---|---|---|---|
| exemple.jsonl | 6 | atlas | answer 1, clarify 1, compare 1, email_draft 1, no_action 1, task_draft 1 |
| test.jsonl | 12 | clinique-tilleuls | answer 2, clarify 3, compare 1, dossier 1, email_draft 2, event_draft 1, no_action 1, task_draft 1 |
| holdout.jsonl | 24 | horizons-2026, nordstock-relais | answer 5, clarify 5, compare 2, dossier 2, email_draft 4, event_draft 2, no_action 2, task_draft 2 |
| test2.jsonl | 25 | grain-ocre-croix-rousse, pierre-angle-crm | answer 4, clarify 5, compare 2, dossier 2, email_draft 6, event_draft 2, no_action 2, task_draft 2 |
| slots.jsonl | 35 | reprises de test.jsonl, holdout.jsonl et test2.jsonl | personne, projet, échéance |

Unités extraites :

| split | projets | unités extraites | unités annotées alignées sur une unité extraite |
|---|---|---|---|
| train | 7 | 119 | 119 / 119 |
| val | 2 | 34 | 34 / 34 |
| test | 3 | 69 | 69 / 70 |
| test2 | 2 | 45 | 45 / 47 |

## 5. Licences et vie privée

**Personnes.** Tous les projets, personnes et adresses sont fictifs. Les adresses utilisent le domaine
réservé example.org, et le schéma refuse toute autre adresse. Aucune donnée personnelle réelle n'est utilisée.

**Modèles utilisés.** Licences vérifiées sur les métadonnées Hugging Face (API `huggingface.co/api/models`)
et, pour les modèles Ollama, avec `ollama show`, le 4 novembre 2026 :

| modèle | rôle | licence |
|---|---|---|
| intfloat/multilingual-e5-base | encodeur de base, fine-tuné | MIT |
| intfloat/multilingual-e5-small | comparaison de taille | MIT |
| BAAI/bge-m3 | référence | MIT |
| BAAI/bge-reranker-v2-m3 | reranking (non intégré à l'agent) | Apache 2.0 |
| MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 | classifieur de relations (fine-tuné) | MIT |
| qwen3:8b (Qwen/Qwen3-8B) | génération de train et val, extraction, rédaction | Apache 2.0 |
| qwen3.5:4b | option de rédaction | Apache 2.0 |
| llama3.1:8b | premier test, rejeté | Llama 3.1 Community License |

Les modèles que j'ai entraînés (`models/`) dérivent d'e5-base, d'e5-small et de mDeBERTa, tous sous licence
MIT. Ils ne sont pas versionnés.

**Sorties de Claude.** Le scénario Atlas, le jeu de test, test2 et les questions assistées (idées en partie
suggérées par une IA, reformulées par Claude) ont été produits avec Claude (Anthropic). Leur usage relève des
conditions d'utilisation d'Anthropic, auxquelles je renvoie sans les résumer.

**QMSum.** Le dépôt est sous licence MIT. AMI et ICSI sont sous CC BY 4.0. Les commissions du Parlement gallois
et de la Chambre des communes du Canada sont réutilisables sans but commercial. Je ne redistribue pas ces
données. Le détail et les liens sont dans [qmsum.md](qmsum.md).

## 6. Limites

- Données synthétiques : les réunions sont générées par des modèles de langue, l'oral est plus propre
  qu'en vrai, et les questions et les unités partagent souvent le même vocabulaire. L'évaluation sur QMSum
  montre que les conclusions ne se transfèrent pas telles quelles à de vraies réunions.
- Projets proches : beaucoup de projets suivent le même schéma, un budget proposé puis validé, une date
  fixée puis reportée, un responsable remplacé. C'est voulu pour tester les statuts, mais cela rend les
  splits plus proches entre eux que de vrais projets.
- Petits effectifs : 14 projets, 270 unités, 191 questions, dont 73 sur test et 28 sur test2. Une question
  pèse 1,4 point sur test et 3,6 points sur test2.
- Questions assistées : les idées des 33 questions `assisted` venaient en partie de suggestions d'une IA, que
  j'ai choisies ou adaptées, puis reformulées par Claude. Elles sont plus variées que les questions générées,
  mais elles ne sont pas écrites par de vrais utilisateurs.
- Vérité terrain produite par les générateurs : les annotations viennent du même modèle que le texte. Je
  les ai relues, mais je n'ai pas fait d'accord inter-annotateurs.

# Rédaction des réponses par un modèle local

Pour les intentions answer et compare, l'agent affichait la première unité trouvée, sans qu'aucun modèle de
langue ne lise les résultats. Sur grain-ocre-croix-rousse (test2), « On a le droit de mettre notre logo sur la
vitrine ? » répondait par la question de Lou (u-014) au lieu de la réponse d'Inaya (u-018), et les
contradictions (deux vendeurs selon Victor, u-015, trois selon Kofi, u-021) n'étaient pas signalées.

Une étape de rédaction a été ajoutée (`sae/agent/writer.py`). Elle est encadrée par le code et évaluée à part.
**Les métriques de recherche (docs/reranking_et_systeme.md, docs/chunking.md…) et la rédaction sont évaluées
séparément** : la recherche, la mémoire et l'abstention en règles sont les mêmes pour tous les systèmes
comparés ici, seule la façon de rédiger la réponse change. Aucun résultat déjà écrit dans docs/ n'a été modifié.

```bash
python scripts/agent.py --writer qwen3:8b --chat          # ligne de commande (none par défaut)
python scripts/eval_writer.py --splits train val          # mise au point
python scripts/eval_writer.py --splits test test2         # mesure unique
```

## 1. Fonctionnement

**Ce qui ne change pas.** Les décisions restent en règles : intention, hypothèses, demandes de précision,
destinataires, brouillons, et l'abstention de l'agent (stratégie score), qui passe avant la rédaction. La
rédaction ne remplace que le texte et les sources d'une réponse answer ou compare ; le type de résultat ne
change pas.

**Entrée.** La question, le nom du projet et les K premières unités après la mémoire (K = 6, réglable de 5 à 8
par `--writer-k`), chacune avec son identifiant court (u-018), sa date, sa réunion, son locuteur, son type, son
statut et sa phrase exacte (plus le résumé). Le statut vient de la mémoire de l'agent : « remplacée » si la
mémoire (relations prédites par défaut) rend l'unité inactive, sinon « en vigueur », « proposée »,
« hypothétique » ou « annulée ». Le statut annoté « superseded » n'est jamais montré tel quel au modèle : une
unité dont les relations prédites ont manqué le remplacement apparaît « en vigueur », comme pour l'agent.

**Consignes** (français, 1 à 3 phrases, uniquement à partir des unités) : vérifier d'abord qu'une unité répond
au sujet précis de la question (le responsable du projet n'est pas le responsable de la formation) ; sinon
dire « Je n'ai pas trouvé cette information dans les réunions. » ; une question posée en réunion n'est pas une
réponse ; privilégier les unités en vigueur et présenter les anciennes comme telles ; signaler un désaccord
entre unités en vigueur ; citer les identifiants utilisés.

**Sortie JSON contrainte** : `answer`, `cited_ids`, `found`, `disagreement` (schéma Pydantic passé à Ollama
dans `format`, comme pour l'extraction).

**Vérification par le code.** La réponse est rejetée, et l'agent revient à l'affichage en règles (première
unité), si :

- un identifiant cité ne fait pas partie des unités fournies ;
- le modèle dit avoir trouvé l'information sans citer d'unité, ou la réponse est vide ;
- la sortie ne respecte pas le schéma ;
- un nombre ou un nom propre de la réponse est absent des unités citées (texte, locuteur, réunion) et du nom du
  projet. Cette règle a été ajoutée pendant la mise au point, après un cas où le modèle attribuait le budget du
  projet Atlas à un « salon professionnel de Lyon » qui n'existe dans aucune réunion.

**Modèle et mémoire GPU.**

- **Modèles.** qwen3:8b par défaut, qwen3.5:4b en option. Avec qwen3.5:4b, Ollama ignore le schéma
  quand la réflexion est désactivée, et avec la réflexion il met environ 200 s sans rien renvoyer. Pour ce
  modèle, le JSON est donc demandé dans le prompt, puis validé par le même schéma.
- **Libération de la mémoire.** La mémoire et les relations sont calculées au démarrage. La mémoire GPU du
  classifieur de relations est ensuite libérée (`gc`, `torch.cuda.empty_cache`), et seul le bi-encodeur reste
  chargé.
- **Contrôle au démarrage.** `build_agent` compare la mémoire libre aux besoins du modèle et prévient si elle
  ne suffit pas.
- **Taille dans Ollama.** Mesurée avec num_ctx 4096 : 5 700 Mio pour qwen3:8b. qwen3.5:4b, multimodal, en
  prend 6 200 : il n'économise pas de mémoire ici, et l'avertissement ne le propose donc pas. S'il manque de
  la place, Ollama met une partie des couches sur le CPU : c'est plus lent, mais ça fonctionne.
- **Ollama non lancé.** Si Ollama ne répond pas ou si le modèle est absent, l'agent l'écrit et revient à
  l'affichage en règles.

**Intégration.**

- **Ligne de commande.** L'option `--writer` (none, qwen3:8b ou qwen3.5:4b) est ajoutée à scripts/agent.py et
  scripts/eval_agent.py. Elle vaut none par défaut, ce qui ne change rien aux évaluations existantes.
- **Interface.** La rédaction est activée par défaut (qwen3:8b), avec un réglage dans « Options avancées ».
  Les sources affichées dans « Détails » sont les unités citées par le modèle. Si le modèle signale un
  désaccord, une ligne l'indique sous la réponse.
- **Les deux cas cités plus haut.** Le logo reçoit maintenant la réponse d'Inaya (« Non, le bail interdit le
  flocage… », u-014 et u-018). Pour les vendeurs, la réponse donne les deux estimations et `disagreement` vaut
  true.

## 2. Protocole

Toutes les questions answer et compare des scénarios (`queries` des fichiers de data/scenarios), avec la
recherche de l'agent (models/e5b-both-type-s1, type dans le texte) et la mémoire prédite (models/relations-cross,
seuil 0,45). Quatre systèmes sur les mêmes questions :

| système | ce qu'il affiche |
|---|---|
| première unité | l'affichage actuel, sans abstention |
| première unité + abstention | l'agent actuel (abstention en règles, stratégie score) |
| rédaction | le modèle ; « pas trouvé » compte comme une abstention ; rejet : retour à la première unité |
| abstention + rédaction | l'abstention en règles d'abord, la rédaction ensuite : ce que fait l'agent avec `--writer` |

Mesures (unités attendues et remplacements annotés, jamais montrés au modèle) :

- **cite une unité pertinente** : au moins une unité attendue parmi les sources (questions avec réponse) ;
- **remplacée présentée comme en vigueur** : une unité annotée comme remplacée est citée sans aucune de ses
  versions ultérieures (questions avec réponse). La mesure est automatique et un peu large : elle compte aussi
  une ancienne version citée comme contexte (voir les erreurs) ;
- **abstention justifiée** (questions sans réponse) et **abstention à tort** (questions avec réponse) ;
- **rejets** de la vérification et **temps moyen** par réponse rédigée.

Mise au point des consignes sur **train et val uniquement**, en trois tours ; test et test2 ont été mesurés une
seule fois, consignes figées.

## 3. Mise au point (train, val)

| tour | changement | train : cite une unité pertinente | train : remplacée comme en vigueur | train : abstention justifiée (rédaction seule) | rejets |
|---|---|---|---|---|---|
| première unité | — | 0,921 | 0,032 | 0,000 | — |
| 1 | consignes initiales | 0,968 | 0,000 | 0,000 (0/7) | 0,014 |
| 2 | vérifier le sujet précis, réponse fixe si rien ne répond | 0,937 | 0,000 | 0,143 | 0,000 |
| 3 | nom du projet dans le prompt, contrôle des noms et nombres | 0,968 | 0,000 | 0,000 | 0,029 |

Sur val (20 questions), les trois tours donnent 1,000 contre 0,833 pour la première unité.

- **Tour 1.** Sur les 9 questions sans réponse, le modèle répondait par un sujet voisin, par exemple
  « Julien est le responsable de la gamme » pour « Qui forme les agents ? ».
- **Tour 2.** La réponse fixe a fait apparaître une invention (le budget d'Atlas attribué au salon de Lyon) et
  deux citations moins bonnes.
- **Tour 3.** Le contrôle des noms et des nombres rejette l'invention.
- **Ce qui reste.** Le modèle ne s'abstient presque jamais de lui-même sur train. Les questions sans réponse
  restent couvertes par l'abstention en règles, qui passe avant la rédaction.

## 4. Résultats (test, test2 ; qwen3:8b, mesure unique)

**test** : 68 questions (58 avec réponse, 10 sans) · 2 rejets (0,029) · 6,3 s par réponse · 1 désaccord signalé

| système | cite une unité pertinente | remplacée présentée comme en vigueur | abstention justifiée | abstention à tort |
|---|---|---|---|---|
| première unité | 0,828 | 0,103 | 0,000 | 0,000 |
| première unité + abstention | 0,810 | 0,103 | 0,500 | 0,017 |
| rédaction | **0,966** | 0,052 | 0,600 | 0,000 |
| abstention + rédaction | 0,948 | 0,052 | **0,900** | 0,017 |

**test2** : 25 questions (21 avec réponse, 4 sans) · aucun rejet · 6,4 s par réponse · aucun désaccord signalé

| système | cite une unité pertinente | remplacée présentée comme en vigueur | abstention justifiée | abstention à tort |
|---|---|---|---|---|
| première unité | 0,810 | 0,095 | 0,000 | 0,000 |
| première unité + abstention | 0,810 | 0,095 | 0,250 | 0,000 |
| rédaction | **0,905** | **0,000** | 0,750 | 0,000 |
| abstention + rédaction | 0,905 | 0,000 | 0,750 | 0,000 |

Par intention, « cite une unité pertinente » passe de 0,846 à 0,981 (answer) et de 0,667 à 0,833 (compare) sur
test, de 0,789 à 0,895 (answer) et reste à 1,000 (compare) sur test2.

**Temps.** Les temps ont été mesurés pendant qu'une autre interface Streamlit occupait 3 Go du GPU. Ollama n'a
donc mis que 2 380 des 5 700 Mio de qwen3:8b sur le GPU (environ 400 Mio libres au moment de charger le
modèle), et le reste tournait sur le CPU. Sur un GPU libre, il faut s'attendre à moins. Pour comparaison,
l'affichage en règles prend environ 8 ms (docs/reranking_et_systeme.md).

## 5. Vérification manuelle (10 réponses de test2)

Questions q-01 à q-09 et q-11 de grain-ocre-croix-rousse : chaque affirmation est comparée aux unités citées.

| question | réponse (résumée) | unités citées | appui |
|---|---|---|---|
| q-01 inauguration, finalement ? | reportée au 28 novembre | u-016 | oui |
| q-02 budget des travaux ? | validé à 95 000 euros | u-006 | oui |
| q-03 qui valide les devis ? | Kofi | u-010 | oui |
| q-04 recrutement du responsable ? | Victor initialement, Romane a repris (14 novembre) | u-012, u-017 | oui, ancienne version présentée comme telle |
| q-05 floquer la vitrine ? | non, le bail l'interdit ; adhésif amovible | u-014, u-018 | oui |
| q-06 bail signé, à quelle condition ? | signé après trois mois de franchise | u-008, u-011 | oui |
| q-07 stand au marché avant l'ouverture ? | « aucun stand n'est prévu » | u-001 | **non** : u-001 ne parle que de l'emplacement |
| q-08 jours d'ouverture ? | mardi au dimanche, dimanche 9 h – 13 h | u-019 | oui |
| q-09 évolution du nombre de vendeurs ? | deux, puis trois le week-end | u-015, u-021 | oui (désaccord non marqué, présenté comme une évolution) |
| q-11 qui ajoute les commerçants ? | Lou | u-022 | oui |

Une affirmation sans appui sur 10 réponses (q-07). L'unité attendue (u-005, l'idée d'un stand éphémère)
n'était pas fournie : les relations prédites la donnent à tort comme remplacée par l'inauguration (erreur déjà
relevée dans docs/dossiers_et_slots.md), donc la mémoire la fait descendre. Le modèle a conclu à l'absence au lieu
de dire qu'il ne trouvait pas, et le contrôle des noms et nombres ne peut pas voir une affirmation négative.

## 6. Erreurs et limites

- **Rejets à tort** (test) : « réduit de 9 000 euros » (45 000 − 38 000, calcul exact mais absent du texte) et
  « réunion du 8 septembre 2026 » (date de l'unité, fournie au modèle mais au format 2026-09-08). Les deux
  reviennent à la première unité : la vérification est volontairement stricte.
- **Mesure des unités remplacées un peu large** : les 3 cas comptés sur test (nordstock-relais q-03, q-09,
  a-06) citent l'acceptation conditionnelle de l'exclusivité (u-007) comme contexte, avec la garantie du taux de
  service (u-016), et disent bien que l'exclusivité est validée ; la mesure les compte parce que u-017 (la
  validation) n'est pas citée. À la lecture, aucune ne présente l'ancienne version comme actuelle.
- **Abstention** : le modèle seul s'abstient sur test (6/10) et test2 (3/4), mais presque jamais sur train ;
  l'abstention en règles reste nécessaire, et la combinaison donne 9/10 sur test.
- **La rédaction ne corrige pas la recherche** : si l'unité attendue n'est pas dans les K premières (lyon
  q-01 sur train, u-005 sur test2), le modèle ne peut pas la citer.
- **Effectifs petits** : 58 et 21 questions avec réponse, 10 et 4 sans réponse ; une question pèse 1,7 point
  (test), 4,8 points (test2) et 25 points sur l'abstention de test2.
- **Mêmes données pour la mise au point et l'écriture des consignes** : train et val ont des formulations
  proches de test (questions générées) ; test2 est le jeu le plus indépendant.
- **Non-déterminisme** : température 0 et graine fixée, mais Ollama n'est pas strictement reproductible
  (docs/reranking_et_systeme.md, extraction).
- **qwen3.5:4b** : vérifié seulement sur val (cite une unité pertinente 1,000, 2 rejets sur 20, 5,0 s par
  réponse), sans mesure sur test ; il n'économise pas de mémoire GPU ici.

## 7. Non-régression

Avec `--writer none` (valeur par défaut en ligne de commande) : la démonstration de clinique-tilleuls est
identique octet pour octet à la sortie d'avant ce travail, et `eval_agent.py` sur test2 donne toujours 24/25
(0,960), avec un rapport identique.

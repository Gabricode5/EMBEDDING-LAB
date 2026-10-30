# Abstention de l'agent sur les questions sans réponse

Avant ce travail, l'agent répondait toujours avec l'unité la mieux classée, même quand l'information n'existe pas dans les réunions. Le sujet demande de signaler explicitement une information manquante plutôt que d'inventer. L'agent peut maintenant répondre `no_answer` : « Je n'ai pas trouvé cette information dans les réunions », suivi des deux unités les plus proches, présentées comme « sujets proches » et jamais comme réponse. Le résultat n'a alors aucune source, et la raison de l'abstention figure dans le payload.

## Méthode

L'abstention ne concerne que les questions d'information (`_answer` et `_compare`). Pour chacune, l'agent calcule trois indices sur la recherche (`Agent.signals`) :

- **score** : similarité (cosinus e5, après la mémoire) de la 1re unité ;
- **écart** : score de la 1re unité moins celui de la 2e ;
- **lexical** : au moins une racine commune entre les mots du sujet (`topic_words` + `stems`, comme pour le choix du destinataire) et l'une des 3 premières unités. S'il n'y a aucun mot de sujet exploitable, ce test ne compte pas comme un échec.

Stratégies comparées (option `--abstention` de `scripts/agent.py` et `scripts/eval_agent.py`, paramètre `abstention` de `Agent`) :

| stratégie | l'agent s'abstient si… |
|---|---|
| `none` | jamais (comportement historique) |
| `lexical` | aucune racine commune |
| `score` | score < seuil |
| `gap` | écart < seuil |
| `both` | aucune racine commune **et** score < seuil |
| `either` | aucune racine commune **ou** score < seuil |

**Calibrage** (`python scripts/eval_abstention.py --calibrate`, refusé sur un autre split que train et val). On calcule un seuil par intention détectée, sur train + val réunis. Les seuils candidats sont les milieux entre deux valeurs observées consécutives. On retient celui qui fait le moins d'abstentions à tort, puis le plus d'abstentions justifiées, puis qui a la marge la plus large. Une intention sans question sans réponse n'a pas de seuil, ce qui est le cas de `compare` dans les données. La stratégie est ensuite choisie selon le même ordre (moins d'abstentions à tort, puis plus d'abstentions justifiées, puis meilleure 1re source). Les seuils ont été figés dans `ABSTENTION_THRESHOLDS` (`sae/agent/engine.py`) **avant** toute mesure sur test et test2.

**Compromis retenu.** Refuser une question qui a une réponse est plus gênant que répondre à côté sur une question sans réponse : dans le second cas, la source citée permet à l'utilisateur de voir l'erreur, alors que dans le premier, l'information est perdue. On impose donc zéro abstention à tort sur train + val, et on maximise les abstentions justifiées sous cette contrainte. `either` en détecte plus (8/9) mais refuse 3 bonnes questions sur 81 : elle est écartée.

## Résultats sur train + val (calibrage)

Effectifs faibles : 9 questions sans réponse en tout, dont **seulement 2 dans val**.

| stratégie | seuil | abstention justifiée (sans réponse) | abstention à tort (avec réponse) | 1re source correcte (répondues) |
|---|---|---|---|---|
| none | - | 0/9 (0,00) | 0/81 (0,00) | 69/81 (0,85) |
| lexical | - | 3/9 (0,33) | 3/81 (0,04) | 67/78 (0,86) |
| **score** | answer < 0,5121 | **6/9 (0,67)** | **0/81 (0,00)** | 69/81 (0,85) |
| gap | answer < 0,0031 | 1/9 (0,11) | 0/81 (0,00) | 69/81 (0,85) |
| both | answer < 0,5379 | 2/9 (0,22) | 0/81 (0,00) | 69/81 (0,85) |
| either | answer < 0,5121 | 8/9 (0,89) | 3/81 (0,04) | 67/78 (0,86) |

Détail de `score` : train 5/7 justifiées et 0/63 à tort ; val 1/2 justifiées et 0/18 à tort. **Stratégie retenue : `score`**, désormais utilisée par défaut (`--abstention none` rétablit l'ancien comportement). L'écart entre 1re et 2e unité ne sépare pas les cas : les questions de comparaison, qui ont une réponse, ont naturellement des écarts proches de 0.

## Résultat final sur test et test2 (mesuré une seule fois, réglage figé)

| split | abstention justifiée | abstention à tort | 1re source correcte (répondues) |
|---|---|---|---|
| test | 5/10 (0,50) | 2/58 (0,03) | 47/56 (0,84) |
| test2 | 1/4 (0,25) | 0/21 (0,00) | 15/21 (0,71) |
| **test + test2** | **6/14 (0,43)** | **2/79 (0,03)** | 62/77 (0,81) |

Pour comparaison, `none` donne 0/14 justifiées, 0/79 à tort et 63/79 en 1re source. Les deux abstentions à tort de test sont :

- une question très informelle (« Salut, est-ce que tu pourrais me confirmer la date de démarrage au bloc, la version qui tient maintenant ? », score 0,499) ;
- une question de comparaison classée `answer` par les règles d'intention (« Comment le budget a-t-il évolué entre la première estimation et la validation ? », 0,379). Elle reçoit le seuil de `answer`, alors que les comparaisons ont des scores plus bas.

Sur test, `lexical` et `both` ne font aucune abstention à tort, mais en détectent moins (3/14). Le réglage n'a pas été modifié après cette observation.

Non-régression : `eval_agent.py` sur test2 donne 24/25 (identique), la démo `clinique-tilleuls` est identique au caractère près, et `data/agent_eval/test.jsonl` et `holdout.jsonl` ne montrent aucun changement de type de sortie ni de destinataire. Les demandes d'action ne passent pas par l'abstention.

## Limites

- **Effectifs très faibles** : 9 questions sans réponse pour calibrer (2 dans val), 14 pour la mesure finale (4 dans test2). Une question de plus ou de moins fait varier les taux de 10 à 25 points.
- **Marge étroite** : avec ce seuil, rien ne change entre 0,509 et 0,515. Un léger changement de formulation ou d'encodeur suffit à faire basculer une question.
- **Biais de calibrage** : l'encodeur e5 a été fine-tuné sur les questions de train, donc les scores de train sont optimistes pour les questions avec réponse. val n'a pas ce biais, mais elle est minuscule.
- **Questions sans réponse stéréotypées** : en train/val, elles suivent deux modèles (« Quel budget a été alloué à … ? », « Qui s'occupe de former … ? »). La capacité à repérer d'autres formes d'information absente n'est mesurée que par test et test2.
- **Pas de seuil pour `compare`** : il n'existe aucune question de comparaison sans réponse. Une comparaison mal reconnue comme `answer` peut donc être refusée à tort.
- **Question informelle** : les formules de politesse diluent la similarité. Une piste, à valider sur train + val, serait de calculer le score sur `topic_query` plutôt que sur le message brut.

# Évaluation des dossiers et des paramètres extraits des demandes

Deux mesures du sujet qui manquaient : la qualité des dossiers produits par l'agent (`_build_dossier`) et
l'exactitude des paramètres (personne, projet, échéance) extraits des demandes par `understand`.
Rien n'a été réglé : l'agent n'a pas été modifié (hormis le retrait d'un import inutilisé), et test et test2
ont été mesurés une seule fois.

```bash
python scripts/eval_dossier.py --splits test test2
python scripts/eval_slots.py --file data/agent_eval/slots.jsonl
```

## 1. Dossiers

### Ce que contient un dossier

`_build_dossier` prend toutes les unités du projet qui ne sont pas inactives (statut annoté superseded ou
cancelled, ou, avec les relations prédites, unités ayant un successeur prédit) et les range par type :

| section | contenu |
|---|---|
| Décisions en vigueur | décisions |
| Engagements et actions | engagements et actions, avec « (responsable : X, échéance : Y) » si l'unité a un responsable |
| Risques | risques |
| Propositions et idées non tranchées | idées |
| Historique des changements | un remplacement de la mémoire par affirmation, qui cite [ancienne, nouvelle] |

Chaque affirmation (`Claim`) reprend le statement et cite l'unité ; le schéma interdit une affirmation sans
unité. Le statut n'est pas filtré : une décision `proposed` non remplacée apparaîtrait en « en vigueur ».

### Faits attendus

Construits à partir des annotations de `data/scenarios`, par projet :

| section | faits attendus |
|---|---|
| Décisions en vigueur | décisions au statut active |
| Engagements et actions | engagements et actions au statut active, avec leur responsable annoté |
| Risques | risques au statut active |
| Propositions et idées | idées non remplacées ni annulées (active, proposed, hypothetical) |
| Historique | relations `supersedes` annotées (ancienne → nouvelle) |

### Mesures

- **couverture** : part des faits attendus cités dans la bonne section ;
- **citations exactes** : part des affirmations dont l'unité citée appartient aux faits attendus de leur section
  (une unité remplacée en « en vigueur », un fait en « engagements », un remplacement non annoté dans
  l'historique sont inexacts) ;
- **obsolètes présentées comme actuelles** : part des affirmations des quatre sections « actuelles » qui citent
  une unité annotée superseded ou cancelled ;
- **non étayées** : part des affirmations sans unité, ou citant une unité absente de la mémoire ;
- **responsable correct** : parmi les engagements couverts qui ont un responsable annoté, part dont le
  responsable affiché est le bon.

Trois conditions : unités et relations annotées ; unités annotées et relations prédites
(`models/relations-cross`, seuil 0,45) ; unités extraites (`data/extracted/<split>`) et relations prédites.
Pour les unités extraites, chaque unité citée est ramenée à l'unité annotée par l'alignement déjà calculé
(`data/extracted/<split>/alignment`) ; une unité extraite non alignée compterait comme citation inexacte
(aucune ne l'est en pratique). Le dossier n'utilise pas la recherche : l'agent est construit avec BM25.

### Résultats

Entre parenthèses, la taille du dénominateur.

**test** (3 projets, 46 faits attendus)

| condition | couverture | citations exactes | obsolètes présentées comme actuelles | non étayées | responsable correct |
|---|---|---|---|---|---|
| annotées · relations annotées | 1,000 (46) | 1,000 (46) | 0,000 (32) | 0,000 | 1,000 (11) |
| annotées · relations prédites | 0,783 (46) | 0,818 (44) | 0,167 (36) | 0,000 | 1,000 (10) |
| extraites · relations prédites | 0,652 (46) | 0,682 (44) | 0,132 (38) | 0,000 | 0,800 (10) |

**test2** (2 projets, 31 faits attendus)

| condition | couverture | citations exactes | obsolètes présentées comme actuelles | non étayées | responsable correct |
|---|---|---|---|---|---|
| annotées · relations annotées | 1,000 (31) | 1,000 (31) | 0,000 (21) | 0,000 | 1,000 (6) |
| annotées · relations prédites | 0,839 (31) | 0,867 (30) | 0,091 (22) | 0,000 | 1,000 (6) |
| extraites · relations prédites | 0,484 (31) | 0,517 (29) | 0,250 (28) | 0,000 | 1,000 (6) |

Couverture par section :

| split · condition | décisions | engagements | risques | idées | historique |
|---|---|---|---|---|---|
| test · annotées, relations annotées | 13/13 | 11/11 | 3/3 | 5/5 | 14/14 |
| test · annotées, relations prédites | 12/13 | 10/11 | 3/3 | 5/5 | 6/14 |
| test · extraites, relations prédites | 9/13 | 10/11 | 3/3 | 4/5 | 4/14 |
| test2 · annotées, relations annotées | 10/10 | 6/6 | 3/3 | 2/2 | 10/10 |
| test2 · annotées, relations prédites | 10/10 | 6/6 | 3/3 | 1/2 | 6/10 |
| test2 · extraites, relations prédites | 5/10 | 6/6 | 1/3 | 2/2 | 1/10 |

### Erreurs typiques

- **Remplacement manqué → information périmée présentée comme actuelle** (relations prédites) : « La formation
  des équipes du bloc opératoire est confiée à Inès » reste dans « Engagements et actions » alors qu'elle a été
  réattribuée à Agnès ; de même « Hélène est chargée de réserver le transport en car », « Grégoire est chargé de
  négocier le contrat de niveau de service ». Les changements de responsable sont les plus souvent manqués.
- **Faux remplacement** : « Lou évoque un stand éphémère au marché » → « remplacé par : L'inauguration est fixée
  au 14 novembre » ; le classifieur relie deux unités voisines sans rapport de remplacement. Un faux remplacement
  retire aussi l'ancienne unité des sections actuelles (« Rachid s'engage à présenter le budget » disparaît
  des engagements).
- **Confirmation prise pour un remplacement** : « Le démarrage du logiciel au bloc le 17 novembre est confirmé »
  est prédit comme remplaçant « Le démarrage au bloc est reporté au 17 novembre », que l'annotation garde en
  vigueur ; le report disparaît des décisions. Même confusion sur le CRM : la proposition du 16 novembre est
  reliée directement au report du 30 novembre, en sautant la décision du 16 novembre.
- **Erreur de type à l'extraction → mauvaise section** (unités extraites) : « Le contrat avec le domaine des
  Grands Chênes est signé définitivement » extrait comme fait disparaît du dossier ; « Mei Chen-Aubert doit
  reprendre le SLA » (décision annotée) part en engagement ; « Nadia Ferhat propose de remplacer le tableur »
  (fait annoté) part en idée.
- **Statement extrait faux** : « Romane Aubertin a confirmé la date du 14 novembre pour le bail » (l'inauguration,
  pas le bail), déjà vu dans l'évaluation de l'agent (docs/extraction.md).
- **Mauvais responsable** (extraites, test) : « Hélène Kowalski » annotée, « Camille Vasseur » affichée ; un
  engagement extrait sans responsable.

### Lecture

Avec la mémoire annotée, le dossier est parfait : il applique exactement les annotations. Toute la perte vient
de la mémoire et de l'extraction. Les relations prédites coûtent 16 à 22 points de couverture, presque tous
dans l'historique, et chaque remplacement manqué laisse une information périmée en section actuelle
(9 à 17 % des affirmations actuelles). L'extraction ajoute les erreurs de type, qui déplacent ou font
disparaître des faits, et prive le classifieur de relations de remplacements (1 sur 10 sur test2) :
sur test2, le dossier extrait ne couvre plus que la moitié des faits et un quart de ses affirmations actuelles
sont périmées.

## 2. Paramètres des demandes (slots)

### Ce que fait `understand`

Règles, sans modèle :

- **projet** : identifiant, nom complet ou nom propre du titre (« NordStock », « Tilleuls ») présent dans le
  message, si un seul projet correspond ; sinon le projet passé en contexte, sinon le seul projet chargé ;
- **personne** : nom complet ou prénom d'un participant, s'il est unique (d'abord dans le projet, puis partout) ;
- **échéance** : première correspondance d'une expression régulière (jour de la semaine, « 12 mars »,
  « demain », « fin du mois », éventuellement précédé de avant / d'ici / pour / le), en minuscules sans accents.

### Vérité terrain : data/agent_eval/slots.jsonl

35 demandes d'action tirées de `test.jsonl`, `holdout.jsonl` et `test2.jsonl` (mails, tâches, événements, et
les demandes d'action qui appellent une précision ; les 5 phrases hypothétiques sont exclues). Annotations
proposées puis validées explicitement avant toute mesure :

- **personne** : nom complet de la personne nommée dans le message ; un rôle (« au responsable du budget ») ou un
  pronom (« Envoie-lui ») donne null ;
- **projet** : seulement si le message nomme le projet ; ni le sujet ni la personne ne suffisent ;
- **échéance** : texte tel qu'écrit, préposition comprise ; comparaison en minuscules et sans accents.

25 demandes ont les trois paramètres à null et 10 en ont au moins un : 5 personnes, 1 projet, 7 échéances.
Cas tranchés : n-07 « relancer NordStock » → projet nordstock-relais (nom propre du titre, même règle que
l'agent) ; p-08 et p-09 (« la bascule ») → projet null ; p-08 → échéance « avant la bascule » conservée ;
p-09 « réunion lundi sur la bascule du 30 novembre » → échéance « lundi ».

Pour la mesure, `understand` reçoit tous les projets du split de la demande et **aucun projet par défaut** :
le projet prédit vient du seul message (dans `eval_agent.py`, le projet est au contraire fourni en contexte).

### Résultats

| paramètre | exactitude (35) | si renseigné | prédit à tort (attendu null) |
|---|---|---|---|
| personne | 1,000 | 1,000 (5) | 0 |
| projet | 1,000 | 1,000 (1) | 0 |
| échéance | 0,971 | 0,857 (7) | 0 |

Une seule erreur : p-08 « valider le test d'export **avant la bascule** » — échéance relative à un événement,
que l'expression régulière ne couvre pas (elle ne reconnaît que des dates et des jours).

### Lecture

Les règles sont exactes sur ces demandes, mais l'épreuve est facile : les personnes sont toujours nommées dans
la même tournure (« Crée une tâche pour Prénom : … »), les échéances sont des jours de la semaine, et la plupart
des demandes désignent la personne par son rôle, ce qui ne demande que de ne rien prédire. Le cas manqué montre
la limite attendue des règles : une échéance relative (« avant la bascule », « après le comité ») n'est pas
reconnue, et elle n'est d'ailleurs pas résolue en date.

## 3. Limites

- **Effectifs très petits** : 5 projets et 77 faits attendus pour les dossiers ; pour les paramètres, 35
  demandes mais seulement 5 personnes, 1 projet et 7 échéances renseignés. Une erreur pèse 14 points sur les
  échéances renseignées, 100 sur le projet.
- **Score parfait par construction avec la mémoire annotée** : les faits attendus reprennent les mêmes types que
  les sections de l'agent ; la seule différence de règle est le statut (l'agent ne filtre pas proposed ni
  hypothetical), et test/test2 ne contiennent ni décision proposed ni risque hypothétique : ce test ne peut pas
  la prendre en défaut. Les 1,000 mesurent la cohérence, pas la qualité rédactionnelle.
- **« Non étayées » toujours nul par construction** : le schéma impose une unité par affirmation, et l'agent ne
  cite que des unités de la mémoire. La mesure reste utile comme garde-fou si le dossier est un jour rédigé
  par un modèle de langue.
- **On ne vérifie que la citation, pas le texte** : une affirmation est jugée par l'unité qu'elle cite et sa
  section ; un statement extrait faux mais aligné sur la bonne unité compte comme exact. Seul le responsable
  affiché est contrôlé dans le texte ; l'échéance affichée ne l'est pas.
- **Dépendance à l'alignement** pour les unités extraites (recouvrement des evidence, docs/extraction.md).
- **Vérité terrain des paramètres** proposée par le même auteur que l'évaluation, puis validée à la main ; les
  demandes ont été écrites avec les mêmes tournures que celles que les règles reconnaissent.
- Pas de mesure du dossier sur train/val : il n'y avait rien à régler.

# Interface de démonstration

Interface web (Streamlit) qui appelle l'agent et affiche ses résultats. Elle ne modifie pas l'agent : elle
réutilise `build_agent` et `DEMO` de `scripts/agent.py`, et ses sorties sont identiques à celles de la ligne
de commande. Aucune action n'est exécutée : mails, tâches et événements restent des brouillons.

## Lancer l'interface

```bash
pip install -r requirements.txt          # streamlit>=1.30
streamlit run app/streamlit_app.py       # puis ouvrir http://localhost:8501
```

Il faut les modèles locaux `models/e5b-both-type-s1` (recherche) et, pour la mémoire prédite,
`models/relations-cross`. Le premier chargement prend une dizaine de secondes ; les modèles sont ensuite gardés
en mémoire (trois configurations au plus). Le GPU de 8 Go ne doit pas être occupé par Ollama en même temps.

## Barre latérale

- **Projet** : Clinique des Tilleuls, Séminaire Horizons, Relais NordStock (test), Boutique Grain d'Ocre,
  CRM Pierre d'Angle (test2), Atlas (train, pour sa démonstration). Le jeu de données se déduit du projet.
  Changer de projet repart d'une conversation vide.
- **Démonstration** : joue les 7 cas du sujet (Clinique des Tilleuls et Atlas seulement).
- **Effacer la conversation**.
- **Options avancées** (repliées) : mémoire des remplacements (prédite par défaut, annotations, aucune),
  abstention (seuil de similarité par défaut, ou les autres stratégies), et la taille de la mémoire ;
  rédaction des réponses par un modèle local (qwen3:8b par défaut, qwen3.5:4b, aucune ; docs/redaction.md).
  Il faut qu'Ollama tourne (`ollama serve`) ; sinon l'interface le signale et affiche la première unité trouvée.

## Ce qui est affiché

- conversation vide : un message d'accueil en deux phrases et quatre boutons, « Une décision », « Ce qui a
  changé », « Un dossier », « Un mail » ; leurs questions sont tirées des réunions du projet
  (`Agent.examples`), et un clic envoie la question complète dans la conversation ;
- « bonjour », « merci », « aide », « tu sais faire quoi ? » : le message d'aide de l'agent (résultat `help`) ;
- la réponse elle-même, puis un bloc replié « Détails » : historique (« cette information remplace… »),
  sources (date, réunion, locuteur, statut, phrase exacte), intention détectée et sa raison, paramètres
  extraits, raison de l'abstention, JSON des brouillons ;
- brouillon de mail, de tâche ou d'événement : une carte avec destinataire, objet, corps et raison du
  destinataire ; la mention « non envoyé, à confirmer » et l'avertissement d'adresse manquante restent visibles ;
- dossier : le Markdown et un bouton pour le télécharger en `.md` ;
- demande de précision, absence d'action, absence de réponse : un encadré, avec les sujets proches pour une
  absence de réponse.

## Dérouler la démonstration

1. Projet **Clinique des Tilleuls**, options avancées par défaut.
2. Montrer l'accueil et cliquer sur un exemple, puis « Effacer la conversation ».
3. Bouton **Démonstration** : les 7 cas du sujet s'affichent dans l'ordre (décision finale, proposition ou
   décision, comparaison, dossier, mail à la bonne personne, demande trop vague, phrase hypothétique), avec
   leur numéro en petit au-dessus de la question. Ouvrir « Détails » sur le cas 1 pour montrer la source.
4. Questions libres dans la zone de saisie, par exemple :
   - « Quel est le nom du traiteur choisi pour la fête de fin d'année ? » : absence de réponse, avec les sujets
     proches ;
   - « Crée une tâche pour Olivier : vérifier les gardes de nuit avant vendredi. » : proposition de tâche.
5. Pour montrer l'effet de la mémoire, rejouer la démonstration avec la mémoire **annotations** puis **aucune**
   (options avancées) : avec la mémoire prédite, le mail sur la formation au bloc part bien à Agnès
   Rousseau-Lin, mais la question « Qui a été désigné pour la formation au bloc ? » répond encore Inès
   (réattribution manquée, docs/dossiers_et_slots.md) ; avec les annotations, elle répond Agnès.

## Mode comparaison

Interrupteur « Comparer avec un modèle non entraîné » dans la barre latérale (coupé par défaut). Chaque question
est traitée par deux systèmes, affichés côte à côte, chacun avec sa réponse et son bloc « Détails » :

- **à gauche**, un système de référence non entraîné : e5-base d'origine sans mémoire (par défaut), e5-base
  d'origine avec mémoire, bge-m3 sans mémoire, ou BM25 sans mémoire. Sans rédaction (case « Rédaction aussi à
  gauche » décochée par défaut), il affiche la première unité trouvée, avec l'abstention en règles ;
- **à droite**, le système complet, avec les réglages des options avancées (e5-base fine-tuné
  `models/e5b-both-type-s1`, mémoire prédite, rédaction qwen3:8b par défaut).

Les deux systèmes sont construits par le même `build_agent` (scripts/agent.py) ; seul l'encodeur change, et la
mémoire pour les variantes « sans mémoire » (`--relations none`). **Les unités sont encodées avec leur type dans
le texte (« Décision : … ») des deux côtés**, comme pour le modèle fine-tuné : seule la différence d'entraînement
compte. Les seuils d'abstention ont été calibrés sur les scores du modèle fine-tuné : à gauche, ils ne se
déclenchent presque jamais.

Le mode est plus lent : chaque question passe par les deux systèmes (et deux rédactions si la case est cochée).
Le second encodeur n'est chargé que quand le mode est activé, et libéré dès qu'il est coupé ou que le système
de gauche change. Avant de charger un modèle, l'interface décharge le modèle de rédaction d'Ollama, qui le
recharge à la question suivante. Sans cela, Ollama occupe la mémoire libre et le chargement échoue, faute de
place sur le GPU de 8 Go. Ces cas ont été vérifiés sans erreur de mémoire : les quatre systèmes de gauche, avec
et sans rédaction à gauche. Il vaut mieux ne pas avoir deux interfaces ouvertes en même temps.

### Questions qui montrent la différence

Vérifiées dans l'interface, avec les réglages par défaut (gauche : e5-base d'origine sans mémoire ni
rédaction ; droite : système complet) :

| projet | question | gauche (non entraîné) | droite (système complet) |
|---|---|---|---|
| Séminaire Horizons | « Quelles sont les dates du séminaire ? » (décision en vigueur) | Hélène doit envoyer les invitations avant le 31 octobre | le séminaire a lieu les 11 et 12 décembre |
| Clinique des Tilleuls | « Le budget de 45 000 euros a-t-il été validé ? » (piège de statut) | la proposition de Baptiste (45 000 euros, à faire valider) | la direction a validé 38 000 euros |
| Relais NordStock | « Le contrat de 85 000 euros a-t-il été validé ? » (piège de statut) | la proposition de Grégoire (85 000 euros) | le contrat est validé à 72 000 euros |
| Relais NordStock | « Qui négocie le contrat de niveau de service ? » (changement de responsable) | Grégoire | initialement Grégoire, puis Mei à sa place |

Pour le changement de responsable, c'est la rédaction qui fait la différence. Les relations prédites ont
manqué cette réattribution : sans rédaction, la droite répond aussi « Grégoire ». Même chose pour « Qui est
chargé de la formation des équipes du bloc ? » à la Clinique des Tilleuls (Inès, puis Agnès avec la rédaction).
Avec bge-m3 à gauche, les pièges de statut ne marchent plus toujours : bge-m3 trouve lui aussi les 38 000 et
72 000 euros validés. Le contraste le plus net est obtenu avec e5-base ou BM25.

Équivalent en ligne de commande : `python scripts/agent.py --project clinique-tilleuls --demo`.

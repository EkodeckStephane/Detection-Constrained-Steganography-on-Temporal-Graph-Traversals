# ADR-002 : définition du canal de parcours temporel

## Statut

Accepté provisoirement le 12 juin 2026.

## Contexte

La formulation initiale ne précisait pas suffisamment ce qu'un observateur
reçoit. Sans artefact observable, la sécurité, la capacité et la robustesse ne
peuvent pas être définies. Par ailleurs, réordonner une liste d'arêtes
temporelles modifierait sa sémantique et rapprocherait inutilement la méthode
de BIND.

## Décision

Le stego est une trace temporelle transmise, composée uniquement de
transitions admissibles dans le graphe de référence. Le graphe, les arêtes et
les timestamps ne sont pas modifiés. Le message influence la sélection des
continuations de la trace.

Le système est donc classé comme stéganographie générative/sélective de
parcours, et non comme insertion par modification d'un graphe.

## Conséquences

- la capacité se mesure principalement en bits par transition;
- la correction exige une liste de candidats identique chez Alice et Bob;
- la sécurité compare des traces naturelles et des traces stego;
- les datasets principaux doivent contenir des trajectoires ou sessions
  réellement observées;
- TGB peut servir à l'apprentissage temporel sans suffire seul à valider la
  naturalité des parcours;
- Graph-Stega devient une baseline conceptuelle obligatoire.

## Alternatives rejetées

- réordonner les événements temporels : détruit potentiellement la causalité;
- ajouter des arêtes : modification détectable et chevauchement avec AdaBIND;
- synthétiser un graphe complet : problème différent, proche de BYNIS;
- appeler « cover naturel » un parcours aléatoire construit : hypothèse non
  défendable sans données de trajectoires observées.

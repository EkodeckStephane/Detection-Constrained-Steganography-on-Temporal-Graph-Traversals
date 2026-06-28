# Protocole de vérification d'antériorité

## Fenêtre et sources

- période : 1998 au 12 juin 2026;
- sources : Scopus, Web of Science, IEEE Xplore, ACM Digital Library,
  SpringerLink, ScienceDirect, arXiv et Google Scholar;
- recherche arrière et avant à partir de Lee, Graph-Stega, ADG, iMEC, TGN,
  FreStega et des revues de stéganographie de graphes ou de maillages.

## Requête conceptuelle

Les familles suivantes sont combinées :

1. `steganograph*`, `information hiding`, `covert channel`;
2. `graph`, `network`, `temporal graph`, `dynamic graph`, `trajectory`,
   `walk`, `path`, `session`;
3. `RNN`, `GRU`, `LSTM`, `GNN`, `TGN`, `autoregressive`, `next event`;
4. `fuzzy`, `adaptive rate`, `entropy control`, `risk controller`;
5. `steganalysis`, `undetectability`, `distribution preserving`,
   `minimum entropy coupling`.

Chaque base reçoit une requête adaptée à sa syntaxe. Les chaînes exactes, la
date, le nombre de résultats et le fichier exporté sont archivés dans
`literature/searches/`.

## Inclusion

- méthode encodant ou détectant une information cachée;
- graphe utilisé comme cover, espace de chemins ou modèle de canal;
- adaptation probabiliste, temporelle ou guidée par détectabilité pertinente;
- article, conférence, préprint substantiel ou thèse avec méthode vérifiable.

## Exclusion

- « graph » désignant seulement un graphique de résultats;
- GNN utilisé uniquement pour la stéganalyse d'images sans apport transférable;
- résumé sans texte accessible;
- doublon d'une version plus complète;
- article sans protocole ou formulation suffisamment précise.

## Procédure

1. déduplication par DOI, arXiv et titre normalisé;
2. tri titre/résumé avec motif d'exclusion;
3. lecture intégrale des travaux proches;
4. extraction dans la matrice de comparaison;
5. recherche des citations entrantes et sortantes;
6. second passage ciblé sur chaque revendication « première ».

## Critère de clôture

La revue est close lorsque deux itérations successives de snowballing
n'ajoutent aucun travail modifiant la matrice d'originalité. Toute nouvelle
publication avant soumission déclenche une mise à jour.

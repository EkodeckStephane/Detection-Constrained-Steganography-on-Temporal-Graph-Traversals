# Feuille de route vers un article Q1

## Phase 1 — verrouillage scientifique

- terminer la revue systématique et la matrice de comparaison;
- formaliser le modèle de menace et le canal de communication;
- vérifier l'antériorité de chaque revendication d'originalité;
- choisir la revue cible avant de figer la longueur du manuscrit.

**Sortie :** protocole enregistré, revendications provisoires et critères
d'arrêt.

## Phase 2 — reproduction des baselines

- réimplémenter BIND, AdaBIND et BYNIS;
- reproduire les résultats essentiels de Lee;
- implémenter les parcours uniforme, DeepWalk et node2vec;
- écrire des tests de décodage exact et de déterminisme par clé.

**Sortie :** rapport de reproduction avec écarts expliqués.

## Phase 3 — pipeline des données réelles

- intégrer TGB et les datasets JODIE;
- produire les splits causaux et fiches de provenance;
- mesurer entropie conditionnelle, répétition, burstiness et motifs;
- sélectionner quatre domaines présentant des dynamiques distinctes.

**Sortie :** tableau descriptif figé des datasets.

## Phase 4 — modèle de probabilité du parcours

- entraîner TGN/GAT + GRU à prédire la prochaine interaction;
- calibrer les probabilités;
- comparer GRU, TGN seul et Transformer temporel léger;
- vérifier la généralisation inter-dataset.

**Sortie :** modèle de couverture validé indépendamment de la stéganographie.

## Phase 5 — encodage et décodage

- chiffrer et blanchir le message;
- implémenter range coding ou groupement adaptatif;
- garantir la synchronisation déterministe par clé;
- ajouter CRC puis BCH/LDPC;
- caractériser la capacité théorique et empirique.

**Sortie :** encodeur-décodeur exact avec tests de propriété.

## Phase 6 — contrôleur flou

- définir les variables linguistiques sans utiliser le test final;
- commencer par entropie, risque, charge restante et risque de blocage;
- apprendre ou ajuster les conséquences Takagi-Sugeno sur validation;
- comparer à des seuils fixes et à une petite politique MLP.

**Sortie :** preuve que le flou apporte un gain propre et interprétable.

## Phase 7 — stéganalyse adversariale

- entraîner un détecteur interne avec gradient reversal ou jeu alterné;
- constituer des détecteurs externes GCN, GAT et Transformer;
- empêcher toute fuite de messages, fenêtres ou graphes entre splits;
- mesurer transfert et détection hors distribution.

**Sortie :** évaluation de furtivité crédible, non limitée à l'adversaire vu.

## Phase 8 — robustesse et ablations

- conduire toutes les attaques du protocole;
- retirer successivement GNN, GRU, flou, code entropique et adversaire;
- tracer les fronts de Pareto et intervalles de confiance;
- analyser les échecs, pas uniquement les moyennes.

**Sortie :** paquet de résultats final et gelé.

## Phase 9 — rédaction de l'article

- écrire d'abord méthodes et protocole, puis résultats;
- limiter les revendications à ce que démontrent les expériences;
- produire un dépôt de réplication anonymisé;
- effectuer relecture technique, statistique et linguistique séparément.

**Sortie :** manuscrit conforme à la revue cible et supplément reproductible.

## Phase 10 — mémoire

- développer le contexte, les preuves, les choix négatifs et les détails de
  reproduction absents de l'article;
- aligner les résultats avec l'article sans copier mécaniquement son texte;
- inclure limites, menaces à la validité et perspectives.

## Critères d'abandon ou de pivot

- capacité utile trop faible sur trois datasets;
- AUC externe durablement supérieure à 0,65 au débit cible;
- absence de gain du contrôleur flou face à un seuil simple;
- dépendance à un seul dataset;
- impossibilité de reproduire les baselines de manière équitable.

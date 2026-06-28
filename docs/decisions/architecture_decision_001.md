# ADR-001 : architecture scientifique initiale

## Décision

Adopter un encodeur temporel GAT/TGN, suivi d'un GRU de parcours, d'un codeur
entropique et d'un contrôleur flou Takagi-Sugeno.

## Raisons

- le GNN représente le contexte structurel;
- le GRU représente l'ordre causal du parcours avec moins de paramètres qu'un
  LSTM;
- le codage entropique rapproche la distribution stego de la distribution
  naturelle;
- le contrôleur flou rend la politique de débit inspectable;
- le stéganalyste adversarial matérialise la furtivité dans l'objectif.

## Alternatives rejetées

- RNN seul : représentation structurelle insuffisante;
- logique floue seule : règles sans estimation apprise de plausibilité;
- système hybride BIND/BYNIS/AdaBIND : hypothèse scientifique trop diffuse;
- diffusion de graphes comme méthode principale : extraction exacte et passage
  à l'échelle trop risqués pour le premier article.

## Révision

Cette décision sera réévaluée après les baselines et l'étude de capacité
empirique sur deux datasets réels.

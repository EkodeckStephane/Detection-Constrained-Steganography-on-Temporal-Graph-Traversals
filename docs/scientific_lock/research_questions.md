# Questions de recherche et hypothèses falsifiables

## Question principale

À charge et fiabilité comparables, l'encodage dans une distribution causale de
parcours temporels réels réduit-il la détectabilité par rapport aux méthodes
statiques et aux parcours heuristiques?

## Questions secondaires

### RQ1 — modèle de couverture

Un modèle temporel calibré reproduit-il mieux les continuations et les
statistiques de trajectoires qu'un parcours uniforme, DeepWalk ou node2vec?

**H1.** Sur au moins trois domaines réels, le modèle réduit significativement
la log-loss et la divergence des statistiques de parcours par rapport au
meilleur parcours heuristique.

### RQ2 — encodage distributionnel

Le codeur entropique conserve-t-il la distribution du modèle à charge utile
non nulle?

**H2.** À distribution \(q_\theta\) figée, l'écart cover-stego est inférieur à
celui de l'affectation fixe de bits, sans erreur de décodage dans le canal
passif.

### RQ3 — contrôle flou

Le contrôleur flou apporte-t-il un gain propre face à un seuil fixe et à une
politique MLP de complexité comparable?

**H3.** Son hypervolume sur le front capacité-détectabilité-fiabilité dépasse
celui du meilleur contrôle concurrent d'au moins 5 %, avec intervalle
bootstrap excluant zéro.

### RQ4 — stéganalyse externe

Le gain subsiste-t-il contre des détecteurs non utilisés pendant
l'entraînement et en transfert temporel ou inter-domaine?

**H4.** Au point d'exploitation retenu, l'AUC externe reste au plus à 0,60 sur
au moins trois domaines et est inférieure d'au moins 0,05 à celle de BIND ou
du meilleur parcours heuristique à charge appariée.

### RQ5 — robustesse

Quel compromis entre redondance, débit et récupération est obtenu sous
altération du canal?

**H5.** Le code correcteur réduit le BER après attaques définies sans augmenter
la détectabilité au-delà du seuil d'exploitation.

### RQ6 — décomposition de la distorsion

Quelle part de la détectabilité provient de l'erreur du modèle, du contrôleur
et du codeur?

**H6.** La distorsion locale du codeur reste inférieure à celle de
l'affectation fixe, tandis que l'abstention réduit la distorsion du contrôleur
sans annuler le débit utile.

## Règles d'interprétation

- Une hypothèse rejetée est rapportée, jamais reformulée après lecture du test.
- Une différence statistique sans taille d'effet utile n'est pas un succès.
- Le contrôleur flou est retiré du titre s'il ne bat pas le seuil fixe.
- Le terme « distribution-preserving » ne s'applique qu'à la distribution du
  modèle; l'alignement avec le domaine réel est évalué séparément.

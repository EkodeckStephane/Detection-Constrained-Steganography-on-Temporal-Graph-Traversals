# Critères de succès, d'arrêt et de pivot

## Portes avant expérimentation finale

1. canal et artefact observable définis sans ambiguïté;
2. décodage exact démontré par tests de propriété;
3. au moins trois datasets de niveau A avec trajectoires ou sessions réelles;
4. baselines reproduites avec écarts expliqués;
5. modèle temporel mieux calibré que les heuristiques;
6. détecteurs externes gelés avant le test final.
7. coût par transition et quatre modes d'abstention activables séparément;
8. décomposition modèle--contrôleur--codeur enregistrée pour chaque campagne.

## Point d'exploitation

Il est choisi sur validation par optimisation multiobjectif, puis figé. Les
seuils minimaux sont :

- BER passif égal à zéro avant bruit;
- BER après correction inférieur ou égal à \(10^{-3}\) au niveau d'attaque
  retenu;
- AUC externe inférieure ou égale à 0,60;
- aucune dégradation significative de l'utilité aval;
- gain présent sur au moins trois domaines réels.

## Pivots obligatoires

- **Flou vers contrôle simple** : si l'hypervolume ne dépasse pas le seuil
  fixe de 5 %, retirer la logique floue de la contribution principale.
- **TGN-GRU vers modèle plus simple** : si GRU ou mémoire temporelle n'améliore
  ni calibration ni détectabilité, conserver le modèle le plus parcimonieux.
- **Distribution-preserving vers distortion-aware** : si l'écart entre
  \(q_\theta\) et les covers réels domine l'erreur d'encodage, optimiser
  explicitement l'alignement au domaine, à la manière de FreStega.
- **TGB vers trajectoires** : si les parcours ne possèdent pas de sémantique
  naturelle vérifiable, TGB devient seulement un banc de préentraînement.
- **Abandon du canal** : si l'AUC externe reste supérieure à 0,65 à toute
  charge utile exploitable sur trois datasets.

## Interdictions méthodologiques

- sélectionner un seuil sur le test;
- présenter une amélioration interne comme sécurité générale;
- masquer un échec par moyenne entre domaines;
- utiliser des données synthétiques dans les tableaux scientifiques;
- revendiquer une sécurité parfaite contre la distribution réelle.

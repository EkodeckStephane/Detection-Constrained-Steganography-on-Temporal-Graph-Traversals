# Phase 6/10 - Controleur flou

Date de demarrage : 24 juin 2026.

## Statut scientifique

Cette phase dispose maintenant d'une **baseline deterministe de controle**
executee sur grille synthetique et sur sorties reelles du modele temporel de
couverture. Les resultats produits par `phase6_fuzzy_controller_baseline`
testent les modes `EMBED`, `COVER`, `PAUSE` et `STOP`, et comparent le
controleur a un seuil fixe apparie.

Avant revendication Q1, cette limite doit etre levee par :

- ajustement des consequences sur validation seulement;
- comparaison appariee contre une MLP de taille comparable;
- evaluation avec detecteurs externes et utilite aval, pas seulement avec
  entropie/fragilite du modele de couverture;
- selection du point d'exploitation sans consulter le test final.

## Resultat actuel

La phase 6 dispose d'un controleur deterministe de type Takagi-Sugeno pour les
six entrees verrouillees :

- entropie predictive;
- incertitude de calibration;
- risque steganalytique;
- pression de charge utile;
- risque de cul-de-sac;
- fragilite du canal.

Le controleur produit un mode parmi `EMBED`, `COVER`, `PAUSE` et `STOP`, ainsi
qu'un nombre local de bits. Une baseline a seuil fixe d'entropie est fournie
pour l'ablation obligatoire. La campagne actuelle ajoute une evaluation sur
tgbl-wiki, MOOC, LastFM, GeoLife discretise et T-Drive discretise.

## Execution

```powershell
python code\scripts\run_phase6_fuzzy_controller.py
```

Sorties :

- `results/tables/phase6_fuzzy_controller_baseline.json`
- `results/tables/phase6_fuzzy_controller_baseline.csv`
- `results/tables/phase6_fuzzy_controller_real_data.csv`

## Limite restante avant article

Les consequences floues sont encore reglees manuellement. La prochaine etape
consiste a les ajuster sur validation, sans regarder le test final, puis a les
comparer a une MLP et a des detecteurs externes. Les resultats Phase 6 actuels
sont donc suffisants pour verrouiller l'interface et l'ablation a seuil fixe,
mais pas encore pour conclure sur le gain scientifique complet du controleur
flou.

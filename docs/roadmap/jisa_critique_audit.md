# Audit critique pour refonte JISA

Date : 28 juin 2026.

## Diagnostic global

Le manuscrit actuel contient deja plusieurs formulations prudentes, notamment
sur la faiblesse du debit et le caractere borne des adversaires. Cependant, il
reste vulnerable aux critiques JISA pour quatre raisons :

1. la nouveaute est encore presentee comme une combinaison de briques ;
2. le seuil AUC `0.60` joue un role trop central et parait arbitraire ;
3. la comparaison avec AdaBIND/BIND/BYNIS melange des paradigmes et unites ;
4. la capacite tres faible n'est pas encore transformee en question
   scientifique ou cas d'usage credible.

## Cartographie critique -> action

| Critique | Emplacement actuel | Action requise |
|---|---|---|
| Architecture de briques existantes | Introduction, conclusion | Recentrer sur le canal comportemental temporel et l'abstention sous risque ; declasser TGN/GRU/flou/couplage comme outils. |
| Nouveau support insuffisant | Introduction, related work | Expliciter les contraintes propres aux traversales temporelles : causalite, continuations valides, impasses, synchronisation et non-modification topologique. |
| Controleur flou superflu | Method, results | Renommer comme politique d'abstention ; comparer a seuil simple et MLP ; deplacer les details flous hors contribution principale. |
| Securite circulaire | Security, experiments, results | Remplacer le seuil unique par courbes risque-capacite, intervalles, tests deux-echantillons et adversaires renforces. |
| Seuil AUC arbitraire | Security, experiments, results, conclusion | Ne plus le presenter comme preuve ou condition de publication ; l'utiliser comme point de fonctionnement conservateur et preregistre. |
| Capacite quasi nulle | Results, validated scope, conclusion | Mesurer l'entropie disponible ; tester range coding contraint ; sinon justifier un cas d'usage faible debit. |
| Comparaisons biaisees avec BIND/AdaBIND/BYNIS | Related work, Table `sota-temporal` | Deplacer en discussion/annexe ou requalifier comme comparaison de paradigmes, pas comme comparaison directe de performance. |
| Modele de couverture mediocre | Results | Assumer le modele comme baseline conservatrice ; ajouter diagnostics de calibration, cold-start et distribution naturelle vs stego. |
| Preservation de distribution non prouvee | Title/method/security/metrics | Remplacer par "detection-constrained" ou "distribution-audited"; ajouter MMD/Wasserstein/tests deux-echantillons. |
| Robustesse active limitee | Security/results | Presenter comme fiabilite sous budget borne ; ajouter explicitement les attaques non couvertes. |
| Terminologie opaque | Tout le manuscrit | Reduire les expressions comme "rank-gated hardening" et "distortion-gated coupling"; preferer des descriptions operationnelles. |

## Recommandations section par section

### Introduction

- Remplacer la phrase de contribution par un probleme de securite :
  dissimuler un signal dans des flux temporels reels sans modifier le graphe et
  sous surveillance temporelle.
- Supprimer l'impression que le papier revendique une simple combinaison de
  composants.
- Reformuler les contributions :
  1. canal de traversale temporelle non structurel ;
  2. abstention explicite sous contrainte de detectabilite ;
  3. protocole d'evaluation adversariale borne.

### Related work

- Conserver BIND/AdaBIND/BYNIS comme contraste conceptuel, pas comme baseline
  directe.
- Developper davantage la steganographie comportementale et les flux
  d'interaction, car c'est la famille la plus proche.
- Deplacer le tableau AdaBIND quantitatif vers une annexe ou discussion avec
  avertissement fort sur les unites.

### Method

- Renommer la sous-section "Fuzzy rate and abstention policy" en politique
  d'abstention et de debit.
- Presenter la logique floue comme une implementation testee, pas comme une
  innovation.
- Ajouter le range coding contraint comme piste de codec et distinguer :
  entropie disponible, bits tentes, bits utiles, distorsion de codage.

### Security

- Conserver la proposition conditionnelle actuelle, car elle evite l'erreur
  d'une fausse triangularite KL.
- Ajouter une borne en variation totale locale plus lisible pour les reviewers.
- Presenter l'adversaire white-box comme borne empirique, pas comme preuve
  d'indetectabilite.

### Experiments/results

- Ajouter des mesures distributionnelles directes : MMD, Wasserstein ou tests
  deux-echantillons sur features temporelles.
- Ajouter un audit de capacite : entropie conditionnelle sure vs debit actuel.
- Tester un prototype de range coding contraint pour savoir si le codec ou le
  canal explique la faible capacite.
- Renforcer les adversaires ou, au minimum, expliciter les limites des
  detecteurs actuels.

### Conclusion

- Retirer le langage de nouveaute par combinaison.
- Conclure sur un baseline faible debit audite, pas sur une securite generale.
- Faire des limites une force : elles definissent les conditions go/no-go pour
  JISA.

## Decisions immediates

1. Ne pas promettre de "provable indistinguishability".
2. Ne pas utiliser "distribution-preserving" comme revendication centrale sans
   mesures distributionnelles directes.
3. Ne pas conserver AdaBIND comme comparaison principale.
4. Prioriser l'audit de capacite et le range coding contraint avant une
   reecriture finale.


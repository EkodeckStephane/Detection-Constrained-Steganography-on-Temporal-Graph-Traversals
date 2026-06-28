# Plan experimental renforce pour JISA

Date : 28 juin 2026.

## Objectif

Transformer les critiques JISA en experiences verifiables. Les priorites sont :

1. expliquer la faible capacite par une decomposition entropie disponible vs
   perte de codec ;
2. mesurer directement la proximite distributionnelle naturel/stego ;
3. renforcer l'evaluation adversariale sans presenter les resultats comme une
   preuve absolue.

## Experience E1 : audit de capacite

### Question

La faible capacite vient-elle du canal lui-meme, des contraintes de furtivite,
ou du codec local actuel ?

### Mesures

Pour chaque dataset et split test :

- entropie moyenne de la distribution candidate ;
- fraction de transitions avec au moins deux continuations ;
- fraction de transitions qui passent les contraintes de securite ;
- entropie disponible sur les transitions faisables ;
- bits actuellement consommes par transition ;
- perte codec = entropie faisable - bits consommes ;
- taux de couverture/abstention ;
- distorsion locale TV/KL moyenne.

### Sorties

- `results/tables/phase11_capacity_audit.csv`
- `results/tables/phase11_capacity_audit.json`
- section manuscrit : "Capacity audit".

### Critere de decision

Si l'entropie faisable est proche du debit actuel, le goulot est le canal et le
papier doit assumer un regime faible debit. Si l'entropie faisable est beaucoup
plus grande que le debit actuel, le range coding contraint devient prioritaire.

## Experience E2 : range coding contraint

### Question

Un codec sequentiel a precision finie peut-il exploiter mieux l'entropie sure
que le backend local actuel, sans degrader la detectabilite ?

### Prototype

- conserver l'ordre canonique des candidats ;
- convertir les probabilites en masses entieres ;
- accumuler les bits sur plusieurs transitions lorsque les intervalles locaux
  ne consomment pas un nombre entier optimal ;
- enregistrer la distorsion de codage et le debit recupere ;
- verifier le round-trip Alice-Bob sur tests unitaires.

### Sorties

- module ou fonctions de codec ajoutees a `code/src/stego/coding.py` ;
- tests unitaires dans `code/tests/test_phase5_coding.py` ;
- `results/tables/phase11_range_codec.csv/json`.

### Critere de decision

Le codec est utile seulement s'il augmente les bits utiles par transition au
point conservateur tout en maintenant la synchronisation exacte et une
distorsion locale non superieure au backend actuel.

## Experience E3 : mesures distributionnelles directes

### Question

Les traces stego restent-elles proches des traces naturelles sur des features
temporelles observables, au-dela de l'AUC des detecteurs ?

### Mesures minimales

- MMD RBF sur les features publiques existantes ;
- distance de Wasserstein 1D par feature ;
- tests de permutation sur difference de moyenne ;
- Jensen-Shannon sur features discretisees : rang, top-action, self-loop,
  unseen-context.

### Sorties

- `results/tables/phase11_distribution_audit.csv`
- `results/tables/phase11_distribution_audit.json`
- figures ou tableaux pour article.

### Critere de decision

Ces mesures doivent remplacer le slogan "distribution-preserving". Si elles
sont defavorables, le manuscrit doit parler de "detection-constrained" et
documenter les ecarts.

## Experience E4 : adversaires renforces

### Question

Les resultats tiennent-ils contre des adversaires sequentiels et graphes plus
proches des critiques JISA ?

### Priorite

1. Exploiter les resultats existants Phase 10 comme baseline.
2. Ajouter, si faisable, un detecteur sequence leger sur les records publics.
3. Rapporter explicitement ce qui reste hors perimetre : red-team tiers,
   temporal GNN industriel, attaque active non bornee.

### Sorties

- `results/tables/phase11_adversarial_audit.csv/json` si nouveau detecteur ;
- sinon tableau de consolidation des adversaires existants.

## Ordre d'implementation

1. E1 capacity audit : priorite maximale, faible risque, repond directement a
   la critique de debit.
2. E3 distribution audit : priorite maximale, repond au titre et a la critique
   "distribution-preserving".
3. E2 range coding : seulement apres E1, pour savoir si le codec peut vraiment
   gagner.
4. E4 adversaires renforces : depend du temps et des donnees disponibles.

## Conditions avant refonte manuscrit

- E1 executee et interpretee.
- E3 executee et interpretee.
- Decision documentee sur E2 : gain mesurable ou limite du canal.
- Le seuil `0.60` remplace par courbes ou tableaux risque-capacite lorsque
  possible.


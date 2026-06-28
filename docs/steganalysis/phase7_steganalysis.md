# Phase 7/10 - Steganalyse adversariale

Date de demarrage : 26 juin 2026.

## Resultat actuel

La phase 7 dispose d'une campagne executable de steganalyse externe :

- generation de paires appariees naturel/stego sur validation et test;
- modele public de couverture `temporal_source_previous_destination_backoff`;
- codeur range de phase 5 avec un bit maximum par transition;
- durcissement adaptatif par rang : seules les actions encodees dans la moitie
  superieure du classement local sont acceptees;
- politique conservatrice : `COVER` lorsque la distorsion locale depasse les
  seuils predefinis;
- detecteurs externes tabulaires `linear`, `forest` et `mlp`;
- metriques AUC, balanced accuracy et EER.

Execution :

```powershell
python code\scripts\run_phase7_steganalysis.py
```

Sorties :

- `results/tables/phase7_steganalysis.json`
- `results/tables/phase7_steganalysis.csv`
- `results/tables/phase7_steganalysis_records.csv`

## Verdict

Avec la politique conservatrice durcie, les detecteurs externes restent sous
le seuil preregistre `AUC_ext <= 0,60` sur les cinq datasets evalues. Le pire
cas observe est TGBL-Wiki avec une AUC de 0,576 pour le detecteur `forest`.
T-Drive cellules descend a 0,537 pour le meme detecteur.

Ce resultat ferme le blocage de detectabilite immediate, mais au prix d'un
debit faible :

- tgbl-wiki : 0,0995 bit tente par transition;
- MOOC : 0,0008;
- LastFM : 0,0007;
- GeoLife cellules : 0,0022;
- T-Drive cellules : 0,1007.

La revendication publiable doit donc porter sur un point d'exploitation tres
conservateur, pas sur un haut debit.

## Interpretation

Le signal detecte ne vient pas de champs internes du codeur : les features
d'entree sont limitees aux informations observables ou derivables du modele
public de couverture. Le controle de distorsion reduit fortement les fuites
comportementales, en particulier probabilite de l'action, rang et surprise.

La prochaine correction doit chercher un meilleur front debit/AUC, notamment
par apprentissage adversarial des seuils sur validation et par une evaluation
multi-seed plus large.

# Notes Phase 12 pour la revision majeure JISA

Date : 28 juin 2026.

## Artefacts produits

- `code/scripts/run_phase12_capacity_detectability_sweep.py`
- `code/scripts/run_phase12_oracle_leakage_audit.py`
- `code/scripts/run_phase12_cover_sensitivity.py`
- `code/scripts/run_phase12_practical_payloads.py`
- `results/tables/phase12_capacity_detectability_sweep.csv`
- `results/tables/phase12_capacity_detectability_summary.csv`
- `results/tables/phase12_capacity_detectability_sweep.json`
- `results/figures/phase12_capacity_detectability_sweep.pdf`
- `results/figures/phase12_capacity_detectability_sweep.png`
- `results/tables/phase12_oracle_leakage_audit.csv`
- `results/tables/phase12_oracle_leakage_correlations.csv`
- `results/tables/phase12_oracle_leakage_audit.json`
- `results/tables/phase12_cover_sensitivity.csv`
- `results/tables/phase12_cover_sensitivity.json`
- `results/tables/phase12_practical_payloads.csv`
- `results/tables/phase12_practical_payloads.json`

## Capacite-detectabilite

Le sweep mesure cinq points d'exploitation : `ultra_stealth`,
`conservative`, `balanced`, `open_rank`, `two_bit_probe`.

Meilleurs points sous le budget public AUC 0.60 :

| Dataset | Point | Bits/transition | Max public AUC |
|---|---:|---:|---:|
| TGBL-Wiki | open_rank | 0.1338 | 0.59295 |
| T-Drive | balanced | 0.1106 | 0.56561 |
| MOOC | open_rank | 0.0254 | 0.52380 |
| LastFM | two_bit_probe | 0.0536 | 0.52476 |
| GeoLife | balanced | 0.0007 | 0.50007 |

Les regimes plus agressifs franchissent le budget sur certains domaines :
TGBL-Wiki atteint 0.2129 bit/transition avec AUC 0.60737, T-Drive atteint
0.2680 avec AUC 0.64494, GeoLife atteint 0.3086 avec AUC 0.76022.

## Cas d'usage courts

Sous les meilleurs points avec AUC publique au plus 0.60 :

| Dataset | 8 bits | 32 bits | 64 bits | 128 bits |
|---|---:|---:|---:|---:|
| TGBL-Wiki | 60 | 240 | 479 | 957 |
| T-Drive | 73 | 290 | 579 | 1158 |
| LastFM | 150 | 598 | 1195 | 2389 |
| MOOC | 315 | 1260 | 2520 | 5040 |
| GeoLife | 11429 | 45715 | 91429 | 182858 |

Lecture : le canal soutient des charges courtes sur TGBL-Wiki, T-Drive,
LastFM et MOOC ; GeoLife demande des traversales longues au point mesure.

## Oracle-leakage

L'audit Phase 12 utilise trois familles de classifieurs sur les variables
internes : gradient boosting, random forest et MLP.

| Dataset | Best bounded AUC | Best oracle AUC | Epsilon oracle | Internal-only epsilon |
|---|---:|---:|---:|---:|
| TGBL-Wiki | 0.52952 | 0.53716 | 0.00764 | 0.01894 |
| T-Drive | 0.52139 | 0.53294 | 0.01155 | 0.01675 |
| MOOC | 0.50087 | 0.50087 | 0.00000 | 0.00044 |
| LastFM | 0.50000 | 0.50000 | 0.00000 | 0.00000 |
| GeoLife | 0.50000 | 0.50000 | 0.00000 | 0.00000 |

Les correlations les plus elevees avec l'etiquette apparaissent sur
`bits_consumed` : 0.13894 pour TGBL-Wiki et 0.13052 pour T-Drive.

## Sensibilite couverture

La synthese Phase 12 relie les metriques de couverture aux regimes mesures.
Le debit utile depend surtout de la presence d'etats faisables dans le domaine
et de la contrainte de rang, plutot que de la seule top-1 accuracy.
Les prochaines ameliorations doivent donc cibler la calibration des masses de
probabilite et la stabilite sous steganalyse, en plus de la prediction top-1.

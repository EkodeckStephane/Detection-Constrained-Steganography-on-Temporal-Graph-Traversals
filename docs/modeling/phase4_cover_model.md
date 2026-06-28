# Phase 4/10 - Modele de couverture probabiliste

Date de demarrage : 24 juin 2026.

## Objectif

La phase 4 commence par un contrat mesurable pour la prediction de la prochaine
destination conditionnelle a la source. Ce contrat precede les modeles TGN/GAT,
GRU et Transformer leger afin que les futurs modeles neuronaux soient compares
a une base causale reproductible.

## Baselines gelees

La baseline `source_destination_frequency` estime uniquement
`P(destination | source)` sur le split `train`. Les splits `validation` et
`test` ne mettent pas a jour les compteurs. Les sources inconnues utilisent le
prior global des destinations et les destinations jamais vues en entrainement
sont envoyees vers un bucket explicite `unknown`.

Une seconde baseline temporelle `temporal_source_previous_destination_backoff`
conditionne la destination suivante sur `(source, destination precedente)`,
puis retombe sur `source` et sur le prior global. Elle produit aussi les
candidats, l'entropie et les signaux de cold-start consommes par le controleur
flou de la phase 6.

Une troisieme campagne `phase4_neural_cover_model` entraine des modeles GRU et
Transformer legers sur des echantillons causaux. Elle valide le chemin
PyTorch, les historiques par source et la comparaison neurale minimale avant
les variantes TGN/GAT completes.

La configuration est :

- `experiments/real_world/phase4_cover_model.yaml`
- script : `python code/scripts/run_phase4_cover_model.py`
- sorties : `results/tables/phase4_cover_model_baseline.json` et `.csv`
- `experiments/real_world/phase4_temporal_cover_model.yaml`
- script : `python code/scripts/run_phase4_temporal_cover_model.py`
- sorties : `results/tables/phase4_temporal_cover_model.json` et `.csv`
- `experiments/real_world/phase4_neural_cover_model.yaml`
- script : `python code/scripts/run_phase4_neural_cover_model.py`
- sorties : `results/tables/phase4_neural_cover_model.json` et `.csv`
- `experiments/real_world/phase4_spatial_discretization.yaml`
- script : `python code/scripts/run_phase4_spatial_discretization.py`
- sortie : `results/tables/phase4_spatial_discretization.json`

## Metriques

Les metriques retenues sont :

- NLL moyenne en bits;
- perplexite;
- exactitude top-1;
- expected calibration error sur la confiance top-1;
- fraction de sources inconnues;
- fraction de destinations inconnues.

## Portee actuelle

La premiere iteration couvre les flux d'interactions de niveau B :

- tgbl-wiki;
- MOOC/JODIE;
- LastFM/JODIE.

La discretisation spatiale de niveau A est maintenant gelee pour les
evaluations rapides :

- GeoLife -> `datasets/processed/geolife_cells/events.parquet`;
- T-Drive -> `datasets/processed/t_drive_cells/events.parquet`;
- grille : 0,01 degre, origine `(-90, -180)`;
- echantillon de validation : 100 000 points maximum par split.

GeoLife et T-Drive sont donc inclus dans la baseline temporelle sous forme de
transitions de cellules. Les resultats restent sensibles au choix de resolution
spatiale et devront etre confirmes par une analyse de sensibilite.

## Critere de passage

Les modeles TGN/GAT + GRU de la suite devront battre cette baseline sur NLL et
calibration validation, puis conserver le gain sur test sans re-entrainement
sur les donnees futures.

La campagne neurale actuelle ne remplace pas encore le TGN/GAT complet : elle
utilise des historiques de destinations et des echantillons limites. Elle
resout le blocage d'implementation GRU/Transformer, mais l'article devra encore
entrainer la version graphe-temporale complete et calibrer ses probabilites sur
validation.

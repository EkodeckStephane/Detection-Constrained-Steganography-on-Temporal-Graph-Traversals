# Variante : cover models neuronaux

Ce répertoire documente la variante créée pour lever le point 4 du reviewer simulé : le modèle de cover actuel (`TemporalBackoffModel`) est trop simpliste et pourrait limiter la capacité utile.

## Objectif

Comparer, dans le même pipeline stéganographique, quatre modèles de cover :

1. `backoff` — modèle non-neural source-plus-history actuellement utilisé dans l'article.
2. `gru` — modèle de séquence neuronal (`NeuralSequenceModel` avec `kind="gru"`).
3. `transformer` — modèle de séquence neuronal (`NeuralSequenceModel` avec `kind="transformer"`).
4. `temporal_graph_encoder` — encodeur temporel de graphe léger (`TemporalGraphCoverModel`).

## Fichiers

| Fichier | Description |
|---|---|
| `experiments/variant_neural_cover/variant_cover_models.yaml` | Configuration principale de la variante. |
| `experiments/variant_neural_cover/test_variant_cover_models.yaml` | Configuration de test réduite. |
| `experiments/variant_neural_cover/lee_baselines_temporal.yaml` | Configuration pour exécuter BIND/AdaBIND sur les datasets temporels. |
| `experiments/variant_neural_cover/proposition_on_terrorists.yaml` | Configuration pour exécuter la proposition sur `terrorists-911`. |
| `code/src/models/cover_model.py` | Interface unifiée `CoverModel` et wrappers pour backoff, GRU, Transformer et GNN temporel. |
| `code/scripts/variant_neural_cover/run_variant_cover_models.py` | Script principal de la variante. |
| `code/scripts/variant_neural_cover/run_lee_baselines_temporal.py` | Adaptation de BIND/AdaBIND aux datasets temporels. |
| `code/scripts/variant_neural_cover/run_proposition_on_terrorists.py` | Exécution de la proposition sur le dataset statique de Lee. |
| `code/scripts/variant_neural_cover/generate_comparison_tables.py` | Génération des tableaux comparatifs. |
| `results/tables/variant_neural_cover/` | Résultats bruts et tableaux comparatifs. |

## Exécution

```bash
# Variante cover models (peut être long)
python code/scripts/variant_neural_cover/run_variant_cover_models.py

# BIND/AdaBIND sur datasets temporels
python code/scripts/variant_neural_cover/run_lee_baselines_temporal.py

# Proposition sur terrorists-911
python code/scripts/variant_neural_cover/run_proposition_on_terrorists.py

# Génération des tableaux comparatifs
python code/scripts/variant_neural_cover/generate_comparison_tables.py
```

## Décision

La variante a été exécutée sur 5 datasets avec 5 seeds. Les résultats détaillés sont dans `comparison_report.md`.

**Décision** : la variante n'est pas retenue pour l'article.

- Les modèles neuronaux (GRU, Transformer) peuvent augmenter localement la capacité, mais la détectabilité externe devient inacceptable (AUC jusqu'à 0,99 sur MOOC).
- L'encodeur temporel de graphe est instable et atteint parfois des AUC > 0,80 (GeoLife cells).
- Le modèle backoff reste le seul à maintenir systématiquement l'AUC externe sous le seuil de 0,60 avec une variance faible.

La variante reste documentée ici comme réponse technique au reviewer et comme base pour d'éventuelles explorations futures.

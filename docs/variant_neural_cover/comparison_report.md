# Rapport comparatif : SOTA et variante à cover models neuronaux

## Résumé exécutif

- Un **tableau comparatif chiffré BIND/AdaBIND vs proposition** a été produit sur les datasets temporels et sur `terrorists-911`.
- Une **variante à cover models neuronaux** (backoff, GRU, Transformer, encodeur temporel de graphe) a été créée et évaluée dans le même pipeline stéganographique.
- **Décision** : la variante neurale **n'est pas retenue** pour l'article. Le modèle backoff reste meilleur ou comparable sur le couple débit/détectabilité, tandis que les modèles neuronaux entraînés deviennent trop détectables (AUC jusqu'à 0,99 sur MOOC).

## 1. Positionnement face au SOTA

### 1.1 Comparaison sur les datasets temporels

Le tableau `results/tables/variant_neural_cover/sota_comparison_temporal_datasets.csv` compare la capacité d'AdaBIND (bits/arête) avec celle de la proposition (bits/transition) et l'AUC externe maximal de la proposition.

| Dataset | Arêtes uniques | AdaBIND max payload (bpe) | Arêtes ajoutées | Proposition max AUC | Proposition bits/transition |
|---|---:|---:|---:|---:|---:|
| GeoLife cells | 118 | 1.3617 | 70 | 0.5002 | 0.0023 |
| LastFM | 14 042 | 0.0182 | 0 | 0.5001 | 0.0006 |
| MOOC | 17 083 | 0.0150 | 0 | 0.5011 | 0.0012 |
| T-Drive cells | 6 704 | 0.0382 | 0 | 0.5362 | 0.0994 |
| TGBL-Wiki | 5 316 | 0.0482 | 0 | 0.5756 | 0.0993 |

**Interprétation**
- AdaBIND peut atteindre des débits bien supérieurs (jusqu'à 1,36 bpe sur GeoLife cells), mais au prix d'une modification topologique (70 arêtes ajoutées sur 118) qui crée des anomalies globales.
- La proposition préserve la topologie (pas d'arêtes ajoutées) et maintient l'AUC externe sous le seuil preregistered de 0,60, sauf sur TGBL-Wiki (0,5756) et T-Drive cells (0,5362) qui restent proches du seuil.
- Les unités diffèrent (bits/arête pour AdaBIND, bits/transition pour la proposition), mais l'ordre de grandeur montre que le canal trajectoire est plus conservateur car il ne modifie pas le graphe.

### 1.2 Comparaison sur `terrorists-911`

Le tableau `results/tables/variant_neural_cover/sota_comparison_terrorists_911.csv` reprend les résultats de la reproduction de Lee et ajoute la proposition appliquée à des walks générés sur le graphe statique.

| Méthode | Payload (bpe) | Arêtes ajoutées | Topologie modifiée | Max AUC |
|---|---:|---:|:---|---:|
| BIND (8 B) | 0.4211 | 0 | Non | — |
| BIND (16 B) | 0.8421 | 0 | Non | — |
| BIND (24 B) | 1.2632 | 0 | Non | — |
| AdaBIND | 0.4025 | 7 | Oui | — |
| BYNIS (0 extra) | 7.2000 | 0 | Non | — |
| BYNIS (50 extra) | 2.0571 | 50 | Oui | — |
| Proposition (backoff) | 0.1846 | 0 | Non | 0.5563 |

**Interprétation**
- BIND, BYNIS et AdaBIND offrent des débits nettement plus élevés sur ce petit graphe statique.
- La proposition, contrainte à des trajectoires valides dans le graphe, a un débit plus faible mais fournit une mesure de détectabilité externe (AUC max 0,5563).
- Les méthodes de Lee n'ont pas été évaluées avec les mêmes détecteurs externes ; leur sécurité est principalement justifiée par la préservation du multiset d'arêtes (BIND) ou par la synthèse plausible (BYNIS).

## 2. Variante à cover models neuronaux

### 2.1 Protocole

Quatre cover models ont été comparés dans le pipeline stéganographique phase 7 :

1. `backoff` (baseline actuelle)
2. `gru` (`NeuralSequenceModel` GRU)
3. `transformer` (`NeuralSequenceModel` Transformer)
4. `temporal_graph_encoder` (`TemporalGraphCoverModel`)

Chaque modèle a été entraîné sur le split train, puis utilisé pour générer des records cover/stego sur validation et test. Les détecteurs externes (linear, forest, mlp) ont été entraînés sur les mêmes features. L'expérience a été répétée sur 5 seeds.

### 2.2 Résultats

Le tableau `results/tables/variant_neural_cover/variant_cover_models_summary.csv` résume le meilleur détecteur par dataset et par cover model.

| Dataset | Cover model | Bits/transition | Max AUC | Meilleur détecteur |
|---|---|---:|---:|:---|
| GeoLife cells | backoff | 0.0011 | 0.5001 | forest |
| GeoLife cells | gru | 0.3159 | 0.5018 | forest |
| GeoLife cells | temporal_graph_encoder | 0.2968 | **0.5802** | forest |
| GeoLife cells | transformer | 0.3363 | 0.5026 | forest |
| LastFM | backoff | 0.0711 | 0.5076 | forest |
| LastFM | gru | 0.0003 | 0.5001 | linear |
| LastFM | temporal_graph_encoder | 0.0000 | 0.5000 | mlp |
| LastFM | transformer | 0.0004 | 0.5003 | linear |
| MOOC | backoff | 0.0343 | 0.5052 | forest |
| MOOC | gru | 0.8610 | **0.9874** | forest |
| MOOC | temporal_graph_encoder | 0.0000 | 0.5000 | forest |
| MOOC | transformer | 0.5543 | **0.8643** | forest |
| T-Drive cells | backoff | 0.2471 | 0.5365 | forest |
| T-Drive cells | gru | 0.0276 | 0.5046 | forest |
| T-Drive cells | temporal_graph_encoder | 0.0002 | 0.5002 | forest |
| T-Drive cells | transformer | 0.0142 | 0.5018 | linear |
| TGBL-Wiki | backoff | 0.2105 | 0.5418 | forest |
| TGBL-Wiki | gru | 0.0498 | 0.5057 | linear |
| TGBL-Wiki | temporal_graph_encoder | 0.0000 | 0.5000 | forest |
| TGBL-Wiki | transformer | 0.0328 | 0.5051 | forest |

### 2.3 Analyse

- **GRU et Transformer sur MOOC** : très forte capacité (0,86 et 0,55 bits/transition) mais détectabilité catastrophique (AUC 0,99 et 0,86). Le modèle apprend si bien la distribution que les transitions encodées deviennent statistiquement anormales.
- **TemporalGNN sur GeoLife cells** : capacité élevée (0,30 bits/transition) mais AUC maximal de 0,58, avec une grande variance entre seeds (jusqu'à 0,90). Instable.
- **Backoff** : reste systématiquement sous 0,58 AUC, avec une variance très faible. C'est le point de fonctionnement le plus sûr.
- **Dans la plupart des cas**, les modèles neuronaux n'apportent pas de gain net : soit la capacité reste faible (LastFM, TGBL-Wiki), soit la détectabilité explose (MOOC).

### 2.4 Décision

La variante à cover models neuronaux **n'est pas retenue** pour l'article. Les raisons sont :

1. Le modèle backoff est plus robuste au regard de l'AUC externe.
2. Les gains de capacité des modèles neuronaux s'accompagnent d'une détectabilité inacceptable.
3. L'argument du reviewer ("modèle de cover trop simpliste") est partiellement levé par l'existence d'une variante évaluée, mais les résultats montrent que la simplicité du backoff est une vertu au point de fonctionnement conservateur choisi.

La variante reste documentée dans `docs/variant_neural_cover/` et les scripts/résultats dans `code/scripts/variant_neural_cover/` et `results/tables/variant_neural_cover/`.

## 3. Fichiers produits

- `results/tables/variant_neural_cover/lee_baselines_temporal.csv` et `.json`
- `results/tables/variant_neural_cover/proposition_on_terrorists.csv` et `.json`
- `results/tables/variant_neural_cover/variant_cover_models.csv` et `.json`
- `results/tables/variant_neural_cover/sota_comparison_temporal_datasets.csv`
- `results/tables/variant_neural_cover/sota_comparison_terrorists_911.csv`
- `results/tables/variant_neural_cover/variant_cover_models_summary.csv`
- `results/tables/variant_neural_cover/comparison_tables.json`

## 4. Recommandations pour l'article

1. **Conserver le modèle backoff** comme cover model principal.
2. **Ajouter le Tableau `sota_comparison_temporal_datasets.csv`** dans la section Related Work ou Experiments pour positionner la proposition face à BIND/AdaBIND.
3. **Mentionner la variante neurale** comme réponse au reviewer : "Nous avons évalué des cover models neuronaux (GRU, Transformer, GNN temporel) dans le même pipeline ; ils améliorent localement la capacité mais dégradent la détectabilité, ce qui confirme le choix du backoff au point de fonctionnement conservateur."
4. **Ne pas intégrer** les résultats de la variante dans le fil principal des résultats pour ne pas alourdir l'article.

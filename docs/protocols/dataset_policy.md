# Politique des datasets

## Regle de publication

Les tableaux et conclusions du manuscrit et du memoire utilisent uniquement
des traces reelles traitees. Les donnees synthetiques peuvent servir aux tests
unitaires, au debogage et au calibrage initial; elles ne figurent pas dans les
resultats scientifiques.

## Datasets utilises

Les resultats courants reposent sur cinq flux:

- `tgbl-wiki`: interactions temporelles Wikipedia;
- `mooc`: activite educative JODIE;
- `lastfm`: ecoute utilisateur-artiste JODIE;
- `geolife_cells`: trajectoires GeoLife discretisees en cellules;
- `t_drive_cells`: trajectoires T-Drive discretisees en cellules.

TGBL-Wiki, MOOC et LastFM sont traites comme flux d'interactions. GeoLife et
T-Drive fournissent les traces de mobilite de niveau A apres discretisation
spatiale. Porto Taxi et d'autres sources restent des candidats futurs lorsque
la licence, le telechargement et la segmentation reproductible sont disponibles.

## Decoupage

- entrainement: premiers evenements ou sequences selon le pipeline;
- validation: bloc causal suivant;
- test: bloc causal final;
- aucun melange aleatoire des evenements;
- tous les normaliseurs et hyperparametres sont ajustes avant le test final.

Pour les trajectoires, l'unite de decoupage est la sequence complete lorsque le
pipeline source le permet. La discretisation spatiale utilise une grille
deterministe et une analyse de sensibilite sur plusieurs tailles de cellules.

## Schemas canoniques

Les flux d'interactions utilisent:

`event_id, source, destination, timestamp, label, sequence_id, split`.

Les trajectoires utilisent:

`point_id, sequence_id, entity_id, timestamp, latitude, longitude, split`.

Les espaces d'identifiants bipartites sont explicitement separes. Par exemple,
TGB Wiki utilise les prefixes `user:` et `item:` afin de ne pas creer de
boucles artificielles.

## Tracabilite

Chaque dataset doit avoir une fiche dans `datasets/metadata/` contenant:
source, version, licence, checksum, date de telechargement, script de
pretraitement et statistiques avant/apres transformation.

Les donnees brutes tierces ne sont pas redistribuees si leur licence ne le
permet pas. L'artefact publiable documente alors comment les reacquerir et
comment reproduire les donnees traitees.

# ADR-003 : pipeline causal des données réelles

## Statut

Accepté le 13 juin 2026.

## Décision

Deux schémas sont conservés séparément : événements temporels et points de
trajectoire. Les trajectoires ne sont pas transformées prématurément en graphe
spatial. Cette transformation dépendra d'une discrétisation apprise ou choisie
sur l'entraînement uniquement.

Les interactions TGB sont divisées par timestamp, sans répartir un même
timestamp entre deux splits. GeoLife est divisé par trajectoire complète.
T-Drive est segmenté aux changements de jour et après 20 minutes
d'interruption; les frontières de splits sont des fins de journée.

Toute séquence franchissant une frontière est mise en quarantaine. Les
coordonnées T-Drive hors de l'enveloppe large Chine
`latitude=[18,54], longitude=[73,135]` sont exclues et comptabilisées.

## Conséquences

- absence de fuite de séquence entre entraînement, validation et test;
- ordre temporel strict entre les trois blocs;
- conservation des données brutes et des checksums;
- possibilité de comparer plusieurs discrétisations sans retraiter les ZIP;
- déséquilibre possible des tailles de splits lorsque les frontières
  naturelles sont privilégiées.

## Limites actives

- les termes de redistribution T-Drive ne sont pas explicites;
- l'échantillon T-Drive disponible contient moins de taxis et de points que
  les chiffres annoncés dans l'article;
- Porto exige une authentification Kaggle;
- le panel inter-domaines final n'est pas encore gelé.

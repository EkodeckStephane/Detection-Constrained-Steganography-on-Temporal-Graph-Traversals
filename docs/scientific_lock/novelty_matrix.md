# Matrice d'antériorité

Légende : oui = composant central; partiel = composant voisin; non = absent.

| Travail | Chemin | Temporel | Distribution apprise | Encodage distributionnel | Débit adaptatif | Flou | Stéganalyse externe |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lee: BIND | non | non | non | non | partiel | non | non |
| Lee: AdaBIND | non | non | non | non | oui, par ajout d'arêtes | non | non |
| Lee: BYNIS | non | non | non | non | partiel | non | non |
| Graph-Stega | oui, graphe de connaissances | non | partiel | non | non | non | partiel |
| ADG | non | séquentiel textuel | oui | oui, approché | oui | non | oui |
| iMEC | non | autorégressif | oui | oui, avec garantie sur le modèle | non | non | oui |
| ADLM-stega | non | séquentiel textuel | oui | partiel | oui, par entropie | non | oui |
| FreStega | non | séquentiel textuel | oui | oui | oui | non | oui |
| Adaptive 3D Mesh FPD | non | non | non | oui, coût/STC | oui | non | oui |
| Joshi et Bhand 2026 | non | non | non | non | oui | oui, Mamdani | classique seulement |
| Proposition | oui | oui | oui, causale | oui | oui, débit et abstention | oui | oui, détecteurs non vus |

## Revendications abandonnées

1. « Première stéganographie par parcours de graphe » : faux à cause de
   Graph-Stega.
2. « Première stéganographie utilisant la logique floue » : faux.
3. « Première adaptation du débit par entropie » : faux, notamment ADLM-stega
   et FreStega.
4. « Sécurité parfaite » : non défendable vis-à-vis d'une distribution réelle
   inconnue.

## Revendications provisoires autorisées

- première étude identifiée de l'encodage stéganographique dans des parcours
  de graphes temporels appris causalement;
- premier contrôleur flou identifié combinant débit, abstention, calibration,
  risque de détection et fragilité d'un parcours temporel;
- première évaluation identifiée de ce canal sur trajectoires ou sessions
  réelles avec stéganalyse externe et transfert inter-domaine.

Le terme « première » reste interdit dans le manuscrit tant que les recherches
Scopus, Web of Science, IEEE Xplore, ACM DL, SpringerLink, ScienceDirect et
Google Scholar n'ont pas été consignées et dédupliquées.

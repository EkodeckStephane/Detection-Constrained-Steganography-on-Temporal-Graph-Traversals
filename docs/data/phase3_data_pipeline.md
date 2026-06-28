# Phase 3/10 - Pipeline causal des données réelles

Date de gel : 13 juin 2026.

## Résultat

Cinq datasets réels sont acquis et transformés dans deux schémas canoniques :

- niveau A : GeoLife 1.3 et l'échantillon officiel T-Drive;
- niveau B : tgbl-wiki-v2, MOOC/JODIE et LastFM/JODIE.

Les données brutes sont conservées avec leurs checksums. Les sorties Parquet
sont séparées en entraînement, validation et test. Le validateur indépendant
confirme l'ordre temporel strict et l'absence de trajectoire ou session commune
entre splits.

## Panel

| Dataset | Niveau | Domaine | Taille utile |
|---|---:|---|---:|
| GeoLife | A | mobilité personnelle multimodale | 24 867 361 points |
| T-Drive | A | mobilité taxi urbaine | 10 055 095 points |
| tgbl-wiki | B | édition collaborative | 157 474 interactions |
| MOOC | B | apprentissage en ligne | 411 749 interactions |
| LastFM | B | écoute musicale | 1 293 103 interactions |

GeoLife conserve 18 668 trajectoires après quarantaine de deux trajectoires
franchissant les frontières. T-Drive produit 211 404 sessions, segmentées aux
changements de jour et aux interruptions supérieures à 20 minutes.

## Faits structurants

Sur les flux bipartites, l'entropie conditionnelle de la destination vaut :

- tgbl-wiki : 1,197 bit, soit 2,293 continuations effectives;
- MOOC : 4,880 bits;
- LastFM : 7,034 bits.

La fraction d'arêtes répétées vaut respectivement 88,4 %, 56,7 % et 88,0 %.
Un contrôleur de débit fixe serait donc difficile à défendre : les marges
d'encodage varient fortement selon le domaine et le contexte local.

GeoLife présente une médiane de 506 points par trajectoire, une durée médiane
de 2 527,5 secondes et un intervalle médian de deux secondes. T-Drive présente
20 points par session, une durée médiane de 5 408 secondes et un intervalle
médian de 300 secondes.

## Contrôles de fuite

- TGB/JODIE : un timestamp identique appartient à un seul split.
- GeoLife : une trajectoire entière appartient à un split; deux trajectoires
  traversant une frontière sont exclues.
- T-Drive : une session entière appartient à un split; les frontières sont
  placées à minuit et aucune session n'est mise en quarantaine.
- Les espaces utilisateurs et items sont préfixés afin d'éviter les collisions
  d'identifiants bipartites.
- La discrétisation spatiale est différée : elle sera ajustée uniquement sur
  l'entraînement.

## Limites et licences

GeoLife est soumis à la licence Microsoft Research, usage non commercial.
La page T-Drive ne fournit pas de termes explicites de redistribution.
L'échantillon T-Drive disponible contient 8 911 fichiers et 10,09 millions de
points bruts, contre 10 357 taxis et environ 15 millions de points annoncés
dans l'article. Les termes SNAP de MOOC et LastFM doivent encore être validés
avant redistribution dans un dépôt public.

L'archive GeoLife contient 18 670 fichiers PLT, alors que son guide en annonce
17 621. Cet écart est conservé dans la fiche de provenance, pas corrigé
silencieusement.

Porto Taxi n'est pas téléchargé automatiquement, car Kaggle exige une
authentification et l'acceptation des règles de compétition.

## Verdict

La phase 3 est techniquement validée :

- cinq datasets réels et quatre familles comportementales;
- 16 tests automatisés réussis;
- checksums et fiches de provenance;
- Parquet compressés et scripts reproductibles;
- splits causalement vérifiés par un programme indépendant.

Le panel est suffisant pour commencer la phase 4 de modélisation. Avant le gel
du dépôt Q1, T-Drive et SNAP devront obtenir une décision explicite de
redistribution ou être remplacés dans le paquet public.

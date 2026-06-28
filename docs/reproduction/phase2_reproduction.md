# Phase 2/10 - Reproduction des méthodes de référence

Date de gel : 13 juin 2026.

## Objet

Cette phase reproduit BIND, AdaBIND et BYNIS de Lee et al., puis met en place
les parcours uniforme, DeepWalk et node2vec. Elle ne cherche pas encore à
valider la méthode proposée. Son rôle est d'établir des baselines exécutables,
déterministes et suffisamment contrôlées pour les comparaisons ultérieures.

## Provenance

- dépôt officiel : `https://github.com/dwgoon/hmg`;
- commit figé : `b7499454be42aa24b61825b2f87f9083539b02fa`;
- licence : BSD 3-Clause;
- réseau réel de contrôle : `terrorists-911`, 62 sommets et 152 arêtes;
- SHA-256 du fichier : `0897b4df9fcddbb3ed42b04e4188dde1d72de0f72011d56f54d55315bfeb03a4`.

Le snapshot minimal du code officiel est conservé dans
`literature/external/lee_hmg_snapshot`. BIND a été comparé directement à
l'oracle HMG : pour le même graphe, le même message et la même clé, la liste
d'arêtes produite est strictement identique.

## Résultats essentiels

Les quatre classes d'arêtes de BIND comptent respectivement 51, 39, 37 et 25
arêtes. La borne équilibrée publiée vaut donc 200 bits. Les charges utiles de
8, 16 et 24 octets sont récupérées exactement, sans changement du multiensemble
des arêtes, pour 0,421, 0,842 et 1,263 bit par arête.

Un message de huit octets `0xff`, choisi pour surreprésenter la classe `11`,
fait échouer BIND. AdaBIND le rend encodable après l'ajout de sept arêtes en
sept itérations. La récupération est exacte, mais la topologie est modifiée.

BYNIS récupère exactement un message de 18 octets. Son débit est de 7,2 bits
par arête sans camouflage, puis de 2,057 bits par arête avec 50 arêtes
supplémentaires. Ce chiffre élevé ne doit pas être comparé directement à BIND
ou à notre canal : BYNIS synthétise le graphe porteur.

Sur 100 attaques par niveau, un seul échange adjacent, touchant au plus 1,32 %
des positions, réduit le taux de récupération exacte de BIND à 42 %. Avec
quatre échanges, soit 5,26 % des positions touchées, ce taux tombe à 3 %.
Cette fragilité justifie l'intégration future d'un code correcteur et d'une
politique explicite de synchronisation.

## Décisions d'implémentation

La réimplémentation conserve le format de longueur, les classes de parité et
la permutation MT19937 de HMG. Deux corrections de robustesse sont assumées :

1. AdaBIND réinitialise le meilleur candidat à chaque itération, afin d'éviter
   la réutilisation d'une arête devenue invalide dans le code publié.
2. Les quatre comptes de parité sont recalculés après chaque ajout. Cela
   garantit que le changement de degré des deux extrémités et de toutes leurs
   arêtes incidentes est pris en compte.

Les générateurs de parcours sont déterministes par graine. DeepWalk est traité
comme un générateur de corpus de marches uniformes; node2vec applique le biais
du second ordre défini par `p` et `q`.

## Verdict

La phase 2 est validée pour son objectif de reproduction logicielle :

- 10 tests automatisés réussissent;
- BIND correspond exactement à l'oracle officiel;
- BIND, AdaBIND et BYNIS décodent sans erreur en canal passif;
- les changements topologiques sont distingués explicitement;
- les parcours uniforme, DeepWalk et node2vec sont testés et déterministes;
- la campagne et ses sorties JSON/CSV sont figées.

Cette phase ne reproduit pas encore toutes les grandes campagnes OGB de
l'article et ne constitue pas une validation Q1. La suite doit construire le
pipeline de trajectoires et sessions réelles, avec splits temporels causaux.

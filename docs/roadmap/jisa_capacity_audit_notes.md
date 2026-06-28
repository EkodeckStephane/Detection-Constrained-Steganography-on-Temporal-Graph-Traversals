# Notes sur l'audit de capacite JISA

Date : 28 juin 2026.

## Resultat principal

L'audit `phase11_capacity_audit` separe trois quantites :

1. l'entropie moyenne de la distribution candidate ;
2. l'entropie sure apres contraintes de detection et de rang ;
3. les bits consommes par le codec local exactement decodable.

Le diagnostic indique que la capacite publiee depend fortement de la rarete des
etats faisables. Sur MOOC, LastFM et GeoLife, les transitions qui passent les
contraintes sont tres rares. Sur TGBL-Wiki et T-Drive, il existe davantage
d'entropie sure que le codec local n'en recupere.

## Consequence codec

Le backend `range` local precedent pouvait annoncer un bit consomme lorsque
l'action emise correspondait a plusieurs prefixes binaires. Cette situation
cree une ambiguite de decodage pour une interface qui recupere les bits a partir
de chaque action separement. Le codec local a donc ete corrige pour ne consommer
un prefixe que lorsque l'action selectionnee correspond a un intervalle entier
singleton.

Cette correction rend le contrat Alice-Bob exact, mais elle reduit le debit
local. Elle clarifie la distinction entre :

- un codage local par action, simple et exactement decodable ;
- un vrai range coding sequentiel, qui accumule l'information sur plusieurs
  actions et peut exploiter des intervalles plus larges.

## Lecture par dataset

- TGBL-Wiki : entropie sure par transition superieure au debit local ; un range
  coding sequentiel peut recuperer une partie de la perte.
- T-Drive cells : meme profil que TGBL-Wiki, avec une perte codec visible.
- MOOC : la rarete des etats faisables domine ; le codec peut peu aider au
  point conservateur actuel.
- LastFM : entropie brute elevee, mais contraintes et decodage local annulent
  pratiquement le debit.
- GeoLife cells : faisabilite faible et distribution dominee par des actions
  tres probables ; le gain codec dependra d'une politique de securite moins
  restrictive ou d'un meilleur modele.

## Decision

L'etape suivante cote capacite est un prototype de range coding sequentiel a
precision finie. L'objectif est de recuperer une partie de l'entropie sure sur
TGBL-Wiki et T-Drive tout en gardant :

- synchronisation exacte ;
- distorsion locale mesuree ;
- contraintes adversariales conservees ;
- separation entre entropie disponible, debit tente et debit recupere.


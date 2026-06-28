# Notes sur l'audit adversarial consolide

Date : 28 juin 2026.

## Objectif

L'audit `phase11_adversarial_audit` rassemble les resultats adversariaux
coherents avec le codec local corrige :

- detecteurs externes sur features publiques ;
- steganalystes neuraux independants sur sequences publiques ;
- gradient boosting white-box borne sur diagnostics publics ;
- audit oracle avec variables d'instrumentation.

## Resultat principal

Les maxima AUC publics bornes restent faibles sur les cinq datasets. Les deux
cas les plus informatifs sont :

- TGBL-Wiki : max public borne 0.540, oracle 0.537 ;
- T-Drive cells : max public borne 0.523, oracle 0.533.

Les autres datasets restent proches de 0.50, principalement parce que le regime
conservateur encode tres peu.

## Portee

L'audit couvre plusieurs familles de detecteurs, mais il reste un protocole
interne. Les couches suivantes pour un dossier JISA complet sont :

1. red-team tiers ;
2. steganalystes temporal-GNN plus larges ;
3. transfert inter-domaine ;
4. adversaire actif non borne.

## Consequence pour le manuscrit

Le manuscrit doit parler de "bounded steganalysis" et de "empirical security
under the declared adversarial budget". Les resultats doivent etre presentes
comme une evaluation par couches, avec une colonne separant les features
publiques et les variables oracle.


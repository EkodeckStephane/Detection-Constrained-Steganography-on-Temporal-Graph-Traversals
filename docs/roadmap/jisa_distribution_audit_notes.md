# Notes sur l'audit distributionnel JISA

Date : 28 juin 2026.

## Objectif

L'audit `phase11_distribution_audit` mesure directement la proximite entre
traces naturelles et traces stego sur les features publiques Phase 7. Il
complete les AUC de steganalyse avec des distances distributionnelles :

- MMD RBF multivarie ;
- Wasserstein 1D par feature ;
- tests de permutation sur les moyennes ;
- Jensen-Shannon sur features discretes.

## Resultat principal

Les distances sont tres faibles pour GeoLife cells, LastFM et MOOC. Ces trois
datasets correspondent a un regime de tres faible debit avec une forte
proportion de transitions couvertes.

TGBL-Wiki et T-Drive cells montrent les ecarts les plus visibles. Ces deux
datasets portent aussi la plus grande partie du debit utile apres correction du
codec local. Le compromis capacite-detectabilite se concentre donc sur ces deux
domaines.

## Lecture par dataset

- GeoLife cells : MMD proche de zero, Wasserstein moyen inferieur a
  \(10^{-4}\), p-values elevees. Le regime est extremement conservateur.
- LastFM : distances quasi nulles au test, avec debit quasi nul.
- MOOC : distances faibles, debit quasi nul, ecarts de moyenne non
  significatifs dans l'audit actuel.
- T-Drive cells : MMD test \(4.1\times 10^{-4}\), Wasserstein moyen environ
  \(8.4\times 10^{-3}\), p-value minimale \(0.005\). Le signal distributionnel
  existe mais reste limite sur les features publiques.
- TGBL-Wiki : MMD test \(1.4\times 10^{-3}\), Wasserstein moyen environ
  \(2.9\times 10^{-2}\), p-value minimale \(0.005\). C'est le dataset qui porte
  le compromis le plus visible.

## Consequences pour le manuscrit

1. Le titre doit privilegier "detection-constrained" ou "distribution-audited".
2. Les resultats doivent presenter les distances distributionnelles avec les
   AUC, afin de separer detectabilite supervisee et ecart de distribution.
3. Les claims doivent distinguer deux regimes :
   - regime quasi-cover : GeoLife, LastFM, MOOC ;
   - regime faible debit mesurable : TGBL-Wiki, T-Drive.
4. Les courbes capacite-risque doivent focaliser l'analyse sur TGBL-Wiki et
   T-Drive, car ce sont les domaines ou le codec et la politique peuvent encore
   apporter un gain.

## Decision

L'audit distributionnel soutient une redaction centree sur un canal
`detection-constrained` avec mesures distributionnelles directes. Il fournit
une base plus solide que les seules AUC pour discuter la proximite
naturel/stego.


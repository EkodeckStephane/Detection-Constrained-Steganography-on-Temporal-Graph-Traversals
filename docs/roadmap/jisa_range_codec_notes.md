# Notes sur le prototype range coding sequentiel

Date : 28 juin 2026.

## Objectif

Le prototype `phase11_range_codec` estime le gain possible d'un range coding
sequentiel a precision finie. Contrairement au codec local par action, ce
prototype accumule l'information sur des blocs de transitions sures, puis
recupere le prefixe binaire garanti par l'intervalle final.

## Resultat principal

Le gain existe surtout lorsque le dataset fournit assez de transitions sures.
T-Drive cells est le cas le plus favorable : le debit estime passe d'environ
0.030 bit par transition avec le codec local a environ 0.065 bit par transition
avec le prototype sequentiel. TGBL-Wiki montre un gain plus modeste, d'environ
0.006 bit par transition.

Sur MOOC, LastFM et GeoLife, le facteur dominant reste la rarete des transitions
sures au point conservateur. Le codec sequentiel aide peu lorsque la politique
d'abstention laisse trop peu d'occasions d'encoder.

## Lecture par dataset

- TGBL-Wiki : 772 transitions sures sur 10,000 ; gain estime de 0.0061 bit par
  transition.
- T-Drive cells : 879 transitions sures sur 10,000 ; gain estime de 0.0349 bit
  par transition.
- GeoLife cells : peu de transitions sures, mais le prototype recupere mieux
  leur entropie.
- LastFM : quelques transitions sures, gain absolu faible.
- MOOC : transitions sures trop rares dans cette configuration.

## Consequence scientifique

Le range coding sequentiel merite une integration propre pour TGBL-Wiki et
T-Drive. Il repond partiellement a la critique de capacite, mais le principal
levier reste la politique qui determine quelles transitions deviennent sures.

La prochaine version experimentale doit donc separer deux axes :

1. meilleur codec sur les transitions deja sures ;
2. meilleur modele/politique pour augmenter la fraction de transitions sures.

## Consequence pour le manuscrit

Le manuscrit peut presenter le range coding comme une amelioration de codec
fondee sur l'audit de capacite. La formulation doit rester quantitative :
entropy available, local codec rate, sequential range-coding rate, and residual
gap.


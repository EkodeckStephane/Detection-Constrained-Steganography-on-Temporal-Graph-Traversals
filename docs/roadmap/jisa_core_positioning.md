# Cadrage central pour la refonte JISA

Date : 28 juin 2026.

## These centrale

Le manuscrit doit etre reconstruit autour de la these suivante :

> Il est possible de definir un canal steganographique comportemental sur des
> traversales de graphes temporels, dans lequel les bits ne modifient pas la
> topologie mais influencent le choix de continuations valides, avec abstention
> explicite lorsque le risque de detection ou de desynchronisation devient trop
> eleve.

Cette these est volontairement plus faible que "distribution-preserving" ou
"provably indistinguishable". Elle est plus compatible avec les resultats
actuels et avec les critiques JISA.

## Probleme de securite

Les methodes structurelles cachent l'information dans des modifications du
graphe ou dans une topologie synthetisee. Dans des systemes temporels reels,
ces modifications peuvent etre incompatibles avec les contraintes
operationnelles : historiques causaux, continuations disponibles, timestamps,
impasses, cold-start et surveillance sequentielle.

Le probleme traite est donc :

> Comment transmettre un signal discret dans un flux d'interactions temporelles
> sans modifier les sommets, aretes ou timestamps observables, tout en maintenant
> une detectabilite faible face a des steganalystes temporels bornes ?

Le papier ne doit pas pretendre resoudre toute la steganographie sur graphes
temporels. Il doit etablir un cadre, des bornes conditionnelles, et un protocole
experimental strict pour un regime faible debit.

## Contributions autorisees

### C1. Canal non structurel de traversale temporelle

Definition formelle d'un canal ou le cover/stego est une sequence d'actions
temporellement admissibles. L'encodage agit sur le choix parmi les
continuations valides, sans ajout ni suppression d'aretes et sans modification
des timestamps.

### C2. Abstention sous contrainte de detection

La decision d'encodage inclut des modes de non-encodage (`COVER`, `PAUSE`,
`STOP`) lorsque l'entropie, la calibration, le risque de detection ou le risque
de desynchronisation rendent l'insertion dangereuse. Le controleur flou est une
implementation possible, pas la contribution principale.

### C3. Evaluation adversariale bornee et reproductible

Le protocole separe les detecteurs publics, les detecteurs externes, les
adversaires white-box bornes, les audits oracle et les attaques actives de
fiabilite. Les resultats sont empiriques et limites au budget de menace declare.

### C4. Audit de capacite

Le manuscrit doit rapporter l'entropie disponible apres filtrage de securite, la
perte due au codec, et le gain eventuel d'un range coding contraint. Cette
contribution est necessaire pour repondre a la critique de capacite quasi nulle.

## Revendications interdites sans nouvelle preuve

- "Provable indistinguishability" contre la distribution naturelle inconnue.
- "Distribution-preserving" comme titre ou contribution centrale.
- "First" sans audit bibliographique cible sur steganographie comportementale
  et flux d'interaction.
- "Undetectable" sans qualifier l'adversaire, le budget et les intervalles.
- Comparaison directe de debit avec BIND, AdaBIND ou BYNIS sans avertissement
  sur les paradigmes et unites.

## Modele de menace JISA

### Parties

- Alice choisit une trace admissible et encode un message chiffre.
- Bob decode a partir du meme modele, des memes regles de candidats et de la
  meme cle.
- Eve observe la trace publique, connait l'algorithme, le modele, les donnees
  publiques, les seuils, les scripts et les hyperparametres.

### Capacites d'Eve

1. Eve passive publique : utilise seulement les sequences source-action-temps.
2. Eve externe : entraine des detecteurs non utilises dans la selection du point
   d'exploitation.
3. Eve white-box bornee : connait le codec, le modele de couverture, la
   politique d'abstention et les diagnostics publics.
4. Audit oracle : ajoute des variables internes non publiques pour quantifier la
   marge de securite en cas de fuite d'instrumentation.
5. Canal actif borne : supprime, insere, reordonne ou tronque selon un budget
   declare ; cette partie evalue la fiabilite, pas la furtivite.

### Hors perimetre

- Compromission de la cle ou des terminaux.
- Adversaire controlant la generation naturelle des traces.
- Garantie informationnelle parfaite contre \(P^\star\).
- Attaques actives non bornees, suppression en rafale non modelisee et
  substitution adversariale generalisee.

## Formulation de titre candidate

Titre prefere :

> Detection-Constrained Steganography over Temporal Graph Traversals

Alternatives :

- Behavioral Steganography in Temporal Graph Traversals under Bounded
  Steganalysis
- Low-Rate Covert Signaling through Valid Temporal Graph Continuations

Titres a eviter :

- Distribution-Preserving Temporal Graph Steganography
- Provably Indistinguishable Temporal Graph Traversal Steganography
- Adaptive Graph Traversal Steganography with White-Box Security

## Structure manuscrit cible

1. Introduction : probleme de securite, contraintes temporelles, contributions
   sobres.
2. Related work : structurel vs comportemental vs distributionnel, sans tableau
   de superiorite artificiel.
3. Threat model and channel : parties, adversaires, objet observable,
   continuations admissibles.
4. Method : cover model, abstention, codec, synchronisation.
5. Distortion and detectability analysis : borne conditionnelle et limites.
6. Experimental protocol : splits, adversaires, mesures distributionnelles,
   capacite.
7. Results : risque-capacite, distribution, adversaires, robustesse.
8. Limitations and deployment scope : faible debit, adversaires hors perimetre,
   datasets.
9. Conclusion : baseline faible debit audite et feuille de route.

## Phrase de contribution cible

Version courte :

> We introduce a detection-constrained behavioral steganographic channel over
> temporal graph traversals, where secret bits select among valid temporal
> continuations and an explicit abstention policy prevents embedding in
> high-risk states.

Version prudente pour abstract :

> The contribution is not a new temporal graph model or a new cryptographic
> primitive. It is a security-oriented formulation, implementation, and
> evaluation of low-rate covert signaling through valid temporal graph
> continuations under bounded steganalysis.


# Plan d'action pour une refonte compatible JISA

Date de reference : 28 juin 2026.

## Position centrale

Le manuscrit ne doit plus etre positionne comme une contribution de
``distribution-preserving steganography'' avec indistinguabilite prouvee. La
position defendable est plus sobre :

> Une steganographie comportementale sur traversales de graphes temporels, sans
> modification topologique, avec abstention sous contrainte explicite de
> detectabilite.

Le travail doit devenir un article de securite adversariale sur canal
comportemental temporel, et non un article d'integration de briques IA/ML.

## Contributions a conserver

1. Canal non structurel sur graphes temporels : l'information est cachee dans le
   choix de continuations temporellement valides, sans ajout ni suppression
   d'aretes.
2. Abstention sous risque de detection : le systeme accepte de ne pas encoder
   lorsque le contexte est trop fragile.
3. Evaluation adversariale bornee : detection externe, adversaires white-box et
   audits oracle doivent etre presentes comme evaluation empirique, pas comme
   preuve absolue d'indetectabilite.

Les modeles TGN, GRU, Transformer, le controleur flou, le couplage, HMAC ou les
codes correcteurs ne doivent pas etre presentes comme contributions principales.
Ils sont des outils au service du canal et du modele de menace.

## Plan de refonte

1. Auditer le manuscrit actuel en reliant chaque critique reviewer a une action
   concrete : suppression, reformulation, experience supplementaire ou limite
   explicite.
2. Reformuler l'introduction autour d'un probleme de securite non resolu :
   cacher de l'information dans des flux temporels reels sans modifier la
   topologie et sous adversaire temporel fort.
3. Remplacer les revendications fortes par une analyse formelle modeste :
   borne locale de distorsion, accumulation sur trajectoire et lien avec
   l'avantage maximal d'un detecteur via la distance en variation totale.
4. Ajouter une evaluation distributionnelle directe : MMD ou Wasserstein,
   tests deux-echantillons, inter-arrival times, rangs, motifs temporels,
   calibration du modele de couverture et distances entre naturel, couverture
   simulee et stego.
5. Renforcer l'evaluation adversariale : detecteurs sequentiels, GNN ou
   temporal GNN, adversaires white-box, audits oracle, intervalles de confiance
   et tests hors-domaine si possible.
6. Traiter explicitement la capacite : soit augmenter le debit, soit justifier
   un cas d'usage a faible debit comme signal discret, watermark comportemental,
   alerte courte, dead-drop ou transmission de cle courte. La piste prioritaire
   pour augmenter le debit est un codage arithmetique ou range coding contraint,
   utilise comme optimisation de l'entropie sure disponible, pas comme source de
   capacite nouvelle.
7. Retrograder ou simplifier le controleur flou. Le comparer a un seuil simple,
   une MLP et des politiques ablatees. S'il n'apporte pas un gain clair, le
   deplacer en composant secondaire ou en annexe.
8. Recrire le manuscrit comme un papier de securite : probleme, modele de
   menace, canal, borne de distorsion, protocole adversarial, resultats et
   limites.

## Recommandation capacite : codage arithmetique contraint

Le codage arithmetique, ou sa variante range coding, doit etre explore en
priorite pour ameliorer la capacite. Il ne cree pas d'information cachable :
si la politique sure a une entropie faible, le debit restera faible. En
revanche, il peut recuperer les pertes dues aux groupements discrets, aux
seuils de rang, aux arrondis et aux choix de type `floor(log2 k)`.

Objectif technique :

1. Mesurer l'entropie conditionnelle disponible apres contraintes de securite
   et d'abstention.
2. Comparer cette borne au debit actuel afin d'identifier la perte de codec.
3. Implementer un prototype de range coding deterministe a precision finie,
   avec ordre canonique des candidats, intervalles entiers et regles d'arrondi
   partagees par Alice et Bob.
4. Verifier que le gain de debit ne degrade pas les AUC adversariales ni les
   mesures distributionnelles directes.

Critere de succes :

- augmenter les bits utiles par transition aux points de fonctionnement
  conservateurs ;
- conserver la synchronisation exacte Alice-Bob ;
- maintenir les contraintes de detection et de distribution ;
- rapporter separement la borne d'entropie disponible, le debit tente et le
  debit effectivement recupere.

Si le gain est faible, la conclusion doit etre explicite : la faible capacite
vient du canal et des contraintes de furtivite, pas du codec.

## Criteres go/no-go avant resoumission

- Les adversaires renforces restent proches du hasard ou clairement bornes.
- Les mesures distributionnelles soutiennent explicitement le positionnement.
- Le debit est ameliore, idealement via range coding contraint, ou justifie par
  un cas d'usage credible.
- Le seuil de securite n'est plus arbitraire.
- Les comparaisons avec BIND, AdaBIND et BYNIS sont recadrees ou deplacees.
- Les mots ``premier'', ``indetectable'', ``provable'' et
  ``distribution-preserving'' ne sont utilises que si les preuves et mesures les
  soutiennent rigoureusement.

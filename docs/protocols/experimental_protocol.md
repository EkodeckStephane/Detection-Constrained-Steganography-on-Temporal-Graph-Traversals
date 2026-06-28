# Protocole experimental courant

## Question principale

A charge utile faible et fiabilite passive verifiee, un couplage
distributionnel de parcours temporels avec abstention peut-il maintenir les
detecteurs externes sous `AUC_ext <= 0.60` sur des traces reelles traitees?

## Baselines implementees

1. Modele frequenciel source-destination.
2. Modele temporel source plus destination precedente.
3. Baselines GRU et Transformer echantillonnees.
4. Codec EMBED-only pour contrat Alice-Bob.
5. Seuil fixe d'entropie.
6. Controleur flou deterministe.
7. Politique MLP de complexite comparable.
8. Couplage avec abstention et cout local de distorsion.
9. Detecteurs externes lineaire, foret aleatoire et MLP.

Les baselines BIND, AdaBIND, GNN/GAT temporels et steganalystes graphes
profonds restent des exigences futures, pas des resultats deja publies.

## Metriques

### Communication

- bits utiles par transition et bits tentes par transition;
- taux `EMBED`, `COVER`, `PAUSE` et `STOP`;
- BER passif et BER corrige dans le contrat codec;
- variation totale et divergence KL locales.

### Furtivite

- AUC, balanced accuracy et EER de detecteurs externes;
- ablations de caracteristiques;
- robustesse sous sous-echantillonnage, troncature et bruit de timestamp.

### Couverture

- log-loss, perplexite, top-1 accuracy et entropie;
- fractions de contextes et destinations inconnus;
- sensibilite de discretisation spatiale pour les traces de mobilite.

## Statistiques

Les resultats courants utilisent des splits deterministes et des graines
fixes. Les intervalles bootstrap multi-seed, tests apparies, tailles d'effet
et correction de Holm sont des exigences de validation futures avant une
revendication plus forte.

## Selection du point d'exploitation

Le point courant maximise le debit observe sous les contraintes:

- AUC externe au plus 0.60;
- BER passif nul dans le contrat codec;
- emission de couverture lorsque les seuils locaux de variation totale, KL et
  entropie ne sont pas satisfaits.

Les attaques actives de phase 8 servent a tester la robustesse des
caracteristiques de detection. Elles ne prouvent pas encore une communication
fiable sous adversaire actif.

## Critere de succes courant

Le resultat publiable actuel est un point conservateur: tous les detecteurs
externes testes restent sous `AUC_ext <= 0.60`, mais le debit est faible sur
plusieurs datasets. Le manuscrit doit presenter ce compromis explicitement et
ne pas revendiquer un systeme haut debit.

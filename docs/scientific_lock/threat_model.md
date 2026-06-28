# Modèle de menace et canal stéganographique

## Objet transmis

Le cover et le stego sont des traces de parcours temporel, pas des graphes
modifiés. Pour un graphe temporel observé jusqu'à l'instant \(T\), une trace
valide est :

\[
\tau=(v_0,e_1,t_1,v_1,\ldots,e_L,t_L,v_L),\qquad
t_1<\cdots<t_L\leq T.
\]

Chaque transition appartient au graphe disponible et respecte les règles du
domaine. La méthode principale n'ajoute ni sommet ni arête et ne modifie pas
les timestamps. Le canal public transporte la trace, par exemple comme
séquence de références, journal de navigation ou trajectoire partagée.

## Parties

- Alice possède un message, une clé secrète et la version figée du modèle.
- Bob possède la même clé, le même modèle, le même graphe de référence et les
  mêmes règles de construction des candidats.
- Eve observe la trace, connaît l'algorithme, l'architecture, les données
  publiques, les hyperparamètres et peut entraîner ses propres détecteurs.
  Elle ne connaît pas la clé.

Cette hypothèse suit le principe de Kerckhoffs. La sécurité ne doit pas
reposer sur le secret de l'algorithme ou du modèle.

## Encodage

À l'étape \(i\), le modèle causal estime une distribution
\(q_\theta(a_i\mid \tau_{<i},G_{\leq t_i})\) sur les continuations admissibles.
Le message est d'abord chiffré et blanchi. Un codeur entropique sélectionne une
continuation compatible avec les bits. Un contrôleur flou peut réduire le
débit ou s'abstenir selon :

- l'entropie locale;
- l'incertitude ou l'erreur de calibration;
- le risque donné par un stéganalyste;
- la charge restante;
- le risque de blocage et la fragilité du canal.

L'abstention prend quatre formes explicites : `EMBED`, `COVER`, `PAUSE` et
`STOP`. Le symbole `STOP` appartient à l'espace probabiliste et permet de
comparer des traces de longueurs variables.

## Propriété de correction

Dans le canal passif et avec des états synchronisés :

\[
\Pr[\operatorname{Dec}_k(\operatorname{Enc}_k(m,G),G)=m]=1.
\]

Cette propriété doit être testée sur toutes les longueurs, graines et règles
d'abstention. Une méthode incapable de garantir cette synchronisation est
abandonnée.

## Adversaire principal

Eve est passive mais adaptative :

- elle connaît les distributions de covers et peut obtenir des exemples
  stego;
- elle utilise des détecteurs non vus par Alice;
- elle combine caractéristiques locales, séquentielles et globales;
- elle peut tester un autre domaine ou une autre période;
- elle choisit son seuil après apprentissage, mais jamais sur le test final.

L'objectif n'est pas la « sécurité parfaite ». Le modèle \(q_\theta\) n'est
qu'une approximation de la distribution réelle \(P^\star\). Les garanties
seront donc séparées en fidélité à \(q_\theta\) et détectabilité empirique face
à \(P^\star\).

## Canal actif secondaire

Une seconde évaluation autorise Eve ou le canal à supprimer, insérer,
réordonner, tronquer ou retarder des événements. Les codes correcteurs
protègent la fiabilité, mais ne doivent pas être présentés comme une garantie
de furtivité.

## Hors périmètre

- compromission d'Alice ou Bob;
- vol de clé ou du modèle privé;
- analyse juridique ou opérationnelle d'un déploiement clandestin;
- adversaire contrôlant entièrement la génération des covers;
- garantie informationnelle parfaite par rapport à une distribution réelle
  inconnue.

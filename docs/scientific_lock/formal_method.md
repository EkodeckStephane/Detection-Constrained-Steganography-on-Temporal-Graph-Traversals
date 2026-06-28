# Formalisation de la méthode proposée

## 1. Notation

Soit \(G_{\le t}\) le graphe temporel observable jusqu'à l'instant \(t\). Une
trace de longueur au plus \(L_{\max}\) est

\[
\tau=(a_1,\ldots,a_L,\mathtt{STOP}),
\]

où chaque \(a_i\) est une transition admissible et \(\mathtt{STOP}\) termine
la trace. L'historique avant l'étape \(i\) est
\(h_i=(a_{<i},G_{\le t_i})\), et l'ensemble déterministe des actions
admissibles est \(\mathcal A(h_i)\cup\{\mathtt{STOP}\}\).

Quatre distributions sont distinguées :

- \(P^\star\) : distribution réelle, inconnue, des traces naturelles;
- \(Q_\theta\) : distribution causale apprise par TGN/GAT et GRU;
- \(C_\psi\) : distribution après contrôle du débit, du risque et de
  l'abstention;
- \(S_{\theta,\psi,\phi}\) : distribution effectivement induite par
  l'encodage du message chiffré.

Cette séparation interdit de confondre fidélité au modèle et fidélité au
domaine réel.

## 2. Objectif multiobjectif

Pour une politique d'encodage \(\pi=(\theta,\psi,\phi)\), l'objectif
scientifique est

\[
\max_\pi\quad
\mathcal J(\pi)
=R(\pi)
-\lambda_d D(S_\pi,P^\star)
-\lambda_r \operatorname{Risk}_{Eve}(\pi)
-\lambda_e \operatorname{BER}(\pi)
-\lambda_c \operatorname{Cost}(\pi).
\]

Ici :

- \(R(\pi)=\mathbb E[B(\tau)/N(\tau)]\) est le débit utile en bits par
  transition transmise;
- \(D\) est évaluée par KL lorsque les densités sont estimables, et par
  Jensen--Shannon, MMD et tests à deux échantillons dans les expériences;
- \(\operatorname{Risk}_{Eve}\) est une perte propre calibrée, complétée par
  AUC, EER et balanced accuracy de détecteurs externes;
- \(\operatorname{BER}\) mesure l'erreur avant et après correction;
- \(\operatorname{Cost}\) couvre latence, mémoire et longueur induite.

Le point d'exploitation n'est pas obtenu par une somme arbitraire sur le test.
Il est choisi sur validation comme solution de :

\[
\max_\pi R(\pi)
\quad\text{sous}\quad
\mathrm{AUC}_{ext}\le \delta,\;
\mathrm{BER}\le \eta,\;
\Delta U\le \zeta,
\]

avec \(\delta=0{,}60\), \(\eta=10^{-3}\) sous attaque retenue et une borne
\(\zeta\) préenregistrée pour l'utilité aval.

## 3. Modèle causal de continuation

Le modèle produit des logits \(z_\theta(a\mid h_i)\), puis

\[
Q_\theta(a\mid h_i)
=\operatorname{softmax}_{a\in\mathcal A(h_i)\cup\{\mathtt{STOP}\}}
\left(z_\theta(a\mid h_i)/T\right).
\]

Il est entraîné uniquement sur le passé du split causal. Sa calibration est
mesurée par log-loss, Brier score, ECE et calibration par groupe de degré,
heure, longueur de trace et domaine. Une calibration post-hoc est ajustée sur
validation seulement.

## 4. Coût guidé par la détectabilité

Pour chaque continuation admissible :

\[
c_i(a)=
\alpha[-\log Q_\theta(a\mid h_i)]
+\beta\,r_{\mathrm{steg}}(a,h_i)
+\gamma\,r_{\mathrm{canal}}(a,h_i)
+\xi\,r_{\mathrm{cal}}(a,h_i).
\]

Les termes représentent respectivement la surprise sous le modèle, le risque
prévu par le stéganalyste interne, la fragilité aux attaques du canal et
l'incertitude de calibration. La distribution contrôlée de base est

\[
\widetilde C_\psi(a\mid h_i)
\propto Q_\theta(a\mid h_i)\exp[-c_i(a)].
\]

Le signal du détecteur intervient donc dans le coût des actions avant le
codage, et non uniquement comme perte auxiliaire globale.

## 5. Politique floue de débit et d'abstention

Le contrôleur Takagi--Sugeno reçoit :

\[
x_i=(H_i,\;u_i,\;r_i,\;p_i,\;d_i,\;f_i),
\]

avec entropie \(H_i\), incertitude \(u_i\), risque stéganalytique \(r_i\),
pression de charge \(p_i\), risque de cul-de-sac \(d_i\) et fragilité
\(f_i\). Il produit deux sorties :

\[
(\rho_i,\omega_i)=F_\psi(x_i),
\]

où \(\rho_i\) est le débit maximal et \(\omega_i\) la probabilité ou décision
d'abstention. Après quantification déterministe :

\[
b_i\in\{0,1,\ldots,b_{\max}\},\qquad
u_i^{act}\in\{\mathtt{EMBED},\mathtt{COVER},\mathtt{PAUSE},\mathtt{STOP}\}.
\]

- `EMBED` : encoder \(b_i\ge1\) bits;
- `COVER` : émettre une transition échantillonnée sans consommer de bits;
- `PAUSE` : suspendre l'insertion pendant une durée ou un nombre de pas
  déterministe;
- `STOP` : terminer la trace.

`COVER` et `PAUSE` ne sont pas synonymes : le premier produit une action
publique, le second reporte la production. La politique doit limiter les
longues séquences d'abstention afin d'éviter un canal inutilisable.

## 6. Couplage distributionnel et codage

Le message est protégé par chiffrement authentifié puis converti en source
quasi uniforme. À chaque étape `EMBED`, un codeur de type range coding,
distribution matching ou minimum-entropy coupling construit un couplage
\(\Gamma_i\) entre le préfixe binaire et \(C_\psi(\cdot\mid h_i)\).

La propriété recherchée est :

\[
\sum_x\Gamma_i(x,a)=C_\psi(a\mid h_i)
\]

pour chaque action \(a\), exactement lorsque le codeur le permet, sinon avec
une erreur locale mesurée. Le choix d'action est décodable à partir du même
intervalle, du même historique et de la même clé.

## 7. Synchronisation Alice--Bob

La synchronisation repose sur :

1. sérialisation canonique du graphe et des candidats;
2. version et checksum identiques du modèle;
3. arithmétique déterministe ou quantification entière des probabilités;
4. générateur pseudo-aléatoire dérivé de la clé, de l'identifiant de trace et
   de l'indice d'étape;
5. mêmes règles floues, arrondis et priorités;
6. en-tête authentifié contenant version, longueur protégée et paramètres du
   code correcteur;
7. échec explicite plutôt qu'une correction silencieuse d'un état divergent.

Les modes `COVER`, `PAUSE` et `STOP` sont reproductibles à partir de l'état
public et de la clé. Aucun bit latéral non déclaré ne doit être nécessaire.

## 8. Proposition de décomposition de la distorsion

La divergence KL n'obéit pas à une inégalité triangulaire. La borne suivante
requiert donc des hypothèses explicites.

**Proposition.** Supposons que \(S\ll C_\psi\ll Q_\theta\ll P^\star\) et que,
sur le support des traces stego,

\[
\log\frac{Q_\theta(\tau)}{P^\star(\tau)}
\le \varepsilon_{\mathrm{mod}},
\qquad
\log\frac{C_\psi(\tau)}{Q_\theta(\tau)}
\le \varepsilon_{\mathrm{ctrl}}.
\]

Si

\[
D_{\mathrm{KL}}(S\Vert C_\psi)
\le \varepsilon_{\mathrm{code}},
\]

alors

\[
D_{\mathrm{KL}}(S\Vert P^\star)
\le
\varepsilon_{\mathrm{mod}}
+\varepsilon_{\mathrm{ctrl}}
+\varepsilon_{\mathrm{code}}.
\]

**Démonstration.**

\[
\begin{aligned}
D_{\mathrm{KL}}(S\Vert P^\star)
&=\mathbb E_S\left[
\log\frac{S}{C_\psi}
+\log\frac{C_\psi}{Q_\theta}
+\log\frac{Q_\theta}{P^\star}
\right]\\
&=D_{\mathrm{KL}}(S\Vert C_\psi)
+\mathbb E_S\log\frac{C_\psi}{Q_\theta}
+\mathbb E_S\log\frac{Q_\theta}{P^\star}\\
&\le
\varepsilon_{\mathrm{code}}
+\varepsilon_{\mathrm{ctrl}}
+\varepsilon_{\mathrm{mod}}.
\end{aligned}
\]

La proposition est correcte mais volontairement limitée : les deux bornes de
rapport de vraisemblance sont fortes et invérifiables exactement lorsque
\(P^\star\) est inconnue. Dans l'article, elles servent à structurer l'analyse,
pas à annoncer une sécurité absolue. Les expériences estiment séparément les
trois contributions.

## 9. Décomposition séquentielle vérifiable

Pour les distributions autorégressives \(S\) et \(C_\psi\), la règle de chaîne
donne exactement :

\[
D_{\mathrm{KL}}(S\Vert C_\psi)
=\sum_i
\mathbb E_{h_i\sim S}
D_{\mathrm{KL}}\!\left(
S(\cdot\mid h_i)\Vert C_\psi(\cdot\mid h_i)
\right).
\]

Cette identité rend \(\varepsilon_{\mathrm{code}}\) mesurable étape par étape.
Le symbole `STOP` assure que les distributions portent sur un même espace de
traces de longueur variable.

## 10. Algorithme conceptuel

```text
Entrées : graphe G, message m, clé k, modèles gelés
Sortie  : trace stego tau

1. x <- AEAD_Encrypt_And_Whiten(m, k)
2. initialiser état causal, codeur et compteur de pause
3. pour i = 1 ... L_max:
4.     construire canoniquement les actions admissibles
5.     calculer Q_theta, calibration, risques et coûts
6.     calculer (b_i, mode_i) avec le contrôleur déterministe
7.     si mode_i = STOP: émettre STOP et terminer
8.     si mode_i = PAUSE: mettre à jour l'état public sans consommer de bit
9.     si mode_i = COVER: échantillonner sans consommer de bit
10.    si mode_i = EMBED: coupler les bits suivants à C_psi et choisir l'action
11.    mettre à jour la trace, le modèle causal et le codeur
12. vérifier l'authentification et la longueur au décodage
```

## 11. Condition de contribution

La méthode complète n'est retenue que si :

- le couplage réduit la distorsion face au codage fixe;
- l'abstention améliore le front de Pareto;
- le flou bat un seuil fixe et une MLP;
- le risque local améliore la résistance à des détecteurs externes;
- la correction reste exacte dans le canal passif.

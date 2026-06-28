# Dossier de verrouillage scientifique — phase 1/10

Date de gel provisoire : 12 juin 2026.

## Décision

Le projet étudie un canal de stéganographie coverless/sélective dans lequel le
message détermine les choix successifs d'une trace valide sur un graphe
temporel. Le graphe source n'est pas modifié. La distribution de continuation
est apprise causalement; un codeur entropique réalise l'encodage; un
contrôleur flou ajuste le débit ou impose l'abstention.

## Contribution centrale visée

La contribution n'est ni l'usage isolé d'un RNN, ni celui de la logique floue,
ni l'encodage générique par chemin. Elle réside dans la formulation et
l'évaluation d'un canal de parcours temporels réunissant :

1. une distribution conditionnelle apprise sur données réelles;
2. un encodage aussi fidèle que possible à cette distribution;
3. un contrôle interprétable du compromis capacité-risque-fiabilité;
4. une synchronisation déterministe;
5. une stéganalyse externe et un transfert inter-domaine.

## Verdict sur l'originalité

**Avis tranché : prometteur mais non encore publiable comme revendication de
primauté.** Aucun travail identifié ne réunit ces cinq éléments dans des
graphes temporels. En revanche, chaque brique possède des antériorités fortes.
L'article devra démontrer une interaction scientifique entre les briques, pas
une simple juxtaposition.

Le risque principal n'est pas algorithmique. Il est conceptuel : apprendre
une distribution sur des interactions ne suffit pas à prouver qu'une trace
est un cover naturel. Les conclusions principales seront donc fondées sur des
trajectoires ou sessions réellement observées.

## Documents normatifs

- `threat_model.md` : canal, adversaires et garanties;
- `research_questions.md` : RQ et hypothèses falsifiables;
- `novelty_matrix.md` : antériorités et revendications interdites;
- `review_protocol.md` : clôture de la recherche bibliographique;
- `success_and_pivot_criteria.md` : portes et décisions de pivot.
- `formal_method.md` : objectif, couplage, abstention, coût local,
  synchronisation et borne de distorsion.

## Statut de la phase

Le problème, le canal, les hypothèses et les règles d'abandon sont verrouillés.
Le gel est **provisoire** jusqu'à l'export des bases bibliographiques
institutionnelles. L'implémentation peut commencer, mais le mot « première »
reste interdit dans l'article.

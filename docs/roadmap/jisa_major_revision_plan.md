# Plan de revision majeure JISA

Date : 28 juin 2026.

## Sources editoriales officielles

- Guide for Authors JISA :
  `https://www.sciencedirect.com/journal/journal-of-information-security-and-applications/publish/guide-for-authors`
- Aim and scope verifie : JISA couvre la recherche originale et les
  applications orientees pratique en securite de l'information, avec lien
  recherche-industrie, problemes modernes, solutions scientifiques ou
  best-practice, et details techniques.
- Highlights : 3 a 5 puces, chacune de 85 caracteres maximum espaces inclus.
- Declaration d'usage de l'IA generative : section dediee a la fin du
  manuscrit avant les references, lorsque de tels outils ont ete utilises.

## Regles redactionnelles maintenues

- Article presente comme une version 1.0.
- Formulations positives par defaut.
- Negations reservees aux besoins techniques et aux obligations de conformite.
- Differences avec les travaux existants exprimees par objets, contraintes,
  hypotheses, protocoles et resultats mesures.

## Qualification des remarques du reviewer

| Point | Decision | Justification |
|---|---|---|
| Highlights incoherents | Recevable | Erreur presente dans `article/main.tex`. |
| Courbe capacite-detectabilite | Recevable | Renforce le lien pratique attendu par JISA. |
| Cas d'usage faible debit | Recevable | Repond au volet application pratique. |
| Oracle-leakage | Recevable | Complete le modele d'adversaire. |
| Modele de couverture | Recevable avec cadrage | Analyse de sensibilite et impact empirique. |
| Declaration IA trop visible | Partiel | Elsevier demande une section dediee avant references ; raccourcir et standardiser. |
| Limitations | Recevable avec cadrage positif | Presenter les frontieres validees du regime. |

## Actions

1. Corriger les highlights selon le format Elsevier.
2. Standardiser la declaration d'usage des outils IA.
3. Ajouter un sweep capacite-detectabilite multi-seuils.
4. Ajouter une figure et une table de compromis.
5. Ajouter une quantification de cas d'usage courts.
6. Ajouter un audit oracle-leakage : correlations, importance, detecteurs
   internes plus flexibles et bornes empiriques.
7. Ajouter une synthese de sensibilite couverture-calibration-detectabilite.
8. Integrer les resultats dans le manuscrit avec style version 1.0.
9. Recompiler, tester, verifier references et commit.

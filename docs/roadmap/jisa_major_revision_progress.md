# Progression revision majeure JISA

Date de depart : 28 juin 2026.

## Commande de reprise

```powershell
cd "C:\Users\User\Documents\Articles\Saha_Paper"; codex "Continue la revision majeure JISA depuis docs/roadmap/jisa_major_revision_progress.md"
```

## Regles actives

- Plan : `docs/roadmap/jisa_major_revision_plan.md`.
- Regles de style : `docs/roadmap/jisa_writing_rules.md`.
- Ecriture version 1.0, formulations positives, negations uniquement si elles
  servent une precision technique ou une exigence editoriale.
- Sources JISA officielles verifiees avant acceptation des critiques.

## Etat initial

- Commit de depart : `1e86efe refactor: refondre le manuscrit pour JISA`.
- Depot propre au lancement de cette phase.
- Erreur highlights confirmee dans `article/main.tex`.
- Declaration IA presente comme section dediee avant references, conforme au
  placement Elsevier ; formulation a raccourcir.

## Etapes

| Etape | Statut | Notes |
|---|---|---|
| 0. Sources JISA et plan | Termine | Voir `jisa_major_revision_plan.md`. |
| 1. Mesures de suivi | Termine | Journal de progression cree. |
| 2. Conformite editoriale | Termine | Highlights conformes et declaration IA standardisee. |
| 3. Sweep capacite-detectabilite | Termine | Table, JSON et figure Phase 12 produits. |
| 4. Oracle-leakage | Termine | Correlations, detecteurs et epsilons produits. |
| 5. Sensibilite couverture | Termine | Synthese couverture-capacite-AUC produite. |
| 6. Integration manuscrit | Termine | Sections experiences, resultats, portee, conclusion et declarations. |
| 7. Verification et commit | Termine | Checkpoint Git cree. |

## Journal

### 2026-06-28

- Demarrage de la revision majeure demandee par le reviewer.
- Verification officielle JISA effectuee : aim and scope, highlights,
  declaration d'usage des outils IA, data/code availability.
- Decision : accepter les demandes de courbe de compromis, cas d'usage,
  oracle-leakage, sensibilite du modele de couverture et correction des
  highlights.
- Decision : garder une declaration IA dediee avant references, avec une
  formulation plus concise et standard.
- Scripts Phase 12 ajoutes :
  `run_phase12_capacity_detectability_sweep.py`,
  `run_phase12_oracle_leakage_audit.py`,
  `run_phase12_cover_sensitivity.py`,
  `run_phase12_practical_payloads.py`.
- Sweep capacite-detectabilite execute. Sorties :
  `results/tables/phase12_capacity_detectability_sweep.csv`,
  `results/tables/phase12_capacity_detectability_summary.csv`,
  `results/tables/phase12_capacity_detectability_sweep.json`,
  `results/figures/phase12_capacity_detectability_sweep.pdf`,
  `results/figures/phase12_capacity_detectability_sweep.png`.
- Meilleurs points sous AUC publique 0.60 :
  TGBL-Wiki 0.1338 bit/transition a 0.59295 AUC, T-Drive 0.1106
  a 0.56561, MOOC 0.0254 a 0.52380, LastFM 0.0536 a 0.52476,
  GeoLife 0.0007 a 0.50007.
- Cas d'usage courts produits dans
  `results/tables/phase12_practical_payloads.csv` :
  32 bits demandent 240 transitions sur TGBL-Wiki, 290 sur T-Drive,
  598 sur LastFM, 1260 sur MOOC, et 45715 sur GeoLife.
- Audit oracle-leakage execute. Sorties :
  `results/tables/phase12_oracle_leakage_audit.csv`,
  `results/tables/phase12_oracle_leakage_correlations.csv`,
  `results/tables/phase12_oracle_leakage_audit.json`.
- Epsilons oracle empiriques :
  0.00764 sur TGBL-Wiki, 0.01155 sur T-Drive, 0.0 sur MOOC,
  LastFM et GeoLife.
- Sensibilite couverture executee. Sorties :
  `results/tables/phase12_cover_sensitivity.csv` et
  `results/tables/phase12_cover_sensitivity.json`.
- Notes de synthese ajoutees dans
  `docs/roadmap/jisa_phase12_revision_notes.md`.
- Highlights corriges dans `article/highlights.tex`, `article/main.tex` et le
  template CAS. Chaque puce compte moins de 85 caracteres.
- Declaration d'usage des outils IA standardisee selon le placement Elsevier :
  section dediee avant les references.
- Manuscrit integre :
  courbe capacite-detectabilite, meilleurs points sous AUC 0.60, cas d'usage
  courts, audit oracle-leakage, sensibilite du modele de couverture et portee
  validee.
- Mise en page verifiee par rendu PNG de la page de la figure Phase 12.
  La figure a ete simplifiee pour enlever les etiquettes superposees, puis une
  coupure de page a ete ajoutee apres les tables de compromis.
- Compilation PDF reussie via `pdflatex`, `bibtex`, puis passes `pdflatex`.
  PDF produit : `article/main.pdf`, 12 pages.
- Verification finale :
  `pytest code\tests` = 44 passed.
- Verification structurelle :
  `python code\scripts\validate_project.py` = structure valide, 24 PDF checked.
- Verification references :
  `python check_labels_refs.py` = article propre ; deux labels inutilises
  restent signales dans la these (`fig:controller-modes`,
  `fig:temporal-walk`).
- Log LaTeX controle : aucune reference indefinie, aucune demande de rerun,
  aucune erreur fatale.
- Checkpoint Git cree :
  revision majeure JISA, Phase 12, integration manuscrit et PDF compile.

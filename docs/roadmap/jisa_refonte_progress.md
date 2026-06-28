# Progression de la refonte JISA

Date de depart : 28 juin 2026.

## Commande de reprise

```powershell
cd "C:\Users\User\Documents\Articles\Saha_Paper"; codex "Continue l'execution du plan JISA depuis docs/roadmap/jisa_refonte_progress.md"
```

## Regles de travail

- Le plan de reference est `docs/roadmap/jisa_refonte_plan.md`.
- Ce fichier est mis a jour apres chaque etape significative.
- Les changements sont regroupes en checkpoints Git lisibles.
- Les revendications fortes sont interdites sauf preuve ou mesure directe :
  `premier`, `indetectable`, `provable`, `distribution-preserving`.
- Le manuscrit doit etre repositionne comme papier de securite adversariale sur
  canal comportemental temporel.
- Les regles de redaction sont dans `docs/roadmap/jisa_writing_rules.md` :
  rediger comme une version 1.0, privilegier les formulations positives, et
  expliquer les differences par les objets, contraintes et protocoles.

## Etat initial constate

- Commit de verrouillage existant : `edb4b4c`, tag `lock-2026-06-28`.
- Tests avant refonte : `pytest code\tests` = 41 passed.
- Validation structurelle avant refonte : `python code\scripts\validate_project.py`
  = 24 PDF checked.
- Le plan de refonte JISA existe mais n'est pas encore committe.

## Etapes

| Etape | Statut | Notes |
|---|---|---|
| 0. Journal de progression | Termine | Checkpoint `595b22d`. |
| 1. Audit critique du manuscrit | Termine | Voir `docs/roadmap/jisa_critique_audit.md`. |
| 2. Nouvelle contribution et modele de menace | Termine | Voir `docs/roadmap/jisa_core_positioning.md`. |
| 3. Plan experimental renforce | Termine | Voir `docs/roadmap/jisa_experimental_plan.md`. |
| 4. Implementation experimentale | Termine | E1, E2, E3 et E4 executes ; notes d'audit disponibles dans `docs/roadmap`. |
| 5. Refonte manuscrit | Termine | Manuscrit article recadre comme version 1.0 JISA. |
| 6. Verification finale | Termine | Tests, validation projet, references article et compilation PDF executes. |

## Journal

### 2026-06-28

- Demarrage de l'execution du plan JISA.
- Creation du journal de progression pour reprise inter-session.
- Checkpoint Git `595b22d` : ajout du plan de refonte et du journal.
- Audit critique du manuscrit realise dans
  `docs/roadmap/jisa_critique_audit.md`.
- Decisions issues de l'audit :
  1. ne pas promettre de `provable indistinguishability` ;
  2. ne pas faire de `distribution-preserving` la revendication centrale ;
  3. deplacer ou recadrer AdaBIND/BIND/BYNIS comme contraste de paradigmes ;
  4. prioriser l'audit de capacite et le range coding contraint.
- Checkpoint Git `075abdb` : audit des objections JISA.
- Cadrage central etabli dans `docs/roadmap/jisa_core_positioning.md` :
  canal comportemental temporel, abstention sous contrainte de detection,
  adversaires bornes, et audit de capacite.
- Checkpoint Git `d170828` : cadrage central JISA.
- Plan experimental etabli dans `docs/roadmap/jisa_experimental_plan.md`.
- Priorite d'implementation : E1 audit de capacite, puis E3 audit
  distributionnel, puis E2 range coding contraint selon le diagnostic E1.
- Regles de redaction version 1.0 ajoutees dans
  `docs/roadmap/jisa_writing_rules.md`.
- Checkpoint Git `b2f1937` : regles de redaction JISA.
- E1 audit de capacite implemente dans
  `code/scripts/run_phase11_capacity_audit.py` et execute.
- Resultats produits :
  `results/tables/phase11_capacity_audit.csv` et
  `results/tables/phase11_capacity_audit.json`.
- Le backend local `range` a ete corrige pour ne consommer que des prefixes
  exactement decodables par action. Test cible ajoute dans
  `code/tests/test_phase5_coding.py`.
- Diagnostic E1 : TGBL-Wiki et T-Drive presentent une perte codec visible ;
  MOOC, LastFM et GeoLife sont surtout limites par la rarete des etats
  faisables au point conservateur.
- Verification apres E1 : `pytest code\tests` = 42 passed.
- Phase 7 regeneree avec le codec local exactement decodable.
- E3 audit distributionnel implemente dans
  `code/scripts/run_phase11_distribution_audit.py` et execute.
- Resultats produits :
  `results/tables/phase11_distribution_audit.csv` et
  `results/tables/phase11_distribution_audit.json`.
- Notes E3 ajoutees dans `docs/roadmap/jisa_distribution_audit_notes.md`.
- E2 prototype range coding sequentiel ajoute dans `code/src/stego/coding.py`
  avec tests dans `code/tests/test_phase5_coding.py`.
- Estimation E2 executee via `code/scripts/run_phase11_range_codec.py`.
- Resultats produits :
  `results/tables/phase11_range_codec.csv` et
  `results/tables/phase11_range_codec.json`.
- Notes E2 ajoutees dans `docs/roadmap/jisa_range_codec_notes.md`.
- Verification E2 ciblee : `pytest code\tests\test_phase5_coding.py` =
  9 passed.
- Verification apres E2 : `pytest code\tests` = 44 passed.
- Phase 9 independent neural et adaptive steganalysis regenerees avec le codec
  local exactement decodable.
- E4 consolidation adversariale implemente dans
  `code/scripts/run_phase11_adversarial_audit.py` et execute.
- Resultats produits :
  `results/tables/phase11_adversarial_audit.csv` et
  `results/tables/phase11_adversarial_audit.json`.
- Notes E4 ajoutees dans `docs/roadmap/jisa_adversarial_audit_notes.md`.
- Refonte manuscrit executee sur les sections article :
  introduction, related work, methode, securite, experiences, resultats,
  portee validee et conclusion.
- Ligne editoriale appliquee : article presente comme version 1.0, avec
  formulations positives, negations limitees aux besoins techniques, et
  differences exprimees par objets, contraintes, hypotheses et protocoles.
- Titre article mis a jour :
  `Detection-Constrained Steganography over Temporal Graph Traversals`.
- Revendication centrale stabilisee : canal comportemental temporel sous
  contrainte de detection, abstention operationnelle, audits de capacite,
  proximite distributionnelle et steganalyse adversariale bornee.
- Tableaux de resultats remplaces ou consolides autour des audits Phase 11 :
  capacite, steganalyse externe, proximite distributionnelle et adversaires
  adaptatifs.
- Front matter article mis a jour : abstract, highlights et keywords.
- Compilation `latexmk` tentee ; MiKTeX signale l'absence de Perl pour
  executer `latexmk`.
- Compilation directe reussie via `pdflatex`, `bibtex`, `pdflatex`,
  `pdflatex`; PDF produit : `article/main.pdf`.
- Verification finale :
  `pytest code\tests` = 44 passed.
- Verification structurelle :
  `python code\scripts\validate_project.py` = structure valide, 24 PDF checked.
- Verification references :
  `python check_labels_refs.py` = article propre ; deux labels inutilises
  restent signales dans la these (`fig:controller-modes`,
  `fig:temporal-walk`).

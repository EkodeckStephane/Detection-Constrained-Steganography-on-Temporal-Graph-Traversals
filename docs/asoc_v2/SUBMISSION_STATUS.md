# ASOC submission status

## Scientific object

**Detector-constrained causal steganographic control over temporal graph processes whose emitted actions alter their own future admissible support.**

Target manuscript title: **Detector-Constrained Causal Steganography on Temporal Graphs**.

## Evidence status

The submission manuscript is aligned to `experiments/asoc_v2/submission_snapshot.yaml` and `results_submission_snapshot/summary.json`.

- The reserved final holdout remains unopened.
- The confirmatory extension is closed without opening the holdout because exact pre-snapshot policy documents are unavailable for all required actor-action datasets; see `experiments/asoc_v2/confirmatory_status.yaml`.
- T-Drive is the primary full-validation mobility result.
- GeoLife remains a boundary condition and is reported only as screening evidence.
- Actor-action results are policy-validation evidence, not an independent confirmatory test.

## Q1 gate status

The causal/state, synchronization, admissibility, null-channel, integer-arithmetic, split/provenance, and claim-discipline gates are satisfied by the current source and snapshot.

Two submission-readiness items remain open:

1. **Same-protocol comparator evidence.** The snapshot does not contain a frozen comparison of the optimized Takagi-Sugeno policy against the declared fixed-threshold, hand-tuned, detector-unaware, random-admissible, and distribution-preserving reference controllers. No superiority claim should be made until those comparisons are executed on the frozen protocol.
2. **Uniform uncertainty coverage.** T-Drive and LastFM have cluster-bootstrap uncertainty in the snapshot, whereas the TGBL-Wiki and MOOC snapshot records only seed ranges. These regimes should receive the same cluster-aware uncertainty treatment before a strict Gate-10 PASS.

Sequence-aware / neural unseen-Eves belong to the uncompleted confirmatory extension and are not claimed in the submission snapshot.

## Holdout rule

Opening the reserved final holdout using retrospectively reconstructed policies is prohibited. A future independent confirmatory study must begin from a new complete pre-holdout policy bundle and preserve GeoLife's boundary-condition role.

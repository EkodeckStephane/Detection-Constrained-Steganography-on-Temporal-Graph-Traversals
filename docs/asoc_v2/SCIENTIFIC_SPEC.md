# ASOC V2 Scientific Lock

Target journal: Applied Soft Computing (Elsevier)

## Scientific object

The V2 study treats covert communication over temporal graph processes as a **causal constrained sequential decision problem with endogenous support**. At time t, the action emitted by the steganographic policy is part of the public history and therefore changes the admissible continuation set and cover distribution at later steps.

The paper is not organized around a codec, fuzzy controller, repository, or experiment campaign. Those are implementations of the scientific object.

## Core problem

Given a public causal history H_t and admissible action set A(H_t), construct a policy pi that selects an emitted action A_t and local payload rate R_t so as to maximize expected covert throughput while satisfying:

1. causal validity: A_t belongs to A(H_t);
2. sender-receiver state consistency under passive transmission;
3. local and trajectory-level detectability constraints;
4. bounded failure/abstention behavior.

The state evolves with the emitted action:

H_{t+1} = F(H_t, A_t).

This emitted-history recursion is a hard invariant of every experiment.

## Protected contribution

The contribution to protect is not “fuzzy steganography” or “range coding on graphs.” It is:

> detector-constrained causal steganographic control over temporal graph processes whose steganographic actions alter their own future support.

## Contribution structure

### C1 — Endogenous-support formulation

Formalize temporal-graph steganography as a causal channel in which each emitted action changes the future continuation support. Separate the natural process P, learned cover Q_theta, and stego policy S_pi.

### C2 — Soft-computing constrained policy

Treat EMBED/COVER/PAUSE/STOP and local payload as actions of a decision policy. The optimized Takagi-Sugeno policy is an interpretable soft-computing realization and is compared with fixed-threshold, hand-tuned fuzzy, and learned/optimization baselines.

### C3 — Trajectory security relation and sealed evaluation

Use a conditional chain-rule relation from local divergence to trajectory divergence, while keeping empirical security claims bounded to tested detector families. Select policies without consulting the sealed test and evaluate transfer to unseen steganalysers.

## Claims explicitly forbidden until proven

- “first”, “state of the art”, “perfectly secure”, or universal-security claims;
- exact synchronization outside the tested passive synchronized setting;
- an empirical detector as an upper bound on an optimal adversary;
- using mean held-out NLL as a pointwise model-divergence bound;
- any headline number generated before the ASOC V2 causal engine and sealed protocol are frozen.

## Hard scientific invariants

1. The stego history is updated with the **emitted stego action**, not the natural counterfactual action.
2. Natural and stego paths maintain distinct causal histories during paired evaluation.
3. Cover-model fitting, design-Eve fitting, policy selection, and sealed testing use disjoint causal regions.
4. The sealed test is never used to choose a policy, payload point, hyperparameter, or detector.
5. All paper tables are generated from frozen result files; no headline value is manually transcribed.
6. Message seeds are technical stochastic repetitions, not independent experimental units.

## Current target title

**Detector-Constrained Causal Steganography on Temporal Graphs: Soft-Computing Control under Endogenous Support**

The title remains provisional until the final contribution and ASOC scope audit are complete.

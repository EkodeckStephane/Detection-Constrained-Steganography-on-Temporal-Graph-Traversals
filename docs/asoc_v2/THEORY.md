# ASOC V2 — Theory Lock

This note defines the formal claim that may be used by the manuscript. It is intentionally conditional and does not convert empirical model fit into a universal security guarantee.

## 1. Causal path distributions

Let a length-T emitted action path be

`A_1:T = (A_1, ..., A_T)`

with causal history `H_t = A_1:t-1` plus declared public side information. Let

- `P_t(. | H_t)` be the unknown real cover-process conditional distribution;
- `Q_t(. | H_t)` be the learned cover-model distribution, restricted to the domain-valid admissible set;
- `S_t(. | H_t)` be the steganographic policy-induced distribution.

The support condition is required at every reachable history:

`support(S_t(. | H_t)) subseteq support(P_t(. | H_t))`.

For model-relative statements, replace P by Q and require support(S) subseteq support(Q).

## 2. Chain rule for trajectory divergence

For any two causal path distributions S and R defined on the same path space,

`D_KL(S_1:T || R_1:T) = sum_t E_{H_t ~ S}[ D_KL(S_t(.|H_t) || R_t(.|H_t)) ]`.

Therefore, if for every reachable history the local model-relative divergence in **bits** satisfies

`D_KL,2(S_t(.|H_t) || Q_t(.|H_t)) <= delta_t(H_t)`,

then

`D_KL,2(S_1:T || Q_1:T) <= sum_t E_S[delta_t(H_t)]`.

For a uniform local budget `delta`, this becomes at most `T * delta` bits. This is a trajectory accumulation relation, not a claim that the real process P is known.

## 3. Consequence for total variation

Pinsker's inequality is conventionally stated with KL in nats:

`TV(S,R) <= sqrt(D_KL,e(S||R)/2)`.

Since `D_KL,e = ln(2) * D_KL,2`, a bits-based trajectory budget gives

`TV(S_1:T, Q_1:T) <= sqrt( ln(2)/2 * sum_t E_S[delta_t(H_t)] )`.

This upper-bounds the statistical separation of S from Q. It is model-relative unless an additional relation between Q and the real process P is justified.

## 4. Conditional relation to the real process

The exact log-ratio decomposition is

`log(S/P) = log(S/Q) + log(Q/P)`

along the same emitted path. Taking expectation under S yields

`D_KL(S||P) = D_KL(S||Q) + E_S[ log(Q/P) ]`.

This is **not** a triangle inequality for KL and the second term is not `D_KL(Q||P)`.

A real-process upper bound is permitted only under an explicit assumption such as a reachable-history pointwise log-ratio bound

`log_2( Q_t(a|H_t) / P_t(a|H_t) ) <= epsilon_model,t(H_t,a)`.

Under a uniform upper bound `epsilon_model,t`, the path divergence in bits satisfies

`D_KL,2(S_1:T||P_1:T) <= D_KL,2(S_1:T||Q_1:T) + sum_t epsilon_model,t`.

The manuscript must state that such a pointwise model-misspecification bound is an assumption unless it is independently established.

## 5. What empirical NLL can and cannot establish

Held-out NLL/perplexity/calibration measure predictive adequacy of Q on observed cover data. They do **not** directly provide a pointwise upper bound on `log(Q/P)` because P is unknown. Therefore:

- mean held-out NLL may be reported as model adequacy evidence;
- calibration and cold-start diagnostics may be reported as boundary conditions;
- mean NLL must not be substituted for `epsilon_model` in the theorem above.

## 6. Empirical detector interpretation

For a tested family of steganalysers `D_test`, the empirical study estimates quantities such as

`max_{D in D_test} AUC(D)`.

This is evidence against the tested detector family, not an upper bound on the optimal adversary. The paper may use terms such as:

- design-Eve detector budget;
- unseen-Eve transfer evaluation;
- white-box diagnostic;
- oracle-feature diagnostic.

It must not call these empirical attacks an optimal-detector bound.

## 7. Claim permitted in the paper

After the causal generator and experiments pass the hard gates, the formal statement may be summarized as:

> Local divergence constraints compose along the endogenous causal trajectory through the KL chain rule, yielding a model-relative trajectory-divergence bound. Empirical experiments then test whether policies selected under a design-detector budget remain difficult to distinguish for unseen detector families on sealed temporal data.

The two clauses are deliberately separate: the first is formal and model-relative; the second is empirical and detector-family-relative.

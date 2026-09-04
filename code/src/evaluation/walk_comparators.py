from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from controllers.fuzzy import ControlDecision, ControllerInputs, FuzzyRateController, fixed_entropy_threshold
from stego.coding import Candidate
from stego.policy_session import rate_limited_candidates

ComparatorName = Literal[
    "fixed_entropy_threshold",
    "hand_tuned_fuzzy",
    "detector_unaware_causal_arithmetic",
    "random_admissible",
    "coupling_inspired_distribution_preserving",
]


@dataclass(frozen=True)
class ComparatorStep:
    """Public comparator decision and the exact arithmetic support to use.

    ``coding_candidates`` is empty whenever the comparator abstains.  All
    transformations are deterministic functions of the public candidate list
    and controller inputs; no payload bit or secret-dependent action enters the
    comparator decision.
    """

    decision: ControlDecision
    coding_candidates: tuple[Candidate, ...]


_HAND_TUNED = FuzzyRateController(max_bits_per_transition=4, stop_threshold=0.92)


def comparator_step(
    name: ComparatorName,
    *,
    inputs: ControllerInputs,
    candidates: list[Candidate] | tuple[Candidate, ...],
    fixed_threshold: float = 0.55,
) -> ComparatorStep:
    """Return one same-protocol T-Drive comparator action contract.

    The caller remains responsible for COVER/PAUSE emission under walk
    semantics and for applying the shared 128-bit causal arithmetic codec when
    ``decision.mode == 'EMBED'``.
    """

    public_candidates = tuple(_renormalize(candidates))
    if len(public_candidates) < 2:
        return ComparatorStep(
            decision=ControlDecision("COVER", 0, 0.0, 1.0),
            coding_candidates=(),
        )

    if name == "fixed_entropy_threshold":
        decision = fixed_entropy_threshold(
            inputs,
            entropy_threshold=float(fixed_threshold),
            max_bits_per_transition=4,
        )
        return _decision_with_rate_limited_support(decision, public_candidates)

    if name == "hand_tuned_fuzzy":
        decision = _HAND_TUNED.decide(inputs)
        return _decision_with_rate_limited_support(decision, public_candidates)

    if name == "detector_unaware_causal_arithmetic":
        support = tuple(rate_limited_candidates(public_candidates, nominal_bits=4))
        return ComparatorStep(
            decision=ControlDecision("EMBED", 4, 1.0, 0.0),
            coding_candidates=support,
        )

    if name == "random_admissible":
        selected = tuple(public_candidates[: min(16, len(public_candidates))])
        probability = 1.0 / len(selected)
        support = tuple(Candidate(item.action, probability) for item in selected)
        return ComparatorStep(
            decision=ControlDecision("EMBED", 4, 1.0, 0.0),
            coding_candidates=support,
        )

    if name == "coupling_inspired_distribution_preserving":
        # This is deliberately named a coupling-inspired *reference*: it uses
        # the entire public Q|A+ support without the primary controller's
        # rate truncation. It is not claimed to implement a particular
        # published minimum-entropy-coupling algorithm.
        bits = max(1, int(math.ceil(math.log2(len(public_candidates)))))
        return ComparatorStep(
            decision=ControlDecision("EMBED", bits, 1.0, 0.0),
            coding_candidates=public_candidates,
        )

    raise ValueError(f"Unknown comparator: {name}")


def _decision_with_rate_limited_support(
    decision: ControlDecision,
    candidates: tuple[Candidate, ...],
) -> ComparatorStep:
    if decision.mode != "EMBED" or decision.local_payload_bits < 1:
        return ComparatorStep(decision=decision, coding_candidates=())
    support = tuple(
        rate_limited_candidates(
            candidates,
            nominal_bits=decision.local_payload_bits,
        )
    )
    if len(support) < 2:
        return ComparatorStep(
            decision=ControlDecision(
                "COVER",
                0,
                decision.rate_score,
                max(1.0, decision.abstention_score),
            ),
            coding_candidates=(),
        )
    return ComparatorStep(decision=decision, coding_candidates=support)


def _renormalize(candidates: list[Candidate] | tuple[Candidate, ...]) -> list[Candidate]:
    if not candidates:
        return []
    mass = sum(max(0.0, float(item.probability)) for item in candidates)
    if mass <= 0.0:
        raise ValueError("candidate probabilities must contain positive mass")
    return [
        Candidate(item.action, max(0.0, float(item.probability)) / mass)
        for item in candidates
        if float(item.probability) > 0.0
    ]

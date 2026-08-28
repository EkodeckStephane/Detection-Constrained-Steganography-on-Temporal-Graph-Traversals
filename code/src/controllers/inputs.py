from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence

from controllers.fuzzy import ControllerInputs
from stego.coding import Candidate


def build_public_controller_inputs(
    *,
    candidates: Sequence[Candidate],
    top_k: int,
    context_observations: int,
    context_seen: bool,
    steganalysis_risk: float,
    committed_payload_bits: int,
    payload_length: int,
    future_admissible_count: Callable[[Hashable], int] | None = None,
) -> ControllerInputs:
    """Build secret-independent Takagi--Sugeno inputs from public evidence.

    The mapping is fixed before policy optimization:

    - predictive entropy is Shannon entropy normalized by log2(top_k);
    - calibration uncertainty is 1 for unseen contexts and otherwise
      1/sqrt(1+n), where n is the cover-training context count;
    - steganalysis risk is the frozen public design-Eve risk supplied by the
      caller;
    - payload pressure is the uncommitted fraction of the public message;
    - dead-end risk is the candidate probability mass leading to states with
      no admissible continuation (zero for domains without such a callback);
    - channel fragility is one minus normalized log-support size.

    No quantity depends on payload bit values or on the secret-dependent action
    selected by the arithmetic coder.
    """

    if not candidates:
        raise ValueError("at least one candidate is required")
    if top_k < 2:
        raise ValueError("top_k must be at least two")
    if context_observations < 0:
        raise ValueError("context_observations must be non-negative")
    if payload_length < 1:
        raise ValueError("payload_length must be positive")
    if not 0 <= committed_payload_bits <= payload_length:
        raise ValueError("committed_payload_bits must lie within the public payload")

    probabilities = [max(0.0, float(item.probability)) for item in candidates]
    mass = sum(probabilities)
    if mass <= 0:
        raise ValueError("candidate probabilities must contain positive mass")
    probabilities = [value / mass for value in probabilities]

    entropy = -sum(value * math.log2(value) for value in probabilities if value > 0)
    entropy_scale = max(1.0, math.log2(top_k))
    predictive_entropy = _clip(entropy / entropy_scale)

    calibration_uncertainty = 1.0 if not context_seen else _clip(
        1.0 / math.sqrt(1.0 + context_observations)
    )
    payload_pressure = _clip(
        (payload_length - committed_payload_bits) / payload_length
    )

    if future_admissible_count is None:
        dead_end_risk = 0.0
    else:
        dead_end_risk = _clip(
            sum(
                probability
                for candidate, probability in zip(candidates, probabilities, strict=True)
                if future_admissible_count(candidate.action) == 0
            )
        )

    support_size = min(len(candidates), top_k)
    channel_fragility = _clip(
        1.0 - math.log2(max(1, support_size)) / math.log2(top_k)
    )

    return ControllerInputs(
        predictive_entropy=predictive_entropy,
        calibration_uncertainty=calibration_uncertainty,
        steganalysis_risk=_clip(steganalysis_risk),
        payload_pressure=payload_pressure,
        dead_end_risk=dead_end_risk,
        channel_fragility=channel_fragility,
    )


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence

from controllers.fuzzy import ControllerInputs
from stego.coding import Candidate


def effective_backoff_observation_count(
    model: object,
    source: Hashable,
    previous_destination: Hashable | None,
) -> int:
    """Return training evidence at the finest backoff level actually available.

    Exact actor-history evidence is preferred.  If that context is unseen but
    the source itself was observed, the temporal model backs off to its
    source-level distribution, so source-level observations are the relevant
    calibration evidence.  A completely unseen source returns zero; global
    sample size is deliberately not used to erase cold-start uncertainty.
    """

    context_count = int(model.context_observation_count(source, previous_destination))
    if context_count > 0:
        return context_count
    return int(model.source_observation_count(source))


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

    ``context_observations`` is the effective cover-training evidence at the
    finest backoff level actually used by the cover model.  Callers should use
    :func:`effective_backoff_observation_count` for the primary temporal model.

    The fixed mappings are:

    - predictive entropy: Shannon entropy normalized by log2(top_k);
    - calibration uncertainty: 1/sqrt(1+n_eff), with n_eff=0 giving 1;
    - steganalysis risk: the frozen public design-Eve risk supplied by caller;
    - payload pressure: the uncommitted fraction of the public message;
    - dead-end risk: candidate probability mass leading to states with no
      admissible continuation;
    - channel fragility: inverse admissible support size, so a singleton is 1,
      two alternatives are 0.5, and broad support rapidly approaches zero.

    ``context_seen`` remains explicit for provenance/API compatibility but does
    not override source-level backoff evidence.  No quantity depends on payload
    bit values or on the secret-dependent arithmetic action.
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
    del context_seen

    probabilities = [max(0.0, float(item.probability)) for item in candidates]
    mass = sum(probabilities)
    if mass <= 0:
        raise ValueError("candidate probabilities must contain positive mass")
    probabilities = [value / mass for value in probabilities]

    entropy = -sum(value * math.log2(value) for value in probabilities if value > 0)
    entropy_scale = max(1.0, math.log2(top_k))
    predictive_entropy = _clip(entropy / entropy_scale)

    calibration_uncertainty = _clip(1.0 / math.sqrt(1.0 + context_observations))
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

    support_size = max(1, min(len(candidates), top_k))
    channel_fragility = _clip(1.0 / support_size)

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

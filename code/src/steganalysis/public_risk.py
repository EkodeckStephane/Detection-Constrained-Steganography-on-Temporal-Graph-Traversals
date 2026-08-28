from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from steganalysis.detectors import OrientedDetector, worst_case_design_risk
from steganalysis.samples import FEATURE_COLUMNS
from stego.coding import Candidate


@dataclass(frozen=True)
class PublicRiskEnvelope:
    """Secret-independent detectability advantage attached to one public state."""

    worst_risk: float
    action_risks: tuple[tuple[Hashable, float], ...]
    candidate_count: int


@dataclass(frozen=True)
class PublicReferenceRisk:
    """Detectability advantage of a deterministic public reference action.

    The primary ASOC V2 controller uses the top-probability cover action as a
    low-cost, secret-independent state-risk proxy. The underlying oriented Eve
    posterior is converted to advantage above chance: 0.5 -> 0 risk, 1 -> 1
    risk. Security is not inferred from this proxy: every frozen policy is
    certified using adversarial AUC on the actions it actually emits.
    """

    action: Hashable
    risk: float
    candidate_count: int


def steganalysis_advantage(score: float | np.ndarray) -> float | np.ndarray:
    """Map an oriented stego posterior to normalized advantage above chance."""

    values = np.asarray(score, dtype=float)
    transformed = np.clip(2.0 * (values - 0.5), 0.0, 1.0)
    if np.ndim(score) == 0:
        return float(transformed)
    return transformed


def candidate_public_feature_matrix(
    *,
    source: Hashable,
    previous_action: Hashable | None,
    candidates: Sequence[Candidate],
    gap: float,
    context_seen: bool,
    training_destinations: frozenset[Hashable] | set[Hashable],
) -> tuple[np.ndarray, tuple[Hashable, ...]]:
    """Build Eve-visible features for every admissible candidate action."""

    normalized = _normalized_candidates(candidates)
    rows = [
        observed_action_feature_vector(
            source=source,
            action=candidate.action,
            previous_action=previous_action,
            candidates=normalized,
            gap=gap,
            context_seen=context_seen,
            training_destinations=training_destinations,
        )
        for candidate in normalized
    ]
    return np.vstack(rows), tuple(candidate.action for candidate in normalized)


def observed_action_feature_vector(
    *,
    source: Hashable,
    action: Hashable,
    previous_action: Hashable | None,
    candidates: Sequence[Candidate],
    gap: float,
    context_seen: bool,
    training_destinations: frozenset[Hashable] | set[Hashable],
) -> np.ndarray:
    """Return the canonical 12 public Eve features for an observed action.

    The action may be outside the learned top-k candidate list. This is needed
    for actor-action COVER/PAUSE, where the observed natural destination is
    passed through even when Q assigns it only backoff mass outside top-k. The
    fallback probability and rank convention exactly matches the paired-sample
    feature contract: half the smallest retained probability and rank k+1.
    """

    normalized = _normalized_candidates(candidates)
    probabilities = {candidate.action: float(candidate.probability) for candidate in normalized}
    ranked_actions = [candidate.action for candidate in normalized]
    count = len(normalized)
    minimum_probability = min(probabilities.values())
    probability = probabilities.get(action, minimum_probability * 0.5)
    rank = ranked_actions.index(action) + 1 if action in ranked_actions else count + 1
    values = np.asarray(list(probabilities.values()), dtype=float)
    entropy = float(-(values[values > 0] * np.log2(values[values > 0])).sum())

    feature = {
        "action_probability": float(probability),
        "surprise_bits": -math.log2(max(float(probability), np.finfo(float).tiny)),
        "rank_fraction": rank / count,
        "is_top_action": float(rank == 1),
        "entropy_bits": entropy,
        "top_probability": float(values.max()),
        "candidate_count": float(count),
        "unseen_context": float(not context_seen),
        "unseen_destination": float(action not in training_destinations),
        "same_as_previous": float(previous_action == action),
        "self_loop": float(source == action),
        "log_inter_event_gap": math.log1p(max(0.0, float(gap))),
    }
    return np.asarray([float(feature[name]) for name in FEATURE_COLUMNS], dtype=float)


def public_reference_risk(
    detectors: Mapping[str, OrientedDetector],
    *,
    source: Hashable,
    previous_action: Hashable | None,
    candidates: Sequence[Candidate],
    gap: float,
    context_seen: bool,
    training_destinations: frozenset[Hashable] | set[Hashable],
) -> PublicReferenceRisk:
    """Score the deterministic top-Q cover action before secret action selection."""

    matrix, actions = candidate_public_feature_matrix(
        source=source,
        previous_action=previous_action,
        candidates=candidates,
        gap=gap,
        context_seen=context_seen,
        training_destinations=training_destinations,
    )
    scores = worst_case_design_risk(detectors, matrix[:1])
    risks = steganalysis_advantage(scores)
    return PublicReferenceRisk(
        action=actions[0],
        risk=float(risks[0]),
        candidate_count=len(actions),
    )


def public_state_risk_envelope(
    detectors: Mapping[str, OrientedDetector],
    *,
    source: Hashable,
    previous_action: Hashable | None,
    candidates: Sequence[Candidate],
    gap: float,
    context_seen: bool,
    training_destinations: frozenset[Hashable] | set[Hashable],
) -> PublicRiskEnvelope:
    """Return conservative maximum detectability advantage for a public state."""

    matrix, actions = candidate_public_feature_matrix(
        source=source,
        previous_action=previous_action,
        candidates=candidates,
        gap=gap,
        context_seen=context_seen,
        training_destinations=training_destinations,
    )
    action_scores = worst_case_design_risk(detectors, matrix)
    action_risk_values = np.asarray(steganalysis_advantage(action_scores), dtype=float)
    action_risks = tuple(
        (action, float(risk))
        for action, risk in zip(actions, action_risk_values, strict=True)
    )
    return PublicRiskEnvelope(
        worst_risk=float(np.max(action_risk_values)),
        action_risks=action_risks,
        candidate_count=len(actions),
    )


def _normalized_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    if not candidates:
        raise ValueError("at least one admissible candidate is required")
    probabilities = np.asarray([float(item.probability) for item in candidates], dtype=float)
    if np.any(probabilities < 0) or float(probabilities.sum()) <= 0:
        raise ValueError("candidate probabilities must contain positive non-negative mass")
    probabilities = probabilities / probabilities.sum()
    return [
        Candidate(item.action, float(probability))
        for item, probability in zip(candidates, probabilities, strict=True)
    ]

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
    """Map an oriented stego posterior to normalized advantage above chance.

    Oriented detector scores are constructed so larger values indicate stego.
    A score at or below 0.5 supplies no positive steganalytic evidence, while a
    score of 1 represents maximal evidence. This scale matches fuzzy membership
    functions whose zero means no risk rather than chance-level classification.
    """

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

    if not candidates:
        raise ValueError("at least one admissible candidate is required")
    probabilities = np.asarray([float(item.probability) for item in candidates], dtype=float)
    if np.any(probabilities < 0) or float(probabilities.sum()) <= 0:
        raise ValueError("candidate probabilities must contain positive non-negative mass")
    probabilities = probabilities / probabilities.sum()
    entropy = float(-(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0])).sum())
    top_probability = float(probabilities.max())
    count = len(candidates)
    log_gap = math.log1p(max(0.0, float(gap)))

    rows: list[list[float]] = []
    actions: list[Hashable] = []
    for index, (candidate, probability) in enumerate(zip(candidates, probabilities, strict=True)):
        action = candidate.action
        rank = index + 1
        feature = {
            "action_probability": float(probability),
            "surprise_bits": -math.log2(max(float(probability), np.finfo(float).tiny)),
            "rank_fraction": rank / count,
            "is_top_action": float(rank == 1),
            "entropy_bits": entropy,
            "top_probability": top_probability,
            "candidate_count": float(count),
            "unseen_context": float(not context_seen),
            "unseen_destination": float(action not in training_destinations),
            "same_as_previous": float(previous_action == action),
            "self_loop": float(source == action),
            "log_inter_event_gap": log_gap,
        }
        rows.append([float(feature[name]) for name in FEATURE_COLUMNS])
        actions.append(action)
    return np.asarray(rows, dtype=float), tuple(actions)


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

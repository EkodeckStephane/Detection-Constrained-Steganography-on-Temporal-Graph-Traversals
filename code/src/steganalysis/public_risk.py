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
    """Secret-independent detectability risk attached to one public state."""

    worst_risk: float
    action_risks: tuple[tuple[Hashable, float], ...]
    candidate_count: int


def candidate_public_feature_matrix(
    *,
    source: Hashable,
    previous_action: Hashable | None,
    candidates: Sequence[Candidate],
    gap: float,
    context_seen: bool,
    training_destinations: frozenset[Hashable] | set[Hashable],
) -> tuple[np.ndarray, tuple[Hashable, ...]]:
    """Build Eve-visible features for every admissible candidate action.

    The matrix depends only on state shared by Alice and Bob: current source,
    previous emitted action, current admissible candidate distribution, timing,
    and cover-model training support. It never depends on payload bits or on the
    action that an encoder would select for those bits.
    """

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
    """Return the conservative maximum design-Eve risk for a public state.

    Risk is maximized first over frozen design-Eves for each action, then over
    all currently admissible actions. Therefore Alice and Bob obtain exactly
    the same value before any secret-dependent action selection occurs.
    """

    matrix, actions = candidate_public_feature_matrix(
        source=source,
        previous_action=previous_action,
        candidates=candidates,
        gap=gap,
        context_seen=context_seen,
        training_destinations=training_destinations,
    )
    action_risk_values = worst_case_design_risk(detectors, matrix)
    action_risks = tuple(
        (action, float(risk))
        for action, risk in zip(actions, action_risk_values, strict=True)
    )
    return PublicRiskEnvelope(
        worst_risk=float(np.max(action_risk_values)),
        action_risks=action_risks,
        candidate_count=len(actions),
    )

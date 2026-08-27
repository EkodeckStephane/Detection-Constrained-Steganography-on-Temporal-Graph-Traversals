from __future__ import annotations

import numpy as np
import pytest

from steganalysis.detectors import OrientedDetector
from steganalysis.public_risk import candidate_public_feature_matrix, public_state_risk_envelope
from stego.coding import Candidate


class _ProbabilityRiskDetector:
    """Use action probability itself as a deterministic public risk score."""

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        score = np.clip(x[:, 0], 0.0, 1.0)
        return np.column_stack([1.0 - score, score])


def _detectors() -> dict[str, OrientedDetector]:
    return {
        "eve": OrientedDetector(
            detector=_ProbabilityRiskDetector(),
            reverse_score=False,
            calibration_auc=0.75,
        )
    }


def test_public_risk_depends_on_state_candidates_not_payload_bits() -> None:
    candidates = [Candidate("B", 0.7), Candidate("C", 0.2), Candidate("D", 0.1)]

    first = public_state_risk_envelope(
        _detectors(),
        source="A",
        previous_action=None,
        candidates=candidates,
        gap=3.0,
        context_seen=True,
        training_destinations={"B", "C", "D"},
    )
    # Calling the same public state again represents a different secret message:
    # no payload argument exists, so the risk must be identical by construction.
    second = public_state_risk_envelope(
        _detectors(),
        source="A",
        previous_action=None,
        candidates=candidates,
        gap=3.0,
        context_seen=True,
        training_destinations={"B", "C", "D"},
    )

    assert first == second
    assert first.worst_risk == pytest.approx(0.7)
    assert first.candidate_count == 3


def test_public_risk_takes_worst_admissible_action() -> None:
    candidates = [Candidate("B", 0.55), Candidate("C", 0.45)]

    result = public_state_risk_envelope(
        _detectors(),
        source="A",
        previous_action="A",
        candidates=candidates,
        gap=0.0,
        context_seen=False,
        training_destinations={"B", "C"},
    )

    assert result.worst_risk == pytest.approx(0.55)
    assert dict(result.action_risks) == pytest.approx({"B": 0.55, "C": 0.45})


def test_public_feature_matrix_matches_primary_eve_feature_contract() -> None:
    matrix, actions = candidate_public_feature_matrix(
        source="A",
        previous_action="B",
        candidates=[Candidate("B", 0.6), Candidate("A", 0.4)],
        gap=9.0,
        context_seen=True,
        training_destinations={"A", "B"},
    )

    assert actions == ("B", "A")
    assert matrix.shape == (2, 12)
    assert np.isfinite(matrix).all()

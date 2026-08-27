from __future__ import annotations

import numpy as np
import pytest

from steganalysis.detectors import adversarial_auc, orient_detector, worst_case_design_risk


class _FixedDetector:
    def __init__(self, positive_scores: list[float]) -> None:
        self._scores = np.asarray(positive_scores, dtype=float)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = self._scores[: len(x)]
        return np.column_stack([1.0 - scores, scores])


def test_adversarial_auc_is_orientation_invariant() -> None:
    assert adversarial_auc(0.65) == pytest.approx(0.65)
    assert adversarial_auc(0.35) == pytest.approx(0.65)
    assert adversarial_auc(0.50) == pytest.approx(0.50)


def test_adversarial_auc_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        adversarial_auc(-0.01)
    with pytest.raises(ValueError):
        adversarial_auc(1.01)


def test_design_eve_orientation_is_frozen_on_calibration_data() -> None:
    x = np.zeros((4, 1), dtype=float)
    y = np.asarray([0, 0, 1, 1], dtype=int)
    detector = _FixedDetector([0.9, 0.8, 0.2, 0.1])

    oriented = orient_detector(detector, x, y)

    assert oriented.reverse_score
    assert oriented.calibration_auc == pytest.approx(0.0)
    assert oriented.risk(x).tolist() == pytest.approx([0.1, 0.2, 0.8, 0.9])


def test_worst_case_design_risk_uses_maximum_oriented_score() -> None:
    x = np.zeros((3, 1), dtype=float)
    y = np.asarray([0, 1, 1], dtype=int)
    first = orient_detector(_FixedDetector([0.1, 0.7, 0.8]), x, y)
    second = orient_detector(_FixedDetector([0.2, 0.4, 0.9]), x, y)

    risk = worst_case_design_risk({"first": first, "second": second}, x)

    assert risk.tolist() == pytest.approx([0.2, 0.7, 0.9])

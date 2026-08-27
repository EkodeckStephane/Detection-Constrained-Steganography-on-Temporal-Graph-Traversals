from __future__ import annotations

import pytest

from steganalysis.detectors import adversarial_auc


def test_adversarial_auc_is_orientation_invariant() -> None:
    assert adversarial_auc(0.65) == pytest.approx(0.65)
    assert adversarial_auc(0.35) == pytest.approx(0.65)
    assert adversarial_auc(0.50) == pytest.approx(0.50)


def test_adversarial_auc_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        adversarial_auc(-0.01)
    with pytest.raises(ValueError):
        adversarial_auc(1.01)

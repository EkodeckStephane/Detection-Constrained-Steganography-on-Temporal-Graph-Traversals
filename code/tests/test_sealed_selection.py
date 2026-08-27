from __future__ import annotations

import pytest

from evaluation.sealed_selection import (
    ValidationOperatingPoint,
    record_sealed_test_result,
    select_operating_point,
)


def test_selection_uses_validation_upper_bound_and_maximizes_payload() -> None:
    points = [
        ValidationOperatingPoint("low", 0.05, 0.54, 0.57),
        ValidationOperatingPoint("best", 0.10, 0.56, 0.59),
        ValidationOperatingPoint("too_detectable", 0.20, 0.58, 0.61),
    ]

    frozen = select_operating_point(points, detector_auc_budget=0.60)

    assert frozen.name == "best"
    assert frozen.payload_bits_per_transition == 0.10
    assert frozen.validation_max_design_eve_auc_ci_upper == 0.59


def test_no_point_can_be_selected_if_validation_bound_exceeds_budget() -> None:
    with pytest.raises(ValueError, match="No validation operating point"):
        select_operating_point(
            [ValidationOperatingPoint("x", 0.10, 0.60, 0.64)],
            detector_auc_budget=0.60,
        )


def test_bad_sealed_test_result_does_not_change_frozen_choice() -> None:
    frozen = select_operating_point(
        [ValidationOperatingPoint("chosen", 0.10, 0.55, 0.58)],
        detector_auc_budget=0.60,
    )

    result = record_sealed_test_result(
        frozen,
        test_payload_bits_per_transition=0.09,
        test_max_design_eve_auc=0.72,
        test_max_unseen_eve_auc=0.81,
        passive_decode_success_rate=1.0,
        passive_state_mismatch_rate=0.0,
    )

    assert result.frozen is frozen
    assert result.frozen.name == "chosen"
    assert result.test_max_unseen_eve_auc == 0.81

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ValidationOperatingPoint:
    name: str
    payload_bits_per_transition: float
    max_design_eve_auc: float
    max_design_eve_auc_ci_upper: float


@dataclass(frozen=True)
class FrozenOperatingPoint:
    name: str
    payload_bits_per_transition: float
    validation_max_design_eve_auc: float
    validation_max_design_eve_auc_ci_upper: float
    detector_auc_budget: float


@dataclass(frozen=True)
class SealedTestResult:
    frozen: FrozenOperatingPoint
    test_payload_bits_per_transition: float
    test_max_design_eve_auc: float
    test_max_unseen_eve_auc: float
    passive_decode_success_rate: float
    passive_state_mismatch_rate: float


def select_operating_point(
    validation_points: Iterable[ValidationOperatingPoint],
    *,
    detector_auc_budget: float,
) -> FrozenOperatingPoint:
    """Select the highest-payload point using validation information only.

    The API deliberately accepts no test metrics. A candidate is eligible only
    when the upper endpoint of its validation uncertainty interval for the worst
    design-Eve AUC is within the prespecified detector budget.
    """

    if not 0.5 <= detector_auc_budget <= 1.0:
        raise ValueError("detector_auc_budget must be in [0.5, 1.0]")

    points = list(validation_points)
    if not points:
        raise ValueError("At least one validation operating point is required")

    for point in points:
        if point.payload_bits_per_transition < 0:
            raise ValueError("payload must be non-negative")
        if not 0.0 <= point.max_design_eve_auc <= 1.0:
            raise ValueError("AUC must be in [0, 1]")
        if not 0.0 <= point.max_design_eve_auc_ci_upper <= 1.0:
            raise ValueError("AUC confidence bound must be in [0, 1]")
        if point.max_design_eve_auc_ci_upper < point.max_design_eve_auc:
            raise ValueError("AUC CI upper endpoint cannot be below the point estimate")

    eligible = [
        point
        for point in points
        if point.max_design_eve_auc_ci_upper <= detector_auc_budget
    ]
    if not eligible:
        raise ValueError("No validation operating point satisfies the detector budget")

    selected = max(
        eligible,
        key=lambda point: (
            point.payload_bits_per_transition,
            -point.max_design_eve_auc_ci_upper,
            point.name,
        ),
    )
    return FrozenOperatingPoint(
        name=selected.name,
        payload_bits_per_transition=selected.payload_bits_per_transition,
        validation_max_design_eve_auc=selected.max_design_eve_auc,
        validation_max_design_eve_auc_ci_upper=selected.max_design_eve_auc_ci_upper,
        detector_auc_budget=float(detector_auc_budget),
    )


def record_sealed_test_result(
    frozen: FrozenOperatingPoint,
    *,
    test_payload_bits_per_transition: float,
    test_max_design_eve_auc: float,
    test_max_unseen_eve_auc: float,
    passive_decode_success_rate: float,
    passive_state_mismatch_rate: float,
) -> SealedTestResult:
    """Attach final test measurements to an already frozen operating point.

    This function performs no selection and returns no alternative operating
    point. Test outcomes therefore cannot change the frozen design decision.
    """

    bounded = {
        "test_max_design_eve_auc": test_max_design_eve_auc,
        "test_max_unseen_eve_auc": test_max_unseen_eve_auc,
        "passive_decode_success_rate": passive_decode_success_rate,
        "passive_state_mismatch_rate": passive_state_mismatch_rate,
    }
    if test_payload_bits_per_transition < 0:
        raise ValueError("test payload must be non-negative")
    for name, value in bounded.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    return SealedTestResult(
        frozen=frozen,
        test_payload_bits_per_transition=float(test_payload_bits_per_transition),
        test_max_design_eve_auc=float(test_max_design_eve_auc),
        test_max_unseen_eve_auc=float(test_max_unseen_eve_auc),
        passive_decode_success_rate=float(passive_decode_success_rate),
        passive_state_mismatch_rate=float(passive_state_mismatch_rate),
    )

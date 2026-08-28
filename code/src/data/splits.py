from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalCutoffs:
    train_end: float
    validation_end: float
    train_fraction: float
    validation_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class FourWayTemporalCutoffs:
    cover_train_end: float
    eve_train_end: float
    policy_validation_end: float
    cover_train_fraction: float
    eve_train_fraction: float
    policy_validation_fraction: float
    sealed_test_fraction: float


@dataclass(frozen=True)
class FiveWayTemporalCutoffs:
    cover_train_end: float
    eve_train_end: float
    policy_validation_end: float
    development_test_end: float
    cover_train_fraction: float
    eve_train_fraction: float
    policy_validation_fraction: float
    development_test_fraction: float
    final_holdout_fraction: float


def assign_causal_splits(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, TemporalCutoffs]:
    if frame.empty:
        raise ValueError("Cannot split an empty table")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("A positive test fraction is required")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Frame must be sorted chronologically before splitting")

    timestamps = frame["timestamp"].to_numpy()
    train_position = min(len(frame) - 1, max(0, int(np.ceil(train_fraction * len(frame))) - 1))
    validation_position = min(
        len(frame) - 1,
        max(train_position + 1, int(np.ceil((train_fraction + validation_fraction) * len(frame))) - 1),
    )
    train_end = float(timestamps[train_position])
    validation_end = float(timestamps[validation_position])
    if validation_end <= train_end:
        later = timestamps[timestamps > train_end]
        if later.size == 0:
            raise ValueError("At least three temporal regions are required")
        validation_end = float(later[min(len(later) - 1, max(0, len(later) // 2))])

    result = frame.copy()
    result["split"] = np.where(
        result["timestamp"] <= train_end,
        "train",
        np.where(result["timestamp"] <= validation_end, "validation", "test"),
    )
    fractions = result["split"].value_counts(normalize=True)
    if set(fractions.index) != {"train", "validation", "test"}:
        raise ValueError("The timestamps do not permit three non-empty causal splits")
    return result, TemporalCutoffs(
        train_end=train_end,
        validation_end=validation_end,
        train_fraction=float(fractions["train"]),
        validation_fraction=float(fractions["validation"]),
        test_fraction=float(fractions["test"]),
    )


def assign_four_way_causal_splits(
    frame: pd.DataFrame,
    *,
    cover_train_fraction: float = 0.55,
    eve_train_fraction: float = 0.15,
    policy_validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, FourWayTemporalCutoffs]:
    """Create four non-overlapping chronological regions for sealed evaluation.

    This legacy ASOC V2 split remains available for reproduction of development
    pilots. New final experiments must use ``assign_five_way_causal_splits`` so
    that pilot-inspected observations cannot enter the publication holdout.
    """

    if frame.empty:
        raise ValueError("Cannot split an empty table")
    fractions_requested = (
        cover_train_fraction,
        eve_train_fraction,
        policy_validation_fraction,
    )
    if any(value <= 0 for value in fractions_requested):
        raise ValueError("All design-region fractions must be positive")
    if sum(fractions_requested) >= 1:
        raise ValueError("A positive sealed-test fraction is required")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Frame must be sorted chronologically before splitting")

    timestamps = frame["timestamp"].to_numpy(dtype=float)
    if np.unique(timestamps).size < 4:
        raise ValueError("At least four distinct timestamps are required")

    cover_end = _cutoff_at_fraction(timestamps, cover_train_fraction)
    eve_end = _strict_later_cutoff(
        timestamps,
        _cutoff_at_fraction(timestamps, cover_train_fraction + eve_train_fraction),
        previous=cover_end,
    )
    policy_end = _strict_later_cutoff(
        timestamps,
        _cutoff_at_fraction(
            timestamps,
            cover_train_fraction + eve_train_fraction + policy_validation_fraction,
        ),
        previous=eve_end,
    )
    if policy_end >= float(timestamps[-1]):
        raise ValueError("The requested fractions leave no non-empty sealed test")

    result = frame.copy()
    result["split"] = np.select(
        [
            result["timestamp"] <= cover_end,
            result["timestamp"] <= eve_end,
            result["timestamp"] <= policy_end,
        ],
        ["cover_train", "eve_train", "policy_validation"],
        default="sealed_test",
    )

    expected = {"cover_train", "eve_train", "policy_validation", "sealed_test"}
    observed = set(result["split"].unique())
    if observed != expected:
        raise ValueError("The timestamps do not permit four non-empty causal regions")
    if result.groupby("timestamp")["split"].nunique().max() != 1:
        raise AssertionError("Timestamp ties crossed a causal split boundary")

    fractions = result["split"].value_counts(normalize=True)
    return result, FourWayTemporalCutoffs(
        cover_train_end=cover_end,
        eve_train_end=eve_end,
        policy_validation_end=policy_end,
        cover_train_fraction=float(fractions["cover_train"]),
        eve_train_fraction=float(fractions["eve_train"]),
        policy_validation_fraction=float(fractions["policy_validation"]),
        sealed_test_fraction=float(fractions["sealed_test"]),
    )


def assign_five_way_causal_splits(
    frame: pd.DataFrame,
    *,
    cover_train_fraction: float = 0.55,
    eve_train_fraction: float = 0.15,
    policy_validation_fraction: float = 0.15,
    development_test_fraction: float = 0.05,
) -> tuple[pd.DataFrame, FiveWayTemporalCutoffs]:
    """Create a fresh publication holdout after pilot-inspected observations.

    Regions are chronological and timestamp ties remain intact:

    - ``cover_train`` fits the cover model;
    - ``eve_train`` fits and orients design-Eves;
    - ``policy_validation`` tunes and freezes the controller;
    - ``development_test`` contains the early post-validation region that may
      be inspected during engineering diagnostics;
    - ``final_holdout`` is the untouched publication test and must not affect
      any model, detector, threshold, controller, baseline or narrative choice.

    The default 55/15/15/5/10 allocation preserves the original 55/15/15
    design regions while quarantining the pilot-accessible start of the old
    15% test block and reserving its last 10% for final evaluation.
    """

    if frame.empty:
        raise ValueError("Cannot split an empty table")
    requested = (
        cover_train_fraction,
        eve_train_fraction,
        policy_validation_fraction,
        development_test_fraction,
    )
    if any(value <= 0 for value in requested):
        raise ValueError("All pre-holdout fractions must be positive")
    if sum(requested) >= 1:
        raise ValueError("A positive final-holdout fraction is required")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Frame must be sorted chronologically before splitting")

    timestamps = frame["timestamp"].to_numpy(dtype=float)
    if np.unique(timestamps).size < 5:
        raise ValueError("At least five distinct timestamps are required")

    cumulative = np.cumsum(np.asarray(requested, dtype=float))
    cover_end = _cutoff_at_fraction(timestamps, cumulative[0])
    eve_end = _strict_later_cutoff(
        timestamps,
        _cutoff_at_fraction(timestamps, cumulative[1]),
        previous=cover_end,
    )
    policy_end = _strict_later_cutoff(
        timestamps,
        _cutoff_at_fraction(timestamps, cumulative[2]),
        previous=eve_end,
    )
    development_end = _strict_later_cutoff(
        timestamps,
        _cutoff_at_fraction(timestamps, cumulative[3]),
        previous=policy_end,
    )
    if development_end >= float(timestamps[-1]):
        raise ValueError("The requested fractions leave no non-empty final holdout")

    result = frame.copy()
    result["split"] = np.select(
        [
            result["timestamp"] <= cover_end,
            result["timestamp"] <= eve_end,
            result["timestamp"] <= policy_end,
            result["timestamp"] <= development_end,
        ],
        ["cover_train", "eve_train", "policy_validation", "development_test"],
        default="final_holdout",
    )
    expected = {
        "cover_train",
        "eve_train",
        "policy_validation",
        "development_test",
        "final_holdout",
    }
    observed = set(result["split"].unique())
    if observed != expected:
        raise ValueError("The timestamps do not permit five non-empty causal regions")
    if result.groupby("timestamp")["split"].nunique().max() != 1:
        raise AssertionError("Timestamp ties crossed a causal split boundary")

    fractions = result["split"].value_counts(normalize=True)
    return result, FiveWayTemporalCutoffs(
        cover_train_end=cover_end,
        eve_train_end=eve_end,
        policy_validation_end=policy_end,
        development_test_end=development_end,
        cover_train_fraction=float(fractions["cover_train"]),
        eve_train_fraction=float(fractions["eve_train"]),
        policy_validation_fraction=float(fractions["policy_validation"]),
        development_test_fraction=float(fractions["development_test"]),
        final_holdout_fraction=float(fractions["final_holdout"]),
    )


def _cutoff_at_fraction(timestamps: np.ndarray, fraction: float) -> float:
    position = min(
        len(timestamps) - 1,
        max(0, int(np.ceil(float(fraction) * len(timestamps))) - 1),
    )
    return float(timestamps[position])


def _strict_later_cutoff(
    timestamps: np.ndarray,
    proposed: float,
    *,
    previous: float,
) -> float:
    if proposed > previous:
        return float(proposed)
    later = timestamps[timestamps > previous]
    if later.size == 0:
        raise ValueError("Insufficient distinct timestamps for causal regions")
    return float(later[0])

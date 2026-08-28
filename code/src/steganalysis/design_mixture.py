from __future__ import annotations

import math
from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd

from steganalysis.detectors import OrientedDetector, fit_detector, orient_detector
from steganalysis.samples import FEATURE_COLUMNS
from stego.coding import Candidate


def make_actor_action_design_records(
    model: object,
    frame: pd.DataFrame,
    *,
    intensities: Sequence[float] = (0.05, 0.20, 0.50, 1.00),
    seed: int = 20260827,
) -> pd.DataFrame:
    """Build balanced cover/stego records for actor-action design-Eves.

    Each intensity is a separate causal replay.  With probability ``intensity``
    the reference generator samples an action from the current causal cover
    distribution Q; otherwise it leaves the observed natural carrier action
    unchanged.  The stego history is always updated with the action actually
    emitted.  Natural records are duplicated per intensity so the combined
    binary detector sample remains balanced.

    This generator is used only to train design-Eves.  It does not define the
    proposed arithmetic policy and it never accesses policy-validation or test
    regions.
    """

    required = {"source", "destination", "timestamp"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("design-Eve training frame must be non-empty")
    if not intensities or any(not 0.0 <= float(value) <= 1.0 for value in intensities):
        raise ValueError("design intensities must lie in [0, 1]")

    ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for intensity_index, raw_intensity in enumerate(intensities):
        intensity = float(raw_intensity)
        rng = np.random.default_rng(seed + 1009 * intensity_index)
        natural_previous: dict[Hashable, Hashable] = {}
        stego_previous: dict[Hashable, Hashable] = {}
        natural_time: dict[Hashable, float] = {}
        stego_time: dict[Hashable, float] = {}

        for pair_id, (source, natural_action, timestamp) in enumerate(
            ordered[["source", "destination", "timestamp"]].itertuples(index=False, name=None)
        ):
            natural_prev = natural_previous.get(source)
            stego_prev = stego_previous.get(source)
            natural_gap = float(timestamp) - natural_time.get(source, float(timestamp))
            stego_gap = float(timestamp) - stego_time.get(source, float(timestamp))
            natural_candidates = model.candidate_distribution(source, natural_prev)
            stego_candidates = model.candidate_distribution(source, stego_prev)
            if not natural_candidates or not stego_candidates:
                raise ValueError("design generator encountered an empty cover distribution")

            if rng.random() < intensity and len(stego_candidates) >= 2:
                probabilities = np.asarray(
                    [float(item.probability) for item in stego_candidates], dtype=float
                )
                probabilities /= probabilities.sum()
                stego_action = stego_candidates[
                    int(rng.choice(len(stego_candidates), p=probabilities))
                ].action
                modified = stego_action != natural_action
            else:
                stego_action = natural_action
                modified = False

            rows.append(
                {
                    "pair_id": pair_id,
                    "design_intensity": intensity,
                    "source": str(source),
                    "label": 0,
                    "modified": False,
                    **_feature(
                        model,
                        source=source,
                        previous=natural_prev,
                        action=natural_action,
                        candidates=natural_candidates,
                        gap=natural_gap,
                    ),
                }
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "design_intensity": intensity,
                    "source": str(source),
                    "label": 1,
                    "modified": modified,
                    **_feature(
                        model,
                        source=source,
                        previous=stego_prev,
                        action=stego_action,
                        candidates=stego_candidates,
                        gap=stego_gap,
                    ),
                }
            )
            natural_previous[source] = natural_action
            stego_previous[source] = stego_action
            natural_time[source] = float(timestamp)
            stego_time[source] = float(timestamp)

    return pd.DataFrame(rows)


def fit_actor_action_design_eves(
    model: object,
    frame: pd.DataFrame,
    *,
    detector_names: Sequence[str] = ("linear", "forest"),
    intensities: Sequence[float] = (0.05, 0.20, 0.50, 1.00),
    detector_fit_fraction: float = 0.75,
    seed: int = 20260827,
) -> dict[str, OrientedDetector]:
    """Fit and orient design-Eves using a chronological internal split."""

    if not 0.0 < detector_fit_fraction < 1.0:
        raise ValueError("detector_fit_fraction must lie in (0, 1)")
    records = make_actor_action_design_records(
        model,
        frame,
        intensities=intensities,
        seed=seed,
    )
    event_count = int(records["pair_id"].max()) + 1
    cutoff = max(1, min(event_count - 1, int(math.floor(detector_fit_fraction * event_count))))
    fit = records.loc[records["pair_id"] < cutoff]
    calibration = records.loc[records["pair_id"] >= cutoff]
    if fit["label"].nunique() != 2 or calibration["label"].nunique() != 2:
        raise ValueError("both detector-fit and orientation blocks require both classes")

    x_fit = fit[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_fit = fit["label"].to_numpy(dtype=int)
    x_cal = calibration[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_cal = calibration["label"].to_numpy(dtype=int)

    result: dict[str, OrientedDetector] = {}
    for name in detector_names:
        detector = fit_detector(name, x_fit, y_fit, seed=seed)
        result[name] = orient_detector(detector, x_cal, y_cal)
    return result


def _feature(
    model: object,
    *,
    source: Hashable,
    previous: Hashable | None,
    action: Hashable,
    candidates: Sequence[Candidate],
    gap: float,
) -> dict[str, float]:
    probabilities = np.asarray([float(item.probability) for item in candidates], dtype=float)
    probabilities /= probabilities.sum()
    actions = [item.action for item in candidates]
    if action in actions:
        index = actions.index(action)
        action_probability = float(probabilities[index])
    else:
        index = len(actions)
        action_probability = float(max(probabilities.min() * 0.5, np.finfo(float).tiny))
    entropy = float(
        -(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0])).sum()
    )
    feature = {
        "action_probability": action_probability,
        "surprise_bits": -math.log2(max(action_probability, np.finfo(float).tiny)),
        "rank_fraction": (index + 1) / max(1, len(actions)),
        "is_top_action": float(index == 0),
        "entropy_bits": entropy,
        "top_probability": float(probabilities.max()),
        "candidate_count": float(len(actions)),
        "unseen_context": float(not model.has_context(source, previous)),
        "unseen_destination": float(action not in model.destinations),
        "same_as_previous": float(previous == action),
        "self_loop": float(source == action),
        "log_inter_event_gap": math.log1p(max(0.0, float(gap))),
    }
    return {name: float(feature[name]) for name in FEATURE_COLUMNS}

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from steganalysis.detectors import adversarial_auc


@dataclass(frozen=True)
class WorstAucBootstrap:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    detector_point_estimates: dict[str, float]
    confidence_level: float
    resamples: int
    valid_resamples: int


def cluster_bootstrap_worst_adversarial_auc(
    labels: Sequence[int] | np.ndarray,
    score_map: Mapping[str, Sequence[float] | np.ndarray],
    clusters: Sequence[object] | np.ndarray,
    *,
    resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 20260827,
) -> WorstAucBootstrap:
    """Cluster bootstrap the worst orientation-invariant design-Eve AUC.

    Whole trajectories/sources are resampled with replacement. For every
    bootstrap replicate we compute ``max(AUC, 1-AUC)`` for each frozen
    design-Eve and then take the maximum across Eves. The confidence interval
    is therefore on the actual worst-adversary statistic used by policy
    selection, including dependence between detector scores.
    """

    y = np.asarray(labels, dtype=int)
    cluster_values = np.asarray(clusters, dtype=object)
    if y.ndim != 1 or cluster_values.ndim != 1 or len(y) != len(cluster_values):
        raise ValueError("labels and clusters must be aligned one-dimensional arrays")
    if len(y) < 2 or np.unique(y).size != 2:
        raise ValueError("binary labels with both classes are required")
    if resamples < 100:
        raise ValueError("at least 100 bootstrap resamples are required")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if not score_map:
        raise ValueError("at least one detector score vector is required")

    scores: dict[str, np.ndarray] = {}
    for name, values in score_map.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or len(array) != len(y):
            raise ValueError(f"score vector {name!r} is not aligned with labels")
        scores[name] = array

    detector_points = {
        name: adversarial_auc(float(roc_auc_score(y, values)))
        for name, values in scores.items()
    }
    point = max(detector_points.values())

    unique_clusters, inverse = np.unique(cluster_values.astype(str), return_inverse=True)
    member_indices = [np.flatnonzero(inverse == index) for index in range(len(unique_clusters))]
    if not member_indices:
        raise ValueError("at least one cluster is required")

    rng = np.random.default_rng(seed)
    bootstrap_values: list[float] = []
    cluster_count = len(member_indices)
    for _ in range(resamples):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        indices = np.concatenate([member_indices[index] for index in sampled])
        sampled_y = y[indices]
        if np.unique(sampled_y).size != 2:
            continue
        worst = 0.5
        for name, values in scores.items():
            raw = float(roc_auc_score(sampled_y, values[indices]))
            worst = max(worst, adversarial_auc(raw))
        bootstrap_values.append(worst)

    if not bootstrap_values:
        raise ValueError("no valid bootstrap replicate contained both classes")

    values = np.asarray(bootstrap_values, dtype=float)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return WorstAucBootstrap(
        point_estimate=float(point),
        ci_lower=float(lower),
        ci_upper=float(upper),
        detector_point_estimates=detector_points,
        confidence_level=float(confidence_level),
        resamples=int(resamples),
        valid_resamples=len(values),
    )

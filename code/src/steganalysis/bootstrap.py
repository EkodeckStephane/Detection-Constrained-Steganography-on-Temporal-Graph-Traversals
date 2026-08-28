from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numba import njit, prange
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
    batch_size: int = 128,
) -> WorstAucBootstrap:
    """Cluster bootstrap the worst orientation-invariant design-Eve AUC.

    Whole trajectories/sources are resampled with replacement. Each resample is
    represented exactly by cluster multiplicities. For every frozen detector we
    compute the weighted Mann--Whitney AUC, including exact half-credit for score
    ties, then take ``max(AUC, 1-AUC)`` and finally the maximum across design-Eves.

    The statistic is algebraically identical to materializing every replicated
    observation and calling ``roc_auc_score``. The weighted scan is compiled so
    that 10,000 cluster resamples remain practical on the full validation sets.
    """

    y = np.asarray(labels, dtype=np.int8)
    cluster_values = np.asarray(clusters, dtype=object)
    if y.ndim != 1 or cluster_values.ndim != 1 or len(y) != len(cluster_values):
        raise ValueError("labels and clusters must be aligned one-dimensional arrays")
    if len(y) < 2 or np.unique(y).size != 2:
        raise ValueError("binary labels with both classes are required")
    if resamples < 100:
        raise ValueError("at least 100 bootstrap resamples are required")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not score_map:
        raise ValueError("at least one detector score vector is required")

    scores: dict[str, np.ndarray] = {}
    for name, values in score_map.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or len(array) != len(y):
            raise ValueError(f"score vector {name!r} is not aligned with labels")
        if not np.isfinite(array).all():
            raise ValueError(f"score vector {name!r} contains non-finite values")
        scores[name] = array

    detector_points = {
        name: adversarial_auc(float(roc_auc_score(y, values)))
        for name, values in scores.items()
    }
    point = max(detector_points.values())

    unique_clusters, inverse = np.unique(cluster_values.astype(str), return_inverse=True)
    cluster_count = len(unique_clusters)
    if cluster_count == 0:
        raise ValueError("at least one cluster is required")
    inverse = inverse.astype(np.int32, copy=False)

    prepared = {
        name: _prepare_sorted_detector(y, values, inverse)
        for name, values in scores.items()
    }

    rng = np.random.default_rng(seed)
    bootstrap_values: list[float] = []
    generated = 0
    while generated < resamples:
        current_batch = min(batch_size, resamples - generated)
        sampled = rng.integers(
            0,
            cluster_count,
            size=(current_batch, cluster_count),
        )
        multiplicities = np.zeros((current_batch, cluster_count), dtype=np.int32)
        batch_rows = np.repeat(np.arange(current_batch), cluster_count)
        np.add.at(
            multiplicities,
            (batch_rows, sampled.ravel()),
            1,
        )

        worst = np.full(current_batch, 0.5, dtype=float)
        valid = np.ones(current_batch, dtype=bool)
        for labels_sorted, clusters_sorted, scores_sorted in prepared.values():
            detector_values = _weighted_adversarial_auc_replicates(
                multiplicities,
                labels_sorted,
                clusters_sorted,
                scores_sorted,
            )
            detector_valid = np.isfinite(detector_values)
            valid &= detector_valid
            worst = np.maximum(worst, np.nan_to_num(detector_values, nan=0.5))

        bootstrap_values.extend(worst[valid].tolist())
        generated += current_batch

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


def _prepare_sorted_detector(
    labels: np.ndarray,
    scores: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(scores, kind="stable")
    return (
        labels[order].astype(np.int8, copy=False),
        clusters[order].astype(np.int32, copy=False),
        scores[order].astype(np.float64, copy=False),
    )


@njit(cache=True, parallel=True)
def _weighted_adversarial_auc_replicates(
    multiplicities: np.ndarray,
    labels_sorted: np.ndarray,
    clusters_sorted: np.ndarray,
    scores_sorted: np.ndarray,
) -> np.ndarray:
    """Compute exact weighted AUC* for many bootstrap multiplicity vectors."""

    replicate_count = multiplicities.shape[0]
    observation_count = labels_sorted.shape[0]
    output = np.empty(replicate_count, dtype=np.float64)

    for replicate in prange(replicate_count):
        total_positive = 0.0
        total_negative = 0.0
        cumulative_negative = 0.0
        numerator = 0.0
        index = 0

        while index < observation_count:
            score = scores_sorted[index]
            group_positive = 0.0
            group_negative = 0.0
            cursor = index
            while cursor < observation_count and scores_sorted[cursor] == score:
                weight = multiplicities[replicate, clusters_sorted[cursor]]
                if labels_sorted[cursor] == 1:
                    group_positive += weight
                    total_positive += weight
                else:
                    group_negative += weight
                    total_negative += weight
                cursor += 1

            numerator += group_positive * (
                cumulative_negative + 0.5 * group_negative
            )
            cumulative_negative += group_negative
            index = cursor

        denominator = total_positive * total_negative
        if denominator <= 0.0:
            output[replicate] = np.nan
        else:
            raw_auc = numerator / denominator
            output[replicate] = max(raw_auc, 1.0 - raw_auc)

    return output

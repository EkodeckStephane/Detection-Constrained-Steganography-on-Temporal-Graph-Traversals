from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import sparse
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
    batch_size: int = 64,
) -> WorstAucBootstrap:
    """Cluster bootstrap the worst orientation-invariant design-Eve AUC.

    Whole trajectories/sources are resampled with replacement. For every
    bootstrap replicate we compute ``max(AUC, 1-AUC)`` for each frozen
    design-Eve and then take the maximum across Eves. The confidence interval
    is therefore on the actual worst-adversary statistic used by policy
    selection, including dependence between detector scores.

    The implementation is algebraically equivalent to materializing every
    replicated observation and calling ``roc_auc_score`` on each bootstrap
    sample. It instead represents a resample by cluster multiplicities and
    computes the exact weighted Mann--Whitney AUC over fixed score-tie groups.
    This preserves the original statistic while making 10,000 cluster
    resamples practical on the full validation regions.
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

    grouped = {
        name: _cluster_score_groups(y, values, inverse, cluster_count)
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
        multiplicities = np.zeros((current_batch, cluster_count), dtype=float)
        batch_rows = np.repeat(np.arange(current_batch), cluster_count)
        np.add.at(
            multiplicities,
            (batch_rows, sampled.ravel()),
            1.0,
        )

        worst = np.full(current_batch, 0.5, dtype=float)
        valid = np.ones(current_batch, dtype=bool)
        for positive_groups, negative_groups in grouped.values():
            positive = np.asarray(positive_groups.T.dot(multiplicities.T).T)
            negative = np.asarray(negative_groups.T.dot(multiplicities.T).T)
            positive_total = positive.sum(axis=1)
            negative_total = negative.sum(axis=1)
            detector_valid = (positive_total > 0) & (negative_total > 0)
            valid &= detector_valid

            negative_before = np.cumsum(negative, axis=1) - negative
            numerator = (
                positive * (negative_before + 0.5 * negative)
            ).sum(axis=1)
            denominator = positive_total * negative_total
            raw_auc = np.divide(
                numerator,
                denominator,
                out=np.full(current_batch, np.nan, dtype=float),
                where=denominator > 0,
            )
            detector_adversarial = np.maximum(raw_auc, 1.0 - raw_auc)
            worst = np.maximum(worst, np.nan_to_num(detector_adversarial, nan=0.5))

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


def _cluster_score_groups(
    labels: np.ndarray,
    scores: np.ndarray,
    cluster_inverse: np.ndarray,
    cluster_count: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Count positive/negative observations by cluster and equal-score group."""

    _, score_group = np.unique(scores, return_inverse=True)
    group_count = int(score_group.max()) + 1
    positive_mask = labels == 1
    negative_mask = labels == 0

    positive = sparse.coo_matrix(
        (
            np.ones(int(positive_mask.sum()), dtype=float),
            (cluster_inverse[positive_mask], score_group[positive_mask]),
        ),
        shape=(cluster_count, group_count),
    ).tocsr()
    negative = sparse.coo_matrix(
        (
            np.ones(int(negative_mask.sum()), dtype=float),
            (cluster_inverse[negative_mask], score_group[negative_mask]),
        ),
        shape=(cluster_count, group_count),
    ).tocsr()
    return positive, negative

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from steganalysis.bootstrap import cluster_bootstrap_worst_adversarial_auc


def test_cluster_bootstrap_is_half_for_identical_pair_scores() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=int)
    clusters = np.asarray(["a", "a", "b", "b", "c", "c"], dtype=object)
    scores = np.asarray([0.4, 0.4, 0.6, 0.6, 0.5, 0.5], dtype=float)

    result = cluster_bootstrap_worst_adversarial_auc(
        labels,
        {"linear": scores},
        clusters,
        resamples=200,
        seed=11,
    )

    assert result.point_estimate == pytest.approx(0.5)
    assert result.ci_lower == pytest.approx(0.5)
    assert result.ci_upper == pytest.approx(0.5)


def test_cluster_bootstrap_treats_reversed_scores_as_detectable() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    clusters = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"], dtype=object)
    reversed_scores = np.asarray([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])

    result = cluster_bootstrap_worst_adversarial_auc(
        labels,
        {"reversed": reversed_scores},
        clusters,
        resamples=200,
        seed=13,
    )

    assert result.point_estimate == pytest.approx(1.0)
    assert result.ci_upper == pytest.approx(1.0)


def test_cluster_bootstrap_takes_worst_eve_per_resample() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    clusters = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"], dtype=object)
    weak = np.asarray([0.4, 0.6, 0.5, 0.5, 0.45, 0.55, 0.5, 0.5])
    strong = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])

    result = cluster_bootstrap_worst_adversarial_auc(
        labels,
        {"weak": weak, "strong": strong},
        clusters,
        resamples=200,
        seed=17,
    )

    assert result.detector_point_estimates["strong"] >= result.detector_point_estimates["weak"]
    assert result.point_estimate == result.detector_point_estimates["strong"]
    assert result.ci_upper >= result.point_estimate


def test_vectorized_cluster_bootstrap_matches_materialized_reference_exactly() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    clusters = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"], dtype=object)
    score_map = {
        "linear": np.asarray([0.2, 0.8, 0.4, 0.6, 0.5, 0.5, 0.7, 0.3, 0.45, 0.55]),
        "forest": np.asarray([0.3, 0.7, 0.35, 0.65, 0.6, 0.4, 0.8, 0.2, 0.5, 0.5]),
    }
    resamples = 300
    seed = 29

    result = cluster_bootstrap_worst_adversarial_auc(
        labels,
        score_map,
        clusters,
        resamples=resamples,
        seed=seed,
        batch_size=37,
    )
    reference = _materialized_reference(labels, score_map, clusters, resamples=resamples, seed=seed)

    assert result.valid_resamples == len(reference)
    assert result.ci_lower == pytest.approx(np.quantile(reference, 0.025), abs=1e-12)
    assert result.ci_upper == pytest.approx(np.quantile(reference, 0.975), abs=1e-12)


def _materialized_reference(
    labels: np.ndarray,
    score_map: dict[str, np.ndarray],
    clusters: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> np.ndarray:
    unique_clusters, inverse = np.unique(clusters.astype(str), return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique_clusters))]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        sampled = rng.integers(0, len(unique_clusters), size=len(unique_clusters))
        indices = np.concatenate([members[index] for index in sampled])
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size != 2:
            continue
        worst = 0.5
        for scores in score_map.values():
            raw = float(roc_auc_score(sampled_labels, scores[indices]))
            worst = max(worst, raw, 1.0 - raw)
        values.append(worst)
    return np.asarray(values, dtype=float)

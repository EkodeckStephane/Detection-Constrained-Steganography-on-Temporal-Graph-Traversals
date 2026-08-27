from __future__ import annotations

import numpy as np
import pytest

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

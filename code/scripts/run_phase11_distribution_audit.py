from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(np.asarray(left, dtype=float))
    right = np.sort(np.asarray(right, dtype=float))
    count = min(len(left), len(right))
    if count == 0:
        return 0.0
    return float(np.mean(np.abs(left[:count] - right[:count])))


def _median_gamma(matrix: np.ndarray) -> float:
    if len(matrix) < 2:
        return 1.0
    sample = matrix[: min(len(matrix), 1000)]
    diffs = sample[:, None, :] - sample[None, :, :]
    distances = np.sum(diffs * diffs, axis=2)
    positive = distances[distances > 0]
    if len(positive) == 0:
        return 1.0
    median = float(np.median(positive))
    return 1.0 / max(2.0 * median, np.finfo(float).eps)


def _rbf_kernel(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    diffs = left[:, None, :] - right[None, :, :]
    return np.exp(-gamma * np.sum(diffs * diffs, axis=2))


def _mmd_rbf(left: np.ndarray, right: np.ndarray, *, max_rows: int, rng: np.random.Generator) -> float:
    count = min(len(left), len(right), max_rows)
    if count < 2:
        return 0.0
    left_idx = rng.choice(len(left), size=count, replace=False)
    right_idx = rng.choice(len(right), size=count, replace=False)
    left_sample = left[left_idx]
    right_sample = right[right_idx]
    pooled = np.vstack([left_sample, right_sample])
    scale = pooled.std(axis=0)
    scale[scale == 0] = 1.0
    left_sample = (left_sample - pooled.mean(axis=0)) / scale
    right_sample = (right_sample - pooled.mean(axis=0)) / scale
    gamma = _median_gamma(np.vstack([left_sample, right_sample]))
    k_xx = _rbf_kernel(left_sample, left_sample, gamma)
    k_yy = _rbf_kernel(right_sample, right_sample, gamma)
    k_xy = _rbf_kernel(left_sample, right_sample, gamma)
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    denom = count * (count - 1)
    value = k_xx.sum() / denom + k_yy.sum() / denom - 2.0 * k_xy.mean()
    return float(max(0.0, value))


def _js_divergence(left: pd.Series, right: pd.Series) -> float:
    left_counts = left.value_counts(normalize=True)
    right_counts = right.value_counts(normalize=True)
    labels = sorted(set(left_counts.index) | set(right_counts.index), key=str)
    p = np.array([left_counts.get(label, 0.0) for label in labels], dtype=float)
    q = np.array([right_counts.get(label, 0.0) for label in labels], dtype=float)
    m = 0.5 * (p + q)
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def _permutation_p_value(
    left: np.ndarray,
    right: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    observed = abs(float(np.mean(left) - np.mean(right)))
    pooled = np.concatenate([left, right])
    count_left = len(left)
    exceed = 1
    for _ in range(permutations):
        rng.shuffle(pooled)
        diff = abs(float(np.mean(pooled[:count_left]) - np.mean(pooled[count_left:])))
        if diff >= observed:
            exceed += 1
    return exceed / (permutations + 1)


def main() -> None:
    sys.path.insert(0, str(ROOT / "code/src"))

    from steganalysis.samples import FEATURE_COLUMNS

    rng = np.random.default_rng(20260628)
    records_path = ROOT / "results/tables/phase7_steganalysis_records.csv"
    records = pd.read_csv(records_path)
    feature_columns = [column for column in FEATURE_COLUMNS if column in records.columns]
    discrete_columns = [
        column
        for column in (
            "is_top_action",
            "unseen_context",
            "unseen_destination",
            "same_as_previous",
            "self_loop",
            "embedding_feasible",
        )
        if column in records.columns
    ]
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "campaign": "phase11_distribution_audit",
        "source_records": str(records_path.relative_to(ROOT)),
        "feature_columns": feature_columns,
        "discrete_columns": discrete_columns,
        "datasets": {},
    }
    for (dataset, split), group in records.groupby(["dataset", "split"], sort=True):
        natural = group.loc[group["label"] == 0]
        stego = group.loc[group["label"] == 1]
        left = natural[feature_columns].to_numpy(dtype=float)
        right = stego[feature_columns].to_numpy(dtype=float)
        feature_rows = []
        for feature in feature_columns:
            natural_values = natural[feature].to_numpy(dtype=float)
            stego_values = stego[feature].to_numpy(dtype=float)
            feature_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "feature": feature,
                    "wasserstein": _wasserstein_1d(natural_values, stego_values),
                    "mean_natural": float(np.mean(natural_values)),
                    "mean_stego": float(np.mean(stego_values)),
                    "abs_mean_difference": abs(float(np.mean(natural_values) - np.mean(stego_values))),
                    "permutation_p": _permutation_p_value(
                        natural_values.copy(),
                        stego_values.copy(),
                        permutations=199,
                        rng=rng,
                    ),
                }
            )
        rows.extend(feature_rows)
        js_values = {
            feature: _js_divergence(natural[feature], stego[feature])
            for feature in discrete_columns
        }
        mmd = _mmd_rbf(left, right, max_rows=2000, rng=rng)
        dataset_summary = {
            "rows_per_label": int(min(len(natural), len(stego))),
            "mmd_rbf": mmd,
            "mean_wasserstein": float(np.mean([row["wasserstein"] for row in feature_rows])),
            "max_wasserstein": float(max(row["wasserstein"] for row in feature_rows)),
            "max_abs_mean_difference": float(max(row["abs_mean_difference"] for row in feature_rows)),
            "min_permutation_p": float(min(row["permutation_p"] for row in feature_rows)),
            "mean_discrete_js_bits": float(np.mean(list(js_values.values()))) if js_values else 0.0,
            "max_discrete_js_bits": float(max(js_values.values())) if js_values else 0.0,
            "discrete_js_bits": js_values,
        }
        summary["datasets"].setdefault(dataset, {})[split] = dataset_summary

    output_dir = ROOT / "results/tables"
    csv_output = output_dir / "phase11_distribution_audit.csv"
    json_output = output_dir / "phase11_distribution_audit.json"
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

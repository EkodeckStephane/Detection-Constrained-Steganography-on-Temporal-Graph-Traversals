from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

INTERNAL_COLUMNS = [
    "bits_consumed",
    "local_total_variation",
    "local_kl_bits",
]


def _load_records() -> pd.DataFrame:
    path = ROOT / "results/tables/phase9_adaptive_steganalysis_records.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Run code/scripts/run_phase9_adaptive_steganalysis.py before the Phase 12 audit."
        )
    return pd.read_csv(path)


def _detectors(seed: int) -> dict[str, Any]:
    return {
        "gradient_boosting": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=300,
                        max_depth=4,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=1,
        ),
        "mlp": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        alpha=1e-3,
                        max_iter=400,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def _auc(model: Any, train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> float:
    model.fit(train[columns].to_numpy(dtype=float), train["label"].to_numpy(dtype=int))
    scores = model.predict_proba(test[columns].to_numpy(dtype=float))[:, 1]
    return float(roc_auc_score(test["label"].to_numpy(dtype=int), scores))


def _pearson(values: pd.Series, labels: pd.Series) -> float:
    if float(values.std()) == 0.0:
        return 0.0
    value = float(np.corrcoef(values.to_numpy(dtype=float), labels.to_numpy(dtype=float))[0, 1])
    return 0.0 if np.isnan(value) else value


def _finite(value: float) -> float:
    value = float(value)
    return 0.0 if np.isnan(value) or np.isinf(value) else value


def main() -> None:
    sys.path.insert(0, str(ROOT / "code/src"))

    from steganalysis.adaptive_eve import BOUNDED_WHITE_BOX_FEATURE_COLUMNS

    records = _load_records()
    rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "campaign": "phase12_oracle_leakage_audit",
        "source_records": "results/tables/phase9_adaptive_steganalysis_records.csv",
        "datasets": {},
    }
    seed = 20260628
    bounded_columns = [column for column in BOUNDED_WHITE_BOX_FEATURE_COLUMNS if column in records.columns]
    oracle_columns = [*bounded_columns, *[column for column in INTERNAL_COLUMNS if column in records.columns]]
    internal_columns = [column for column in INTERNAL_COLUMNS if column in records.columns]

    for dataset, data in records.groupby("dataset", sort=False):
        train = data.loc[data["split"] == "validation"].copy()
        test = data.loc[data["split"] == "test"].copy()
        dataset_summary: dict[str, Any] = {
            "correlations": {},
            "detectors": {},
        }
        for column in internal_columns:
            pearson = _pearson(test[column], test["label"])
            spearman = _finite(test[[column, "label"]].corr(method="spearman").iloc[0, 1])
            stego_mean = float(test.loc[test["label"] == 1, column].mean())
            natural_mean = float(test.loc[test["label"] == 0, column].mean())
            corr_record = {
                "dataset": dataset,
                "feature": column,
                "pearson_with_label": pearson,
                "spearman_with_label": spearman,
                "natural_mean": natural_mean,
                "stego_mean": stego_mean,
                "absolute_mean_gap": abs(stego_mean - natural_mean),
            }
            corr_rows.append(corr_record)
            dataset_summary["correlations"][column] = corr_record

        best_bounded = 0.0
        best_oracle = 0.0
        best_internal_only = 0.0
        for name, detector in _detectors(seed).items():
            bounded_auc = _auc(detector, train, test, bounded_columns)
            oracle_auc = _auc(_detectors(seed)[name], train, test, oracle_columns)
            internal_auc = _auc(_detectors(seed)[name], train, test, internal_columns)
            best_bounded = max(best_bounded, bounded_auc)
            best_oracle = max(best_oracle, oracle_auc)
            best_internal_only = max(best_internal_only, internal_auc)
            detector_record = {
                "dataset": dataset,
                "detector": name,
                "bounded_auc": bounded_auc,
                "oracle_auc": oracle_auc,
                "internal_only_auc": internal_auc,
                "oracle_minus_bounded_auc": oracle_auc - bounded_auc,
                "internal_advantage_over_random": internal_auc - 0.5,
            }
            rows.append(detector_record)
            dataset_summary["detectors"][name] = detector_record

        dataset_summary["best_bounded_auc"] = best_bounded
        dataset_summary["best_oracle_auc"] = best_oracle
        dataset_summary["best_internal_only_auc"] = best_internal_only
        dataset_summary["empirical_oracle_epsilon"] = max(0.0, best_oracle - best_bounded)
        dataset_summary["empirical_internal_epsilon"] = max(0.0, best_internal_only - 0.5)
        summary["datasets"][dataset] = dataset_summary

    output_dir = ROOT / "results/tables"
    with (output_dir / "phase12_oracle_leakage_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "phase12_oracle_leakage_correlations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(corr_rows[0]))
        writer.writeheader()
        writer.writerows(corr_rows)
    (output_dir / "phase12_oracle_leakage_audit.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

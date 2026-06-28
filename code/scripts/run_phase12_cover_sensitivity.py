from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _load_cover_metrics() -> pd.DataFrame:
    temporal = pd.read_csv(ROOT / "results/tables/phase4_temporal_cover_model.csv")
    temporal = temporal.loc[temporal["split"] == "test"].copy()
    baseline_path = ROOT / "results/tables/phase4_cover_model_baseline.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        baseline = baseline.loc[baseline["split"] == "test"].copy()
        baseline["model_family"] = "source_only_backoff"
        temporal["model_family"] = "source_previous_destination_backoff"
        common = sorted(set(temporal.columns).intersection(set(baseline.columns)))
        return pd.concat([baseline[common], temporal[common]], ignore_index=True)
    temporal["model_family"] = "source_previous_destination_backoff"
    return temporal


def _safe_corr(frame: pd.DataFrame, left: str, right: str) -> float:
    subset = frame[[left, right]].dropna()
    if len(subset) < 2:
        return 0.0
    if float(subset[left].std()) == 0.0 or float(subset[right].std()) == 0.0:
        return 0.0
    return float(subset[left].corr(subset[right], method="spearman"))


def main() -> None:
    cover = _load_cover_metrics()
    sweep = pd.read_csv(ROOT / "results/tables/phase12_capacity_detectability_summary.csv")
    conservative = sweep.loc[sweep["point"] == "conservative"].copy()
    best_under_budget = (
        sweep.loc[sweep["max_public_auc"] <= 0.60]
        .sort_values(["dataset", "bits_per_transition"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .head(1)
        .rename(
            columns={
                "point": "best_budget_point",
                "bits_per_transition": "best_budget_bits_per_transition",
                "max_public_auc": "best_budget_max_public_auc",
            }
        )
    )
    conservative = conservative.rename(
        columns={
            "bits_per_transition": "conservative_bits_per_transition",
            "max_public_auc": "conservative_max_public_auc",
            "embed_rate": "conservative_embed_rate",
        }
    )
    merged = cover.merge(
        conservative[
            [
                "dataset",
                "conservative_bits_per_transition",
                "conservative_embed_rate",
                "conservative_max_public_auc",
            ]
        ],
        on="dataset",
        how="left",
    ).merge(
        best_under_budget[
            [
                "dataset",
                "best_budget_point",
                "best_budget_bits_per_transition",
                "best_budget_max_public_auc",
            ]
        ],
        on="dataset",
        how="left",
    )
    rows = merged.to_dict(orient="records")
    correlations: dict[str, Any] = {}
    primary = merged.loc[merged["model_family"] == "source_previous_destination_backoff"].copy()
    for cover_metric in [
        "top1_accuracy",
        "mean_nll_bits",
        "perplexity",
        "unseen_context_fraction",
        "unseen_destination_fraction",
    ]:
        if cover_metric in primary.columns:
            correlations[cover_metric] = {
                "with_conservative_bits": _safe_corr(
                    primary, cover_metric, "conservative_bits_per_transition"
                ),
                "with_conservative_auc": _safe_corr(
                    primary, cover_metric, "conservative_max_public_auc"
                ),
                "with_best_budget_bits": _safe_corr(
                    primary, cover_metric, "best_budget_bits_per_transition"
                ),
            }

    summary = {
        "campaign": "phase12_cover_sensitivity",
        "sources": [
            "results/tables/phase4_temporal_cover_model.csv",
            "results/tables/phase4_cover_model_baseline.csv",
            "results/tables/phase12_capacity_detectability_summary.csv",
        ],
        "correlations": correlations,
        "records": rows,
    }
    output_dir = ROOT / "results/tables"
    with (output_dir / "phase12_cover_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "phase12_cover_sensitivity.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "results/tables"
VARIANT_DIR = RESULTS_DIR / "variant_neural_cover"


def _round(value: float | None, decimals: int = 4) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), decimals)


def _max_auc_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby("dataset")
    rows = []
    for dataset, group in grouped:
        max_row = group.loc[group["auc_mean"].idxmax()]
        rows.append({
            "dataset": dataset,
            "max_auc_detector": max_row["detector"],
            "max_auc": _round(max_row["auc_mean"]),
            "attempted_bits_per_transition": _round(max_row["attempted_bits_per_transition_mean"]),
        })
    return pd.DataFrame(rows)


def build_sota_temporal_table() -> pd.DataFrame:
    """Compare BIND/AdaBIND vs proposition on the temporal datasets (one row per dataset)."""
    lee = pd.read_csv(VARIANT_DIR / "lee_baselines_temporal.csv")
    # Keep the largest AdaBIND payload per dataset.
    lee_ada = lee[lee["method"] == "AdaBIND"].copy()
    lee_ada["payload_bpe"] = pd.to_numeric(lee_ada["payload_bpe"], errors="coerce")
    lee_best = lee_ada.loc[lee_ada.groupby("dataset")["payload_bpe"].idxmax()]
    lee_best = lee_best.rename(columns={
        "payload_bpe": "ada_max_payload_bpe",
        "added_edges": "ada_added_edges_at_max",
        "message_bytes": "ada_message_bytes_at_max",
    })

    prop = pd.read_csv(RESULTS_DIR / "phase7_steganalysis_multiseed.csv")
    prop_summary = _max_auc_by_dataset(prop)
    prop_summary = prop_summary.rename(columns={
        "max_auc": "proposition_max_auc",
        "attempted_bits_per_transition": "proposition_bits_per_transition",
    })

    merged = pd.merge(
        lee_best[["dataset", "n_edges", "ada_max_payload_bpe", "ada_added_edges_at_max", "ada_message_bytes_at_max"]],
        prop_summary,
        on="dataset",
        how="outer",
    )
    for col in ("ada_max_payload_bpe", "ada_added_edges_at_max", "ada_message_bytes_at_max"):
        merged[col] = merged[col].apply(lambda x: _round(x) if col != "ada_added_edges_at_max" else (int(x) if pd.notna(x) else None))
    merged["proposition_bits_per_transition"] = merged["proposition_bits_per_transition"].apply(lambda x: _round(x))
    merged["proposition_max_auc"] = merged["proposition_max_auc"].apply(lambda x: _round(x))
    return merged


def build_sota_terrorists_table() -> pd.DataFrame:
    """Compare BIND/AdaBIND/BYNIS vs proposition on terrorists-911."""
    lee = pd.read_csv(RESULTS_DIR / "phase2_lee_reproduction.csv")
    # Take BIND rows and BYNIS rows; AdaBIND single row.
    rows = []
    for _, row in lee.iterrows():
        method = str(row["method"])
        if method == "BIND":
            rows.append({
                "method": f"BIND ({int(row['message_bytes'])} B)",
                "payload_bpe": _round(row["payload_bpe"]),
                "added_edges": 0,
                "topology_modified": False,
            })
        elif method == "BYNIS":
            rows.append({
                "method": f"BYNIS ({int(row['extra_edges'])} extra)",
                "payload_bpe": _round(row["payload_bpe"]),
                "added_edges": int(row["extra_edges"]),
                "topology_modified": int(row["extra_edges"]) > 0,
            })
        elif method == "AdaBIND":
            rows.append({
                "method": "AdaBIND",
                "payload_bpe": _round(row.get("payload_bpe")),
                "added_edges": int(row.get("added_edges", 0)) if pd.notna(row.get("added_edges")) else None,
                "topology_modified": True,
            })
    lee_df = pd.DataFrame(rows)

    prop = pd.read_csv(VARIANT_DIR / "proposition_on_terrorists.csv")
    max_auc_row = prop.loc[prop["auc"].idxmax()]
    prop_row = pd.DataFrame([{
        "method": "Proposition (backoff)",
        "payload_bpe": _round(prop.iloc[0]["attempted_bits_per_transition"]),
        "added_edges": 0,
        "topology_modified": False,
        "max_auc": _round(max_auc_row["auc"]),
    }])

    # Lee baselines do not have AUC from phase2; leave blank.
    lee_df["max_auc"] = np.nan
    combined = pd.concat([lee_df, prop_row], ignore_index=True)
    combined["max_auc"] = pd.to_numeric(combined["max_auc"], errors="coerce")
    return combined


def build_variant_table() -> pd.DataFrame | None:
    """Compare cover models within the variant pipeline."""
    path = VARIANT_DIR / "variant_cover_models.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    summary = []
    grouped = df.groupby(["dataset", "cover_model"])
    for (dataset, cover_model), group in grouped:
        max_row = group.loc[group["auc_mean"].idxmax()]
        summary.append({
            "dataset": dataset,
            "cover_model": cover_model,
            "attempted_bits_per_transition": _round(max_row["attempted_bits_per_transition_mean"]),
            "max_auc": _round(max_row["auc_mean"]),
            "max_auc_detector": max_row["detector"],
        })
    return pd.DataFrame(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison tables for SOTA and variant cover models.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VARIANT_DIR,
        help="Directory where comparison tables are written.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sota_temporal = build_sota_temporal_table()
    sota_temporal.to_csv(args.output_dir / "sota_comparison_temporal_datasets.csv", index=False)

    sota_terrorists = build_sota_terrorists_table()
    sota_terrorists.to_csv(args.output_dir / "sota_comparison_terrorists_911.csv", index=False)

    variant = build_variant_table()
    if variant is not None:
        variant.to_csv(args.output_dir / "variant_cover_models_summary.csv", index=False)

    combined: dict[str, Any] = {
        "sota_temporal_datasets": sota_temporal.to_dict(orient="records"),
        "sota_terrorists_911": sota_terrorists.to_dict(orient="records"),
    }
    if variant is not None:
        combined["variant_cover_models"] = variant.to_dict(orient="records")

    (args.output_dir / "comparison_tables.json").write_text(
        json.dumps(combined, indent=2),
        encoding="utf-8",
    )

    print("=== SOTA temporal datasets ===")
    print(sota_temporal.to_string(index=False))
    print("\n=== SOTA terrorists-911 ===")
    print(sota_terrorists.to_string(index=False))
    if variant is not None:
        print("\n=== Variant cover models ===")
        print(variant.to_string(index=False))


if __name__ == "__main__":
    main()

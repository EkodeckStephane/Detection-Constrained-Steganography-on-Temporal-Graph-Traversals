from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/src"))

from models.cover_model import (
    BackoffCoverModel,
    CoverModel,
    train_neural_sequence_cover_model,
    train_temporal_gnn_cover_model,
)
from steganalysis.detectors import fit_detector, metrics_to_dict, score_detector
from steganalysis.samples import SampleConfig, feature_matrix, make_steganalysis_records


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _load_phase3_catalog() -> list[dict[str, Any]]:
    with (ROOT / "results/tables/phase3_dataset_statistics.json").open(encoding="utf-8") as handle:
        return json.load(handle)["datasets"]


def _event_catalog(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in [*_load_phase3_catalog(), *config.get("event_datasets", [])]}


def _read_event_dataset(record: dict[str, Any], *, max_rows_per_split: int) -> pd.DataFrame:
    columns = ["source", "destination", "timestamp", "split"]
    counts = {"train": 0, "validation": 0, "test": 0}
    frames = []
    parquet = pq.ParquetFile(ROOT / record["processed_path"])
    for batch in parquet.iter_batches(batch_size=50_000, columns=columns):
        chunk = batch.to_pandas()
        selected = []
        for split in ("train", "validation", "test"):
            remaining = max_rows_per_split - counts[split]
            if remaining <= 0:
                continue
            subset = chunk.loc[chunk["split"] == split].head(remaining)
            if not subset.empty:
                counts[split] += len(subset)
                selected.append(subset)
        if selected:
            frames.append(pd.concat(selected, ignore_index=True))
        if all(value >= max_rows_per_split for value in counts.values()):
            break
    if not frames:
        raise ValueError(f"No rows read for {record['id']}")
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["timestamp"], kind="stable")
        .reset_index(drop=True)
    )


def _build_cover_model(
    cover_config: dict[str, Any],
    frame: pd.DataFrame,
    seed: int,
) -> CoverModel:
    kind = cover_config["kind"]
    params = cover_config.get("params", {})
    if kind == "backoff":
        return BackoffCoverModel(
            prior_strength=float(params.get("prior_strength", 8.0)),
            top_k=int(params.get("top_k", 32)),
        ).fit(frame)
    if kind in ("gru", "transformer"):
        return train_neural_sequence_cover_model(
            frame,
            kind=kind,
            context_length=int(params["context_length"]),
            embedding_dim=int(params["embedding_dim"]),
            hidden_dim=int(params["hidden_dim"]),
            epochs=int(params["epochs"]),
            batch_size=int(params["batch_size"]),
            learning_rate=float(params["learning_rate"]),
            seed=seed,
            top_k=int(params.get("top_k", 32)),
        )
    if kind == "temporal_gnn":
        return train_temporal_gnn_cover_model(
            frame,
            embedding_dim=int(params["embedding_dim"]),
            hidden_dim=int(params["hidden_dim"]),
            epochs=int(params["epochs"]),
            batch_size=int(params["batch_size"]),
            learning_rate=float(params["learning_rate"]),
            seed=seed,
            patience=int(params.get("patience", 5)),
            dropout=float(params.get("dropout", 0.2)),
            top_k=int(params.get("top_k", 32)),
        )
    raise ValueError(f"Unknown cover model kind: {kind}")


def _metric_records(summary: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows = []
    for dataset, detectors in summary["detectors"].items():
        comm = summary["communication"][dataset]
        for detector, metrics in detectors.items():
            row = {
                "seed": seed,
                "dataset": dataset,
                "detector": detector,
                "cover_model": summary["cover_model"],
                "attempted_bits_per_transition": comm["attempted_bits_per_transition"],
            }
            row.update(metrics)
            rows.append(row)
    return rows


def run_single_configuration(
    config: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    cover_config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    sample_config = SampleConfig(
        max_bits_per_transition=int(config["codec"]["max_bits_per_transition"]),
        seed=seed,
        max_local_total_variation=float(config["codec"]["max_local_total_variation"]),
        max_local_kl_bits=float(config["codec"]["max_local_kl_bits"]),
        min_entropy_bits=float(config["codec"]["min_entropy_bits"]),
        cover_when_unsafe=bool(config["codec"]["cover_when_unsafe"]),
        codec_backend=str(config["codec"].get("codec_backend", "range")),
        min_encoded_probability=float(config["codec"].get("min_encoded_probability", 0.0)),
        max_encoded_surprise_bits=float(config["codec"].get("max_encoded_surprise_bits", "inf")),
        max_encoded_rank_fraction=float(config["codec"].get("max_encoded_rank_fraction", 1.0)),
        require_encoded_top_action=bool(config["codec"].get("require_encoded_top_action", False)),
        require_encoded_self_loop=bool(config["codec"].get("require_encoded_self_loop", False)),
    )
    all_records = []
    summary: dict[str, Any] = {
        "campaign": config["campaign"],
        "date": str(config["date"]),
        "cover_model": cover_config["name"],
        "codec": config["codec"],
        "seed": seed,
        "detectors": {},
        "communication": {},
    }
    for dataset_id in config["datasets"]:
        frame = _read_event_dataset(
            catalog[dataset_id],
            max_rows_per_split=int(config["max_rows_per_split"]),
        )
        cover_model = _build_cover_model(cover_config, frame, seed)
        validation = make_steganalysis_records(
            cover_model,
            frame.loc[frame["split"] == "validation"],
            split="validation",
            config=sample_config,
        )
        test = make_steganalysis_records(
            cover_model,
            frame.loc[frame["split"] == "test"],
            split="test",
            config=sample_config,
        )
        validation.insert(0, "dataset", dataset_id)
        test.insert(0, "dataset", dataset_id)
        all_records.extend([validation, test])
        stego_test = test.loc[test["label"] == 1]
        summary["communication"][dataset_id] = {
            "test_transitions": int(len(stego_test)),
            "embed_rate": float((stego_test["stego_mode"] == "EMBED").mean()),
            "cover_rate": float((stego_test["stego_mode"] == "COVER").mean()),
            "attempted_bits_per_transition": float(stego_test["bits_consumed"].mean()),
            "mean_local_total_variation": float(stego_test["local_total_variation"].mean()),
            "mean_local_kl_bits": float(stego_test["local_kl_bits"].mean()),
        }
        x_train, y_train = feature_matrix(validation)
        x_test, y_test = feature_matrix(test)
        summary["detectors"][dataset_id] = {}
        for detector_name in config["detectors"]:
            detector = fit_detector(detector_name, x_train, y_train, seed=seed)
            summary["detectors"][dataset_id][detector_name] = metrics_to_dict(
                score_detector(detector, x_test, y_test)
            )
    records = pd.concat(all_records, ignore_index=True)
    records.insert(0, "seed", seed)
    records.insert(1, "cover_model", cover_config["name"])
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run steganalysis with multiple cover models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/variant_neural_cover/variant_cover_models.yaml",
        help="Path to the variant cover models YAML config.",
    )
    args = parser.parse_args()
    config = _load_yaml(args.config)
    catalog = _event_catalog(config)
    seeds = [int(s) for s in config.get("seeds", [config["seed"]])]
    cover_configs = config["cover_models"]

    output_dir = ROOT / "results/tables/variant_neural_cover"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_per_seed_summaries: list[dict[str, Any]] = []
    all_per_seed_records: list[pd.DataFrame] = []

    for cover_config in cover_configs:
        print(f"\n=== Cover model: {cover_config['name']} ===", file=sys.stderr)
        for seed in seeds:
            print(f"Running seed {seed} ...", file=sys.stderr)
            summary, records = run_single_configuration(config, catalog, cover_config, seed)
            all_per_seed_summaries.append(summary)
            all_per_seed_records.append(records)

    metric_rows: list[dict[str, Any]] = []
    for summary in all_per_seed_summaries:
        metric_rows.extend(_metric_records(summary, summary["seed"]))
    metrics_df = pd.DataFrame(metric_rows)

    aggregated = []
    grouped = metrics_df.groupby(["dataset", "cover_model", "detector"])
    for (dataset, cover_model, detector), group in grouped:
        row: dict[str, Any] = {
            "dataset": dataset,
            "cover_model": cover_model,
            "detector": detector,
            "attempted_bits_per_transition_mean": float(group["attempted_bits_per_transition"].mean()),
            "attempted_bits_per_transition_std": float(group["attempted_bits_per_transition"].std()),
            "n_seeds": int(len(group)),
        }
        for metric in ("auc", "accuracy", "balanced_accuracy", "eer"):
            if metric in group.columns:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = float(group[metric].std())
                row[f"{metric}_min"] = float(group[metric].min())
                row[f"{metric}_max"] = float(group[metric].max())
        aggregated.append(row)

    aggregated_df = pd.DataFrame(aggregated)
    all_records = pd.concat(all_per_seed_records, ignore_index=True)

    all_records.to_csv(output_dir / "variant_cover_models_records.csv", index=False)
    aggregated_df.to_csv(output_dir / "variant_cover_models.csv", index=False)
    json_output = output_dir / "variant_cover_models.json"
    json_output.write_text(
        json.dumps(
            {
                "campaign": config["campaign"],
                "date": str(config["date"]),
                "codec": config["codec"],
                "seeds": seeds,
                "cover_models": cover_configs,
                "aggregated": aggregated,
                "per_seed": all_per_seed_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(aggregated_df.to_string(index=False))


if __name__ == "__main__":
    main()

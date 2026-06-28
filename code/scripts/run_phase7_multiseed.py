from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]


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


def _metric_records(summary: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows = []
    for dataset, detectors in summary["detectors"].items():
        comm = summary["communication"][dataset]
        for detector, metrics in detectors.items():
            row = {
                "seed": seed,
                "dataset": dataset,
                "detector": detector,
                "attempted_bits_per_transition": comm["attempted_bits_per_transition"],
            }
            row.update(metrics)
            rows.append(row)
    return rows


def run_single_seed(
    config: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    sys.path.insert(0, str(ROOT / "code/src"))

    from models.temporal import TemporalBackoffModel
    from steganalysis.detectors import fit_detector, metrics_to_dict, score_detector
    from steganalysis.samples import SampleConfig, feature_matrix, make_steganalysis_records

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
        "model": config["model"],
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
        model = TemporalBackoffModel(
            prior_strength=float(config["model"]["prior_strength"]),
            top_k=int(config["model"]["top_k"]),
        ).fit(frame.loc[frame["split"] == "train"])
        validation = make_steganalysis_records(
            model,
            frame.loc[frame["split"] == "validation"],
            split="validation",
            config=sample_config,
        )
        test = make_steganalysis_records(
            model,
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
    return summary, records


def main() -> None:
    config = _load_yaml(ROOT / "experiments/real_world/phase7_steganalysis.yaml")
    catalog = _event_catalog(config)
    seeds = [int(s) for s in config.get("seeds", [config["seed"]])]
    per_seed_summaries = []
    per_seed_records = []
    for seed in seeds:
        print(f"Running seed {seed} ...", file=sys.stderr)
        summary, records = run_single_seed(config, catalog, seed)
        per_seed_summaries.append(summary)
        per_seed_records.append(records)

    metric_rows = []
    for summary in per_seed_summaries:
        metric_rows.extend(_metric_records(summary, summary["seed"]))
    metrics_df = pd.DataFrame(metric_rows)

    aggregated = []
    grouped = metrics_df.groupby(["dataset", "detector"])
    for (dataset, detector), group in grouped:
        row: dict[str, Any] = {
            "dataset": dataset,
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
    all_records = pd.concat(per_seed_records, ignore_index=True)

    output_dir = ROOT / "results/tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records.to_csv(output_dir / "phase7_steganalysis_multiseed_records.csv", index=False)
    aggregated_df.to_csv(output_dir / "phase7_steganalysis_multiseed.csv", index=False)
    json_output = output_dir / "phase7_steganalysis_multiseed.json"
    json_output.write_text(
        json.dumps(
            {
                "campaign": config["campaign"],
                "date": str(config["date"]),
                "model": config["model"],
                "codec": config["codec"],
                "seeds": seeds,
                "aggregated": aggregated,
                "per_seed": per_seed_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(aggregated_df.to_string(index=False))


if __name__ == "__main__":
    main()

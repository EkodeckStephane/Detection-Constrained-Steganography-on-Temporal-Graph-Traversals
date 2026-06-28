from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]


OPERATING_POINTS = [
    {
        "point": "ultra_stealth",
        "max_bits_per_transition": 1,
        "max_local_total_variation": 0.10,
        "max_local_kl_bits": 0.15,
        "min_entropy_bits": 1.00,
        "max_encoded_rank_fraction": 0.25,
    },
    {
        "point": "conservative",
        "max_bits_per_transition": 1,
        "max_local_total_variation": 0.35,
        "max_local_kl_bits": 0.60,
        "min_entropy_bits": 0.50,
        "max_encoded_rank_fraction": 0.50,
    },
    {
        "point": "balanced",
        "max_bits_per_transition": 1,
        "max_local_total_variation": 0.50,
        "max_local_kl_bits": 0.90,
        "min_entropy_bits": 0.25,
        "max_encoded_rank_fraction": 0.75,
    },
    {
        "point": "open_rank",
        "max_bits_per_transition": 1,
        "max_local_total_variation": 0.75,
        "max_local_kl_bits": 1.25,
        "min_entropy_bits": 0.00,
        "max_encoded_rank_fraction": 1.00,
    },
    {
        "point": "two_bit_probe",
        "max_bits_per_transition": 2,
        "max_local_total_variation": 0.75,
        "max_local_kl_bits": 1.50,
        "min_entropy_bits": 0.00,
        "max_encoded_rank_fraction": 1.00,
    },
]


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
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp"], kind="stable").reset_index(drop=True)


def _sample_config(base_codec: dict[str, Any], point: dict[str, Any], seed: int) -> Any:
    from steganalysis.samples import SampleConfig

    codec = {**base_codec, **point}
    return SampleConfig(
        max_bits_per_transition=int(codec["max_bits_per_transition"]),
        seed=seed,
        max_local_total_variation=float(codec["max_local_total_variation"]),
        max_local_kl_bits=float(codec["max_local_kl_bits"]),
        min_entropy_bits=float(codec["min_entropy_bits"]),
        cover_when_unsafe=bool(codec.get("cover_when_unsafe", True)),
        codec_backend=str(codec.get("codec_backend", "range")),
        min_encoded_probability=float(codec.get("min_encoded_probability", 0.0)),
        max_encoded_surprise_bits=float(codec.get("max_encoded_surprise_bits", "inf")),
        max_encoded_rank_fraction=float(codec.get("max_encoded_rank_fraction", 1.0)),
        require_encoded_top_action=bool(codec.get("require_encoded_top_action", False)),
        require_encoded_self_loop=bool(codec.get("require_encoded_self_loop", False)),
    )


def _plot_tradeoff(summary_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(summary_rows)
    output_dir = ROOT / "results/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for dataset, data in frame.groupby("dataset", sort=False):
        ordered = data.sort_values("bits_per_transition")
        ax.plot(
            ordered["bits_per_transition"],
            ordered["max_public_auc"],
            marker="o",
            linewidth=1.6,
            label=str(dataset),
        )
    ax.axhline(0.60, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Payload (bits per transition)")
    ax.set_ylabel("Maximum public-feature AUC")
    ax.set_title("Capacity-detectability operating curve")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / "phase12_capacity_detectability_sweep.pdf")
    fig.savefig(output_dir / "phase12_capacity_detectability_sweep.png", dpi=220)
    plt.close(fig)


def main() -> None:
    sys.path.insert(0, str(ROOT / "code/src"))

    from models.temporal import TemporalBackoffModel
    from steganalysis.detectors import fit_detector, metrics_to_dict, score_detector
    from steganalysis.samples import feature_matrix, make_steganalysis_records

    config = _load_yaml(ROOT / "experiments/real_world/phase7_steganalysis.yaml")
    catalog = _event_catalog(config)
    seed = int(config["seed"])
    detectors = list(config["detectors"])
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "campaign": "phase12_capacity_detectability_sweep",
        "source_config": "experiments/real_world/phase7_steganalysis.yaml",
        "operating_points": OPERATING_POINTS,
        "datasets": {},
    }

    for dataset_id in config["datasets"]:
        frame = _read_event_dataset(catalog[dataset_id], max_rows_per_split=int(config["max_rows_per_split"]))
        model = TemporalBackoffModel(
            prior_strength=float(config["model"]["prior_strength"]),
            top_k=int(config["model"]["top_k"]),
        ).fit(frame.loc[frame["split"] == "train"])
        dataset_summary = []
        for index, point in enumerate(OPERATING_POINTS):
            sample_config = _sample_config(config["codec"], point, seed + index * 101)
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
            stego = test.loc[test["label"] == 1]
            bits_per_transition = float(stego["bits_consumed"].mean())
            embed_rate = float((stego["stego_mode"] == "EMBED").mean())
            x_train, y_train = feature_matrix(validation)
            x_test, y_test = feature_matrix(test)
            detector_metrics = {}
            for detector_name in detectors:
                detector = fit_detector(detector_name, x_train, y_train, seed=seed + index)
                metrics = metrics_to_dict(score_detector(detector, x_test, y_test))
                detector_metrics[detector_name] = metrics
                rows.append(
                    {
                        "dataset": dataset_id,
                        "point": point["point"],
                        "detector": detector_name,
                        "bits_per_transition": bits_per_transition,
                        "embed_rate": embed_rate,
                        "auc": metrics["auc"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "eer": metrics["eer"],
                    }
                )
            max_auc = max(value["auc"] for value in detector_metrics.values())
            summary_row = {
                "dataset": dataset_id,
                "point": point["point"],
                "bits_per_transition": bits_per_transition,
                "embed_rate": embed_rate,
                "max_public_auc": max_auc,
                "capacity_at_auc_060": bits_per_transition if max_auc <= 0.60 else 0.0,
            }
            summary_rows.append(summary_row)
            dataset_summary.append({**summary_row, "detectors": detector_metrics})
        summary["datasets"][dataset_id] = dataset_summary

    output_dir = ROOT / "results/tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output = output_dir / "phase12_capacity_detectability_sweep.csv"
    summary_csv_output = output_dir / "phase12_capacity_detectability_summary.csv"
    json_output = output_dir / "phase12_capacity_detectability_sweep.json"
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with summary_csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    json_output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    _plot_tradeoff(summary_rows)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

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

from baselines.graph_stego import (
    BindCapacityError,
    bind_capacity_bits,
    bind_decode,
    bind_encode,
    adabind_encode,
    edge_type_counts,
)


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


def _edges_from_frame(frame: pd.DataFrame) -> list[tuple[Any, Any]]:
    """Return unique directed edges from the event stream."""
    pairs = frame[["source", "destination"]].drop_duplicates()
    return [tuple(row) for row in pairs.itertuples(index=False)]


def _message_of_size(size: int, seed: int) -> bytes:
    rng = np.random.RandomState(seed)
    return bytes(rng.randint(0, 256, size=size, dtype=np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BIND/AdaBIND on temporal event datasets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/variant_neural_cover/lee_baselines_temporal.yaml",
        help="Path to the Lee baselines temporal YAML config.",
    )
    args = parser.parse_args()
    config = _load_yaml(args.config)
    catalog = _event_catalog(config)
    seed = int(config["seed"])
    output_dir = ROOT / "results/tables/variant_neural_cover"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for dataset_id in config["datasets"]:
        frame = _read_event_dataset(
            catalog[dataset_id],
            max_rows_per_split=int(config["max_rows_per_split"]),
        )
        # Use the full observed edge set as the static cover graph.
        edges = _edges_from_frame(frame)
        n_edges = len(edges)
        n_nodes = len(set(node for edge in edges for node in edge))
        capacity = bind_capacity_bits(edges, directed=True)

        row_common = {
            "dataset": dataset_id,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "bind_capacity_bits": capacity,
            "bind_capacity_bpe": capacity / n_edges if n_edges else 0.0,
        }

        for message_bytes in config.get("message_bytes", [8, 16, 32]):
            message = _message_of_size(message_bytes, seed)
            # BIND
            try:
                bind_result = bind_encode(edges, message, seed=seed, directed=True)
                bind_recovered = bind_decode(bind_result.edges, seed=seed, directed=True)
                rows.append({
                    **row_common,
                    "method": "BIND",
                    "message_bytes": message_bytes,
                    "payload_bits": bind_result.payload_bits,
                    "payload_bpe": bind_result.payload_bits / n_edges if n_edges else 0.0,
                    "header_bits": bind_result.header_bits,
                    "selected_edges": bind_result.selected_edges,
                    "roundtrip_exact": bind_recovered == message,
                    "added_edges": 0,
                    "topology_modified": False,
                    "error": None,
                })
            except BindCapacityError as exc:
                rows.append({
                    **row_common,
                    "method": "BIND",
                    "message_bytes": message_bytes,
                    "payload_bits": None,
                    "payload_bpe": None,
                    "header_bits": None,
                    "selected_edges": None,
                    "roundtrip_exact": False,
                    "added_edges": None,
                    "topology_modified": False,
                    "error": str(exc),
                })

            # AdaBIND
            try:
                ada_result = adabind_encode(
                    edges,
                    message,
                    seed=seed,
                    max_iterations=int(config["adabind"]["max_iterations"]),
                    sampled_edges=int(config["adabind"]["sampled_edges"]),
                    extra_target_edges=int(config["adabind"]["extra_target_edges"]),
                )
                ada_recovered = bind_decode(ada_result.edges, seed=seed, directed=True)
                rows.append({
                    **row_common,
                    "method": "AdaBIND",
                    "message_bytes": message_bytes,
                    "payload_bits": ada_result.payload_bits,
                    "payload_bpe": ada_result.payload_bits / len(ada_result.edges) if ada_result.edges else 0.0,
                    "header_bits": ada_result.header_bits,
                    "selected_edges": None,
                    "roundtrip_exact": ada_recovered == message,
                    "added_edges": len(ada_result.added_edges),
                    "topology_modified": len(ada_result.added_edges) > 0,
                    "error": None,
                })
            except BindCapacityError as exc:
                rows.append({
                    **row_common,
                    "method": "AdaBIND",
                    "message_bytes": message_bytes,
                    "payload_bits": None,
                    "payload_bpe": None,
                    "header_bits": None,
                    "selected_edges": None,
                    "roundtrip_exact": False,
                    "added_edges": None,
                    "topology_modified": True,
                    "error": str(exc),
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "lee_baselines_temporal.csv", index=False)
    json_output = output_dir / "lee_baselines_temporal.json"
    json_output.write_text(
        json.dumps(
            {
                "campaign": config["campaign"],
                "date": str(config["date"]),
                "seed": seed,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import math
from collections.abc import Hashable, Sequence
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


def _entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values[values > 0]
    return float(-(values * np.log2(values)).sum())


def _rank_features(source: Hashable, action: Hashable, candidates: Sequence[Any]) -> dict[str, float | int]:
    probabilities = {candidate.action: candidate.probability for candidate in candidates}
    ranked_actions = [candidate.action for candidate in candidates]
    probability = probabilities[action]
    rank = ranked_actions.index(action) + 1
    return {
        "probability": probability,
        "surprise_bits": -math.log2(max(probability, np.finfo(float).tiny)),
        "rank_fraction": rank / max(1, len(ranked_actions)),
        "is_top_action": int(rank == 1),
        "self_loop": int(source == action),
    }


def _allowed_candidates(source: Hashable, candidates: Sequence[Any], codec_config: dict[str, Any]) -> list[Any]:
    allowed = []
    for candidate in candidates:
        features = _rank_features(source, candidate.action, candidates)
        if features["probability"] < float(codec_config.get("min_encoded_probability", 0.0)):
            continue
        if features["surprise_bits"] > float(codec_config.get("max_encoded_surprise_bits", "inf")):
            continue
        if features["rank_fraction"] > float(codec_config.get("max_encoded_rank_fraction", 1.0)):
            continue
        if bool(codec_config.get("require_encoded_top_action", False)) and not features["is_top_action"]:
            continue
        if bool(codec_config.get("require_encoded_self_loop", False)) and not features["self_loop"]:
            continue
        allowed.append(candidate)
    total = sum(candidate.probability for candidate in allowed)
    if total <= 0:
        return []
    from stego.coding import Candidate

    return [Candidate(candidate.action, candidate.probability / total) for candidate in allowed]


def _local_encode(bits: Sequence[int], candidates: Sequence[Any], codec_config: dict[str, Any]) -> Any:
    from stego.coding import encode_next_action, encode_next_action_range

    backend = str(codec_config.get("codec_backend", "range"))
    max_bits = int(codec_config["max_bits_per_transition"])
    if backend == "range":
        return encode_next_action_range(bits, candidates, max_bits=max_bits)
    if backend == "quantized":
        return encode_next_action(bits, candidates, max_bits=max_bits)
    raise ValueError(f"Unknown codec backend: {backend}")


def _arithmetic_bits(
    candidate_steps: Sequence[Sequence[Any]],
    *,
    rng: np.random.Generator,
    precision_bits: int,
    block_steps: int,
) -> int:
    from stego.coding import decode_trace_arithmetic_prefix, encode_trace_arithmetic

    recovered = 0
    for offset in range(0, len(candidate_steps), block_steps):
        block = candidate_steps[offset : offset + block_steps]
        if not block:
            continue
        bits = rng.integers(0, 2, size=precision_bits).tolist()
        encoded = encode_trace_arithmetic(bits, block, precision_bits=precision_bits)
        decoded = decode_trace_arithmetic_prefix(
            encoded.actions,
            block,
            precision_bits=precision_bits,
        )
        if decoded != bits[: len(decoded)]:
            raise AssertionError("arithmetic prefix mismatch")
        recovered += len(decoded)
    return recovered


def main() -> None:
    sys.path.insert(0, str(ROOT / "code/src"))

    from models.temporal import TemporalBackoffModel

    config = _load_yaml(ROOT / "experiments/real_world/phase7_steganalysis.yaml")
    codec_config = config["codec"]
    catalog = _event_catalog(config)
    precision_bits = 48
    block_steps = 64
    rng = np.random.default_rng(int(config["seed"]))
    rows = []
    summary: dict[str, Any] = {
        "campaign": "phase11_range_codec",
        "source_campaign": config["campaign"],
        "precision_bits": precision_bits,
        "block_steps": block_steps,
        "datasets": {},
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
        previous_by_source: dict[Hashable, Hashable] = {}
        safe_steps = []
        local_bits = 0
        safe_entropy = 0.0
        test = frame.loc[frame["split"] == "test"].sort_values(["timestamp"], kind="stable")
        for record in test[["source", "destination", "timestamp"]].itertuples(index=False):
            source, natural_destination, _timestamp = record
            previous = previous_by_source.get(source)
            candidates = model.candidate_distribution(source, previous)
            entropy_bits = _entropy([candidate.probability for candidate in candidates])
            local_probe_bits = rng.integers(0, 2, size=int(codec_config["max_bits_per_transition"])).tolist()
            local = _local_encode(local_probe_bits, candidates, codec_config)
            state_feasible = (
                entropy_bits >= float(codec_config["min_entropy_bits"])
                and local.local_total_variation <= float(codec_config["max_local_total_variation"])
                and local.local_kl_bits <= float(codec_config["max_local_kl_bits"])
            )
            allowed = _allowed_candidates(source, candidates, codec_config)
            if state_feasible and len(allowed) >= 2:
                safe_steps.append(allowed)
                safe_entropy += _entropy([candidate.probability for candidate in allowed])
            encoded_features = (
                _rank_features(source, local.action, candidates)
                if local.action in {candidate.action for candidate in candidates}
                else {"rank_fraction": 1.0}
            )
            if (
                local.bits_consumed > 0
                and state_feasible
                and encoded_features["rank_fraction"] <= float(codec_config.get("max_encoded_rank_fraction", 1.0))
            ):
                local_bits += local.bits_consumed
            previous_by_source[source] = natural_destination
        arithmetic_bits = _arithmetic_bits(
            safe_steps,
            rng=rng,
            precision_bits=precision_bits,
            block_steps=block_steps,
        )
        transitions = len(test)
        row = {
            "dataset": dataset_id,
            "test_transitions": int(transitions),
            "safe_steps": int(len(safe_steps)),
            "safe_step_rate": len(safe_steps) / transitions,
            "safe_entropy_bits_per_transition": safe_entropy / transitions,
            "local_bits_per_transition": local_bits / transitions,
            "arithmetic_bits_per_transition": arithmetic_bits / transitions,
            "arithmetic_bits_per_safe_step": arithmetic_bits / max(1, len(safe_steps)),
            "arithmetic_gain_over_local": (arithmetic_bits - local_bits) / transitions,
        }
        rows.append(row)
        summary["datasets"][dataset_id] = row

    output_dir = ROOT / "results/tables"
    csv_output = output_dir / "phase11_range_codec.csv"
    json_output = output_dir / "phase11_range_codec.json"
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

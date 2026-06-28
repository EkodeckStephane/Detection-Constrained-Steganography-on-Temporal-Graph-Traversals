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


def _renormalized_entropy(candidates: Sequence[Any], allowed: Sequence[Hashable]) -> float:
    allowed_set = set(allowed)
    probabilities = [candidate.probability for candidate in candidates if candidate.action in allowed_set]
    total = sum(probabilities)
    if total <= 0:
        return 0.0
    return _entropy([probability / total for probability in probabilities])


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


def _allowed_actions(source: Hashable, candidates: Sequence[Any], codec_config: dict[str, Any]) -> list[Hashable]:
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
        allowed.append(candidate.action)
    return allowed


def _encode(bits: Sequence[int], candidates: Sequence[Any], codec_config: dict[str, Any]) -> Any:
    from stego.coding import encode_next_action, encode_next_action_range

    backend = str(codec_config.get("codec_backend", "range"))
    max_bits = int(codec_config["max_bits_per_transition"])
    if backend == "range":
        return encode_next_action_range(bits, candidates, max_bits=max_bits)
    if backend == "quantized":
        return encode_next_action(bits, candidates, max_bits=max_bits)
    raise ValueError(f"Unknown codec backend: {backend}")


def main() -> None:
    sys.path.insert(0, str(ROOT / "code/src"))

    from models.temporal import TemporalBackoffModel

    config = _load_yaml(ROOT / "experiments/real_world/phase7_steganalysis.yaml")
    codec_config = config["codec"]
    catalog = _event_catalog(config)
    rng = np.random.default_rng(int(config["seed"]))
    rows = []
    summary: dict[str, Any] = {
        "campaign": "phase11_capacity_audit",
        "source_campaign": config["campaign"],
        "model": config["model"],
        "codec": codec_config,
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
        split_rows = []
        test = frame.loc[frame["split"] == "test"].sort_values(["timestamp"], kind="stable")
        for record in test[["source", "destination", "timestamp"]].itertuples(index=False):
            source, natural_destination, _timestamp = record
            previous = previous_by_source.get(source)
            candidates = model.candidate_distribution(source, previous)
            entropy_bits = _entropy([candidate.probability for candidate in candidates])
            allowed = _allowed_actions(source, candidates, codec_config)
            safe_entropy_bits = _renormalized_entropy(candidates, allowed)
            bits = rng.integers(0, 2, size=int(codec_config["max_bits_per_transition"])).tolist()
            encoded = _encode(bits, candidates, codec_config)
            encoded_features = _rank_features(source, encoded.action, candidates)
            state_feasible = (
                entropy_bits >= float(codec_config["min_entropy_bits"])
                and encoded.local_total_variation <= float(codec_config["max_local_total_variation"])
                and encoded.local_kl_bits <= float(codec_config["max_local_kl_bits"])
            )
            selected_allowed = encoded.action in set(allowed)
            can_embed = encoded.bits_consumed > 0 and state_feasible and selected_allowed
            current_bits = encoded.bits_consumed if can_embed else 0
            row = {
                "dataset": dataset_id,
                "split": "test",
                "candidate_count": len(candidates),
                "candidate_entropy_bits": entropy_bits,
                "allowed_candidate_count": len(allowed) if state_feasible else 0,
                "safe_entropy_bits": safe_entropy_bits if state_feasible and allowed else 0.0,
                "codec_bits_consumed": current_bits,
                "state_feasible": int(state_feasible),
                "selected_allowed": int(selected_allowed),
                "can_embed": int(can_embed),
                "encoded_rank_fraction": encoded_features["rank_fraction"],
                "encoded_probability": encoded_features["probability"],
                "local_total_variation": encoded.local_total_variation if can_embed else 0.0,
                "local_kl_bits": encoded.local_kl_bits if can_embed else 0.0,
                "natural_destination_seen": int(natural_destination in model.destinations),
                "context_seen": int(model.has_context(source, previous)),
            }
            rows.append(row)
            split_rows.append(row)
            previous_by_source[source] = natural_destination
        data = pd.DataFrame(split_rows)
        transitions = len(data)
        summary["datasets"][dataset_id] = {
            "test_transitions": int(transitions),
            "mean_candidate_entropy_bits": float(data["candidate_entropy_bits"].mean()),
            "mean_safe_entropy_bits_per_transition": float(data["safe_entropy_bits"].mean()),
            "mean_safe_entropy_bits_when_feasible": float(
                data.loc[data["safe_entropy_bits"] > 0, "safe_entropy_bits"].mean()
            )
            if bool((data["safe_entropy_bits"] > 0).any())
            else 0.0,
            "mean_codec_bits_per_transition": float(data["codec_bits_consumed"].mean()),
            "state_feasible_rate": float(data["state_feasible"].mean()),
            "selected_allowed_rate": float(data["selected_allowed"].mean()),
            "embed_rate": float(data["can_embed"].mean()),
            "mean_allowed_candidate_count": float(data["allowed_candidate_count"].mean()),
            "codec_gap_bits_per_transition": float(
                (data["safe_entropy_bits"] - data["codec_bits_consumed"]).clip(lower=0).mean()
            ),
        }

    output_dir = ROOT / "results/tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output = output_dir / "phase11_capacity_audit.csv"
    json_output = output_dir / "phase11_capacity_audit.json"
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

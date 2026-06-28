from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/src"))

from baselines.walks import uniform_random_walk
from models.cover_model import BackoffCoverModel
from steganalysis.detectors import fit_detector, metrics_to_dict, score_detector
from steganalysis.samples import SampleConfig, feature_matrix, make_steganalysis_records


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _load_edges(path: Path) -> list[tuple[int, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        next(rows)
        return [tuple(map(int, row)) for row in rows]


def _build_frame_from_walks(
    edges: list[tuple[int, int]],
    *,
    n_walks: int,
    walk_length: int,
    seed: int,
) -> pd.DataFrame:
    import networkx as nx

    graph = nx.DiGraph(edges)
    rng = np.random.RandomState(seed)
    nodes = list(graph.nodes)
    records = []
    timestamp = 0
    for walk_index in range(n_walks):
        start = nodes[int(rng.randint(0, len(nodes)))]
        walk = uniform_random_walk(graph, start, walk_length, seed=int(rng.randint(0, 2**31 - 1)))
        for source, destination in zip(walk[:-1], walk[1:]):
            records.append({
                "source": source,
                "destination": destination,
                "timestamp": timestamp,
            })
            timestamp += 1

    frame = pd.DataFrame(records)
    # Causal split: first 60% train, next 20% validation, last 20% test.
    n = len(frame)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    frame["split"] = "train"
    frame.loc[train_end:val_end, "split"] = "validation"
    frame.loc[val_end:, "split"] = "test"
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the proposed steganographic channel on terrorists-911.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/variant_neural_cover/proposition_on_terrorists.yaml",
        help="Path to the proposition-on-terrorists YAML config.",
    )
    args = parser.parse_args()
    config = _load_yaml(args.config)
    seed = int(config["seed"])

    edges = _load_edges(ROOT / config["dataset_path"])
    frame = _build_frame_from_walks(
        edges,
        n_walks=int(config["n_walks"]),
        walk_length=int(config["walk_length"]),
        seed=seed,
    )

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

    cover_model = BackoffCoverModel(
        prior_strength=float(config["model"]["prior_strength"]),
        top_k=int(config["model"]["top_k"]),
    ).fit(frame.loc[frame["split"] == "train"])

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
    validation.insert(0, "dataset", "terrorists-911")
    test.insert(0, "dataset", "terrorists-911")

    stego_test = test.loc[test["label"] == 1]
    communication = {
        "test_transitions": int(len(stego_test)),
        "embed_rate": float((stego_test["stego_mode"] == "EMBED").mean()),
        "cover_rate": float((stego_test["stego_mode"] == "COVER").mean()),
        "attempted_bits_per_transition": float(stego_test["bits_consumed"].mean()),
        "mean_local_total_variation": float(stego_test["local_total_variation"].mean()),
        "mean_local_kl_bits": float(stego_test["local_kl_bits"].mean()),
    }

    x_train, y_train = feature_matrix(validation)
    x_test, y_test = feature_matrix(test)
    detectors: dict[str, dict[str, float]] = {}
    for detector_name in config["detectors"]:
        detector = fit_detector(detector_name, x_train, y_train, seed=seed)
        detectors[detector_name] = metrics_to_dict(score_detector(detector, x_test, y_test))

    summary = {
        "campaign": config["campaign"],
        "date": str(config["date"]),
        "dataset": "terrorists-911",
        "n_edges": len(edges),
        "n_nodes": len(set(node for edge in edges for node in edge)),
        "walks": {
            "n_walks": int(config["n_walks"]),
            "walk_length": int(config["walk_length"]),
            "total_transitions": len(frame),
        },
        "communication": communication,
        "detectors": detectors,
    }

    output_dir = ROOT / "results/tables/variant_neural_cover"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_output = output_dir / "proposition_on_terrorists.json"
    json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for detector_name, metrics in detectors.items():
        rows.append({
            "dataset": "terrorists-911",
            "detector": detector_name,
            "attempted_bits_per_transition": communication["attempted_bits_per_transition"],
            **metrics,
        })
    pd.DataFrame(rows).to_csv(output_dir / "proposition_on_terrorists.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

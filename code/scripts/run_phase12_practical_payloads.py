from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PAYLOADS = [
    {"payload": "control_command", "bits": 8},
    {"payload": "session_marker", "bits": 32},
    {"payload": "compact_metadata", "bits": 64},
    {"payload": "short_capability_token", "bits": 128},
]


def _transitions(bits: int, rate: float) -> int | str:
    if rate <= 0.0:
        return "outside_measured_regime"
    return int(math.ceil(bits / rate))


def main() -> None:
    sweep = pd.read_csv(ROOT / "results/tables/phase12_capacity_detectability_summary.csv")
    selected = sweep.loc[sweep["max_public_auc"] <= 0.60].copy()
    if selected.empty:
        raise ValueError("No operating point under the 0.60 AUC budget was found.")
    selected = (
        selected.sort_values(["dataset", "bits_per_transition"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .head(1)
    )
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "campaign": "phase12_practical_payloads",
        "source": "results/tables/phase12_capacity_detectability_summary.csv",
        "payloads": PAYLOADS,
        "datasets": {},
    }
    for row in selected.itertuples(index=False):
        dataset_records = []
        for payload in PAYLOADS:
            record = {
                "dataset": row.dataset,
                "operating_point": row.point,
                "max_public_auc": float(row.max_public_auc),
                "bits_per_transition": float(row.bits_per_transition),
                "payload": payload["payload"],
                "payload_bits": int(payload["bits"]),
                "required_transitions": _transitions(int(payload["bits"]), float(row.bits_per_transition)),
            }
            rows.append(record)
            dataset_records.append(record)
        summary["datasets"][row.dataset] = dataset_records

    output_dir = ROOT / "results/tables"
    with (output_dir / "phase12_practical_payloads.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "phase12_practical_payloads.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

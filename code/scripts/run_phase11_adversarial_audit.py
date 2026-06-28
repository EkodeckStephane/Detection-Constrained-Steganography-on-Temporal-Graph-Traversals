from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _auc(record: dict[str, Any]) -> float:
    return float(record["auc"])


def main() -> None:
    phase7 = _load_json(ROOT / "results/tables/phase7_steganalysis.json")
    independent = _load_json(ROOT / "results/tables/phase9_independent_neural_steganalysis.json")
    adaptive = _load_json(ROOT / "results/tables/phase9_adaptive_steganalysis.json")
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "campaign": "phase11_adversarial_audit",
        "sources": [
            "results/tables/phase7_steganalysis.json",
            "results/tables/phase9_independent_neural_steganalysis.json",
            "results/tables/phase9_adaptive_steganalysis.json",
        ],
        "datasets": {},
        "remaining_scope": [
            "third_party_red_team",
            "larger_temporal_gnn_steganalysts",
            "cross_domain_detector_transfer",
            "unbounded_active_adversary",
        ],
    }
    for dataset in phase7["detectors"]:
        external = phase7["detectors"][dataset]
        neural = independent["detectors"][dataset]
        adaptive_detectors = adaptive["detectors"][dataset]
        bounded = adaptive_detectors["bounded_white_box_gradient_boosting"]
        oracle = adaptive_detectors["oracle_leakage_gradient_boosting"]
        dataset_rows = []
        for name, metrics in external.items():
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "family": "external_public_features",
                    "detector": name,
                    "auc": _auc(metrics),
                    "auc_ci_high": "",
                }
            )
        for name, metrics in neural.items():
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "family": "independent_neural_public_sequence",
                    "detector": name,
                    "auc": _auc(metrics),
                    "auc_ci_high": "",
                }
            )
        dataset_rows.append(
            {
                "dataset": dataset,
                "family": "bounded_white_box_public_diagnostics",
                "detector": "gradient_boosting",
                "auc": _auc(bounded),
                "auc_ci_high": float(bounded.get("auc_ci_high", bounded["auc"])),
            }
        )
        dataset_rows.append(
            {
                "dataset": dataset,
                "family": "oracle_instrumentation_leakage",
                "detector": "gradient_boosting",
                "auc": _auc(oracle),
                "auc_ci_high": float(oracle.get("auc_ci_high", oracle["auc"])),
            }
        )
        rows.extend(dataset_rows)
        summary["datasets"][dataset] = {
            "max_external_auc": max(_auc(metrics) for metrics in external.values()),
            "max_independent_neural_auc": max(_auc(metrics) for metrics in neural.values()),
            "bounded_white_box_auc": _auc(bounded),
            "bounded_white_box_auc_ci_high": float(bounded.get("auc_ci_high", bounded["auc"])),
            "oracle_auc": _auc(oracle),
            "oracle_auc_ci_high": float(oracle.get("auc_ci_high", oracle["auc"])),
            "max_bounded_public_auc": max(
                max(_auc(metrics) for metrics in external.values()),
                max(_auc(metrics) for metrics in neural.values()),
                _auc(bounded),
            ),
        }

    output_dir = ROOT / "results/tables"
    csv_output = output_dir / "phase11_adversarial_audit.csv"
    json_output = output_dir / "phase11_adversarial_audit.json"
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path


# Validate only assets that are part of this repository's reproducibility contract.
# Manuscript/thesis trees and external literature snapshots are intentionally not
# required here: they are not redistributed as executable project dependencies.
REQUIRED_PATHS = (
    "README.md",
    "datasets/metadata/manifest.yaml",
    "experiments/asoc_v2/protocol.yaml",
    "docs/asoc_v2/SCIENTIFIC_SPEC.md",
    "docs/scientific_lock/novelty_matrix.md",
    "code/requirements.txt",
    "code/src/data/splits.py",
    "code/src/models/temporal.py",
    "code/src/models/admissibility.py",
    "code/src/stego/causal_arithmetic.py",
    "code/src/controllers/fuzzy.py",
    "code/src/steganalysis/detectors.py",
    "code/tests/test_steganalysis_samples.py",
    "code/tests/test_sealed_selection.py",
    "code/tests/test_admissibility.py",
    ".github/workflows/asoc-v2-ci.yml",
)

OPTIONAL_JSON_PATHS = (
    "results/tables/phase3_dataset_statistics.json",
)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    missing = [path for path in REQUIRED_PATHS if not (root / path).exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Project validation failed. Missing:\n{formatted}")

    invalid_json = []
    for relative in OPTIONAL_JSON_PATHS:
        path = root / relative
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                json.load(handle)
        except (OSError, json.JSONDecodeError):
            invalid_json.append(relative)
    if invalid_json:
        formatted = "\n".join(f"- {path}" for path in invalid_json)
        raise SystemExit(f"Invalid JSON files:\n{formatted}")

    empty_required = [
        path for path in REQUIRED_PATHS
        if (root / path).is_file() and (root / path).stat().st_size == 0
    ]
    if empty_required:
        formatted = "\n".join(f"- {path}" for path in empty_required)
        raise SystemExit(f"Required files are empty:\n{formatted}")

    print(
        "ASOC V2 project structure is valid "
        f"({len(REQUIRED_PATHS)} required paths checked)."
    )


if __name__ == "__main__":
    main()

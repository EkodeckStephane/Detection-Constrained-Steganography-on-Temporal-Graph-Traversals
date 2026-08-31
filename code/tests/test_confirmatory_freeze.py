from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from evaluation.confirmatory_freeze import (
    REQUIRED_CONFIRMATORY_DATASETS,
    build_confirmatory_freeze_manifest,
    load_confirmatory_freeze_manifest,
    save_confirmatory_freeze_manifest,
    verify_confirmatory_freeze_manifest,
)

SNAPSHOT_COMMIT = "2d97df289ae1faaf05c0c7bb656b94a8e6dab898"


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")
    else:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _policy(dataset_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "role": "confirmatory_fixed_policy",
        "selection_region": "policy_validation",
        "source_submission_snapshot_commit": SNAPSHOT_COMMIT,
        "source_commit_sha": SNAPSHOT_COMMIT,
        "may_change_after_holdout": False,
        "selected_policy": {"weights": [0.2, 0.4], "stop_threshold": 0.92},
        "message_seeds": [11, 23, 37, 53, 71],
        "design_metrics": {"worst_adversarial_auc": 0.58},
    }


def _common(tmp_path: Path) -> dict[str, Path]:
    return {
        "submission_snapshot_path": _write(tmp_path / "snapshot.yaml", {"frozen": True}),
        "protocol_path": _write(tmp_path / "protocol.yaml", {"budget": 0.60}),
        "split_manifest_path": _write(tmp_path / "splits.yaml", {"holdout": 0.05}),
        "confirmatory_extension_path": _write(tmp_path / "extension.yaml", {"opened": False}),
        "geolife_boundary_path": _write(
            tmp_path / "geolife.yaml",
            {
                "dataset_id": "geolife_cells",
                "role": "boundary_condition",
                "source_submission_snapshot_commit": SNAPSHOT_COMMIT,
                "may_change_after_holdout": False,
                "retuning_after_snapshot": "forbidden",
            },
        ),
    }


def test_missing_actor_policy_prevents_confirmatory_freeze(tmp_path: Path) -> None:
    common = _common(tmp_path)
    policy_paths = {
        "t_drive_cells": _write(tmp_path / "t_drive.yaml", _policy("t_drive_cells")),
    }
    with pytest.raises(ValueError, match="exact policy documents are missing"):
        build_confirmatory_freeze_manifest(
            submission_snapshot_commit=SNAPSHOT_COMMIT,
            freeze_commit_sha="freeze-sha",
            policy_paths=policy_paths,
            **common,
        )


def test_complete_policy_bundle_can_be_fingerprinted_and_verified(tmp_path: Path) -> None:
    common = _common(tmp_path)
    policy_paths = {
        dataset_id: _write(tmp_path / f"{dataset_id}.yaml", _policy(dataset_id))
        for dataset_id in REQUIRED_CONFIRMATORY_DATASETS
    }
    kwargs = {
        "submission_snapshot_commit": SNAPSHOT_COMMIT,
        "freeze_commit_sha": "freeze-sha",
        "policy_paths": policy_paths,
        **common,
    }
    manifest = build_confirmatory_freeze_manifest(**kwargs)
    path = tmp_path / "freeze.json"
    save_confirmatory_freeze_manifest(manifest, path)
    loaded = load_confirmatory_freeze_manifest(path)
    assert loaded == manifest
    verify_confirmatory_freeze_manifest(loaded, **kwargs)

    policy_paths["mooc"].write_text(
        policy_paths["mooc"].read_text(encoding="utf-8") + "\n# mutation\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="differs from the pre-holdout freeze"):
        verify_confirmatory_freeze_manifest(loaded, **kwargs)

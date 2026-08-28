from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evaluation.final_holdout import (
    build_freeze_manifest,
    design_only_frame,
    final_holdout_frame,
    load_freeze_manifest,
    save_freeze_manifest,
)


def _files(tmp_path: Path) -> tuple[Path, Path]:
    protocol = tmp_path / "protocol.yaml"
    dataset_manifest = tmp_path / "datasets.yaml"
    protocol.write_text("campaign: asoc_v2\n", encoding="utf-8")
    dataset_manifest.write_text("datasets: [a, b]\n", encoding="utf-8")
    return protocol, dataset_manifest


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4, 5],
            "split": [
                "cover_train",
                "eve_train",
                "policy_validation",
                "development_test",
                "final_holdout",
            ],
        }
    )


def test_design_view_excludes_final_holdout() -> None:
    design = design_only_frame(_frame())
    assert "final_holdout" not in set(design["split"])
    assert len(design) == 4


def test_freeze_manifest_round_trip_and_verified_open(tmp_path: Path) -> None:
    protocol, datasets = _files(tmp_path)
    policy = {"weights": [0.1, 0.2], "stop": 0.9}
    design_results = {"validation_auc_upper": 0.59, "payload_rate": 0.2}
    manifest = build_freeze_manifest(
        commit_sha="abc123",
        protocol_path=protocol,
        dataset_manifest_path=datasets,
        frozen_policy=policy,
        design_results=design_results,
    )
    destination = tmp_path / "freeze.json"
    save_freeze_manifest(manifest, destination)
    loaded = load_freeze_manifest(destination)
    assert loaded == manifest

    holdout = final_holdout_frame(
        _frame(),
        manifest=loaded,
        commit_sha="abc123",
        protocol_path=protocol,
        dataset_manifest_path=datasets,
        frozen_policy=policy,
        design_results=design_results,
    )
    assert holdout["timestamp"].tolist() == [5]


def test_holdout_refuses_policy_or_design_drift(tmp_path: Path) -> None:
    protocol, datasets = _files(tmp_path)
    policy = {"weights": [0.1, 0.2]}
    design_results = {"payload_rate": 0.2}
    manifest = build_freeze_manifest(
        commit_sha="abc123",
        protocol_path=protocol,
        dataset_manifest_path=datasets,
        frozen_policy=policy,
        design_results=design_results,
    )

    with pytest.raises(ValueError, match="frozen pre-holdout"):
        final_holdout_frame(
            _frame(),
            manifest=manifest,
            commit_sha="abc123",
            protocol_path=protocol,
            dataset_manifest_path=datasets,
            frozen_policy={"weights": [0.1, 0.3]},
            design_results=design_results,
        )
    with pytest.raises(ValueError, match="frozen pre-holdout"):
        final_holdout_frame(
            _frame(),
            manifest=manifest,
            commit_sha="abc123",
            protocol_path=protocol,
            dataset_manifest_path=datasets,
            frozen_policy=policy,
            design_results={"payload_rate": 0.21},
        )


def test_holdout_refuses_protocol_drift(tmp_path: Path) -> None:
    protocol, datasets = _files(tmp_path)
    policy = {"weights": [0.1]}
    design_results = {"payload_rate": 0.2}
    manifest = build_freeze_manifest(
        commit_sha="abc123",
        protocol_path=protocol,
        dataset_manifest_path=datasets,
        frozen_policy=policy,
        design_results=design_results,
    )
    protocol.write_text("campaign: changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen pre-holdout"):
        final_holdout_frame(
            _frame(),
            manifest=manifest,
            commit_sha="abc123",
            protocol_path=protocol,
            dataset_manifest_path=datasets,
            frozen_policy=policy,
            design_results=design_results,
        )


def test_holdout_must_be_nonempty(tmp_path: Path) -> None:
    protocol, datasets = _files(tmp_path)
    policy = {"weights": [0.1]}
    design_results = {"payload_rate": 0.2}
    manifest = build_freeze_manifest(
        commit_sha="abc123",
        protocol_path=protocol,
        dataset_manifest_path=datasets,
        frozen_policy=policy,
        design_results=design_results,
    )
    frame = _frame().loc[_frame()["split"] != "final_holdout"]
    with pytest.raises(ValueError, match="final holdout is empty"):
        final_holdout_frame(
            frame,
            manifest=manifest,
            commit_sha="abc123",
            protocol_path=protocol,
            dataset_manifest_path=datasets,
            frozen_policy=policy,
            design_results=design_results,
        )

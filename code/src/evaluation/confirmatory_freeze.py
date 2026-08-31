from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

REQUIRED_CONFIRMATORY_DATASETS = (
    "tgbl-wiki",
    "mooc",
    "lastfm",
    "t_drive_cells",
)
BOUNDARY_DATASET = "geolife_cells"


@dataclass(frozen=True)
class ConfirmatoryFreezeManifest:
    submission_snapshot_commit: str
    freeze_commit_sha: str
    submission_snapshot_sha256: str
    protocol_sha256: str
    split_manifest_sha256: str
    confirmatory_extension_sha256: str
    frozen_policy_sha256: dict[str, str]
    frozen_policy_bundle_sha256: str
    geolife_boundary_sha256: str

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(asdict(self))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return sha256_bytes(payload)


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def validate_exact_policy_document(
    document: Mapping[str, Any],
    *,
    dataset_id: str,
    submission_snapshot_commit: str,
) -> None:
    if document.get("dataset_id") != dataset_id:
        raise ValueError(f"policy dataset_id mismatch for {dataset_id}")
    if document.get("role") != "confirmatory_fixed_policy":
        raise ValueError(f"{dataset_id} is not marked confirmatory_fixed_policy")
    if document.get("selection_region") != "policy_validation":
        raise ValueError(f"{dataset_id} policy was not selected on policy_validation")
    if document.get("may_change_after_holdout") is not False:
        raise ValueError(f"{dataset_id} policy is not immutable after holdout")
    source_commit = document.get("source_submission_snapshot_commit")
    if source_commit != submission_snapshot_commit:
        raise ValueError(f"{dataset_id} policy does not point to the frozen submission snapshot")
    selected_policy = document.get("selected_policy")
    if not isinstance(selected_policy, Mapping) or not selected_policy:
        raise ValueError(f"{dataset_id} selected_policy is missing or empty")
    seeds = document.get("message_seeds")
    if not isinstance(seeds, list) or seeds != [11, 23, 37, 53, 71]:
        raise ValueError(f"{dataset_id} message seeds differ from the frozen protocol")
    design_metrics = document.get("design_metrics")
    if not isinstance(design_metrics, Mapping) or not design_metrics:
        raise ValueError(f"{dataset_id} design_metrics are missing")


def validate_boundary_document(
    document: Mapping[str, Any],
    *,
    submission_snapshot_commit: str,
) -> None:
    if document.get("dataset_id") != BOUNDARY_DATASET:
        raise ValueError("GeoLife boundary document has the wrong dataset_id")
    if document.get("role") != "boundary_condition":
        raise ValueError("GeoLife must remain a boundary_condition")
    if document.get("may_change_after_holdout") is not False:
        raise ValueError("GeoLife boundary role may not change after holdout")
    source_commit = document.get("source_submission_snapshot_commit")
    if source_commit != submission_snapshot_commit:
        raise ValueError("GeoLife boundary document does not point to the frozen snapshot")
    if document.get("retuning_after_snapshot") != "forbidden":
        raise ValueError("GeoLife retuning must remain forbidden")


def build_confirmatory_freeze_manifest(
    *,
    submission_snapshot_commit: str,
    freeze_commit_sha: str,
    submission_snapshot_path: Path,
    protocol_path: Path,
    split_manifest_path: Path,
    confirmatory_extension_path: Path,
    policy_paths: Mapping[str, Path],
    geolife_boundary_path: Path,
) -> ConfirmatoryFreezeManifest:
    if not submission_snapshot_commit.strip():
        raise ValueError("submission_snapshot_commit must be non-empty")
    if not freeze_commit_sha.strip():
        raise ValueError("freeze_commit_sha must be non-empty")

    missing = [
        dataset_id
        for dataset_id in REQUIRED_CONFIRMATORY_DATASETS
        if dataset_id not in policy_paths or not policy_paths[dataset_id].is_file()
    ]
    if missing:
        raise ValueError(
            "cannot freeze confirmatory campaign; exact policy documents are missing for: "
            + ", ".join(missing)
        )

    policy_hashes: dict[str, str] = {}
    for dataset_id in REQUIRED_CONFIRMATORY_DATASETS:
        path = policy_paths[dataset_id]
        document = load_yaml(path)
        validate_exact_policy_document(
            document,
            dataset_id=dataset_id,
            submission_snapshot_commit=submission_snapshot_commit,
        )
        policy_hashes[dataset_id] = sha256_file(path)

    boundary_document = load_yaml(geolife_boundary_path)
    validate_boundary_document(
        boundary_document,
        submission_snapshot_commit=submission_snapshot_commit,
    )

    bundle_hash = canonical_sha256(
        [(dataset_id, policy_hashes[dataset_id]) for dataset_id in REQUIRED_CONFIRMATORY_DATASETS]
    )
    return ConfirmatoryFreezeManifest(
        submission_snapshot_commit=submission_snapshot_commit.strip(),
        freeze_commit_sha=freeze_commit_sha.strip(),
        submission_snapshot_sha256=sha256_file(submission_snapshot_path),
        protocol_sha256=sha256_file(protocol_path),
        split_manifest_sha256=sha256_file(split_manifest_path),
        confirmatory_extension_sha256=sha256_file(confirmatory_extension_path),
        frozen_policy_sha256=policy_hashes,
        frozen_policy_bundle_sha256=bundle_hash,
        geolife_boundary_sha256=sha256_file(geolife_boundary_path),
    )


def save_confirmatory_freeze_manifest(
    manifest: ConfirmatoryFreezeManifest,
    path: Path,
) -> None:
    payload = {**asdict(manifest), "fingerprint": manifest.fingerprint}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_confirmatory_freeze_manifest(path: Path) -> ConfirmatoryFreezeManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = payload.pop("fingerprint", None)
    manifest = ConfirmatoryFreezeManifest(**payload)
    if fingerprint != manifest.fingerprint:
        raise ValueError("confirmatory freeze fingerprint mismatch")
    return manifest


def verify_confirmatory_freeze_manifest(
    manifest: ConfirmatoryFreezeManifest,
    **build_kwargs: Any,
) -> None:
    expected = build_confirmatory_freeze_manifest(**build_kwargs)
    if expected != manifest:
        raise ValueError("current confirmatory state differs from the pre-holdout freeze")

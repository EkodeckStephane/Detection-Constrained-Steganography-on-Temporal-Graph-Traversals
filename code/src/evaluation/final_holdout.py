from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FinalFreezeManifest:
    """Immutable evidence that every design choice predates holdout access."""

    commit_sha: str
    protocol_sha256: str
    dataset_manifest_sha256: str
    policy_sha256: str
    design_results_sha256: str

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256_bytes(payload)


def build_freeze_manifest(
    *,
    commit_sha: str,
    protocol_path: Path,
    dataset_manifest_path: Path,
    frozen_policy: Any,
    design_results: Any,
) -> FinalFreezeManifest:
    if not commit_sha.strip():
        raise ValueError("commit_sha must be non-empty")
    return FinalFreezeManifest(
        commit_sha=commit_sha.strip(),
        protocol_sha256=sha256_file(protocol_path),
        dataset_manifest_sha256=sha256_file(dataset_manifest_path),
        policy_sha256=canonical_sha256(frozen_policy),
        design_results_sha256=canonical_sha256(design_results),
    )


def save_freeze_manifest(manifest: FinalFreezeManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**asdict(manifest), "fingerprint": manifest.fingerprint}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_freeze_manifest(path: Path) -> FinalFreezeManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = payload.pop("fingerprint", None)
    manifest = FinalFreezeManifest(**payload)
    if fingerprint is not None and fingerprint != manifest.fingerprint:
        raise ValueError("freeze manifest fingerprint mismatch")
    return manifest


def verify_freeze_manifest(
    manifest: FinalFreezeManifest,
    *,
    commit_sha: str,
    protocol_path: Path,
    dataset_manifest_path: Path,
    frozen_policy: Any,
    design_results: Any,
) -> None:
    expected = build_freeze_manifest(
        commit_sha=commit_sha,
        protocol_path=protocol_path,
        dataset_manifest_path=dataset_manifest_path,
        frozen_policy=frozen_policy,
        design_results=design_results,
    )
    if expected != manifest:
        raise ValueError("current design state differs from the frozen pre-holdout manifest")


def design_only_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return all non-publication regions while guaranteeing holdout exclusion."""

    if "split" not in frame.columns:
        raise ValueError("frame must contain split")
    return frame.loc[frame["split"] != "final_holdout"].copy()


def final_holdout_frame(
    frame: pd.DataFrame,
    *,
    manifest: FinalFreezeManifest,
    commit_sha: str,
    protocol_path: Path,
    dataset_manifest_path: Path,
    frozen_policy: Any,
    design_results: Any,
) -> pd.DataFrame:
    """Expose publication holdout only after the frozen state is verified."""

    verify_freeze_manifest(
        manifest,
        commit_sha=commit_sha,
        protocol_path=protocol_path,
        dataset_manifest_path=dataset_manifest_path,
        frozen_policy=frozen_policy,
        design_results=design_results,
    )
    if "split" not in frame.columns:
        raise ValueError("frame must contain split")
    result = frame.loc[frame["split"] == "final_holdout"].copy()
    if result.empty:
        raise ValueError("final holdout is empty")
    if set(result["split"].unique()) != {"final_holdout"}:
        raise AssertionError("non-holdout rows escaped the final gate")
    return result

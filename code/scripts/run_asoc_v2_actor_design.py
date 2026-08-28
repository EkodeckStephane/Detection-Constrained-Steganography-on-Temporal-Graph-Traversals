from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))

from evaluation.actor_action_design import (  # noqa: E402
    certification_to_dict,
    certify_actor_candidate,
    fit_actor_design_eves,
    load_actor_design_prefix,
    propose_actor_candidates,
    select_certified_actor_policy,
)
from models.temporal import TemporalBackoffModel  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASOC V2 actor-action policy search and full validation certification"
    )
    parser.add_argument("--dataset-id", required=True, choices=("tgbl-wiki", "mooc", "lastfm"))
    parser.add_argument("--raw-events", required=True, type=Path)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "experiments/asoc_v2/split_manifest.yaml",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "experiments/asoc_v2/protocol.yaml",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--search-rows", type=int, default=3000)
    parser.add_argument("--shortlist-size", type=int, default=12)
    parser.add_argument("--certify-limit", type=int, default=12)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--search-only", action="store_true")
    args = parser.parse_args()

    frame, split_spec = load_actor_design_prefix(
        args.raw_events,
        args.split_manifest,
        args.dataset_id,
    )
    cover = frame.loc[frame["split"] == "cover_train"].copy()
    eve = frame.loc[frame["split"] == "eve_train"].copy()
    validation = frame.loc[frame["split"] == "policy_validation"].copy()

    model = TemporalBackoffModel(
        prior_strength=8.0,
        top_k=32,
        history_mode="actor_history",
    ).fit(cover)
    detectors, eve_summary = fit_actor_design_eves(model, eve)
    shortlist = propose_actor_candidates(
        model,
        detectors,
        validation,
        search_rows=args.search_rows,
        shortlist_size=args.shortlist_size,
    )

    result: dict[str, object] = {
        "campaign": "asoc_v2_causal_detector_constrained_control",
        "stage": "actor_action_design",
        "dataset_id": args.dataset_id,
        "final_holdout_accessed": False,
        "development_test_accessed": False,
        "provenance": {
            "commit_sha": _commit_sha(),
            "raw_events": str(args.raw_events),
            "raw_events_sha256": _sha256(args.raw_events),
            "split_manifest": str(args.split_manifest),
            "split_manifest_sha256": _sha256(args.split_manifest),
            "protocol": str(args.protocol),
            "protocol_sha256": _sha256(args.protocol),
        },
        "split_spec": asdict(split_spec),
        "design_eves": asdict(eve_summary),
        "search": {
            "rows": args.search_rows,
            "shortlist_size": len(shortlist),
            "authority": "proposal_only_not_scientific_certification",
            "candidates": [asdict(item) for item in shortlist],
        },
    }

    if not args.search_only:
        candidates = shortlist[: min(args.certify_limit, len(shortlist))]
        certifications = [
            certify_actor_candidate(
                model,
                detectors,
                validation,
                candidate,
                bootstrap_resamples=args.bootstrap_resamples,
            )
            for candidate in candidates
        ]
        result["certification"] = {
            "rows": len(validation),
            "message_seeds": [11, 23, 37, 53, 71],
            "bootstrap_resamples_per_seed": args.bootstrap_resamples,
            "candidates": [certification_to_dict(item) for item in certifications],
        }
        feasible = [item for item in certifications if item.feasible]
        if feasible:
            selected = select_certified_actor_policy(certifications)
            result["selected_policy"] = certification_to_dict(selected)
        else:
            result["selected_policy"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

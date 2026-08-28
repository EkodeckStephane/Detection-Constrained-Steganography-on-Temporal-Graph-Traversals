from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from controllers.fuzzy import FuzzyWeights
from evaluation.actor_action_design import (
    SearchCandidate,
    certify_actor_candidate,
    load_actor_design_prefix,
)
from models.temporal import TemporalBackoffModel
from steganalysis.detectors import OrientedDetector


class _ChanceDetector:
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        score = np.full(len(x), 0.5, dtype=float)
        return np.column_stack([1.0 - score, score])


def test_actor_design_loader_stops_at_policy_validation_prefix(tmp_path: Path) -> None:
    raw = tmp_path / "events.csv"
    raw.write_text(
        "user,item,timestamp,label,feature\n"
        + "".join(f"{index % 2},{index % 3},{index},0,x\n" for index in range(1, 11)),
        encoding="utf-8",
    )
    manifest = tmp_path / "splits.yaml"
    manifest.write_text(
        "datasets:\n"
        "  demo:\n"
        "    timestamp_cutoffs:\n"
        "      cover_train_end: 3\n"
        "      eve_train_end: 5\n"
        "      policy_validation_end: 7\n"
        "      development_test_end: 9\n"
        "    counts:\n"
        "      cover_train: 3\n"
        "      eve_train: 2\n"
        "      policy_validation: 2\n"
        "      development_test: 2\n"
        "      final_holdout: 1\n",
        encoding="utf-8",
    )

    frame, spec = load_actor_design_prefix(raw, manifest, "demo")

    assert spec.design_rows == 7
    assert len(frame) == 7
    assert frame["timestamp"].max() == 7
    assert frame["split"].value_counts().to_dict() == {
        "cover_train": 3,
        "eve_train": 2,
        "policy_validation": 2,
    }
    assert 8 not in frame["timestamp"].tolist()
    assert 10 not in frame["timestamp"].tolist()


def test_zero_payload_actor_policy_is_exact_natural_trace_and_chance_auc() -> None:
    cover_rows = []
    validation_rows = []
    event_id = 0
    for source in range(4):
        for step in range(12):
            cover_rows.append(
                {
                    "event_id": event_id,
                    "source": f"user:{source}",
                    "destination": f"item:{(source + step) % 4}",
                    "timestamp": event_id,
                    "label": 0,
                    "sequence_id": pd.NA,
                }
            )
            event_id += 1
        for step in range(8):
            validation_rows.append(
                {
                    "event_id": event_id,
                    "source": f"user:{source}",
                    "destination": f"item:{(source + step + 1) % 4}",
                    "timestamp": event_id,
                    "label": 0,
                    "sequence_id": pd.NA,
                }
            )
            event_id += 1
    cover = pd.DataFrame(cover_rows).sort_values("timestamp", kind="stable")
    validation = pd.DataFrame(validation_rows).sort_values("timestamp", kind="stable")
    model = TemporalBackoffModel(
        prior_strength=8.0,
        top_k=4,
        history_mode="actor_history",
    ).fit(cover)
    detectors = {
        "chance": OrientedDetector(
            detector=_ChanceDetector(),
            reverse_score=False,
            calibration_auc=0.5,
        )
    }
    candidate = SearchCandidate(
        weights=FuzzyWeights(
            opportunity_entropy_weight=0.0,
            opportunity_payload_weight=0.0,
            cover_entropy_weight=1.0,
            pause_risk_weight=0.0,
            stop_dead_end_weight=0.0,
            abstention_cover_weight=1.0,
            abstention_pause_weight=0.0,
            abstention_stop_weight=0.0,
        ),
        stop_threshold=0.99,
        surrogate_score=0.0,
        surrogate_nominal_bits_per_transition=0.0,
        surrogate_embed_rate=0.0,
        surrogate_mean_embed_risk=0.0,
    )

    certification = certify_actor_candidate(
        model,
        detectors,
        validation,
        candidate,
        message_seeds=(11,),
        payload_bits=8,
        precision_bits=16,
        max_bits_per_transition=2,
        bootstrap_resamples=200,
    )
    result = certification.seeds[0]

    assert result.bits_per_transition == pytest.approx(0.0)
    assert result.worst_adversarial_auc == pytest.approx(0.5)
    assert result.auc_ci_lower == pytest.approx(0.5)
    assert result.auc_ci_upper == pytest.approx(0.5)
    assert result.passive_state_mismatch_rate == pytest.approx(0.0)
    assert result.passive_decode_success_rate == pytest.approx(1.0)
    assert result.invalid_transition_rate == pytest.approx(0.0)
    assert result.mode_rates["EMBED"] == pytest.approx(0.0)

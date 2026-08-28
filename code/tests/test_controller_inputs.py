from __future__ import annotations

import pytest

from controllers.inputs import build_public_controller_inputs, effective_backoff_observation_count
from stego.coding import Candidate


def test_public_inputs_are_normalized_and_payload_secret_independent() -> None:
    candidates = [Candidate("a", 0.5), Candidate("b", 0.5)]
    first = build_public_controller_inputs(
        candidates=candidates,
        top_k=32,
        context_observations=99,
        context_seen=True,
        steganalysis_risk=0.42,
        committed_payload_bits=8,
        payload_length=32,
    )
    second = build_public_controller_inputs(
        candidates=candidates,
        top_k=32,
        context_observations=99,
        context_seen=True,
        steganalysis_risk=0.42,
        committed_payload_bits=8,
        payload_length=32,
    )
    assert first == second
    assert 0.0 <= first.predictive_entropy <= 1.0
    assert first.calibration_uncertainty == pytest.approx(0.1)
    assert first.payload_pressure == pytest.approx(0.75)
    assert first.steganalysis_risk == pytest.approx(0.42)
    assert first.channel_fragility == pytest.approx(0.5)


def test_cover_calibrated_entropy_scale_changes_only_entropy_mapping() -> None:
    candidates = [Candidate("a", 0.5), Candidate("b", 0.5)]
    default = build_public_controller_inputs(
        candidates=candidates,
        top_k=32,
        context_observations=9,
        context_seen=True,
        steganalysis_risk=0.2,
        committed_payload_bits=0,
        payload_length=32,
    )
    calibrated = build_public_controller_inputs(
        candidates=candidates,
        top_k=32,
        context_observations=9,
        context_seen=True,
        steganalysis_risk=0.2,
        committed_payload_bits=0,
        payload_length=32,
        entropy_scale_bits=2.0,
    )
    assert default.predictive_entropy == pytest.approx(0.2)
    assert calibrated.predictive_entropy == pytest.approx(0.5)
    assert calibrated.calibration_uncertainty == default.calibration_uncertainty
    assert calibrated.steganalysis_risk == default.steganalysis_risk
    assert calibrated.payload_pressure == default.payload_pressure
    assert calibrated.dead_end_risk == default.dead_end_risk
    assert calibrated.channel_fragility == default.channel_fragility


def test_non_positive_entropy_scale_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_public_controller_inputs(
            candidates=[Candidate("a", 0.5), Candidate("b", 0.5)],
            top_k=32,
            context_observations=1,
            context_seen=True,
            steganalysis_risk=0.1,
            committed_payload_bits=0,
            payload_length=32,
            entropy_scale_bits=0.0,
        )


def test_zero_effective_evidence_has_maximum_calibration_uncertainty() -> None:
    inputs = build_public_controller_inputs(
        candidates=[Candidate("a", 0.7), Candidate("b", 0.3)],
        top_k=32,
        context_observations=0,
        context_seen=False,
        steganalysis_risk=0.2,
        committed_payload_bits=0,
        payload_length=32,
    )
    assert inputs.calibration_uncertainty == 1.0


class _BackoffEvidenceStub:
    def context_observation_count(self, source: str, previous: str | None) -> int:
        return 0 if previous == "new-context" else 7

    def source_observation_count(self, source: str) -> int:
        return 25 if source == "seen-source" else 0


def test_effective_backoff_evidence_uses_source_when_exact_context_is_unseen() -> None:
    model = _BackoffEvidenceStub()
    assert effective_backoff_observation_count(model, "seen-source", "old-context") == 7
    assert effective_backoff_observation_count(model, "seen-source", "new-context") == 25
    assert effective_backoff_observation_count(model, "cold-source", "new-context") == 0


def test_dead_end_risk_is_probability_mass_of_fragile_continuations() -> None:
    inputs = build_public_controller_inputs(
        candidates=[Candidate("safe", 0.75), Candidate("dead", 0.25)],
        top_k=4,
        context_observations=20,
        context_seen=True,
        steganalysis_risk=0.3,
        committed_payload_bits=0,
        payload_length=32,
        future_admissible_count=lambda action: 0 if action == "dead" else 3,
    )
    assert inputs.dead_end_risk == pytest.approx(0.25)


def test_singleton_support_is_maximally_fragile() -> None:
    inputs = build_public_controller_inputs(
        candidates=[Candidate("only", 1.0)],
        top_k=32,
        context_observations=5,
        context_seen=True,
        steganalysis_risk=0.1,
        committed_payload_bits=0,
        payload_length=32,
    )
    assert inputs.channel_fragility == 1.0

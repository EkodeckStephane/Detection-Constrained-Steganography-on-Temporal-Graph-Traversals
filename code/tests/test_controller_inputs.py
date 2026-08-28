from __future__ import annotations

import pytest

from controllers.inputs import build_public_controller_inputs
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


def test_unseen_context_has_maximum_calibration_uncertainty() -> None:
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

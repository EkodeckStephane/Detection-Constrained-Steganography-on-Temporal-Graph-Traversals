from __future__ import annotations

import pytest

from controllers.fuzzy import (
    ControllerInputs,
    FuzzyRateController,
    FuzzyWeights,
    fixed_entropy_threshold,
)


def test_fuzzy_controller_embeds_when_entropy_is_high_and_risk_is_low() -> None:
    controller = FuzzyRateController(max_bits_per_transition=4)

    decision = controller.decide(
        ControllerInputs(
            predictive_entropy=0.9,
            calibration_uncertainty=0.05,
            steganalysis_risk=0.05,
            payload_pressure=0.9,
            dead_end_risk=0.0,
            channel_fragility=0.05,
        )
    )

    assert decision.mode == "EMBED"
    assert decision.local_payload_bits >= 1


def test_fuzzy_controller_abstains_under_high_risk() -> None:
    controller = FuzzyRateController(max_bits_per_transition=4)

    decision = controller.decide(
        ControllerInputs(
            predictive_entropy=0.9,
            calibration_uncertainty=0.8,
            steganalysis_risk=0.9,
            payload_pressure=0.7,
            dead_end_risk=0.1,
            channel_fragility=0.8,
        )
    )

    assert decision.mode in {"COVER", "PAUSE"}
    assert decision.local_payload_bits == 0


def test_fuzzy_controller_stops_at_dead_end() -> None:
    controller = FuzzyRateController(max_bits_per_transition=4)

    decision = controller.decide(
        ControllerInputs(
            predictive_entropy=0.4,
            calibration_uncertainty=0.1,
            steganalysis_risk=0.1,
            payload_pressure=0.0,
            dead_end_risk=1.0,
            channel_fragility=0.1,
        )
    )

    assert decision.mode == "STOP"


def test_low_payload_pressure_cannot_by_itself_trigger_stop() -> None:
    controller = FuzzyRateController(max_bits_per_transition=4, stop_threshold=0.92)
    common = dict(
        predictive_entropy=0.8,
        calibration_uncertainty=0.55,
        steganalysis_risk=0.55,
        dead_end_risk=0.0,
        channel_fragility=0.55,
    )

    high_pressure = controller.decide(ControllerInputs(payload_pressure=1.0, **common))
    near_completion = controller.decide(ControllerInputs(payload_pressure=0.01, **common))

    assert high_pressure.mode != "STOP"
    assert near_completion.mode != "STOP"


def test_tunable_opportunity_weight_changes_the_policy() -> None:
    inputs = ControllerInputs(
        predictive_entropy=0.95,
        calibration_uncertainty=0.0,
        steganalysis_risk=0.0,
        payload_pressure=1.0,
        dead_end_risk=0.0,
        channel_fragility=0.0,
    )
    high = FuzzyRateController(
        max_bits_per_transition=4,
        weights=FuzzyWeights(opportunity_entropy_weight=1.0),
    ).decide(inputs)
    suppressed = FuzzyRateController(
        max_bits_per_transition=4,
        weights=FuzzyWeights(opportunity_entropy_weight=0.0),
    ).decide(inputs)

    assert high.mode == "EMBED"
    assert suppressed.mode == "COVER"
    assert high.rate_score > suppressed.rate_score


def test_fuzzy_weights_reject_out_of_range_parameter() -> None:
    with pytest.raises(ValueError):
        FuzzyWeights(pause_risk_weight=1.01)


def test_fixed_threshold_baseline_is_deterministic() -> None:
    inputs = ControllerInputs(0.7, 0.0, 0.0, 0.5, 0.0, 0.0)

    first = fixed_entropy_threshold(inputs, entropy_threshold=0.5, max_bits_per_transition=4)
    second = fixed_entropy_threshold(inputs, entropy_threshold=0.5, max_bits_per_transition=4)

    assert first == second
    assert first.mode == "EMBED"

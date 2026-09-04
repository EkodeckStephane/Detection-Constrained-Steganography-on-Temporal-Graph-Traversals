from __future__ import annotations

import pytest

from controllers.fuzzy import ControllerInputs
from evaluation.walk_comparators import comparator_step
from stego.coding import Candidate


def _inputs(*, entropy: float = 0.8, risk: float = 0.1) -> ControllerInputs:
    return ControllerInputs(
        predictive_entropy=entropy,
        calibration_uncertainty=0.05,
        steganalysis_risk=risk,
        payload_pressure=1.0,
        dead_end_risk=0.0,
        channel_fragility=0.05,
    )


def _candidates(count: int = 20) -> list[Candidate]:
    weights = list(range(count, 0, -1))
    mass = float(sum(weights))
    return [Candidate(f"a{index}", weight / mass) for index, weight in enumerate(weights)]


def test_detector_unaware_ignores_detector_risk_and_uses_top16_q_support() -> None:
    low = comparator_step(
        "detector_unaware_causal_arithmetic",
        inputs=_inputs(risk=0.0),
        candidates=_candidates(),
    )
    high = comparator_step(
        "detector_unaware_causal_arithmetic",
        inputs=_inputs(risk=1.0),
        candidates=_candidates(),
    )
    assert low == high
    assert low.decision.mode == "EMBED"
    assert low.decision.local_payload_bits == 4
    assert len(low.coding_candidates) == 16
    assert sum(item.probability for item in low.coding_candidates) == pytest.approx(1.0)


def test_random_admissible_is_uniform_on_same_top16_action_set() -> None:
    q = _candidates()
    reference = comparator_step(
        "detector_unaware_causal_arithmetic",
        inputs=_inputs(),
        candidates=q,
    )
    random = comparator_step(
        "random_admissible",
        inputs=_inputs(),
        candidates=q,
    )
    assert [item.action for item in random.coding_candidates] == [
        item.action for item in reference.coding_candidates
    ]
    assert len(random.coding_candidates) == 16
    assert {round(item.probability, 12) for item in random.coding_candidates} == {
        round(1.0 / 16.0, 12)
    }


def test_coupling_inspired_reference_uses_full_q_support_without_truncation() -> None:
    result = comparator_step(
        "coupling_inspired_distribution_preserving",
        inputs=_inputs(),
        candidates=_candidates(20),
    )
    assert result.decision.mode == "EMBED"
    assert result.decision.local_payload_bits == 5
    assert len(result.coding_candidates) == 20
    assert [item.action for item in result.coding_candidates] == [
        item.action for item in _candidates(20)
    ]
    assert sum(item.probability for item in result.coding_candidates) == pytest.approx(1.0)


def test_fixed_entropy_threshold_obeys_predeclared_threshold() -> None:
    low = comparator_step(
        "fixed_entropy_threshold",
        inputs=_inputs(entropy=0.54),
        candidates=_candidates(),
        fixed_threshold=0.55,
    )
    high = comparator_step(
        "fixed_entropy_threshold",
        inputs=_inputs(entropy=0.56),
        candidates=_candidates(),
        fixed_threshold=0.55,
    )
    assert low.decision.mode == "COVER"
    assert low.coding_candidates == ()
    assert high.decision.mode == "EMBED"
    assert high.coding_candidates


def test_hand_tuned_fuzzy_is_deterministic_and_secret_independent() -> None:
    first = comparator_step(
        "hand_tuned_fuzzy",
        inputs=_inputs(entropy=0.9, risk=0.05),
        candidates=_candidates(),
    )
    second = comparator_step(
        "hand_tuned_fuzzy",
        inputs=_inputs(entropy=0.9, risk=0.05),
        candidates=_candidates(),
    )
    assert first == second


def test_singleton_support_forces_cover_for_every_comparator() -> None:
    names = [
        "fixed_entropy_threshold",
        "hand_tuned_fuzzy",
        "detector_unaware_causal_arithmetic",
        "random_admissible",
        "coupling_inspired_distribution_preserving",
    ]
    for name in names:
        result = comparator_step(
            name,
            inputs=_inputs(),
            candidates=[Candidate("only", 1.0)],
        )
        assert result.decision.mode == "COVER"
        assert result.coding_candidates == ()

from __future__ import annotations

from controllers.fuzzy import ControllerInputs, FuzzyRateController, FuzzyWeights
from stego.coding import Candidate
from stego.policy_session import SynchronizedPolicySession


def _embed_inputs() -> ControllerInputs:
    return ControllerInputs(
        predictive_entropy=1.0,
        calibration_uncertainty=0.0,
        steganalysis_risk=0.0,
        payload_pressure=1.0,
        dead_end_risk=0.0,
        channel_fragility=0.0,
    )


def _binary_candidates(state: str) -> list[Candidate]:
    return [
        Candidate(f"{state}0", 0.5),
        Candidate(f"{state}1", 0.5),
    ]


def test_alice_bob_arithmetic_state_stays_identical_under_causal_actions() -> None:
    payload = [1, 0, 1, 1]
    alice_controller = FuzzyRateController(max_bits_per_transition=1)
    bob_controller = FuzzyRateController(max_bits_per_transition=1)
    session = SynchronizedPolicySession(
        payload,
        sender_controller=alice_controller,
        receiver_controller=bob_controller,
        precision_bits=16,
    )

    state = "S"
    for _ in range(4):
        candidates = _binary_candidates(state)
        emission = session.emit(inputs=_embed_inputs(), candidates=candidates)
        assert emission.decision.mode == "EMBED"
        assert emission.coder_advanced
        assert emission.sender_decoder_state_match
        state = str(emission.action)

    assert session.complete
    assert session.state_match
    assert session.decoded_payload == payload


def test_cover_mode_does_not_advance_arithmetic_interval() -> None:
    cover_only = FuzzyRateController(
        max_bits_per_transition=1,
        weights=FuzzyWeights(opportunity_entropy_weight=0.0),
    )
    session = SynchronizedPolicySession(
        [1, 0],
        sender_controller=cover_only,
        receiver_controller=cover_only,
        precision_bits=8,
    )
    initial_bits = session.committed_payload_bits

    emission = session.emit(
        inputs=_embed_inputs(),
        candidates=[Candidate("B", 0.6), Candidate("C", 0.4)],
        cover_action="B",
        admissible_actions={"B", "C"},
    )

    assert emission.decision.mode == "COVER"
    assert not emission.coder_advanced
    assert session.committed_payload_bits == initial_bits
    assert session.state_match


def test_public_control_decision_is_independent_of_secret_payload() -> None:
    controller_a = FuzzyRateController(max_bits_per_transition=1)
    controller_b = FuzzyRateController(max_bits_per_transition=1)
    first = SynchronizedPolicySession(
        [0, 0],
        sender_controller=controller_a,
        receiver_controller=controller_b,
        precision_bits=8,
    )
    second = SynchronizedPolicySession(
        [1, 1],
        sender_controller=FuzzyRateController(max_bits_per_transition=1),
        receiver_controller=FuzzyRateController(max_bits_per_transition=1),
        precision_bits=8,
    )
    candidates = [Candidate("left", 0.5), Candidate("right", 0.5)]

    emission_zero = first.emit(inputs=_embed_inputs(), candidates=candidates)
    emission_one = second.emit(inputs=_embed_inputs(), candidates=candidates)

    assert emission_zero.decision == emission_one.decision
    assert emission_zero.decision.mode == "EMBED"
    # Secret messages may select different actions, which is expected.
    assert emission_zero.action != emission_one.action

from __future__ import annotations

import numpy as np

from stego.causal_arithmetic import (
    CausalArithmeticDecoder,
    CausalArithmeticEncoder,
    PROBABILITY_MASS,
    _partition_interval,
    _quantized_probability_weights,
    _rank_candidates,
)
from stego.coding import Candidate


def _endogenous_candidates(history: tuple[str, ...]) -> list[Candidate]:
    # The labels exposed at the next step depend on the action just emitted.
    prefix = "root" if not history else history[-1]
    return [
        Candidate(f"{prefix}:0", 0.5),
        Candidate(f"{prefix}:1", 0.5),
    ]


def test_decoder_recovers_payload_without_encoder_width_metadata() -> None:
    payload = [1, 0, 1, 1, 0, 0, 1, 0]
    alice = CausalArithmeticEncoder(payload, precision_bits=32)
    bob = CausalArithmeticDecoder(payload_length=len(payload), precision_bits=32)
    alice_history: list[str] = []
    bob_history: list[str] = []

    while not alice.complete:
        alice_candidates = _endogenous_candidates(tuple(alice_history))
        bob_candidates = _endogenous_candidates(tuple(bob_history))
        emission = alice.emit(alice_candidates)
        bob.observe(emission.action, bob_candidates)
        alice_history.append(str(emission.action))
        bob_history.append(str(emission.action))

        assert alice.state == bob.state
        assert alice_history == bob_history

    assert bob.complete
    assert bob.decoded_payload() == payload
    assert len(alice_history) == len(payload)


def test_skewed_probabilities_remain_synchronized_under_endogenous_history() -> None:
    payload = [0, 1, 1, 0]
    alice = CausalArithmeticEncoder(payload, precision_bits=32)
    bob = CausalArithmeticDecoder(payload_length=len(payload), precision_bits=32)
    alice_history: list[str] = []
    bob_history: list[str] = []

    def candidates(history: tuple[str, ...]) -> list[Candidate]:
        prefix = "r" if not history else history[-1]
        return [
            Candidate(f"{prefix}:a", 0.7),
            Candidate(f"{prefix}:b", 0.2),
            Candidate(f"{prefix}:c", 0.1),
        ]

    for _ in range(32):
        if alice.complete:
            break
        emission = alice.emit(candidates(tuple(alice_history)))
        bob.observe(emission.action, candidates(tuple(bob_history)))
        alice_history.append(str(emission.action))
        bob_history.append(str(emission.action))
        assert alice.state == bob.state

    assert alice.complete
    assert bob.complete
    assert bob.decoded_payload() == payload


def test_candidate_mismatch_is_exposed_instead_of_hidden_by_oracle_widths() -> None:
    payload = [1, 0]
    alice = CausalArithmeticEncoder(payload, precision_bits=16)
    bob = CausalArithmeticDecoder(payload_length=len(payload), precision_bits=16)

    emission = alice.emit([Candidate("a", 0.5), Candidate("b", 0.5)])

    if emission.action == "a":
        mismatched = [Candidate("b", 1.0)]
    else:
        mismatched = [Candidate("a", 1.0)]

    try:
        bob.observe(emission.action, mismatched)
    except ValueError as exc:
        assert "not decodable" in str(exc)
    else:
        raise AssertionError("Bob must reject a mismatched causal candidate set")


def test_128_bit_partition_uses_exact_integer_width_arithmetic() -> None:
    candidates = _rank_candidates(
        [
            Candidate("dominant", 0.9999990),
            Candidate("tiny-a", 0.0000001),
            Candidate("tiny-b", 0.0000002),
            Candidate("tiny-c", 0.0000007),
        ]
    )
    weights = _quantized_probability_weights(candidates)
    assert sum(weights.values()) == PROBABILITY_MASS
    assert all(weights[item.action] > 0 for item in candidates)

    parent_high = 1 << 128
    intervals = _partition_interval(candidates, 0, parent_high)
    assert intervals[0][1] == 0
    assert intervals[-1][2] == parent_high
    assert all(high > low for _, low, high in intervals)
    assert sum(high - low for _, low, high in intervals) == parent_high


def test_random_skewed_128_bit_distributions_decode_without_drift() -> None:
    rng = np.random.default_rng(20260828)
    payload = rng.integers(0, 2, size=32).astype(int).tolist()
    alice = CausalArithmeticEncoder(payload, precision_bits=128)
    bob = CausalArithmeticDecoder(payload_length=len(payload), precision_bits=128)

    for step in range(256):
        if alice.complete:
            break
        raw = rng.lognormal(mean=0.0, sigma=5.0, size=12)
        raw = raw / raw.sum()
        candidates = [
            Candidate(f"step-{step}:action-{index}", float(probability))
            for index, probability in enumerate(raw)
        ]
        emission = alice.emit(candidates)
        bob.observe(emission.action, candidates)
        assert alice.state == bob.state
        assert alice.committed_prefix_bits == bob.committed_prefix_bits

    assert alice.complete
    assert bob.complete
    assert bob.decoded_payload() == payload

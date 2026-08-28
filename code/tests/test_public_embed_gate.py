from __future__ import annotations

from controllers.gating import PublicEmbedGate


def test_public_gate_is_deterministic_and_payload_independent_by_api() -> None:
    gate = PublicEmbedGate(0.25, seed=17)
    first = [gate.allow_embed(index) for index in range(100)]
    second = [gate.allow_embed(index) for index in range(100)]

    assert first == second


def test_larger_throttle_accepts_a_superset_of_public_events() -> None:
    low = PublicEmbedGate(0.20, seed=23)
    high = PublicEmbedGate(0.60, seed=23)

    for index in range(1000):
        if low.allow_embed(index):
            assert high.allow_embed(index)


def test_zero_and_one_throttles_are_exact_extremes() -> None:
    zero = PublicEmbedGate(0.0, seed=31)
    one = PublicEmbedGate(1.0, seed=31)

    assert not any(zero.allow_embed(index) for index in range(100))
    assert all(one.allow_embed(index) for index in range(100))

from __future__ import annotations

import pandas as pd

from evaluation.walk_design import (
    WalkTransition,
    simulate_detector_unaware_walk,
    zero_payload_walk_check,
)
from models.walk_cover import WalkCoverModel


def _model() -> WalkCoverModel:
    train = pd.DataFrame(
        {
            "source": ["A", "A", "B", "B", "C", "C", "X", "X"],
            "destination": ["B", "X", "C", "X", "D", "X", "B", "C"],
        }
    )
    return WalkCoverModel(prior_strength=1.0, top_k=4).fit(train)


def _natural_walk() -> list[WalkTransition]:
    return [
        WalkTransition("A", "B", 1.0),
        WalkTransition("B", "C", 2.0),
        WalkTransition("C", "D", 3.0),
    ]


def test_zero_payload_walk_uses_same_full_simulator_and_is_exact_identity() -> None:
    trace = zero_payload_walk_check(_model(), _natural_walk(), sequence_id="trip-1")

    assert trace.processed_transitions == 3
    assert trace.emitted_actions == ("B", "C", "D")
    assert trace.state_mismatch_count == 0
    assert trace.committed_payload_bits == 0
    assert trace.natural_features.shape == (3, 12)
    assert (trace.natural_features == trace.stego_features).all()


def test_payload_and_cover_random_streams_are_reproducible() -> None:
    first = simulate_detector_unaware_walk(
        _model(),
        _natural_walk(),
        sequence_id="trip-2",
        intensity=0.5,
        message_seed=23,
        payload_bits=4,
        precision_bits=32,
        nominal_bits=2,
    )
    second = simulate_detector_unaware_walk(
        _model(),
        _natural_walk(),
        sequence_id="trip-2",
        intensity=0.5,
        message_seed=23,
        payload_bits=4,
        precision_bits=32,
        nominal_bits=2,
    )

    assert first.emitted_actions == second.emitted_actions
    assert first.committed_payload_bits == second.committed_payload_bits
    assert first.state_mismatch_count == second.state_mismatch_count == 0
    assert (first.stego_features == second.stego_features).all()


def test_unseen_natural_source_does_not_authorise_embedding_support() -> None:
    model = _model()
    walk = [WalkTransition("UNSEEN", "natural", 1.0)]

    trace = simulate_detector_unaware_walk(
        model,
        walk,
        sequence_id="cold-start",
        intensity=1.0,
        message_seed=11,
        payload_bits=4,
        precision_bits=32,
        nominal_bits=2,
    )

    assert trace.emitted_actions == ("natural",)
    assert trace.committed_payload_bits == 0
    assert trace.dead_end is False

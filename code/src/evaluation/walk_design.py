from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from models.walk_cover import WalkCoverModel
from steganalysis.public_risk import observed_action_feature_vector
from stego.causal_arithmetic import CausalArithmeticDecoder, CausalArithmeticEncoder
from stego.policy_session import rate_limited_candidates
from stego.walk_semantics import WalkDeadEndError, cover_walk_action


@dataclass(frozen=True)
class WalkTransition:
    source: object
    destination: object
    timestamp: float


@dataclass(frozen=True)
class WalkTrace:
    natural_features: np.ndarray
    stego_features: np.ndarray
    emitted_actions: tuple[object, ...]
    committed_payload_bits: int
    complete: bool
    state_mismatch_count: int
    dead_end: bool
    processed_transitions: int


def deterministic_payload(sequence_id: str, *, seed: int, length: int) -> list[int]:
    if length < 1:
        raise ValueError("payload length must be positive")
    rng = np.random.default_rng(_stable_u64(f"payload:{seed}:{sequence_id}"))
    return rng.integers(0, 2, size=length).astype(int).tolist()


def simulate_detector_unaware_walk(
    model: WalkCoverModel,
    transitions: Sequence[WalkTransition],
    *,
    sequence_id: str,
    intensity: float,
    message_seed: int,
    payload_bits: int = 32,
    precision_bits: int = 128,
    nominal_bits: int = 4,
) -> WalkTrace:
    """Generate one paired natural/stego walk under a fixed embedding intensity.

    This routine is used to construct design-Eve traces before policy tuning.
    It deliberately contains no fuzzy-policy signal.  Three independent random
    streams are derived from public identifiers: payload bits, embedding
    opportunities, and COVER/PAUSE continuations.  Consequently a variable
    number of coder operations cannot shift the cover-continuation RNG.

    Natural features are scored under the public likelihood Q at the observed
    natural source.  Stego features are scored under Q at the actually emitted
    source.  Arithmetic embedding is restricted to A(H_t), represented by
    ``model.admissible_distribution``.
    """

    if not 0.0 <= intensity <= 1.0:
        raise ValueError("intensity must lie in [0, 1]")
    if not transitions:
        return WalkTrace(
            natural_features=np.empty((0, 12), dtype=float),
            stego_features=np.empty((0, 12), dtype=float),
            emitted_actions=(),
            committed_payload_bits=0,
            complete=False,
            state_mismatch_count=0,
            dead_end=False,
            processed_transitions=0,
        )

    payload = deterministic_payload(sequence_id, seed=message_seed, length=payload_bits)
    encoder = CausalArithmeticEncoder(payload, precision_bits=precision_bits)
    decoder = CausalArithmeticDecoder(
        payload_length=payload_bits,
        precision_bits=precision_bits,
    )
    opportunity_rng = np.random.default_rng(
        _stable_u64(f"opportunity:{message_seed}:{sequence_id}")
    )
    cover_rng = np.random.default_rng(
        _stable_u64(f"cover:{message_seed}:{sequence_id}")
    )

    emitted_source = transitions[0].source
    previous_natural_action: object | None = None
    previous_stego_action: object | None = None
    previous_timestamp: float | None = None
    natural_rows: list[np.ndarray] = []
    stego_rows: list[np.ndarray] = []
    emitted_actions: list[object] = []
    state_mismatches = 0
    dead_end = False

    for transition in transitions:
        observed_source = transition.source
        natural_action = transition.destination
        timestamp = float(transition.timestamp)
        gap = 0.0 if previous_timestamp is None else max(0.0, timestamp - previous_timestamp)

        natural_q = model.likelihood_distribution(observed_source)
        stego_q = model.likelihood_distribution(emitted_source)
        admissible = model.admissible_distribution(emitted_source)

        natural_feature = observed_action_feature_vector(
            source=observed_source,
            action=natural_action,
            previous_action=previous_natural_action,
            candidates=natural_q,
            gap=gap,
            context_seen=model.has_context(observed_source),
            training_destinations=model.destinations,
        )

        should_embed = bool(
            not encoder.complete
            and len(admissible) >= 2
            and opportunity_rng.random() < intensity
        )
        if should_embed:
            support = rate_limited_candidates(admissible, nominal_bits=nominal_bits)
            emission = encoder.emit(support)
            decoder.observe(emission.action, support)
            stego_action = emission.action
            if encoder.state != decoder.state:
                state_mismatches += 1
        else:
            try:
                stego_action = cover_walk_action(
                    observed_source=observed_source,
                    natural_action=natural_action,
                    emitted_source=emitted_source,
                    admissible_candidates=admissible,
                    cover_rng=cover_rng,
                )
            except WalkDeadEndError:
                dead_end = True
                break

        stego_feature = observed_action_feature_vector(
            source=emitted_source,
            action=stego_action,
            previous_action=previous_stego_action,
            candidates=stego_q,
            gap=gap,
            context_seen=model.has_context(emitted_source),
            training_destinations=model.destinations,
        )
        natural_rows.append(natural_feature)
        stego_rows.append(stego_feature)
        emitted_actions.append(stego_action)

        previous_natural_action = natural_action
        previous_stego_action = stego_action
        previous_timestamp = timestamp
        emitted_source = stego_action

    natural_matrix = (
        np.vstack(natural_rows)
        if natural_rows
        else np.empty((0, 12), dtype=float)
    )
    stego_matrix = (
        np.vstack(stego_rows)
        if stego_rows
        else np.empty((0, 12), dtype=float)
    )
    return WalkTrace(
        natural_features=natural_matrix,
        stego_features=stego_matrix,
        emitted_actions=tuple(emitted_actions),
        committed_payload_bits=encoder.committed_prefix_bits,
        complete=encoder.complete,
        state_mismatch_count=state_mismatches,
        dead_end=dead_end,
        processed_transitions=len(emitted_actions),
    )


def zero_payload_walk_check(
    model: WalkCoverModel,
    transitions: Sequence[WalkTransition],
    *,
    sequence_id: str,
) -> WalkTrace:
    """Run the null-channel hard gate through the same walk simulator."""

    trace = simulate_detector_unaware_walk(
        model,
        transitions,
        sequence_id=sequence_id,
        intensity=0.0,
        message_seed=0,
        payload_bits=1,
        precision_bits=128,
        nominal_bits=1,
    )
    if trace.dead_end:
        raise AssertionError("zero-payload natural walk reached a simulator dead end")
    if trace.processed_transitions != len(transitions):
        raise AssertionError("zero-payload simulator did not preserve walk length")
    natural_actions = tuple(item.destination for item in transitions)
    if trace.emitted_actions != natural_actions:
        raise AssertionError("zero-payload emitted actions differ from natural walk")
    if not np.array_equal(trace.natural_features, trace.stego_features):
        raise AssertionError("zero-payload public feature vectors differ")
    if trace.committed_payload_bits != 0:
        raise AssertionError("zero-payload hard gate advanced the arithmetic coder")
    return trace


def _stable_u64(value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(value.encode("utf-8")).digest()[:8],
        "big",
        signed=False,
    )

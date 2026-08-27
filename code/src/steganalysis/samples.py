from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from models.cover_model import CoverModel
from models.temporal import temporal_history_key
from stego.coding import Candidate, encode_next_action, encode_next_action_range

FEATURE_COLUMNS = [
    "action_probability",
    "surprise_bits",
    "rank_fraction",
    "is_top_action",
    "entropy_bits",
    "top_probability",
    "candidate_count",
    "unseen_context",
    "unseen_destination",
    "same_as_previous",
    "self_loop",
    "log_inter_event_gap",
]

WHITE_BOX_FEATURE_COLUMNS = [
    *FEATURE_COLUMNS,
    "encoder_capacity_bits",
    "encoder_tv_bound",
    "encoder_kl_bound",
    "embedding_feasible",
]

ORACLE_LEAKAGE_COLUMNS = [
    "bits_consumed",
    "local_total_variation",
    "local_kl_bits",
]

TRANSITION_SEMANTICS = frozenset({"actor_action", "walk"})


@dataclass(frozen=True)
class SampleConfig:
    max_bits_per_transition: int
    seed: int
    max_local_total_variation: float = 0.10
    max_local_kl_bits: float = 0.10
    min_entropy_bits: float = 1.0
    cover_when_unsafe: bool = True
    codec_backend: str = "range"
    min_encoded_probability: float = 0.0
    max_encoded_surprise_bits: float = float("inf")
    max_encoded_rank_fraction: float = 1.0
    require_encoded_top_action: bool = False
    require_encoded_self_loop: bool = False
    transition_semantics: str = "actor_action"

    def __post_init__(self) -> None:
        if self.transition_semantics not in TRANSITION_SEMANTICS:
            raise ValueError(
                f"transition_semantics must be one of {sorted(TRANSITION_SEMANTICS)}"
            )


def make_steganalysis_records(
    model: CoverModel,
    frame: pd.DataFrame,
    *,
    split: str,
    config: SampleConfig,
) -> pd.DataFrame:
    """Generate paired natural/stego records with distinct causal histories.

    ``actor_action`` keeps the observed actor/source identity fixed while the
    previous emitted action changes later context. ``walk`` treats each emitted
    destination as the next source node of that sequence, which is required for
    mobility and other node-to-node traversals.

    For walk semantics, COVER has two cases. Before divergence, the observed
    natural continuation is retained exactly, even when it lies outside the
    learned model's top-k support; it is an observed valid transition. After a
    steganographic divergence, the counterfactual natural destination is no
    longer necessarily reachable from the emitted source, so COVER follows the
    highest-probability admissible continuation returned for the actual stego
    source. Consequently, a zero-payload policy is exactly the natural path and
    cannot acquire detectability from the simulator itself.
    """

    rows = []
    rng = np.random.default_rng(config.seed + _stable_offset(split))
    natural_previous_by_history: dict[Hashable, Hashable] = {}
    stego_previous_by_history: dict[Hashable, Hashable] = {}
    previous_timestamp_by_history: dict[Hashable, float] = {}
    ordered = frame.sort_values(["timestamp"], kind="stable")

    columns = ["source", "destination", "timestamp"]
    has_sequence = "sequence_id" in ordered.columns
    if has_sequence:
        columns.append("sequence_id")
    if config.transition_semantics == "walk" and not has_sequence:
        raise ValueError("walk transition semantics require sequence_id")

    for index, record in enumerate(ordered[columns].itertuples(index=False, name=None)):
        observed_source, natural_destination, timestamp = record[0], record[1], record[2]
        sequence_id: Any | None = record[3] if has_sequence else None
        history_key = temporal_history_key(observed_source, sequence_id)
        natural_previous_emitted = natural_previous_by_history.get(history_key)
        stego_previous_emitted = stego_previous_by_history.get(history_key)

        if config.transition_semantics == "walk":
            natural_source = observed_source
            stego_source = (
                stego_previous_emitted
                if stego_previous_emitted is not None
                else observed_source
            )
            natural_model_previous = None
            stego_model_previous = None
        else:
            natural_source = observed_source
            stego_source = observed_source
            natural_model_previous = natural_previous_emitted
            stego_model_previous = stego_previous_emitted

        natural_candidates = model.candidate_distribution(
            natural_source, natural_model_previous
        )
        stego_candidates = model.candidate_distribution(stego_source, stego_model_previous)
        if not natural_candidates or not stego_candidates:
            raise ValueError(
                "Cover model returned an empty admissible candidate set; "
                "ASOC V2 policy code must handle this state as a dead end/STOP rather than encode it"
            )
        bits = rng.integers(0, 2, size=config.max_bits_per_transition).tolist()

        natural_encoded = _encode(bits, natural_candidates, config=config)
        natural_entropy = _entropy(
            [candidate.probability for candidate in natural_candidates]
        )
        natural_encoded_features = _action_position_features(
            source=natural_source,
            action=natural_encoded.action,
            candidates=natural_candidates,
        )
        natural_feasible = _embedding_feasible(
            natural_encoded,
            entropy=natural_entropy,
            encoded_features=natural_encoded_features,
            config=config,
        )

        encoded = _encode(bits, stego_candidates, config=config)
        stego_entropy = _entropy([candidate.probability for candidate in stego_candidates])
        encoded_features = _action_position_features(
            source=stego_source,
            action=encoded.action,
            candidates=stego_candidates,
        )
        can_embed = _embedding_feasible(
            encoded,
            entropy=stego_entropy,
            encoded_features=encoded_features,
            config=config,
        )

        if can_embed:
            stego_action = encoded.action
            stego_bits = encoded.bits_consumed
            stego_tv = encoded.local_total_variation
            stego_kl = encoded.local_kl_bits
            stego_mode = "EMBED"
        elif not config.cover_when_unsafe:
            stego_action = encoded.action
            stego_bits = encoded.bits_consumed
            stego_tv = encoded.local_total_variation
            stego_kl = encoded.local_kl_bits
            stego_mode = "FORCED_EMBED"
        else:
            if config.transition_semantics == "walk" and stego_source != observed_source:
                # The natural counterfactual may be unreachable after a prior
                # stego action. Follow an admissible cover continuation instead.
                stego_action = stego_candidates[0].action
            else:
                # Before divergence this preserves the observed stream exactly.
                stego_action = natural_destination
            stego_bits = 0
            stego_tv = 0.0
            stego_kl = 0.0
            stego_mode = "COVER"

        gap = (
            float(timestamp) - float(previous_timestamp_by_history[history_key])
            if history_key in previous_timestamp_by_history
            else 0.0
        )
        sequence_value = "" if sequence_id is None or _is_missing(sequence_id) else str(sequence_id)

        rows.append(
            {
                "split": split,
                "pair_id": index,
                "source": str(natural_source),
                "observed_source": str(observed_source),
                "sequence_id": sequence_value,
                "label": 0,
                "action": str(natural_destination),
                "previous_action": "" if natural_previous_emitted is None else str(natural_previous_emitted),
                "stego_mode": "NATURAL",
                **_features(
                    source=natural_source,
                    action=natural_destination,
                    previous=natural_model_previous,
                    candidates=natural_candidates,
                    gap=gap,
                    bits_consumed=0,
                    local_total_variation=0.0,
                    local_kl_bits=0.0,
                    encoder_capacity_bits=natural_encoded.bits_consumed,
                    encoder_tv_bound=natural_encoded.local_total_variation,
                    encoder_kl_bound=natural_encoded.local_kl_bits,
                    embedding_feasible=natural_feasible,
                    training_destination_seen=natural_destination in model.destinations,
                    context_seen=model.has_context(natural_source, natural_model_previous),
                ),
            }
        )
        rows.append(
            {
                "split": split,
                "pair_id": index,
                "source": str(stego_source),
                "observed_source": str(observed_source),
                "sequence_id": sequence_value,
                "label": 1,
                "action": str(stego_action),
                "previous_action": "" if stego_previous_emitted is None else str(stego_previous_emitted),
                "stego_mode": stego_mode,
                **_features(
                    source=stego_source,
                    action=stego_action,
                    previous=stego_model_previous,
                    candidates=stego_candidates,
                    gap=gap,
                    bits_consumed=stego_bits,
                    local_total_variation=stego_tv,
                    local_kl_bits=stego_kl,
                    encoder_capacity_bits=encoded.bits_consumed,
                    encoder_tv_bound=encoded.local_total_variation,
                    encoder_kl_bound=encoded.local_kl_bits,
                    embedding_feasible=can_embed,
                    training_destination_seen=stego_action in model.destinations,
                    context_seen=model.has_context(stego_source, stego_model_previous),
                ),
            }
        )

        natural_previous_by_history[history_key] = natural_destination
        stego_previous_by_history[history_key] = stego_action
        previous_timestamp_by_history[history_key] = float(timestamp)
        if hasattr(model, "update_timestamp"):
            model.update_timestamp(stego_source, float(timestamp))

    return pd.DataFrame(rows)


def feature_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return frame[FEATURE_COLUMNS].to_numpy(dtype=float), frame["label"].to_numpy(dtype=int)


def _embedding_feasible(
    encoded: object,
    *,
    entropy: float,
    encoded_features: dict[str, float | int],
    config: SampleConfig,
) -> bool:
    return bool(
        encoded.bits_consumed > 0
        and entropy >= config.min_entropy_bits
        and encoded.local_total_variation <= config.max_local_total_variation
        and encoded.local_kl_bits <= config.max_local_kl_bits
        and encoded_features["action_probability"] >= config.min_encoded_probability
        and encoded_features["surprise_bits"] <= config.max_encoded_surprise_bits
        and encoded_features["rank_fraction"] <= config.max_encoded_rank_fraction
        and (
            not config.require_encoded_top_action
            or bool(encoded_features["is_top_action"])
        )
        and (
            not config.require_encoded_self_loop
            or bool(encoded_features["self_loop"])
        )
    )


def _features(
    *,
    source: Hashable,
    action: Hashable,
    previous: Hashable | None,
    candidates: Sequence[Candidate],
    gap: float,
    bits_consumed: int,
    local_total_variation: float,
    local_kl_bits: float,
    encoder_capacity_bits: int,
    encoder_tv_bound: float,
    encoder_kl_bound: float,
    embedding_feasible: bool,
    training_destination_seen: bool,
    context_seen: bool,
) -> dict[str, float | int]:
    position = _action_position_features(source=source, action=action, candidates=candidates)
    probability = position["action_probability"]
    rank_fraction = position["rank_fraction"]
    is_top_action = position["is_top_action"]
    self_loop = position["self_loop"]
    entropy = _entropy([candidate.probability for candidate in candidates])
    return {
        "action_probability": probability,
        "surprise_bits": position["surprise_bits"],
        "rank_fraction": rank_fraction,
        "is_top_action": is_top_action,
        "entropy_bits": entropy,
        "top_probability": candidates[0].probability,
        "candidate_count": len(candidates),
        "unseen_context": int(not context_seen),
        "unseen_destination": int(not training_destination_seen),
        "same_as_previous": int(previous == action),
        "self_loop": self_loop,
        "log_inter_event_gap": math.log1p(max(0.0, gap)),
        "encoder_capacity_bits": encoder_capacity_bits,
        "encoder_tv_bound": encoder_tv_bound,
        "encoder_kl_bound": encoder_kl_bound,
        "embedding_feasible": int(embedding_feasible),
        "bits_consumed": bits_consumed,
        "local_total_variation": local_total_variation,
        "local_kl_bits": local_kl_bits,
    }


def _action_position_features(
    *,
    source: Hashable,
    action: Hashable,
    candidates: Sequence[Candidate],
) -> dict[str, float | int]:
    probabilities = {candidate.action: candidate.probability for candidate in candidates}
    ranked_actions = [candidate.action for candidate in candidates]
    probability = probabilities.get(action, min(probabilities.values()) * 0.5)
    rank = ranked_actions.index(action) + 1 if action in ranked_actions else len(ranked_actions) + 1
    return {
        "action_probability": probability,
        "surprise_bits": -math.log2(max(probability, np.finfo(float).tiny)),
        "rank_fraction": rank / max(1, len(ranked_actions)),
        "is_top_action": int(rank == 1),
        "self_loop": int(source == action),
    }


def _encode(bits: Sequence[int], candidates: Sequence[Candidate], *, config: SampleConfig):
    if config.codec_backend == "range":
        return encode_next_action_range(bits, candidates, max_bits=config.max_bits_per_transition)
    if config.codec_backend == "quantized":
        return encode_next_action(bits, candidates, max_bits=config.max_bits_per_transition)
    raise ValueError(f"Unknown codec backend: {config.codec_backend}")


def _entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values[values > 0]
    return float(-(values * np.log2(values)).sum())


def _stable_offset(value: str) -> int:
    return sum(ord(character) for character in value)


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False

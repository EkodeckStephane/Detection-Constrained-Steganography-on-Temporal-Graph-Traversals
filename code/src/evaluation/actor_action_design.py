from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import differential_evolution
from sklearn.metrics import roc_auc_score

from controllers.fuzzy import ControllerInputs, FuzzyRateController, FuzzyWeights
from controllers.inputs import build_public_controller_inputs
from data.adapters import read_bipartite_interactions
from models.temporal import TemporalBackoffModel
from steganalysis.bootstrap import WorstAucBootstrap, cluster_bootstrap_worst_adversarial_auc
from steganalysis.detectors import (
    OrientedDetector,
    adversarial_auc,
    fit_detector,
    orient_detector,
)
from steganalysis.public_risk import (
    observed_action_feature_vector,
    public_reference_risk,
)
from stego.causal_arithmetic import CausalArithmeticDecoder, CausalArithmeticEncoder
from stego.policy_session import SynchronizedPolicySession, rate_limited_candidates


@dataclass(frozen=True)
class ActorSplitSpec:
    dataset_id: str
    cover_train_end: float
    eve_train_end: float
    policy_validation_end: float
    cover_train_rows: int
    eve_train_rows: int
    policy_validation_rows: int

    @property
    def design_rows(self) -> int:
        return self.cover_train_rows + self.eve_train_rows + self.policy_validation_rows


@dataclass(frozen=True)
class SearchCandidate:
    weights: FuzzyWeights
    stop_threshold: float
    surrogate_score: float
    surrogate_nominal_bits_per_transition: float
    surrogate_embed_rate: float
    surrogate_mean_embed_risk: float


@dataclass(frozen=True)
class SeedCertification:
    seed: int
    bits_per_transition: float
    completion_rate: float
    abstention_rate: float
    mode_rates: dict[str, float]
    worst_adversarial_auc: float
    auc_ci_lower: float
    auc_ci_upper: float
    detector_adversarial_auc: dict[str, float]
    bootstrap_valid_resamples: int
    passive_decode_success_rate: float
    passive_state_mismatch_rate: float
    invalid_transition_rate: float
    sessions: int
    completed_sessions: int


@dataclass(frozen=True)
class CandidateCertification:
    weights: FuzzyWeights
    stop_threshold: float
    seeds: tuple[SeedCertification, ...]
    feasible: bool
    mean_bits_per_transition: float
    mean_completion_rate: float
    mean_abstention_rate: float
    max_auc_ci_upper: float


@dataclass(frozen=True)
class DesignEveSummary:
    calibration_auc: dict[str, float]
    reverse_score: dict[str, bool]
    fit_rows: int
    calibration_rows: int
    intensity_counts: dict[str, int]


def load_actor_design_prefix(
    raw_path: Path,
    split_manifest_path: Path,
    dataset_id: str,
) -> tuple[pd.DataFrame, ActorSplitSpec]:
    """Load exactly cover+Eve+validation rows and no later region."""

    manifest = _load_yaml(split_manifest_path)
    dataset = manifest["datasets"][dataset_id]
    counts = dataset["counts"]
    cutoffs = dataset["timestamp_cutoffs"]
    if any(not isinstance(counts[name], int) for name in ("cover_train", "eve_train", "policy_validation")):
        raise ValueError(f"{dataset_id} is not an actor-action row-count manifest")
    spec = ActorSplitSpec(
        dataset_id=dataset_id,
        cover_train_end=float(cutoffs["cover_train_end"]),
        eve_train_end=float(cutoffs["eve_train_end"]),
        policy_validation_end=float(cutoffs["policy_validation_end"]),
        cover_train_rows=int(counts["cover_train"]),
        eve_train_rows=int(counts["eve_train"]),
        policy_validation_rows=int(counts["policy_validation"]),
    )
    frame = read_bipartite_interactions(raw_path, nrows=spec.design_rows)
    if len(frame) != spec.design_rows:
        raise AssertionError("design-prefix row count changed after loading")
    if float(frame["timestamp"].max()) > spec.policy_validation_end:
        raise AssertionError("post-validation observation entered the design prefix")

    frame = frame.copy()
    frame["split"] = np.select(
        [
            frame["timestamp"] <= spec.cover_train_end,
            frame["timestamp"] <= spec.eve_train_end,
        ],
        ["cover_train", "eve_train"],
        default="policy_validation",
    )
    observed = frame["split"].value_counts().to_dict()
    expected = {
        "cover_train": spec.cover_train_rows,
        "eve_train": spec.eve_train_rows,
        "policy_validation": spec.policy_validation_rows,
    }
    if observed != expected:
        raise ValueError(f"frozen split counts changed: expected {expected}, observed {observed}")
    return frame, spec


def fit_actor_design_eves(
    model: TemporalBackoffModel,
    eve_frame: pd.DataFrame,
    *,
    detector_names: Sequence[str] = ("linear", "forest"),
    intensities: Sequence[float] = (0.05, 0.20, 0.50, 1.00),
    payload_bits: int = 32,
    precision_bits: int = 128,
    nominal_bits: int = 4,
    seed: int = 20260827,
    fit_fraction: float = 0.75,
) -> tuple[dict[str, OrientedDetector], DesignEveSummary]:
    """Fit design-Eves on detector-unaware causal arithmetic traces.

    Each source is deterministically assigned one declared embedding intensity,
    so the complete Eve-training region is used once rather than duplicating
    natural observations four times. Score orientation is frozen on the final
    chronological quarter of Eve-training and never uses policy validation.
    """

    if not 0 < fit_fraction < 1:
        raise ValueError("fit_fraction must lie in (0, 1)")
    intensity_values = tuple(float(value) for value in intensities)
    if not intensity_values or any(not 0 <= value <= 1 for value in intensity_values):
        raise ValueError("intensities must be non-empty values in [0, 1]")

    x, y, timestamps, assigned = _exploratory_eve_records(
        model,
        eve_frame,
        intensities=intensity_values,
        payload_bits=payload_bits,
        precision_bits=precision_bits,
        nominal_bits=nominal_bits,
        seed=seed,
    )
    unique_times = np.sort(np.unique(timestamps))
    cutoff_position = min(
        len(unique_times) - 2,
        max(0, int(math.ceil(fit_fraction * len(unique_times))) - 1),
    )
    cutoff = unique_times[cutoff_position]
    fit_mask = timestamps <= cutoff
    calibration_mask = timestamps > cutoff
    if np.unique(y[fit_mask]).size != 2 or np.unique(y[calibration_mask]).size != 2:
        raise ValueError("both Eve fit and orientation blocks require cover and stego labels")

    detectors: dict[str, OrientedDetector] = {}
    for offset, name in enumerate(detector_names):
        detector = fit_detector(name, x[fit_mask], y[fit_mask], seed=seed + offset)
        detectors[name] = orient_detector(
            detector,
            x[calibration_mask],
            y[calibration_mask],
        )
    summary = DesignEveSummary(
        calibration_auc={name: float(item.calibration_auc) for name, item in detectors.items()},
        reverse_score={name: bool(item.reverse_score) for name, item in detectors.items()},
        fit_rows=int(fit_mask.sum()),
        calibration_rows=int(calibration_mask.sum()),
        intensity_counts={str(value): int(sum(item == value for item in assigned)) for value in intensity_values},
    )
    return detectors, summary


def propose_actor_candidates(
    model: TemporalBackoffModel,
    detectors: Mapping[str, OrientedDetector],
    validation_frame: pd.DataFrame,
    *,
    search_rows: int = 3000,
    shortlist_size: int = 12,
    population_size: int = 8,
    generations: int = 8,
    seed: int = 20260827,
    max_bits_per_transition: int = 4,
    payload_bits: int = 32,
) -> tuple[SearchCandidate, ...]:
    """Use a deterministic natural-history surrogate only to propose candidates."""

    inputs = _natural_surrogate_inputs(
        model,
        detectors,
        validation_frame,
        search_rows=search_rows,
        payload_bits=payload_bits,
    )
    cache: dict[tuple[float, ...], SearchCandidate] = {}
    weight_fields = tuple(item.name for item in fields(FuzzyWeights))

    def objective(vector: np.ndarray) -> float:
        key = tuple(np.round(vector, 7))
        if key not in cache:
            weights = FuzzyWeights(**dict(zip(weight_fields, vector[:-1], strict=True)))
            controller = FuzzyRateController(
                max_bits_per_transition=max_bits_per_transition,
                stop_threshold=float(vector[-1]),
                weights=weights,
            )
            decisions = [controller.decide(item) for item in inputs]
            embedded = [
                (decision, item)
                for decision, item in zip(decisions, inputs, strict=True)
                if decision.mode == "EMBED"
            ]
            nominal = float(sum(decision.local_payload_bits for decision in decisions) / len(decisions))
            embed_rate = float(len(embedded) / len(decisions))
            mean_risk = float(np.mean([item.steganalysis_risk for _, item in embedded])) if embedded else 1.0
            score = nominal - 0.5 * mean_risk - 0.02 * (1.0 - embed_rate)
            cache[key] = SearchCandidate(
                weights=weights,
                stop_threshold=float(vector[-1]),
                surrogate_score=score,
                surrogate_nominal_bits_per_transition=nominal,
                surrogate_embed_rate=embed_rate,
                surrogate_mean_embed_risk=mean_risk,
            )
        return -cache[key].surrogate_score

    bounds = [(0.0, 1.0)] * len(weight_fields) + [(0.80, 0.99)]
    differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        popsize=population_size,
        maxiter=generations,
        polish=False,
        updating="immediate",
        workers=1,
    )
    ranked = sorted(cache.values(), key=lambda item: item.surrogate_score, reverse=True)
    selected: list[SearchCandidate] = []
    seen: set[tuple[float, ...]] = set()
    for item in ranked:
        key = tuple(round(float(getattr(item.weights, name)), 3) for name in weight_fields) + (
            round(item.stop_threshold, 3),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) >= shortlist_size:
            break
    if not selected:
        raise RuntimeError("policy search produced no candidate")
    return tuple(selected)


def certify_actor_candidate(
    model: TemporalBackoffModel,
    detectors: Mapping[str, OrientedDetector],
    validation_frame: pd.DataFrame,
    candidate: SearchCandidate,
    *,
    message_seeds: Sequence[int] = (11, 23, 37, 53, 71),
    payload_bits: int = 32,
    precision_bits: int = 128,
    max_bits_per_transition: int = 4,
    bootstrap_resamples: int = 10000,
    confidence_level: float = 0.95,
    detector_auc_budget: float = 0.60,
) -> CandidateCertification:
    """Certify one candidate on every validation row and every message seed."""

    natural_features, clusters = _natural_validation_features(model, validation_frame)
    seed_results: list[SeedCertification] = []
    for seed in message_seeds:
        result = _replay_actor_policy(
            model,
            detectors,
            validation_frame,
            natural_features=natural_features,
            clusters=clusters,
            weights=candidate.weights,
            stop_threshold=candidate.stop_threshold,
            message_seed=int(seed),
            payload_bits=payload_bits,
            precision_bits=precision_bits,
            max_bits_per_transition=max_bits_per_transition,
            bootstrap_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
        )
        seed_results.append(result)

    feasible = all(
        item.auc_ci_upper <= detector_auc_budget
        and item.invalid_transition_rate == 0.0
        and item.passive_state_mismatch_rate == 0.0
        and item.passive_decode_success_rate == 1.0
        for item in seed_results
    )
    return CandidateCertification(
        weights=candidate.weights,
        stop_threshold=candidate.stop_threshold,
        seeds=tuple(seed_results),
        feasible=bool(feasible),
        mean_bits_per_transition=float(np.mean([item.bits_per_transition for item in seed_results])),
        mean_completion_rate=float(np.mean([item.completion_rate for item in seed_results])),
        mean_abstention_rate=float(np.mean([item.abstention_rate for item in seed_results])),
        max_auc_ci_upper=float(max(item.auc_ci_upper for item in seed_results)),
    )


def select_certified_actor_policy(
    certifications: Sequence[CandidateCertification],
) -> CandidateCertification:
    feasible = [item for item in certifications if item.feasible]
    if not feasible:
        raise ValueError("no actor-action candidate passed full validation certification")
    return max(
        feasible,
        key=lambda item: (
            item.mean_bits_per_transition,
            item.mean_completion_rate,
            -item.max_auc_ci_upper,
            -item.mean_abstention_rate,
        ),
    )


def certification_to_dict(value: CandidateCertification) -> dict[str, Any]:
    return {
        "weights": asdict(value.weights),
        "stop_threshold": value.stop_threshold,
        "feasible": value.feasible,
        "mean_bits_per_transition": value.mean_bits_per_transition,
        "mean_completion_rate": value.mean_completion_rate,
        "mean_abstention_rate": value.mean_abstention_rate,
        "max_auc_ci_upper": value.max_auc_ci_upper,
        "seeds": [asdict(item) for item in value.seeds],
    }


def _exploratory_eve_records(
    model: TemporalBackoffModel,
    frame: pd.DataFrame,
    *,
    intensities: tuple[float, ...],
    payload_bits: int,
    precision_bits: int,
    nominal_bits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    natural_previous: dict[str, str] = {}
    stego_previous: dict[str, str] = {}
    previous_timestamp: dict[str, float] = {}
    encoders: dict[str, CausalArithmeticEncoder] = {}
    decoders: dict[str, CausalArithmeticDecoder] = {}
    source_intensity: dict[str, float] = {}
    source_rng: dict[str, np.random.Generator] = {}
    rows: list[np.ndarray] = []
    labels: list[int] = []
    timestamps: list[float] = []
    assigned: list[float] = []

    for source, natural_action, timestamp in frame[["source", "destination", "timestamp"]].itertuples(index=False, name=None):
        source = str(source)
        natural_action = str(natural_action)
        previous_natural = natural_previous.get(source)
        previous_stego = stego_previous.get(source)
        gap = max(0.0, float(timestamp) - previous_timestamp.get(source, float(timestamp)))
        natural_candidates = model.candidate_distribution(source, previous_natural)
        stego_candidates = model.candidate_distribution(source, previous_stego)

        if source not in source_intensity:
            intensity = intensities[_stable_u64(f"intensity:{seed}:{source}") % len(intensities)]
            source_intensity[source] = intensity
            source_rng[source] = np.random.default_rng(_stable_u64(f"rng:{seed}:{source}"))
            payload = _payload_for_source(source, seed=seed, length=payload_bits)
            encoders[source] = CausalArithmeticEncoder(payload, precision_bits=precision_bits)
            decoders[source] = CausalArithmeticDecoder(payload_length=payload_bits, precision_bits=precision_bits)
        intensity = source_intensity[source]
        encoder = encoders[source]
        decoder = decoders[source]

        stego_action = natural_action
        if (
            not encoder.complete
            and len(stego_candidates) >= 2
            and source_rng[source].random() < intensity
        ):
            public_support = rate_limited_candidates(stego_candidates, nominal_bits=nominal_bits)
            try:
                emission = encoder.emit(public_support)
                decoder.observe(emission.action, public_support)
                if encoder.state != decoder.state:
                    raise AssertionError("exploratory Eve trace desynchronized")
                stego_action = str(emission.action)
            except ValueError:
                stego_action = natural_action

        rows.append(
            observed_action_feature_vector(
                source=source,
                action=natural_action,
                previous_action=previous_natural,
                candidates=natural_candidates,
                gap=gap,
                context_seen=model.has_context(source, previous_natural),
                training_destinations=model.destinations,
            )
        )
        labels.append(0)
        timestamps.append(float(timestamp))
        assigned.append(intensity)
        rows.append(
            observed_action_feature_vector(
                source=source,
                action=stego_action,
                previous_action=previous_stego,
                candidates=stego_candidates,
                gap=gap,
                context_seen=model.has_context(source, previous_stego),
                training_destinations=model.destinations,
            )
        )
        labels.append(1)
        timestamps.append(float(timestamp))
        assigned.append(intensity)

        natural_previous[source] = natural_action
        stego_previous[source] = stego_action
        previous_timestamp[source] = float(timestamp)

    return (
        np.vstack(rows),
        np.asarray(labels, dtype=int),
        np.asarray(timestamps, dtype=float),
        assigned,
    )


def _natural_surrogate_inputs(
    model: TemporalBackoffModel,
    detectors: Mapping[str, OrientedDetector],
    validation_frame: pd.DataFrame,
    *,
    search_rows: int,
    payload_bits: int,
) -> list[ControllerInputs]:
    enriched = validation_frame[["source", "destination", "timestamp"]].copy()
    enriched["previous_action"] = enriched.groupby("source", sort=False)["destination"].shift(1)
    enriched["previous_timestamp"] = enriched.groupby("source", sort=False)["timestamp"].shift(1)
    if len(enriched) > search_rows:
        positions = np.unique(np.linspace(0, len(enriched) - 1, search_rows, dtype=int))
        enriched = enriched.iloc[positions]

    inputs: list[ControllerInputs] = []
    for row in enriched.itertuples(index=False):
        source = str(row.source)
        previous = None if pd.isna(row.previous_action) else str(row.previous_action)
        gap = 0.0 if pd.isna(row.previous_timestamp) else max(0.0, float(row.timestamp) - float(row.previous_timestamp))
        candidates = model.candidate_distribution(source, previous)
        reference = public_reference_risk(
            detectors,
            source=source,
            previous_action=previous,
            candidates=candidates,
            gap=gap,
            context_seen=model.has_context(source, previous),
            training_destinations=model.destinations,
        )
        inputs.append(
            build_public_controller_inputs(
                candidates=candidates,
                top_k=model.top_k,
                context_observations=model.context_observation_count(source, previous),
                context_seen=model.has_context(source, previous),
                steganalysis_risk=reference.risk,
                committed_payload_bits=0,
                payload_length=payload_bits,
            )
        )
    return inputs


def _natural_validation_features(
    model: TemporalBackoffModel,
    validation_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    previous: dict[str, str] = {}
    previous_timestamp: dict[str, float] = {}
    rows: list[np.ndarray] = []
    clusters: list[str] = []
    for source, action, timestamp in validation_frame[["source", "destination", "timestamp"]].itertuples(index=False, name=None):
        source = str(source)
        action = str(action)
        prior = previous.get(source)
        gap = max(0.0, float(timestamp) - previous_timestamp.get(source, float(timestamp)))
        candidates = model.candidate_distribution(source, prior)
        rows.append(
            observed_action_feature_vector(
                source=source,
                action=action,
                previous_action=prior,
                candidates=candidates,
                gap=gap,
                context_seen=model.has_context(source, prior),
                training_destinations=model.destinations,
            )
        )
        clusters.append(source)
        previous[source] = action
        previous_timestamp[source] = float(timestamp)
    return np.vstack(rows), np.asarray(clusters, dtype=object)


def _replay_actor_policy(
    model: TemporalBackoffModel,
    detectors: Mapping[str, OrientedDetector],
    validation_frame: pd.DataFrame,
    *,
    natural_features: np.ndarray,
    clusters: np.ndarray,
    weights: FuzzyWeights,
    stop_threshold: float,
    message_seed: int,
    payload_bits: int,
    precision_bits: int,
    max_bits_per_transition: int,
    bootstrap_resamples: int,
    confidence_level: float,
) -> SeedCertification:
    controller_a = FuzzyRateController(
        max_bits_per_transition=max_bits_per_transition,
        stop_threshold=stop_threshold,
        weights=weights,
    )
    controller_b = FuzzyRateController(
        max_bits_per_transition=max_bits_per_transition,
        stop_threshold=stop_threshold,
        weights=weights,
    )
    sessions: dict[str, SynchronizedPolicySession] = {}
    payloads: dict[str, list[int]] = {}
    stego_previous: dict[str, str] = {}
    previous_timestamp: dict[str, float] = {}
    stego_rows: list[np.ndarray] = []
    modes: Counter[str] = Counter()
    total_committed_gain = 0
    state_mismatches = 0
    invalid = 0

    for source, natural_action, timestamp in validation_frame[["source", "destination", "timestamp"]].itertuples(index=False, name=None):
        source = str(source)
        natural_action = str(natural_action)
        prior = stego_previous.get(source)
        gap = max(0.0, float(timestamp) - previous_timestamp.get(source, float(timestamp)))
        candidates = model.candidate_distribution(source, prior)
        if source not in sessions:
            payload = _payload_for_source(source, seed=message_seed, length=payload_bits)
            payloads[source] = payload
            sessions[source] = SynchronizedPolicySession(
                payload,
                sender_controller=controller_a,
                receiver_controller=controller_b,
                precision_bits=precision_bits,
            )
        session = sessions[source]

        if session.complete or session.terminated:
            inputs = build_public_controller_inputs(
                candidates=candidates,
                top_k=model.top_k,
                context_observations=model.context_observation_count(source, prior),
                context_seen=model.has_context(source, prior),
                steganalysis_risk=0.0,
                committed_payload_bits=session.committed_payload_bits,
                payload_length=payload_bits,
            )
        else:
            reference = public_reference_risk(
                detectors,
                source=source,
                previous_action=prior,
                candidates=candidates,
                gap=gap,
                context_seen=model.has_context(source, prior),
                training_destinations=model.destinations,
            )
            inputs = build_public_controller_inputs(
                candidates=candidates,
                top_k=model.top_k,
                context_observations=model.context_observation_count(source, prior),
                context_seen=model.has_context(source, prior),
                steganalysis_risk=reference.risk,
                committed_payload_bits=session.committed_payload_bits,
                payload_length=payload_bits,
            )

        before = session.committed_payload_bits
        emission = session.emit(
            inputs=inputs,
            candidates=candidates,
            cover_action=natural_action,
        )
        after = session.committed_payload_bits
        total_committed_gain += max(0, after - before)
        emitted_action = str(emission.action)
        modes[emission.decision.mode] += 1
        if not emission.sender_decoder_state_match:
            state_mismatches += 1
        if emission.decision.mode == "EMBED" and emitted_action not in model.destinations:
            invalid += 1

        stego_rows.append(
            observed_action_feature_vector(
                source=source,
                action=emitted_action,
                previous_action=prior,
                candidates=candidates,
                gap=gap,
                context_seen=model.has_context(source, prior),
                training_destinations=model.destinations,
            )
        )
        stego_previous[source] = emitted_action
        previous_timestamp[source] = float(timestamp)

    stego_features = np.vstack(stego_rows)
    x = np.vstack([natural_features, stego_features])
    y = np.concatenate([
        np.zeros(len(natural_features), dtype=int),
        np.ones(len(stego_features), dtype=int),
    ])
    pair_clusters = np.concatenate([clusters, clusters])
    score_map = {name: detector.risk(x) for name, detector in detectors.items()}
    detector_aucs = {
        name: adversarial_auc(float(roc_auc_score(y, score)))
        for name, score in score_map.items()
    }
    bootstrap: WorstAucBootstrap = cluster_bootstrap_worst_adversarial_auc(
        y,
        score_map,
        pair_clusters,
        resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        seed=20260827 + message_seed,
    )

    completed = 0
    decoded_ok = 0
    for source, session in sessions.items():
        if session.complete:
            completed += 1
            if session.decoded_payload == payloads[source]:
                decoded_ok += 1
    decode_success = 1.0 if completed == 0 else decoded_ok / completed
    transitions = len(validation_frame)
    abstentions = modes["COVER"] + modes["PAUSE"] + modes["STOP"]
    return SeedCertification(
        seed=message_seed,
        bits_per_transition=float(total_committed_gain / transitions),
        completion_rate=float(completed / max(1, len(sessions))),
        abstention_rate=float(abstentions / transitions),
        mode_rates={name: float(modes[name] / transitions) for name in ("EMBED", "COVER", "PAUSE", "STOP")},
        worst_adversarial_auc=float(max(detector_aucs.values())),
        auc_ci_lower=float(bootstrap.ci_lower),
        auc_ci_upper=float(bootstrap.ci_upper),
        detector_adversarial_auc=detector_aucs,
        bootstrap_valid_resamples=int(bootstrap.valid_resamples),
        passive_decode_success_rate=float(decode_success),
        passive_state_mismatch_rate=float(state_mismatches / transitions),
        invalid_transition_rate=float(invalid / transitions),
        sessions=len(sessions),
        completed_sessions=completed,
    )


def _payload_for_source(source: str, *, seed: int, length: int) -> list[int]:
    rng = np.random.default_rng(_stable_u64(f"payload:{seed}:{source}"))
    return rng.integers(0, 2, size=length).astype(int).tolist()


def _stable_u64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big", signed=False)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value

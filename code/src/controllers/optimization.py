from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution

from controllers.fuzzy import FuzzyWeights


@dataclass(frozen=True)
class FuzzyCandidate:
    weights: FuzzyWeights
    stop_threshold: float
    embed_throttle: float = 1.0


@dataclass(frozen=True)
class PolicyEvaluation:
    """Validation-only evidence used to tune a fuzzy policy.

    ``auc_ci_upper`` is the upper confidence bound of the worst oriented
    design-Eve AUC, where oriented AUC is max(AUC, 1-AUC). The optimizer never
    receives development-test or final-holdout metrics.
    """

    payload_rate: float
    completion_rate: float
    abstention_rate: float
    worst_adversarial_auc: float
    auc_ci_upper: float
    invalid_transition_rate: float = 0.0
    state_mismatch_rate: float = 0.0
    passive_decode_success_rate: float = 1.0


@dataclass(frozen=True)
class EvaluatedCandidate:
    candidate: FuzzyCandidate
    evaluation: PolicyEvaluation
    feasible: bool


@dataclass(frozen=True)
class OptimizationResult:
    selected: EvaluatedCandidate
    evaluated: tuple[EvaluatedCandidate, ...]
    detector_auc_budget: float
    seed: int


Evaluator = Callable[[FuzzyCandidate], PolicyEvaluation]


class DetectorConstrainedFuzzyOptimizer:
    """Evolutionary tuning of Takagi--Sugeno consequences and public thinning.

    Differential evolution proposes controller parameters using only the
    policy-validation region. ``embed_throttle`` is a secret-independent public
    thinning probability applied only after the fuzzy controller declares an
    event EMBED-eligible. Every evaluation is cached, and final selection is
    lexicographic: hard-gate feasibility first, then maximum measured payload
    rate, completion, lower adversarial AUC and lower abstention.
    """

    _WEIGHT_FIELDS = (
        "opportunity_entropy_weight",
        "opportunity_payload_weight",
        "cover_entropy_weight",
        "pause_risk_weight",
        "stop_dead_end_weight",
        "abstention_cover_weight",
        "abstention_pause_weight",
        "abstention_stop_weight",
    )

    def __init__(
        self,
        *,
        detector_auc_budget: float = 0.60,
        seed: int = 20260827,
        population_size: int = 8,
        generations: int = 8,
        stop_threshold_bounds: tuple[float, float] = (0.80, 0.99),
        embed_throttle_bounds: tuple[float, float] = (0.0, 1.0),
        cache_decimals: int = 6,
    ) -> None:
        if not 0.5 <= detector_auc_budget <= 1.0:
            raise ValueError("detector_auc_budget must be in [0.5, 1]")
        if population_size < 4:
            raise ValueError("population_size must be at least four")
        if generations < 1:
            raise ValueError("generations must be positive")
        low, high = stop_threshold_bounds
        if not 0 < low < high <= 1:
            raise ValueError("invalid stop-threshold bounds")
        throttle_low, throttle_high = embed_throttle_bounds
        if not 0.0 <= throttle_low < throttle_high <= 1.0:
            raise ValueError("invalid embed-throttle bounds")
        self.detector_auc_budget = float(detector_auc_budget)
        self.seed = int(seed)
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.stop_threshold_bounds = (float(low), float(high))
        self.embed_throttle_bounds = (float(throttle_low), float(throttle_high))
        self.cache_decimals = int(cache_decimals)

    def optimize(self, evaluate: Evaluator) -> OptimizationResult:
        cache: dict[tuple[float, ...], EvaluatedCandidate] = {}

        def objective(vector: np.ndarray) -> float:
            key = tuple(np.round(vector, self.cache_decimals))
            if key not in cache:
                candidate = self._candidate(vector)
                evaluation = evaluate(candidate)
                cache[key] = EvaluatedCandidate(
                    candidate=candidate,
                    evaluation=evaluation,
                    feasible=self._feasible(evaluation),
                )
            item = cache[key]
            return self._penalized_loss(item.evaluation)

        bounds = (
            [(0.0, 1.0)] * len(self._WEIGHT_FIELDS)
            + [self.stop_threshold_bounds, self.embed_throttle_bounds]
        )
        differential_evolution(
            objective,
            bounds=bounds,
            seed=self.seed,
            popsize=self.population_size,
            maxiter=self.generations,
            polish=False,
            updating="immediate",
            workers=1,
        )
        if not cache:
            raise RuntimeError("optimizer evaluated no policy candidates")

        evaluated = tuple(cache.values())
        feasible = [item for item in evaluated if item.feasible]
        pool = feasible if feasible else list(evaluated)
        selected = max(pool, key=self._selection_key)
        return OptimizationResult(
            selected=selected,
            evaluated=evaluated,
            detector_auc_budget=self.detector_auc_budget,
            seed=self.seed,
        )

    def _candidate(self, vector: np.ndarray) -> FuzzyCandidate:
        expected = len(self._WEIGHT_FIELDS) + 2
        if len(vector) != expected:
            raise ValueError(f"expected {expected} optimization parameters")
        values = [float(np.clip(value, 0.0, 1.0)) for value in vector[: len(self._WEIGHT_FIELDS)]]
        weights = FuzzyWeights(**dict(zip(self._WEIGHT_FIELDS, values, strict=True)))
        stop_index = len(self._WEIGHT_FIELDS)
        return FuzzyCandidate(
            weights=weights,
            stop_threshold=float(
                np.clip(
                    vector[stop_index],
                    self.stop_threshold_bounds[0],
                    self.stop_threshold_bounds[1],
                )
            ),
            embed_throttle=float(
                np.clip(
                    vector[stop_index + 1],
                    self.embed_throttle_bounds[0],
                    self.embed_throttle_bounds[1],
                )
            ),
        )

    def _feasible(self, evaluation: PolicyEvaluation) -> bool:
        return bool(
            evaluation.auc_ci_upper <= self.detector_auc_budget
            and evaluation.invalid_transition_rate == 0.0
            and evaluation.state_mismatch_rate == 0.0
            and evaluation.passive_decode_success_rate == 1.0
        )

    def _penalized_loss(self, evaluation: PolicyEvaluation) -> float:
        # Used only to guide evolutionary search. Final scientific selection is
        # lexicographic in _selection_key, not based on this scalarization.
        gate_penalty = (
            max(0.0, evaluation.auc_ci_upper - self.detector_auc_budget)
            + evaluation.invalid_transition_rate
            + evaluation.state_mismatch_rate
            + max(0.0, 1.0 - evaluation.passive_decode_success_rate)
        )
        if gate_penalty > 0:
            return 100.0 + 1000.0 * gate_penalty - 0.01 * evaluation.payload_rate
        return -evaluation.payload_rate - 1e-3 * evaluation.completion_rate

    def _selection_key(self, item: EvaluatedCandidate) -> tuple[float, ...]:
        evaluation = item.evaluation
        if item.feasible:
            return (
                1.0,
                evaluation.payload_rate,
                evaluation.completion_rate,
                -evaluation.worst_adversarial_auc,
                -evaluation.abstention_rate,
            )
        total_violation = (
            max(0.0, evaluation.auc_ci_upper - self.detector_auc_budget)
            + evaluation.invalid_transition_rate
            + evaluation.state_mismatch_rate
            + max(0.0, 1.0 - evaluation.passive_decode_success_rate)
        )
        return (
            0.0,
            -total_violation,
            evaluation.payload_rate,
            evaluation.completion_rate,
            -evaluation.abstention_rate,
        )

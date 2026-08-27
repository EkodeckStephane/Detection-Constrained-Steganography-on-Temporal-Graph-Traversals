from __future__ import annotations

from controllers.optimization import (
    DetectorConstrainedFuzzyOptimizer,
    PolicyEvaluation,
)


def test_evolutionary_fuzzy_optimizer_prefers_highest_feasible_payload() -> None:
    optimizer = DetectorConstrainedFuzzyOptimizer(
        detector_auc_budget=0.60,
        seed=17,
        population_size=4,
        generations=4,
    )

    def evaluate(candidate):
        weight = candidate.weights.opportunity_entropy_weight
        # Feasible while weight <= 0.5. Payload rises monotonically with weight,
        # so the constrained optimum sits near the detectability boundary.
        return PolicyEvaluation(
            payload_rate=weight,
            completion_rate=weight,
            abstention_rate=1.0 - weight,
            worst_adversarial_auc=0.55 + 0.10 * weight,
            auc_ci_upper=0.55 + 0.10 * weight,
        )

    result = optimizer.optimize(evaluate)

    assert result.selected.feasible
    assert result.selected.evaluation.auc_ci_upper <= 0.60
    assert result.selected.evaluation.payload_rate >= 0.35
    assert len(result.evaluated) > 1


def test_hard_causal_gate_overrides_payload_advantage() -> None:
    optimizer = DetectorConstrainedFuzzyOptimizer(
        detector_auc_budget=0.60,
        seed=23,
        population_size=4,
        generations=3,
    )

    def evaluate(candidate):
        weight = candidate.weights.opportunity_entropy_weight
        invalid = 0.01 if weight > 0.55 else 0.0
        return PolicyEvaluation(
            payload_rate=weight,
            completion_rate=weight,
            abstention_rate=1.0 - weight,
            worst_adversarial_auc=0.55,
            auc_ci_upper=0.56,
            invalid_transition_rate=invalid,
        )

    result = optimizer.optimize(evaluate)

    assert result.selected.feasible
    assert result.selected.evaluation.invalid_transition_rate == 0.0
    assert result.selected.evaluation.payload_rate <= 0.55

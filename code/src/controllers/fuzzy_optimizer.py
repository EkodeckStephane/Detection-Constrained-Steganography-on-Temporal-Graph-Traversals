from __future__ import annotations

"""Compatibility surface for the superseded proxy fuzzy optimizer.

ASOC V2 no longer optimizes a scalar proxy over synthetic controller-input
cases. The canonical optimization is detector-constrained and validation-only;
it lives in :mod:`controllers.optimization`.

This module intentionally keeps only the historical helper ``default_weights``
and re-exports the ASOC V2 optimizer types. Any attempt to invoke the old
``evaluate_weights`` / ``optimize_fuzzy_weights`` API fails explicitly instead
of silently producing results under a method that contradicts the paper.
"""

from controllers.fuzzy import FuzzyWeights
from controllers.optimization import (
    DetectorConstrainedFuzzyOptimizer,
    EvaluatedCandidate,
    FuzzyCandidate,
    OptimizationResult,
    PolicyEvaluation,
)


def default_weights() -> FuzzyWeights:
    return FuzzyWeights()


def evaluate_weights(*args, **kwargs):  # type: ignore[no-untyped-def]
    del args, kwargs
    raise RuntimeError(
        "The synthetic proxy objective was retired for ASOC V2. "
        "Use DetectorConstrainedFuzzyOptimizer with a policy-validation "
        "evaluator that returns PolicyEvaluation including worst-Eve AUC UCB."
    )


def optimize_fuzzy_weights(*args, **kwargs):  # type: ignore[no-untyped-def]
    del args, kwargs
    raise RuntimeError(
        "The legacy proxy optimizer was retired for ASOC V2. "
        "Use controllers.optimization.DetectorConstrainedFuzzyOptimizer."
    )


__all__ = [
    "DetectorConstrainedFuzzyOptimizer",
    "EvaluatedCandidate",
    "FuzzyCandidate",
    "OptimizationResult",
    "PolicyEvaluation",
    "default_weights",
    "evaluate_weights",
    "optimize_fuzzy_weights",
]

from __future__ import annotations

"""Retired ASOC V1/Phase-6 proxy optimization entry point.

The historical script optimized fuzzy weights on a synthetic input grid using a
weighted proxy objective. That method is not part of the ASOC V2 scientific
protocol and must not generate manuscript evidence.

Use the canonical ASOC V2 runner under ``code/scripts/run_asoc_v2.py`` once the
four causal regions and frozen design-Eves are available. Controller tuning is
performed by ``controllers.optimization.DetectorConstrainedFuzzyOptimizer`` on
``policy_validation`` only, with the clustered worst-Eve AUC upper confidence
bound as a hard feasibility constraint.
"""


def main() -> None:
    raise SystemExit(
        "This legacy proxy optimizer is retired. Use the ASOC V2 "
        "detector-constrained pipeline; no result file was written."
    )


if __name__ == "__main__":
    main()

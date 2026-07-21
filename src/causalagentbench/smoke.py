"""Smoke report: rejection-loop yield and per-gate statistics.

Run with ``uv run causalagentbench-smoke``. Generates many raw candidates, evaluates the gates, and
prints per-gate pass rates, the acceptance rate, sortability, and the over-adjustment flip rate - a
quick health check that the generator yields sound, non-trivial instances at a reasonable rate.
"""

from __future__ import annotations

import numpy as np

from causalagentbench.config import GeneratorConfig
from causalagentbench.scm.generate import generate
from causalagentbench.scoring.gates import evaluate_gates, generate_gated

_GATES = (
    "structure_valid",
    "confounding_realized",
    "mediation_realized",
    "oracle_recovers",
)


def main() -> None:
    """Print the generator health report."""
    config = GeneratorConfig()
    n_candidates = 500
    counts = dict.fromkeys(_GATES, 0)
    accepted = 0
    varsorts: list[float] = []
    r2sorts: list[float] = []
    flips = 0

    for child in np.random.SeedSequence(20260715).spawn(n_candidates):
        structure_seed, sample_seed = child.spawn(2)
        report = evaluate_gates(generate(structure_seed), config, sample_seed)
        for gate in _GATES:
            counts[gate] += int(getattr(report, gate))
        if report.passed:
            accepted += 1
            varsorts.append(report.varsortability)
            r2sorts.append(report.r2_sortability)
            flips += int(report.overadjust_flips_ranking)

    print(f"Generator health report over {n_candidates} candidates (n_samples={config.n_samples})")
    print("-" * 64)
    for gate in _GATES:
        print(f"  {gate:<22} pass rate: {counts[gate] / n_candidates:6.1%}")
    print("-" * 64)
    print(f"  accepted (all gates):  {accepted / n_candidates:6.1%}  ({accepted}/{n_candidates})")
    if accepted:
        print(f"  varsortability (mean): {np.mean(varsorts):.3f}  [target ~0.5]")
        print(f"  R2-sortability (mean): {np.mean(r2sorts):.3f}")
        print(f"  over-adjust flips rank: {flips / accepted:6.1%} of accepted")

    scm, _ = generate_gated(seed=0, config=config)
    print("-" * 64)
    print("Example accepted instance (seed=0):")
    print(f"  true driver ranking: {scm.true_ranking()}")
    for driver in scm.scaffold.drivers:
        effect = scm.total_effect_on_outcome(driver)
        adjust = scm.scaffold.correct_adjustment.get(driver, ())
        print(f"    {driver:<3} effect={effect:+.3f}  adjust-for={adjust or '{}'}")


if __name__ == "__main__":
    main()

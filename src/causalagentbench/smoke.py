"""Smoke report: rejection-loop yield and per-gate statistics.

Run with ``uv run causalagentbench-smoke``. Generates many raw candidates, evaluates the gates, and
prints per-gate pass rates, the acceptance rate, sortability, and the over-adjustment flip rate - a
quick health check that the generator yields sound, non-trivial instances at a reasonable rate.
"""

from __future__ import annotations

import numpy as np

from causalagentbench.config import GeneratorConfig
from causalagentbench.scm.gadgets import procedural_scaffold
from causalagentbench.scm.generate import generate
from causalagentbench.scm.target_effects import target_effects, with_constructed_confounding, with_constructed_effects
from causalagentbench.scoring.gates import _has_balanced_identifiability, evaluate_gates


def _procedural_report() -> None:
    """Health report for the procedural + construction path (``generate_gated_procedural``).

    Broken down by driver count - the axis its acceptance rate actually varies on. Deliberately reimplements
    ``generate_gated_procedural``'s attempt body stage-by-stage rather than calling it: that function
    retries internally until one instance passes (or raises), so calling it in a loop would yield 500
    *successful* instances, not 500 independent candidates broken down by which stage they failed at, which
    is what this report needs. Keep this in sync by hand if the attempt body in ``scoring/gates.py`` changes.
    """
    config = GeneratorConfig()
    n_attempts = 500
    by_n: dict[int, list[int]] = {}  # non-decoy driver count -> [attempts, topology_ok, constructed, passed]

    for child in np.random.SeedSequence(20260715).spawn(n_attempts):
        topology_seed, weight_seed, target_seed, construction_seed, confounding_seed, sample_seed = child.spawn(6)
        scaffold = procedural_scaffold(topology_seed)
        non_decoy_drivers = scaffold.non_decoy_drivers()
        bucket = by_n.setdefault(len(non_decoy_drivers), [0, 0, 0, 0])
        bucket[0] += 1
        if not _has_balanced_identifiability(scaffold):
            continue
        bucket[1] += 1
        reference = generate(weight_seed, scaffold)
        targets = target_effects(np.random.default_rng(target_seed), non_decoy_drivers)
        scm = with_constructed_effects(reference, targets, np.random.default_rng(construction_seed))
        if scm is None:
            continue
        scm = with_constructed_confounding(scm, np.random.default_rng(confounding_seed))
        if scm is None:
            continue
        bucket[2] += 1
        report = evaluate_gates(scm, config, sample_seed)
        bucket[3] += int(report.passed)

    print(
        f"Procedural health report over {n_attempts} attempts, by non-decoy driver count (n_samples={config.n_samples})"
    )
    print("-" * 64)
    print(f"  {'drivers':<10}{'attempts':<12}{'topology_ok':<13}{'constructed':<13}{'passed':<10}")
    totals = [0, 0, 0, 0]
    for n_drivers, (attempts, topology_ok, constructed, passed) in sorted(by_n.items()):
        topology_rate = topology_ok / attempts
        constructed_rate = constructed / attempts
        passed_rate = passed / attempts
        print(f"  {n_drivers:<10}{attempts:<12}{topology_rate:<13.1%}{constructed_rate:<13.1%}{passed_rate:<10.1%}")
        for i, value in enumerate((attempts, topology_ok, constructed, passed)):
            totals[i] += value
    print("-" * 64)
    print(f"  overall accepted: {totals[3] / totals[0]:.1%}  ({totals[3]}/{totals[0]})")


def main() -> None:
    """Print the generator's health report."""
    _procedural_report()


if __name__ == "__main__":
    main()

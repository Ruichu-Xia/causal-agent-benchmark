"""Soundness gates and the rejection-sampling generator.

The gates are the generator's contract: an instance is accepted only if it demonstrably requires causal
reasoning. Two shortcuts must fail on this scaffold - naive correlation ("no adjustment") and regressing
on every covariate ("adjust for everything", which conditions on the mediator) - while the oracle,
using correct adjustment sets, recovers the truth. The gates double as the property-test suite.

Framing: these are construction-validity checks for the confounded scaffold - they verify the intended
structure was realized (confounding strong enough, mediated effect material). They are NOT a claim that
"correlation always fails" as a law: future unconfounded scaffolds will have correlation succeed, with
gold answer "do not adjust". The full benchmark measures appropriate reliance by scoring a mix of
scaffolds stratified per scaffold, not by filtering a mixed distribution. Here we validate one scaffold.

``evaluate_gates`` locates nodes by ``Role``, so it is robust to relabelling but assumes the
confounder/mediator/decoy role structure. Thresholds are Stage 0 invariants and will migrate into
``GeneratorConfig`` when tunable difficulty arrives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from causalagentbench.baselines import full_adjustment_effects, oracle_effects, ranking
from causalagentbench.config import GeneratorConfig
from causalagentbench.generate import generate
from causalagentbench.metrics import kendall_tau
from causalagentbench.sample import sample
from causalagentbench.scaffold import CANONICAL
from causalagentbench.sortability import r2_sortability, varsortability
from causalagentbench.types import SCM, Role, Scaffold, SeedLike

# --- Stage 0 gate thresholds (future GeneratorConfig fields) ------------------
DECOY_EFFECT_TOL = 1e-9
"""Decoy total effect is a structural zero; tolerate only numerical dust."""
MIN_ABS_EFFECT = 0.10
"""Each true driver must have |standardized effect| at least this large."""
MIN_EFFECT_GAP = 0.05
"""The two true drivers' |effects| must differ by at least this (strict ranking)."""
CONFOUNDING_MARGIN = 0.05
"""Decoy's |corr(Dz, Y)| must beat the weaker driver's correlation by this margin."""
MIN_MEDIATED_EFFECT = 0.10
"""|W[A,M] * W[M,Y]| must be at least this, so over-adjustment biases A materially."""
ORACLE_TAU_THRESHOLD = 0.99
"""Minimum Kendall-tau between the oracle ranking and the truth."""
MAX_ATTEMPTS = 1000
"""Rejection-sampling safety valve."""


@dataclass(frozen=True)
class GateReport:
    """Verdict on one candidate: the four gate outcomes plus diagnostics.

    ``varsortability`` and ``r2_sortability`` are reported diagnostics, not gates: UUMC already
    guarantees population-level neutrality, so per-instance empirical values are informational only.
    Sample-derived fields are ``nan``/``False`` when structural gates fail and sampling is skipped.
    """

    passed: bool
    structure_valid: bool
    confounding_realized: bool
    mediation_realized: bool
    oracle_recovers: bool
    effect_confounded: float
    effect_clean: float
    effect_decoy: float
    effect_gap: float
    decoy_correlation: float
    mediated_effect: float
    oracle_tau: float
    varsortability: float
    r2_sortability: float
    full_adjust_bias: float
    overadjust_flips_ranking: bool


def evaluate_gates(scm: SCM, config: GeneratorConfig, seed: SeedLike) -> GateReport:
    """Evaluate the four soundness gates on ``scm`` (plus sortability diagnostics).

    Structural gates (from the exact SCM) are checked first; the sample-based oracle gate is evaluated
    only if those pass, so most rejections cost no sampling.
    """
    sc = scm.scaffold
    confounded = sc.nodes_with_role(Role.DRIVER_CONFOUNDED)[0]
    clean = sc.nodes_with_role(Role.DRIVER_CLEAN)[0]
    decoy = sc.nodes_with_role(Role.DECOY)[0]
    mediator = sc.nodes_with_role(Role.MEDIATOR)[0]
    outcome_idx = sc.index(sc.outcome)

    # Structural quantities from the exact SCM (cheap, no sampling).
    effect_confounded = abs(scm.total_effect_on_outcome(confounded))
    effect_clean = abs(scm.total_effect_on_outcome(clean))
    effect_decoy = abs(scm.total_effect_on_outcome(decoy))
    weaker_effect = min(effect_confounded, effect_clean)
    effect_gap = abs(effect_confounded - effect_clean)

    # A correlation ranker orders drivers by |corr(., Y)|; the decoy must beat a true driver's
    # correlation (not its effect) for that ranker to mis-rank.
    corr_confounded = abs(float(scm.sigma[sc.index(confounded), outcome_idx]))
    corr_clean = abs(float(scm.sigma[sc.index(clean), outcome_idx]))
    decoy_correlation = abs(float(scm.sigma[sc.index(decoy), outcome_idx]))
    weaker_driver_corr = min(corr_confounded, corr_clean)

    edge_am = scm.W[sc.index(confounded), sc.index(mediator)]
    edge_my = scm.W[sc.index(mediator), outcome_idx]
    mediated_effect = abs(float(edge_am * edge_my))

    structure_valid = (
        effect_decoy <= DECOY_EFFECT_TOL and weaker_effect >= MIN_ABS_EFFECT and effect_gap >= MIN_EFFECT_GAP
    )
    confounding_realized = decoy_correlation > weaker_driver_corr + CONFOUNDING_MARGIN
    mediation_realized = mediated_effect >= MIN_MEDIATED_EFFECT

    oracle_tau = math.nan
    vsort = math.nan
    r2sort = math.nan
    full_adjust_bias = math.nan
    overadjust_flips_ranking = False
    oracle_recovers = False

    if structure_valid and confounding_realized and mediation_realized:
        data = sample(scm, config.n_samples, seed)
        oracle = oracle_effects(data, scm)
        true_effects = [abs(scm.total_effect_on_outcome(d)) for d in sc.drivers]
        oracle_estimates = [abs(oracle[d]) for d in sc.drivers]
        oracle_tau = kendall_tau(true_effects, oracle_estimates)
        vsort = varsortability(data, scm.W)
        r2sort = r2_sortability(scm)
        full = full_adjustment_effects(data, scm)
        full_adjust_bias = abs(full[confounded] - scm.total_effect_on_outcome(confounded))
        overadjust_flips_ranking = ranking(full) != scm.true_ranking()
        oracle_recovers = oracle_tau >= ORACLE_TAU_THRESHOLD

    passed = structure_valid and confounding_realized and mediation_realized and oracle_recovers
    return GateReport(
        passed=passed,
        structure_valid=structure_valid,
        confounding_realized=confounding_realized,
        mediation_realized=mediation_realized,
        oracle_recovers=oracle_recovers,
        effect_confounded=effect_confounded,
        effect_clean=effect_clean,
        effect_decoy=effect_decoy,
        effect_gap=effect_gap,
        decoy_correlation=decoy_correlation,
        mediated_effect=mediated_effect,
        oracle_tau=oracle_tau,
        varsortability=vsort,
        r2_sortability=r2sort,
        full_adjust_bias=full_adjust_bias,
        overadjust_flips_ranking=overadjust_flips_ranking,
    )


def generate_gated(
    seed: int,
    config: GeneratorConfig | None = None,
    scaffold: Scaffold = CANONICAL,
) -> tuple[SCM, GateReport]:
    """Return the first ``(SCM, GateReport)`` that passes all gates, deterministically from ``seed``.

    Each attempt draws independent structure and sample seeds spawned from ``seed``. Raises
    ``RuntimeError`` if no instance passes within ``MAX_ATTEMPTS``.
    """
    config = config or GeneratorConfig()
    for child in np.random.SeedSequence(seed).spawn(MAX_ATTEMPTS):
        structure_seed, sample_seed = child.spawn(2)
        scm = generate(structure_seed, scaffold)
        report = evaluate_gates(scm, config, sample_seed)
        if report.passed:
            return scm, report
    raise RuntimeError(f"no instance passed all gates within {MAX_ATTEMPTS} attempts")

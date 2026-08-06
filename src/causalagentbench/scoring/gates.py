"""Soundness gates and the rejection-sampling generator.

The gates are the generator's contract: an instance is accepted only if it demonstrably requires causal
reasoning. That claim is enforced at two different levels, and the distinction matters.

*Structural* gates (``structure_valid``, ``confounding_realized``, ``mediation_realized``) verify that the
intended structure was realized - drivers separated, confounding present, mediated path material. They are
per-driver properties, and they are NOT a claim that "correlation always fails" as a law: future
unconfounded scaffolds will have correlation succeed, with gold answer "do not adjust".

The *policy* gate (``shortcut_defeated``) checks the proposition the structural ones only proxy for: can a
blind policy - one reading no column names and needing no graph - actually score well here? It is checked
on **estimation error**, not ranking, and that is a finding rather than a preference. Blind policies rank
well no matter how the generator is tuned (marginal correlation reaches tau=+0.627, full adjustment +0.71
at every driver count, against random's +0.014), and forcing those toward chance is infeasible inside the
outcome's unit-variance budget. Estimation separates cleanly instead: correct adjustment recovers the true
effect exactly at the population level, while the best blind policy stays 21-34% of a typical effect away.
See ``SHORTCUT_ERROR_THRESHOLD``. ``max_shortcut_tau`` records the ranking weakness as a diagnostic so it
stays visible rather than implicit.

The gates double as the property-test suite.

``evaluate_gates`` locates nodes by ``Role`` and checks every instance of each role, not just the first - a
procedurally-composed scaffold (``scm/gadgets.py``) can have several, or none, of a given kind (an
instrumented driver has no mediator at all). A threshold check that only inspected the first instance of a
role would silently ignore the rest. Thresholds are Stage 0 invariants and will migrate into
``GeneratorConfig`` when tunable difficulty arrives.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from causalagentbench.config import GeneratorConfig
from causalagentbench.scm.gadgets import RESOLVE_PROBABILITY, procedural_scaffold
from causalagentbench.scm.generate import generate
from causalagentbench.scm.identifiability import unidentifiable_drivers
from causalagentbench.scm.sample import sample
from causalagentbench.scm.sortability import r2_sortability, varsortability
from causalagentbench.scm.target_effects import (
    CONFOUNDING_MARGIN,
    MIN_MEDIATED_SHARE,
    REFERENCE_MIN_EFFECT,
    REFERENCE_MIN_GAP,
    scaled_threshold,
    target_effects,
    with_constructed_confounding,
    with_constructed_effects,
)
from causalagentbench.scoring.baselines import (
    full_adjustment_effects,
    normalized_effect_error,
    oracle_effects,
    ranking,
    shortcut_effects,
)
from causalagentbench.scoring.metrics import kendall_tau
from causalagentbench.types import SCM, FloatArray, Role, Scaffold, SeedLike

# --- Stage 0 gate thresholds (future GeneratorConfig fields) ------------------
DECOY_EFFECT_TOL = 1e-9
"""Decoy total effect is a structural zero; tolerate only numerical dust."""
ORACLE_TAU_THRESHOLD = 0.90
"""Minimum Kendall-tau between the oracle ranking and the truth, at the published sample size.

Ranking the drivers is the task's stated deliverable, so an instance whose true order *no* analyst could
recover from the data is not a hard instance - it is an ill-posed one, and every ranking number reported on
it is noise. This gate keeps them out.

0.90, not the previous 0.99. Both are floors on the same quantity; the difference is what they cost.
Measured over accepted instances, the oracle's own tau at ``n_samples`` has median 1.000 but a p10 of 0.810
- effects are spaced closely enough that sampling noise occasionally swaps an adjacent pair even for a
perfect estimator. 0.99 therefore discarded 43% of survivors to exclude near-misses; 0.90 discards 14% and
still guarantees the oracle orders ~95% of pairs correctly.

Ranking is worth gating for: on the set this admits, oracle tau is 0.934 against 0.367 for blind marginal
correlation. An earlier note here claimed ranking was shortcuttable at every driver count (blind ~0.75) and
used that to demote this gate - that measurement predated ``SHORTCUT_ERROR_THRESHOLD`` rising to 0.25,
which selects against instances where blind policies estimate well, and those turn out to be the same
instances where they rank well."""
SHORTCUT_ERROR_THRESHOLD = 0.25
"""Minimum estimation error any blind policy must still suffer, as a fraction of the instance's median
|true effect|.

The gate this module's docstring always claimed to enforce - "accepted only if it demonstrably requires
causal reasoning" - now checked against *policies* rather than the per-driver structural proxies above,
which cannot see it.

Measured against **estimation**, not ranking, because ranking cannot carry the check. Blind policies rank
far too well to gate on: marginal correlation reaches tau=+0.627 and full adjustment +0.71 at every driver
count, against random's +0.014, and 66% of instances leave the best blind ranker above tau=0.6 even after
this gate. Driving those to chance is infeasible inside the outcome's unit-variance budget - attempting it
(deeper confounding, wider mediation) cut feasible draws from 179/250 to 10/250 at 8 drivers and still left
full adjustment at +0.78. Estimation separates cleanly where ranking does not, and more importantly it
separates *reliably*: correct adjustment recovers the true effect exactly at the population level, while
the best blind policy's error stays above 0.21 even at the 25th percentile.

0.25 chosen from the measured distribution at ``n_candidates=7`` (324 structurally-valid instances:
min 0.09, p05 0.14, p25 0.21, median 0.27). It keeps 62% of them, so roughly one draw in three survives
this gate and the oracle gate together - cheap, since composition costs milliseconds. The value buys a
guarantee worth stating: on every published instance, the best policy that does no causal reasoning is
wrong by at least a quarter of a typical effect, i.e. it cannot score above 0.75 ``effect_accuracy``.
Loosening to 0.15 would keep 94% but leave instances where that ceiling is 0.85."""
MAX_ATTEMPTS = 1000
"""Rejection-sampling safety valve."""

#: `CONFOUNDING_MARGIN` and `MIN_MEDIATED_SHARE` live in `scm/target_effects.py`, next to the driver-effect
#: thresholds - construction and the check that verifies it must share one definition, not two
#: independently-tuned ones (`procedural_scaffold_plan.md` §5a). Both are relative, like the driver-effect
#: thresholds and for the same reason: `MIN_MEDIATED_SHARE` is checked as a fraction of the driver's own
#: effect, and `CONFOUNDING_MARGIN` is scaled here by `scaled_threshold` exactly as construction scales it
#: (`variance_budget_plan.md`). An absolute floor on either one is a demand that grows as a share of the
#: outcome's fixed variance budget as driver count rises, until no candidate can satisfy it.


@dataclass(frozen=True)
class GateReport:
    """Verdict on one candidate: the four gate outcomes plus diagnostics.

    Numeric diagnostics are worst-case aggregates across every instance of the relevant kind (e.g.
    ``min_driver_effect`` is the *weakest* non-decoy driver's effect, not one named driver's) - the
    generalization of what a single threshold check meant when a scaffold had exactly one of each kind.
    ``varsortability`` and ``r2_sortability`` are reported diagnostics, not gates, but for different
    reasons. UUMC does deliver population-level *variance* neutrality, so per-instance ``varsortability`` is
    ensemble noise around 0.5 (measured mean 0.47) and gating on it would reject on chance alone.

    ``r2_sortability`` is **not** neutral - measured mean 0.89 against a 0.5 no-signal baseline (0.70 for the
    observable regress-on-all-others variant) - and that is expected rather than a defect. Decomposed over 25
    instances, almost all of it is irreducible: scoring nodes by the root/non-root split *alone*, with every
    non-root tied, already yields 0.80. A node with no parents has zero variance explained by its parents by
    definition, so no generation scheme avoids that - and it is not a simulation artifact either. Real
    exogenous variables genuinely have low R^2 given the other measured variables, and recovering that is
    ordinary causal discovery (what a CI test does), not a shortcut. Only the remainder is an artifact of this
    parameterization: UUMC draws each node's signal share as ``r = U(0,1)^(1/d)``, so ``E[R^2] = d/(d+2)``
    rises with parent count, and parent count rises toward the outcome sink. Drawing R^2 independent of parent
    count was measured to move 0.89 -> 0.79, i.e. it buys ~0.09 while leaving the legitimate 0.80 untouched -
    not worth replacing a published, verified parameterization and invalidating every dataset built so far.
    So: not gated (the pattern is structural, not a bad draw), reported so it stays visible, and deliberately
    *not* treated as the peer of varsortability - that one was ~all artifact and fully removable, this one is
    mostly real structure. Whether the artifactual remainder is exploitable *for this task* - ranking effects
    and choosing adjustment sets, not recovering a causal order - has not been measured.

    Sample-derived fields are ``nan``/``False``/``None`` when structural gates fail and sampling is skipped.

    ``data`` is the *exact* table the oracle gate was validated against - not a recipe for reconstructing
    one (a seed + sample size that a caller must independently get right and keep in sync). A consumer
    that needs data consistent with what passed the gate (e.g. a released dataset) should use this table
    directly, never redraw independently: a fresh draw, even at a matching sample size, is not guaranteed
    to reproduce the specific pass this report already certified (`procedural_scaffold_plan.md` §12).
    """

    passed: bool
    structure_valid: bool
    confounding_realized: bool
    mediation_realized: bool
    shortcut_defeated: bool
    oracle_recovers: bool
    min_shortcut_error: float  # best blind policy's estimation error, relative; the causal-load certificate
    max_shortcut_tau: float  # best |tau| any blind policy reaches; reported, never gated
    min_driver_effect: float
    max_decoy_effect: float
    min_effect_gap: float
    min_confounding_margin: float
    min_mediated_share: float  # mediated path as a fraction of its own driver's total effect, not a magnitude
    oracle_tau: float
    varsortability: float
    r2_sortability: float
    max_full_adjust_bias: float
    overadjust_flips_ranking: bool
    data: FloatArray | None


def evaluate_gates(scm: SCM, config: GeneratorConfig, seed: SeedLike) -> GateReport:
    """Evaluate the four soundness gates on ``scm``, plus sortability diagnostics.

    Checks every instance of every role, not just the first of each kind. Structural gates (from the
    exact SCM) are checked first; the sample-based oracle gate is evaluated only if those pass, so most
    rejections cost no sampling.
    """
    sc = scm.scaffold
    outcome_idx = sc.index(sc.outcome)
    decoys = sc.nodes_with_role(Role.DECOY)
    non_decoy_drivers = sc.non_decoy_drivers()
    n = len(non_decoy_drivers)

    effect = {d: abs(scm.total_effect_on_outcome(d)) for d in sc.drivers}
    corr = {d: abs(float(scm.sigma[sc.index(d), outcome_idx])) for d in sc.drivers}

    max_decoy_effect = max((effect[d] for d in decoys), default=0.0)
    min_driver_effect = min(effect[d] for d in non_decoy_drivers)
    sorted_effects = sorted(effect[d] for d in non_decoy_drivers)
    min_effect_gap = min((b - a for a, b in itertools.pairwise(sorted_effects)), default=math.inf)

    # A correlation ranker orders drivers by |corr(., Y)|; each decoy must beat the weakest driver's
    # correlation (not its effect) for that ranker to mis-rank it.
    weakest_driver_corr = min(corr[d] for d in non_decoy_drivers)
    min_confounding_margin = min((corr[d] - weakest_driver_corr for d in decoys), default=math.inf)

    # As a share of the driver's own effect, not an absolute magnitude - what makes the over-adjustment trap
    # sharp is the portion of the effect that conditioning on the mediator removes (`MIN_MEDIATED_SHARE`).
    # A driver with no effect at all has no share to measure and cannot host a meaningful trap, so it fails
    # this gate rather than dividing by zero (`structure_valid` independently requires a non-trivial effect).
    mediated_share = {}
    for d in non_decoy_drivers:
        mediator = sc.mediator_of(d)
        if mediator is not None:
            mediated = abs(float(scm.W[sc.index(d), sc.index(mediator)] * scm.W[sc.index(mediator), outcome_idx]))
            mediated_share[d] = mediated / effect[d] if effect[d] > 0 else 0.0
    min_mediated_share = min(mediated_share.values(), default=math.inf)

    structure_valid = (
        max_decoy_effect <= DECOY_EFFECT_TOL
        and min_driver_effect >= scaled_threshold(REFERENCE_MIN_EFFECT, n)
        and min_effect_gap >= scaled_threshold(REFERENCE_MIN_GAP, n)
    )
    confounding_realized = min_confounding_margin > scaled_threshold(CONFOUNDING_MARGIN, n)
    mediation_realized = min_mediated_share >= MIN_MEDIATED_SHARE

    # Closed-form (population, not sampled - `scoring/baselines.py::population_coefficient`), so it runs
    # here with the other structural gates and rejects before the expensive sampling step below.
    true_effects = [effect[d] for d in sc.drivers]
    shortcuts = shortcut_effects(scm)
    # `normalized_effect_error` is the same function `scoring/score.py` grades an agent with, so the
    # property certified here is exactly the property scored there - not a second, similar-looking one.
    min_shortcut_error = min(normalized_effect_error(scm, estimates) for estimates in shortcuts.values())
    shortcut_defeated = min_shortcut_error >= SHORTCUT_ERROR_THRESHOLD
    # Diagnostic only, deliberately not gated: blind policies rank well no matter how the generator is
    # tuned (see `SHORTCUT_ERROR_THRESHOLD`), so this records how weak a readout the ranking is rather
    # than rejecting on it. Reported so that weakness stays visible instead of implicit.
    max_shortcut_tau = max(
        abs(kendall_tau(true_effects, [abs(estimates[d]) for d in sc.drivers])) for estimates in shortcuts.values()
    )

    oracle_tau = math.nan
    vsort = math.nan
    r2sort = math.nan
    max_full_adjust_bias = math.nan
    overadjust_flips_ranking = False
    oracle_recovers = False
    data: FloatArray | None = None

    if structure_valid and confounding_realized and mediation_realized and shortcut_defeated:
        data = sample(scm, config.n_samples, seed)
        oracle = oracle_effects(data, scm)
        oracle_estimates = [abs(oracle[d]) for d in sc.drivers]
        oracle_tau = kendall_tau(true_effects, oracle_estimates)
        vsort = varsortability(data, scm.W)
        r2sort = r2_sortability(scm)
        full = full_adjustment_effects(data, scm)
        max_full_adjust_bias = max((abs(full[d] - scm.total_effect_on_outcome(d)) for d in mediated_share), default=0.0)
        overadjust_flips_ranking = ranking(full) != scm.true_ranking()
        oracle_recovers = oracle_tau >= ORACLE_TAU_THRESHOLD

    passed = (
        structure_valid and confounding_realized and mediation_realized and shortcut_defeated and oracle_recovers
    )
    return GateReport(
        passed=passed,
        structure_valid=structure_valid,
        confounding_realized=confounding_realized,
        mediation_realized=mediation_realized,
        shortcut_defeated=shortcut_defeated,
        oracle_recovers=oracle_recovers,
        min_shortcut_error=min_shortcut_error,
        max_shortcut_tau=max_shortcut_tau,
        min_driver_effect=min_driver_effect,
        max_decoy_effect=max_decoy_effect,
        min_effect_gap=min_effect_gap,
        min_confounding_margin=min_confounding_margin,
        min_mediated_share=min_mediated_share,
        oracle_tau=oracle_tau,
        varsortability=vsort,
        r2_sortability=r2sort,
        max_full_adjust_bias=max_full_adjust_bias,
        overadjust_flips_ranking=overadjust_flips_ranking,
        data=data,
    )


def _has_balanced_identifiability(scaffold: Scaffold) -> bool:
    """True iff the composed scaffold has both an ambiguous driver and a resolved-but-confounded driver.

    Checked among non-decoy drivers only, on the discovered outcome (``unidentifiable_drivers``) - not
    declared by construction. Replaces the old single-forced-slot mechanism ("driver 0 is always
    `confounded_driver`"): that was only ever a proxy for guaranteeing an ambiguous driver exists, and
    never guaranteed the reverse (a driver whose confounding *is* resolvable from data) at all - measured to
    leave 30% of instances with zero identifiable non-clean drivers. See ``procedural_scaffold_plan.md``
    §11.0/§11.3 for the measurement and the reasoning.
    """
    ambiguous = set(unidentifiable_drivers(scaffold))
    non_decoy = scaffold.non_decoy_drivers()
    has_ambiguous = any(d in ambiguous for d in non_decoy)
    has_resolved_confounded = any(d not in ambiguous and scaffold.correct_adjustment.get(d) for d in non_decoy)
    return has_ambiguous and has_resolved_confounded


def generate_gated_procedural(
    seed: int,
    config: GeneratorConfig | None = None,
    n_candidates: int | None = None,
    resolve_probability: float = RESOLVE_PROBABILITY,
) -> tuple[SCM, GateReport]:
    """Return the first ``(SCM, GateReport)`` whose composed topology and constructed effects all pass.

    Deterministic from ``seed``. Flat loop, not nested: every attempt draws a fresh topology
    (``scm/gadgets.py``) and fresh weights together, then constructs every gate-relevant quantity directly
    rather than leaving it to those random weights - mediated effects, driver totals, and decoy confounding
    margins, in that order (``scm/target_effects.py``; left to chance, combined acceptance collapsed to 0%
    at 6+ drivers even after scaling every threshold by driver count, ``procedural_scaffold_plan.md`` §5a).
    The topology gate runs before any weights are even drawn, and the sample-based oracle gate runs only
    after every structural piece is constructed - most rejections should cost nothing
    (``procedural_scaffold_plan.md`` §5). Raises ``RuntimeError`` if no instance passes within
    ``MAX_ATTEMPTS``.

    ``n_candidates`` fixes the total candidate count - drivers plus decoys - instead of leaving it to
    ``seed`` (see ``procedural_scaffold``).
    ``resolve_probability`` controls how much of the composed topology ends up ambiguous vs. resolved,
    for a controlled ambiguity sweep - see ``procedural_scaffold``; the guarantee of at least one of each
    holds at every value.

    The returned ``report.data`` is the exact table the oracle gate certified for this instance. A caller
    that samples its own, different table (a different size, or the same size but a different draw) is
    not guaranteed the same oracle recovery this report already proved - use ``report.data`` directly for
    anything the gate's guarantee needs to hold on.
    """
    config = config or GeneratorConfig()
    for child in np.random.SeedSequence(seed).spawn(MAX_ATTEMPTS):
        topology_seed, weight_seed, target_seed, construction_seed, confounding_seed, sample_seed = child.spawn(6)
        scaffold = procedural_scaffold(
            topology_seed, n_candidates=n_candidates, resolve_probability=resolve_probability
        )
        if not _has_balanced_identifiability(scaffold):
            continue
        reference = generate(weight_seed, scaffold)
        non_decoy_drivers = scaffold.non_decoy_drivers()
        targets = target_effects(np.random.default_rng(target_seed), non_decoy_drivers)
        scm = with_constructed_effects(reference, targets, np.random.default_rng(construction_seed))
        if scm is None:
            continue
        scm = with_constructed_confounding(scm, np.random.default_rng(confounding_seed))
        if scm is None:
            continue
        report = evaluate_gates(scm, config, sample_seed)
        if report.passed:
            return scm, report
    raise RuntimeError(f"no instance passed all gates within {MAX_ATTEMPTS} attempts")

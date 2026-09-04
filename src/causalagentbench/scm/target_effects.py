"""Direct construction of the three outcome-facing properties the gates check, replacing hope with guarantee.

``generate()`` draws every edge weight randomly (UUMC). Three gates - driver separation
(``structure_valid``), the mediation trap (``mediation_realized``), and the decoy's confounding margin
(``confounding_realized``) - all depend on specific edges into or around the outcome landing in a
comfortable range, and pure random draws don't reliably do that at once: measured to collapse combined
acceptance to under 5% even after fixing gate-threshold scaling, because two of the three gates were still
just checked against chance rather than constructed. This module
fixes all three the same way: pick the target value directly, then solve the exact edge weight needed to
hit it, instead of drawing random weights and hoping.

Construction order matters and mirrors the outcome's actual dependency structure: mediator edges first
(so the mediated effect is fixed and protected before anything else touches the outcome's incoming
budget), then confounder->outcome edges get a comfortable starting magnitude (still shrinkable, unlike
the mediator edges - see below), then driver direct effects (solved against the now-fixed mediator
contribution and the confounder edges as they stand after any shrink), then decoy edges last (each needs
to know the *realized* weakest driver correlation, which only exists once driver construction is done).
Every step patches only a node's own incoming edges - never anything upstream of it - so nothing already
fixed is disturbed by a later step.

Every target this module picks is expressed *relative* to the instance it belongs to - the driver-effect
floor and gap via ``scaled_threshold``, the mediated path as a share of its own driver's effect, the decoy
margin as a scaled offset from the realized weakest driver correlation. That is one rule, not three
coincidences: the outcome's incoming variance is capped at 1, so every demand on it competes, and a demand
expressed as a fixed absolute magnitude grows without bound as a share of that cap as driver count rises.
"""

from __future__ import annotations

import numpy as np

from causalagentbench.scm.effects import marginal_covariance, total_effects
from causalagentbench.types import SCM, Role

#: Calibrated at a 3-driver reference point so `scoring/gates.py`'s non-constructed checks stay numerically
#: consistent there; every other n scales from here. Single source of truth, imported by
#: `gates.py` rather than duplicated - two independently-derived formulas for "the guaranteed minimum" is
#: exactly how this module's own gap requirement and the gate's threshold drifted out of sync the first
#: time.
REFERENCE_N = 3
REFERENCE_MIN_EFFECT = 0.10
REFERENCE_MIN_GAP = 0.05
#: 10% headroom above the gate's own floor, so construction reliably clears the check it has to satisfy
#: rather than landing exactly on the boundary.
_MARGIN = 1.1

#: `mediation_realized`'s and `confounding_realized`'s floors. Live here, not in `scoring/gates.py`, for
#: the same reason the driver-effect thresholds do: construction and the check that verifies it must share
#: one definition. Both are *relative* quantities, never fixed magnitudes: everything competing for the
#: outcome's unit-variance budget must scale the same way the achievable effects do (`1/sqrt(N)`), so a
#: threshold pinned to a fixed absolute size grows without bound as a share of that budget and eventually
#: consumes all of it. Being constructed *first* does not make a demand free - it still competes. Measured
#: before this was fixed: mediator edges alone took 0.30 of the budget at 3 drivers but 0.75 at 10, and
#: mediation construction was infeasible in 81% of attempts at 10 drivers, collapsing overall acceptance to
#: 0.3%.
MIN_MEDIATED_SHARE = 0.25
"""|W[driver,mediator] * W[mediator,outcome]| must be at least this *fraction* of that driver's own total
effect, so over-adjustment biases it materially. A fraction in magnitude only - the mediated path may
oppose the direct effect (`_construct_mediator_edge`), which this floor is deliberately blind to.

A share, not a magnitude, on its own merits and not just for the budget: what makes the over-adjustment trap
sharp is the mediated path carrying a material portion of the effect it belongs to - condition on the
mediator and the estimate moves by that share. An absolute floor expresses that faithfully at exactly one
driver count, and for a weak driver it is simply wrong: a fixed 0.10 can exceed the entire total effect it
is supposed to be one component of."""
CONFOUNDING_MARGIN = 0.05
"""Decoy's |corr(decoy, outcome)| must beat the weakest driver's correlation by this margin, scaled to the
instance's driver count via `scaled_threshold`.

The two correlations it sits between both shrink as `1/sqrt(N)`, so an unscaled margin is a growing demand
on a shrinking quantity - the decoy edge needed to clear it eventually exceeds unit variance and the
candidate is rejected, which is what made this the largest single rejection source (§11.6)."""
#: Multiplier range applied to a floor when constructing comfortably above it, so instances vary rather
#: than every one landing on the exact same value.
_SLACK_RANGE = (1.2, 2.0)


def scaled_threshold(reference_value: float, n_drivers: int) -> float:
    """Scale a `REFERENCE_N`-calibrated threshold for a different driver count.

    The `1/sqrt(N)` relationship isn't a tuning choice - it's the same variance-conservation law that made
    naive UUMC sampling shrink driver effects as driver count grew (measured to collapse gate acceptance to
    0% at 7+ drivers): with N drivers competing for the outcome's fixed
    unit-variance budget, a threshold calibrated for one driver count only stays equally hard to clear at
    another if it scales the same way the achievable effects do.
    """
    return reference_value * float(np.sqrt(REFERENCE_N / n_drivers))


def target_effects(rng: np.random.Generator, drivers: tuple[str, ...]) -> dict[str, float]:
    """Signed target total effects, one per driver, with guaranteed minimum |gap| and no positional pattern.

    Magnitudes are drawn with a guaranteed minimum spacing (sorted uniforms over the compressible leftover
    range, then the mandatory gap added back at each step - exact by construction, no rejection sampling
    needed for this piece), using `scaled_threshold` for both the floor and the gap so this always clears
    `scoring/gates.py`'s own `structure_valid` check by construction, not by coincidence. Sign and driver
    assignment are both randomized independently of magnitude order, so neither a driver's identity nor its
    rank in the sorted magnitude list ever predicts its effect - the same anti-memorization principle
    already applied to topology, applied here to effect size.
    """
    n = len(drivers)
    effect_min = _MARGIN * scaled_threshold(REFERENCE_MIN_EFFECT, n)
    gap = _MARGIN * scaled_threshold(REFERENCE_MIN_GAP, n)
    effect_max = effect_min + (n - 1) * gap + 2 * gap  # a modest, comfortably-feasible spread above the floor
    compressible = (effect_max - effect_min) - (n - 1) * gap
    offsets = np.sort(rng.uniform(0.0, compressible, size=n))
    magnitudes = effect_min + offsets + np.arange(n) * gap
    signs = rng.choice((-1.0, 1.0), size=n)
    permuted = rng.permutation(np.array(drivers, dtype=object))
    return dict(zip(permuted, magnitudes * signs, strict=True))


#: Driver->mediator edge magnitude range, chosen directly rather than left to UUMC. A mediator has exactly
#: one parent, so UUMC's own draw for that edge is `r ~ Uniform(0, 1)` in magnitude - it can land
#: arbitrarily close to 0, which would force the mediator->outcome edge (solved from it, below) to blow up
#: to compensate. Bounding it away from zero up front avoids that failure mode entirely.
_MEDIATOR_EDGE_RANGE = (0.3, 0.7)

#: |Portion| of a driver's own total effect routed through its mediator, drawn comfortably above
#: `MIN_MEDIATED_SHARE` so construction clears the gate it has to satisfy rather than landing on the
#: boundary - the role `_SLACK_RANGE` plays for the thresholds that are still absolute. A magnitude only:
#: the mediated path's *sign* is drawn independently (see `_construct_mediator_edge`).
_MEDIATED_SHARE_RANGE = (0.35, 0.60)

#: Confounder->outcome edge magnitude range, chosen directly rather than left to raw UUMC - the same
#: near-zero problem the mediator edge avoids: a confounder can have several co-parents on the outcome
#: (drivers, mediators, other confounders), so its own UUMC-drawn coefficient can land arbitrarily close
#: to 0, leaving no correlation with the outcome for `with_constructed_confounding` to ever beat. Sized
#: close to `_MEDIATOR_EDGE_RANGE` by direct measurement, not guesswork: narrower ranges (e.g. (0.1, 0.25))
#: measurably made `confounding_construct_infeasible` *worse*, not better - a smaller starting value has
#: less room left after the shrink ladder in `with_constructed_effects` (below) backs it off for a
#: driver-target solve, so it beats the decoy margin less often, not more.
#: Widening further past this range keeps helping only marginally while nudging
#: `mediation_construct_infeasible` up - this is the measured sweet spot, not a protected value like the
#: mediator's: it still stays part of the shrink ladder's lever.
_CONFOUNDER_EDGE_RANGE = (0.3, 0.6)

#: A collider's two incoming edges, chosen directly rather than left to raw UUMC - the same near-zero
#: problem the mediator/confounder edges avoid, and measured to matter just as much here: the bias adjusting
#: for a collider induces scales with ``|w1 * w2|``, so a near-zero raw draw on either parent leaves the
#: trap toothless (measured over 200 raw draws: bias collapsing under 0.01 whenever one weight landed near
#: 0, against a reliable 0.04-0.29 once both are constructed away from it).
_COLLIDER_EDGE_RANGE = (0.3, 0.6)


def _construct_collider_edges(
    w: np.ndarray, noise_var: np.ndarray, parent1_idx: int, parent2_idx: int, child_idx: int, rng: np.random.Generator
) -> None:
    """Mutate ``w``/``noise_var`` in place: set a collider's two incoming edges to a comfortable magnitude.

    Valid only for a collider whose two parents are themselves mutually independent exogenous roots - true
    of every ``Role.COLLIDER`` node ``scm/gadgets.py`` constructs, since each collider gadget's two parents
    are both degree-0. With no cross term to account for, the two weights alone fix the child's own unit
    variance (``noise_var = 1 - w1^2 - w2^2``) directly, the same closed form ``_set_single_parent_edge``
    uses for a degree-1 child, generalized to two independent parents instead of one.

    Unlike that helper, this never touches ``sigma``: a ``Role.COLLIDER`` node is a pure sink (no
    outgoing edges, ``scm/gadgets.py``), so nothing downstream needs its covariance propagated
    incrementally - the final ``marginal_covariance`` call in ``with_constructed_effects`` recomputes it
    correctly from the completed ``w`` once every edge, this one included, is set.
    """
    w1 = float(rng.uniform(*_COLLIDER_EDGE_RANGE)) * float(rng.choice((-1.0, 1.0)))
    w2 = float(rng.uniform(*_COLLIDER_EDGE_RANGE)) * float(rng.choice((-1.0, 1.0)))
    w[parent1_idx, child_idx] = w1
    w[parent2_idx, child_idx] = w2
    noise_var[child_idx] = 1.0 - w1 * w1 - w2 * w2


#: Shrink factors tried on the outcome's remaining (confounder) incoming edges, in order, when solving
#: driver targets at full strength (1.0, i.e. no shrink) leaves no room. Adaptive, not a single fixed
#: value: confirmed by direct comparison that a single fixed shrink is worse at every level tried (0.5 ->
#: 2.3%, 0.3 and 0.15 -> 0.0%) than trying 1.0 first and backing off only when needed - different instances
#: need different amounts of room, so one fixed value is either too small to help or large enough to break
#: something else. Mediator edges are excluded from this shrink (see
#: `_construct_mediator_edge`) - they're fixed before this ladder runs, not left for it to squeeze.
_SHRINK_LADDER = (1.0, 0.5, 0.25, 0.1, 0.03)


def _set_single_parent_edge(
    w: np.ndarray, noise_var: np.ndarray, sigma: np.ndarray, parent_idx: int, child_idx: int, edge: float
) -> None:
    """Mutate ``w``/``noise_var``/``sigma`` in place: set ``parent->child`` and propagate its covariance.

    Valid only for a ``child`` whose *only* parent is ``parent_idx`` - the closed-form UUMC formula for a
    degree-1 node's row (``noise_var = 1 - edge^2``, ``Cov(child, other) = edge * Cov(parent, other)``),
    applied directly instead of drawn randomly. Used for both a mediator (parent: its driver) and a decoy
    (parent: its confounder) - the same math either way, so it lives once, not once per caller.
    """
    w[parent_idx, child_idx] = edge
    noise_var[child_idx] = 1.0 - edge * edge
    for other in range(w.shape[0]):
        if other == child_idx:
            continue
        covariance = edge * sigma[parent_idx, other]
        sigma[child_idx, other] = covariance
        sigma[other, child_idx] = covariance
    sigma[child_idx, child_idx] = 1.0


def _construct_mediator_edge(
    w: np.ndarray,
    noise_var: np.ndarray,
    sigma: np.ndarray,
    driver_idx: int,
    mediator_idx: int,
    outcome_idx: int,
    target: float,
    rng: np.random.Generator,
) -> None:
    """Mutate ``w``/``noise_var``/``sigma`` in place to fix one driver's mediator edges.

    Picks ``driver->mediator`` directly (see ``_MEDIATOR_EDGE_RANGE``), then solves ``mediator->outcome``
    so the mediated path carries ``_MEDIATED_SHARE_RANGE`` of ``target`` in *magnitude* - the driver's own
    total-effect target, not a fixed magnitude (see ``MIN_MEDIATED_SHARE``).

    The mediated path's **sign is drawn independently of the target's**, so the indirect effect agrees with
    the direct one only half the time. This costs budget (an opposing path makes the direct edge
    ``(1 + share) * target``, larger than the target it is solving for) and it would be cheaper to always
    share the sign - but always sharing it is unsound in two ways. It deletes *inconsistent mediation*
    (suppression: a direct and an indirect effect that oppose), a real and well-documented structure, and
    it makes ``|direct| < |total|`` hold universally, which turns "adding Z to the adjustment set shrinks my
    estimate" into a perfectly separating test for "Z is a mediator" - measured 100% on true mediators vs.
    53.7% on other variables. The over-adjustment trap is supposed to require reasoning about what is
    downstream of the treatment; a numeric probe that identifies mediators without any structural reasoning
    defeats it. The scaling fix this module exists for is that the mediated
    path is *proportional to its own driver's effect* rather than a fixed magnitude - which is preserved
    either way, independent of sign.

    Only the mediator's own row is touched here - the outcome's ``mediator->outcome`` entry is set as a
    weight only; its own noise/covariance aren't recomputed until later, when its whole row is rebuilt from
    every parent at once (``with_constructed_effects``).
    """
    edge = float(rng.uniform(*_MEDIATOR_EDGE_RANGE)) * float(rng.choice((-1.0, 1.0)))
    _set_single_parent_edge(w, noise_var, sigma, driver_idx, mediator_idx, edge)

    share = float(rng.uniform(*_MEDIATED_SHARE_RANGE)) * float(rng.choice((-1.0, 1.0)))
    w[mediator_idx, outcome_idx] = share * target / edge


def _construct_confounder_edge(w: np.ndarray, confounder_idx: int, outcome_idx: int, rng: np.random.Generator) -> None:
    """Mutate ``w`` in place: set one confounder's direct edge into the outcome to a comfortable magnitude.

    Overwrites whatever `generate()`'s raw UUMC draw left there (see ``_CONFOUNDER_EDGE_RANGE``). Unlike
    ``_construct_mediator_edge``, this only sets the coefficient - it stays part of the shrinkable set in
    ``with_constructed_effects``, so it is deliberately left un-solved and un-protected here.
    """
    w[confounder_idx, outcome_idx] = float(rng.uniform(*_CONFOUNDER_EDGE_RANGE)) * float(rng.choice((-1.0, 1.0)))


def with_constructed_effects(
    reference: SCM, targets: dict[str, float], construction_rng: np.random.Generator
) -> SCM | None:
    """Return ``reference`` with mediator/confounder edges and every driver's outcome edge constructed.

    Returns ``None`` if no shrink level makes the driver-target solve feasible. Three things are
    guaranteed by construction here, in order:

    1. Each driver-with-mediator's ``driver->mediator->outcome`` path is fixed to a comfortable magnitude
       (``_construct_mediator_edge``) - this guarantees ``mediation_realized`` outright, rather than
       leaving it to whatever UUMC happened to draw.
    2. Each confounder's direct edge into the outcome is set to a comfortable, non-tiny magnitude
       (``_construct_confounder_edge``) - so ``with_constructed_confounding`` always has real correlation
       to work with, rather than whatever (possibly near-zero) value UUMC happened to draw.
    3. Every driver archetype has exactly one direct edge into the outcome (``scm/gadgets.py``), so
       ``target - (effect via every other path)`` gives the exact weight needed for that driver's *total*
       effect to hit its target - a closed-form solve, not a search.

    The outcome's total incoming variance is capped at 1 (UUMC's unit-variance guarantee); with the
    mediated contribution now fixed rather than random, only the confounder edges are left to compete with
    driver targets for what's left, so only they are shrunk (see ``_SHRINK_LADDER``) when the straightforward
    solve is infeasible - construction gives them a comfortable *starting* value (step 2), it doesn't
    protect them from this ladder the way mediator edges are protected. Only the outcome's own row is
    patched in that step; it is a terminal sink with no children, so nothing above it in the graph is
    affected.
    """
    scaffold = reference.scaffold
    outcome_idx = scaffold.index(scaffold.outcome)
    n_nodes = len(scaffold.node_names)

    w = reference.W.copy()
    noise_var = reference.noise_var.copy()
    sigma = reference.sigma.copy()
    mediator_mask = np.zeros(n_nodes, dtype=bool)
    for driver in scaffold.drivers:
        mediator = scaffold.mediator_of(driver)
        if mediator is None:
            continue
        # Only decoys are drivers without a target, and no decoy has a mediator (`scm/gadgets.py`) - assert
        # rather than assume, since a mediated driver silently missing a target would now be a KeyError.
        assert driver in targets, f"driver {driver!r} has a mediator but no target effect"
        mediator_idx = scaffold.index(mediator)
        _construct_mediator_edge(
            w, noise_var, sigma, scaffold.index(driver), mediator_idx, outcome_idx, targets[driver], construction_rng
        )
        mediator_mask[mediator_idx] = True

    for confounder in scaffold.nodes_with_role(Role.CONFOUNDER):
        _construct_confounder_edge(w, scaffold.index(confounder), outcome_idx, construction_rng)

    for collider in scaffold.nodes_with_role(Role.COLLIDER):
        parents = scaffold.parents(collider)
        assert len(parents) == 2, f"collider {collider!r} must have exactly two parents, got {parents}"
        _construct_collider_edges(
            w,
            noise_var,
            scaffold.index(parents[0]),
            scaffold.index(parents[1]),
            scaffold.index(collider),
            construction_rng,
        )
        # A collider parent that is ALSO some driver's *sole* parent (``_mbias_driver``'s V, feeding both A
        # and the collider) needs its own edge constructed too - the trap's strength depends on how much of
        # the collider's spurious signal reaches the driver, which a near-zero raw UUMC draw on THIS edge
        # would zero out regardless of how strong the collider's own two edges are (measured: bias collapsing
        # to <0.005 whenever this edge, not the collider's, landed near zero - the plain collider gadget has
        # no such edge, since its own driver has no parent at all). ``_collider_edges`` never has this shape,
        # so this loop is a no-op for it - it exists purely for ``_mbias_driver``'s V.
        for parent in parents:
            for driven in (d for d in scaffold.drivers if scaffold.parents(d) == (parent,)):
                edge = float(construction_rng.uniform(*_COLLIDER_EDGE_RANGE)) * float(
                    construction_rng.choice((-1.0, 1.0))
                )
                _set_single_parent_edge(w, noise_var, sigma, scaffold.index(parent), scaffold.index(driven), edge)

    driver_mask = np.zeros(n_nodes, dtype=bool)
    driver_mask[[scaffold.index(d) for d in targets]] = True

    for shrink in _SHRINK_LADDER:
        shrunk_w = w.copy()
        shrinkable = (shrunk_w[:, outcome_idx] != 0.0) & ~driver_mask & ~mediator_mask
        shrunk_w[shrinkable, outcome_idx] *= shrink

        # Recomputed on the shrunk structure, before any driver edge is overwritten, so each driver's
        # target-minus-non-direct solve is exact rather than circular. Includes each driver's now-fixed
        # mediated contribution automatically, since that's baked into `shrunk_w` already.
        partial_tau = total_effects(shrunk_w)
        for driver, target in targets.items():
            i = scaffold.index(driver)
            non_direct = partial_tau[i, outcome_idx] - shrunk_w[i, outcome_idx]
            shrunk_w[i, outcome_idx] = target - non_direct

        final_noise_var = noise_var.copy()
        incoming = shrunk_w[:, outcome_idx]
        # incoming[outcome_idx] is always 0 (no self-loop), so `sigma`'s own (stale) outcome row/column
        # never enters this bilinear form - only covariance *among* the outcome's parents matters here.
        final_noise_var[outcome_idx] = 1.0 - float(incoming @ sigma @ incoming)
        if final_noise_var[outcome_idx] < 0:
            continue

        return SCM(
            scaffold=scaffold,
            W=shrunk_w,
            noise_var=final_noise_var,
            sigma=marginal_covariance(shrunk_w, final_noise_var),
            tau=total_effects(shrunk_w),
        )
    return None


def with_constructed_confounding(scm: SCM, rng: np.random.Generator) -> SCM | None:
    """Return ``scm`` with every decoy's edge from its confounder solved to beat the weakest driver.

    Returns ``None`` if a confounder's own correlation with the outcome is too small for any edge weight to
    reach the target without exceeding unit variance - historically ~25-27% of attempts, the largest single
    rejection source in `generate_gated_procedural`'s loop, because
    the margin it had to clear was a fixed absolute value while the correlations on both sides of it shrank
    as ``1/sqrt(N)``; the margin now scales with driver count (``CONFOUNDING_MARGIN``). Must run *after*
    driver construction - it needs each driver's realized correlation with the outcome, which only exists
    once ``with_constructed_effects`` has run.

    Each decoy has exactly one parent, its confounder (``scm/gadgets.py``), and no children, so - like the
    outcome in ``with_constructed_effects`` - only its own row is patched; nothing else in the graph is
    affected by any decoy's edge.
    """
    scaffold = scm.scaffold
    outcome_idx = scaffold.index(scaffold.outcome)
    decoys = scaffold.nodes_with_role(Role.DECOY)
    if not decoys:
        return scm
    non_decoy_drivers = scaffold.non_decoy_drivers()
    weakest_driver_corr = min(abs(float(scm.sigma[scaffold.index(d), outcome_idx])) for d in non_decoy_drivers)
    margin = scaled_threshold(CONFOUNDING_MARGIN, len(non_decoy_drivers))

    w = scm.W.copy()
    noise_var = scm.noise_var.copy()
    sigma = scm.sigma.copy()
    for decoy in decoys:
        (confounder,) = scaffold.correct_adjustment[decoy]
        confounder_idx, decoy_idx = scaffold.index(confounder), scaffold.index(decoy)
        confounder_corr = float(sigma[confounder_idx, outcome_idx])
        if confounder_corr == 0.0:
            return None

        target = weakest_driver_corr + float(rng.uniform(*_SLACK_RANGE)) * margin
        edge = target / confounder_corr
        if edge * edge > 1.0:
            return None
        _set_single_parent_edge(w, noise_var, sigma, confounder_idx, decoy_idx, edge)

    return SCM(scaffold=scaffold, W=w, noise_var=noise_var, sigma=sigma, tau=total_effects(w))

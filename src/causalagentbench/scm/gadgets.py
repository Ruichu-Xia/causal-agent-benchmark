"""Procedural scaffold composition: a small catalog of graph fragments, composed per seed.

Each "gadget" below is a self-contained local pattern contributing one candidate driver (or, for a decoy,
one candidate that is a trap): a clean driver, a confounded driver, a richer multi-confounder driver, an
instrumented driver, and a decoy. ``procedural_scaffold`` draws a random count of drivers and decoys per
seed and wires every gadget's terminal edge into one shared outcome - the same anchor-based composition
CausalDS uses to keep the naming/verbalization problem tractable, since a gadget only ever touches the
rest of the graph through the shared outcome, never another gadget directly.

No gadget declares whether its confounder/instrument edge is ambiguous - that is discovered separately
(``scm/identifiability.py``), never assumed here. Whether a driver's ambiguity ends up resolved is also not
tied to its archetype: any confounder- or instrument-bearing driver can independently get the resolving
modifier (``_add_upstream_cause``) or not, discovered the same way. Exactly two driver slots are reserved to
guarantee the composed topology always has both an ambiguous driver and a resolved (identifiable-but-still-
confounded) driver - constructed directly, not left to a topology-gate retry (`procedural_scaffold_plan.md`
§11.3's revision: the original design checked this after the fact and rejected ~32% of raw draws; every
other driver slot stays fully free, exactly as before). Their *archetype* is still drawn, not declared -
only which two properties get guaranteed is fixed, never which gadget shape produces them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causalagentbench.types import Role, Scaffold, SeedLike

OUTCOME = "Y"

#: Stage 0 invariants (mirrors the constants already in ``scoring/gates.py``); migrate to
#: ``GeneratorConfig`` once tunable difficulty is wired up.
CANDIDATE_COUNT_RANGE = (4, 9)
"""Total candidate columns per instance - gadget drivers *plus* decoys, i.e. the task size the agent sees."""
DECOY_COUNT_RANGE = (1, 2)
"""Decoys carved out of the candidate budget, never added on top of it.

At least one, so the correlation trap exists in every instance: measured under the old ``(0, 2)``, 39-42%
of accepted instances contained no decoy at all, leaving the "a strong correlate can have zero effect"
lesson absent from two of every five tasks.

Carved out rather than added because task size must not vary with the draw. A varying candidate count makes
``ranking_tau`` incomparable across instances (its achievable granularity and its chance variance both
depend on the count) and gives an RL policy a varying action-space size. Varying the *split* instead keeps
the size fixed while leaving the decoy count unmemorizable."""
FREE_ARCHETYPES = ("clean", "confounded", "multi_confounder", "instrumented")
#: Chance that a confounder-/instrument-bearing driver gets its ambiguity resolved (§11.1/§11.2). One coin
#: flip per driver, not per confounder - a multi-confounder driver's two confounders are always resolved
#: together, never independently (§11.1: the partial case is real but scores identically to fully ambiguous
#: under the binary abstain schema, so isn't worth tracking separately).
RESOLVE_PROBABILITY = 0.25
#: 0.25 rather than 0.5, chosen so the four quadrants come out equally weighted. This probability sets how
#: many candidates the data can settle on its own, which is exactly the split between "trust the data" and
#: "fall back on the name": at 0.25 an instance averages 3.6 unidentifiable and 3.4 identifiable candidates
#: of 7, giving ~25% in each of `scoring/score.py::Quadrant`'s four cells across the paired naming
#: conditions. Lower is tempting - 0.0 widens the oracle-vs-data-only margin from 25.7% to 28.9% - but it
#: shrinks the identifiable half (where "don't be swayed by a misleading name" is tested) to 3.0 of 7 and
#: drops orientable edges from 72% to 67%, eroding the name-consistency check the whole design rests on.
#: Acceptance is flat across this range, so the choice costs nothing to generate either way.


@dataclass(frozen=True)
class _Gadget:
    """One composed fragment: its nodes/roles, its edges, and its contributed candidate driver."""

    roles: dict[str, Role]
    edges: list[tuple[str, str]]
    driver: str
    adjustment: tuple[str, ...]


def _clean_driver(i: int) -> _Gadget:
    b = f"B{i}"
    return _Gadget(roles={b: Role.DRIVER_CLEAN}, edges=[(b, OUTCOME)], driver=b, adjustment=())


def _confounded_driver(i: int) -> _Gadget:
    c, a, m = f"C{i}", f"A{i}", f"M{i}"
    roles = {c: Role.CONFOUNDER, a: Role.DRIVER_CONFOUNDED, m: Role.MEDIATOR}
    edges = [(c, a), (c, OUTCOME), (a, m), (a, OUTCOME), (m, OUTCOME)]
    return _Gadget(roles=roles, edges=edges, driver=a, adjustment=(c,))


def _multi_confounder_driver(i: int) -> _Gadget:
    c1, c2, a, m = f"C{i}", f"C{i}_2", f"A{i}", f"M{i}"
    roles = {c1: Role.CONFOUNDER, c2: Role.CONFOUNDER, a: Role.DRIVER_CONFOUNDED, m: Role.MEDIATOR}
    edges = [
        (c1, a), (c1, OUTCOME), (c1, c2),  # c1-c2 keeps the collider at `a` shielded, not unshielded
        (c2, a), (c2, OUTCOME),
        (a, m), (a, OUTCOME), (m, OUTCOME),
    ]  # fmt: skip
    return _Gadget(roles=roles, edges=edges, driver=a, adjustment=(c1, c2))


def _instrumented_driver(i: int) -> _Gadget:
    z, a = f"Z{i}", f"A{i}"
    roles = {z: Role.INSTRUMENT, a: Role.DRIVER_CONFOUNDED}
    edges = [(z, a), (a, OUTCOME)]
    return _Gadget(roles=roles, edges=edges, driver=a, adjustment=())


def _decoy(j: int, confounder: str) -> _Gadget:
    dz = f"Dz{j}"
    return _Gadget(roles={dz: Role.DECOY}, edges=[(confounder, dz)], driver=dz, adjustment=(confounder,))


_BUILDERS = {
    "clean": _clean_driver,
    "confounded": _confounded_driver,
    "multi_confounder": _multi_confounder_driver,
    "instrumented": _instrumented_driver,
}


def _resolvable_targets(archetype: str, i: int) -> tuple[str, ...]:
    """The node(s) in driver ``i``'s gadget whose incoming edge could be resolved (§11.1).

    The confounder(s) for ``confounded``/``multi_confounder``, the instrument for ``instrumented``; empty
    for ``clean`` (an exogenous root has nothing to resolve - it's already always identifiable).
    """
    if archetype == "confounded":
        return (f"C{i}",)
    if archetype == "multi_confounder":
        return (f"C{i}", f"C{i}_2")
    if archetype == "instrumented":
        return (f"Z{i}",)
    return ()


def _add_upstream_cause(gadget: _Gadget, target: str, names: tuple[str, str]) -> _Gadget:
    """Return ``gadget`` with two new exogenous, non-adjacent parents added to ``target``.

    Gives ``target`` (a confounder or instrument node) its own v-structure - `X1 -> target <- X2`, `X1`/`X2`
    otherwise unconnected. Ordinary v-structure detection orients both new edges, and Meek rule 1 then
    forces ``target``'s own edge into the driver as well (`procedural_scaffold_plan.md` §11.1) - resolving
    it from data alone while the driver stays genuinely confounded (its `correct_adjustment` is unchanged).
    Never declares this outcome - it's read back off `scm/identifiability.py::unoriented_edges`, exactly as
    for every other edge.
    """
    u1, u2 = names
    roles = {**gadget.roles, u1: Role.UPSTREAM_CAUSE, u2: Role.UPSTREAM_CAUSE}
    edges = [*gadget.edges, (u1, target), (u2, target)]
    return _Gadget(roles=roles, edges=edges, driver=gadget.driver, adjustment=gadget.adjustment)


def _force_resolve(gadget: _Gadget, archetype: str, i: int) -> _Gadget:
    """Unconditionally apply ``_add_upstream_cause`` to every resolvable target in driver ``i``'s gadget.

    One call per driver, not per confounder - a ``multi_confounder`` driver's two confounders are always
    resolved together (§11.1: the partial, one-of-two-resolved outcome is real but scores identically to
    fully ambiguous under the binary abstain schema, so isn't worth tracking separately).
    """
    for j, target in enumerate(_resolvable_targets(archetype, i)):
        names = (f"U{i}_{2 * j + 1}", f"U{i}_{2 * j + 2}")
        gadget = _add_upstream_cause(gadget, target, names)
    return gadget


def _maybe_resolve(
    gadget: _Gadget, archetype: str, i: int, rng: np.random.Generator, resolve_probability: float
) -> _Gadget:
    """Randomly apply ``_force_resolve`` to driver ``i``'s gadget, or don't (``resolve_probability``)."""
    if not _resolvable_targets(archetype, i) or rng.random() >= resolve_probability:
        return gadget
    return _force_resolve(gadget, archetype, i)


#: Archetypes eligible for the forced-resolved *and* forced-ambiguous slots alike: only a confounder's
#: orientation actually changes a driver's identification status (``scm/identifiability.py`` - a
#: confounder always has its own direct edge into the outcome, so whether it points into the driver or
#: not determines whether it must be adjusted for). ``instrumented`` is excluded from both: its adjustment
#: set is always empty and its instrument has no other path to the outcome, so its edge's orientation
#: never changes the driver's effect or adjustment set either way - forcing it "ambiguous" wouldn't
#: guarantee anything, and forcing it "resolved" wouldn't produce a "resolved-but-still-confounded" driver.
#: ``clean`` is excluded from both for the same reason it's excluded from ``FREE_ARCHETYPES``'s resolve
#: coin flip: already always identifiable, nothing to force either way.
_RESOLVABLE_CONFOUNDED_ARCHETYPES = ("confounded", "multi_confounder")
_AMBIGUITY_CAPABLE_ARCHETYPES = _RESOLVABLE_CONFOUNDED_ARCHETYPES


def procedural_scaffold(
    seed: SeedLike, n_candidates: int | None = None, resolve_probability: float = RESOLVE_PROBABILITY
) -> Scaffold:
    """Compose a fresh ``Scaffold`` deterministically from ``seed``.

    Two driver slots are reserved: one forced resolved-but-confounded, one forced ambiguous - the
    guarantee `procedural_scaffold_plan.md` §11.3 needs, constructed directly instead of checked after the
    fact and retried (its original design rejected ~32% of raw draws). Every other slot draws its archetype
    freely from ``FREE_ARCHETYPES`` and, independently, whether its own ambiguity gets resolved
    (``_maybe_resolve``, at ``resolve_probability``) - not special-cased. Decoys attach to a random
    already-placed confounder; neither a decoy's identifiability is assumed here - it's discovered
    afterward, on the composed graph as a whole (a decoy sharing a resolved confounder resolves too, for
    free, via the same Meek-rule propagation - no separate decoy-resolution mechanism needed).

    ``n_candidates`` is the *total* candidate count the agent sees, decoys included; it defaults to a random
    value in ``CANDIDATE_COUNT_RANGE`` drawn from ``seed``. Decoys are drawn first and the remainder become
    gadget drivers, so fixing ``n_candidates`` fixes task size exactly while the decoy count still varies.
    Must leave at least 2 gadget drivers - one slot for each reserved guarantee - so the minimum is
    ``DECOY_COUNT_RANGE[1] + 2``.

    ``resolve_probability`` only affects the free slots - the two reserved ones are always forced either
    way, so the "at least one ambiguous, one resolved" guarantee holds at any value. ``0.0``/``1.0`` push
    every free slot to the same extreme (mostly ambiguous / mostly resolved) for a controlled ambiguity
    sweep; the default ``0.5`` reproduces the original, unparameterized behavior exactly.
    """
    minimum = DECOY_COUNT_RANGE[1] + 2
    if n_candidates is not None and n_candidates < minimum:
        raise ValueError(
            f"n_candidates must be at least {minimum} - up to {DECOY_COUNT_RANGE[1]} decoys plus one "
            f"forced-ambiguous and one forced-resolved driver (§11.3), got {n_candidates}"
        )
    if not 0.0 <= resolve_probability <= 1.0:
        raise ValueError(f"resolve_probability must be in [0, 1], got {resolve_probability}")
    rng = np.random.default_rng(seed)
    if n_candidates is None:
        n_candidates = int(rng.integers(CANDIDATE_COUNT_RANGE[0], CANDIDATE_COUNT_RANGE[1] + 1))
    # Decoys first, drivers from what is left: the total is what must stay fixed, not the split.
    n_decoys = int(rng.integers(DECOY_COUNT_RANGE[0], DECOY_COUNT_RANGE[1] + 1))
    n_gadget_drivers = n_candidates - n_decoys

    resolved_slot, ambiguous_slot = (int(x) for x in rng.choice(n_gadget_drivers, size=2, replace=False))

    gadgets = []
    for i in range(n_gadget_drivers):
        if i == resolved_slot:
            archetype = str(rng.choice(_RESOLVABLE_CONFOUNDED_ARCHETYPES))
            gadget = _force_resolve(_BUILDERS[archetype](i), archetype, i)
        elif i == ambiguous_slot:
            archetype = str(rng.choice(_AMBIGUITY_CAPABLE_ARCHETYPES))
            gadget = _BUILDERS[archetype](i)  # left untouched - ambiguous by construction (§11.0)
        else:
            archetype = str(rng.choice(FREE_ARCHETYPES))
            gadget = _maybe_resolve(_BUILDERS[archetype](i), archetype, i, rng, resolve_probability)
        gadgets.append(gadget)

    # resolved_slot is always confounder-bearing, so this is never empty - not special-cased, still a real
    # guard: it's an emergent consequence of the guarantee above, not something duplicated here.
    confounders = [name for g in gadgets for name, role in g.roles.items() if role is Role.CONFOUNDER]
    if confounders:
        for j in range(n_decoys):
            gadgets.append(_decoy(j, confounder=str(rng.choice(confounders))))

    roles: dict[str, Role] = {OUTCOME: Role.OUTCOME}
    edges: list[tuple[str, str]] = []
    drivers: list[str] = []
    correct_adjustment: dict[str, tuple[str, ...]] = {}
    for gadget in gadgets:
        roles.update(gadget.roles)
        edges.extend(gadget.edges)
        drivers.append(gadget.driver)
        correct_adjustment[gadget.driver] = gadget.adjustment

    return Scaffold(
        node_names=tuple(roles),
        roles=roles,
        edges=tuple(edges),
        drivers=tuple(drivers),
        outcome=OUTCOME,
        correct_adjustment=correct_adjustment,
    )

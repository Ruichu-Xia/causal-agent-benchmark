"""Motif composition: a small catalog of graph fragments wired into one shared outcome, composed per seed.

Each gadget contributes one candidate:

* clean: ``B -> Y``.
* confounded: ``C -> A -> Y``, ``C -> Y``, plus a mediator ``A -> M -> Y``.
* multi_confounder: the same with two linked confounders.
* collider: ``A -> Y``, ``A -> Co <- U -> Y``; adjusting for ``Co`` creates bias.
* mbias: ``V -> A -> Y``, ``V -> Mb <- U -> Y``; adjusting for ``Mb`` creates bias.
* decoy: ``C -> D`` from an existing confounder, with no path to ``Y``.

Every task also gets an outcome child ``Y -> O``: adjusting for it biases every estimate (Cinelli, Forney &
Pearl's "bad control" for a descendant of the outcome).

A confounder-bearing gadget can get two extra exogenous parents of its confounder(s) (``U1 -> C <- U2``).
That v-structure lets the data orient the confounder edge, so the candidate is confounded but its effect is
identified from data alone. Without it, the confounder-driver edges stay undirected in the CPDAG, and the
candidate's effect depends on a direction only the variable names can settle.

The composed graph is the *motif graph* ``G0``. Roles describe positions in ``G0``. The true graph is later
redrawn uniformly from ``G0``'s Markov equivalence class (``scoring/gates.py``), so a node's causal role in
the true graph can differ from its motif role.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causalagentbench.types import Role, Scaffold, SeedLike

OUTCOME = "Y"
OUTCOME_CHILD = "O"

DECOY_COUNT_RANGE = (1, 2)
"""Decoys per task, carved out of the fixed candidate count so task size never varies."""

FREE_ARCHETYPES = ("clean", "confounded", "multi_confounder", "collider", "mbias")

RESOLVE_PROBABILITY = 0.25
"""Chance that a free confounder-bearing gadget gets the upstream causes that make it data-identified."""

_CONFOUNDED_ARCHETYPES = ("confounded", "multi_confounder")


@dataclass(frozen=True)
class _Gadget:
    """One composed fragment: its nodes/roles, its edges, and its candidate."""

    roles: dict[str, Role]
    edges: list[tuple[str, str]]
    candidate: str


def _clean(i: int) -> _Gadget:
    """A driver with only a direct edge into the outcome."""
    b = f"B{i}"
    return _Gadget(roles={b: Role.DRIVER_CLEAN}, edges=[(b, OUTCOME)], candidate=b)


def _confounded(i: int) -> _Gadget:
    """A driver with one confounder and one mediator."""
    c, a, m = f"C{i}", f"A{i}", f"M{i}"
    roles = {c: Role.CONFOUNDER, a: Role.DRIVER_CONFOUNDED, m: Role.MEDIATOR}
    edges = [(c, a), (c, OUTCOME), (a, m), (a, OUTCOME), (m, OUTCOME)]
    return _Gadget(roles=roles, edges=edges, candidate=a)


def _multi_confounder(i: int) -> _Gadget:
    """A driver with two linked confounders and one mediator."""
    c1, c2, a, m = f"C{i}", f"C{i}_2", f"A{i}", f"M{i}"
    roles = {c1: Role.CONFOUNDER, c2: Role.CONFOUNDER, a: Role.DRIVER_CONFOUNDED, m: Role.MEDIATOR}
    edges = [
        (c1, a), (c1, OUTCOME), (c1, c2),  # c1-c2 keeps the collider at `a` shielded
        (c2, a), (c2, OUTCOME),
        (a, m), (a, OUTCOME), (m, OUTCOME),
    ]  # fmt: skip
    return _Gadget(roles=roles, edges=edges, candidate=a)


def _collider(i: int) -> _Gadget:
    """A clean driver plus a collider shared with an independent cause of the outcome."""
    a, u, co = f"A{i}", f"U{i}", f"Co{i}"
    roles = {a: Role.DRIVER_CLEAN, u: Role.CONFOUNDER, co: Role.COLLIDER}
    return _Gadget(roles=roles, edges=[(a, OUTCOME), (u, OUTCOME), (a, co), (u, co)], candidate=a)


def _mbias(i: int) -> _Gadget:
    """A clean driver whose parent and an independent cause of the outcome share a collider (M-bias)."""
    a, v, u, mb = f"A{i}", f"V{i}", f"U{i}", f"Mb{i}"
    roles = {a: Role.DRIVER_CLEAN, v: Role.UPSTREAM_CAUSE, u: Role.CONFOUNDER, mb: Role.COLLIDER}
    return _Gadget(roles=roles, edges=[(v, a), (a, OUTCOME), (v, mb), (u, mb), (u, OUTCOME)], candidate=a)


def _decoy(j: int, confounder: str) -> _Gadget:
    """A candidate caused by an existing confounder, with no path to the outcome in the motif graph."""
    dz = f"Dz{j}"
    return _Gadget(roles={dz: Role.DECOY}, edges=[(confounder, dz)], candidate=dz)


_BUILDERS = {
    "clean": _clean,
    "confounded": _confounded,
    "multi_confounder": _multi_confounder,
    "collider": _collider,
    "mbias": _mbias,
}


def _resolve(gadget: _Gadget, archetype: str, i: int) -> _Gadget:
    """Give each confounder of a confounded gadget two exogenous, non-adjacent parents."""
    targets = (f"C{i}",) if archetype == "confounded" else (f"C{i}", f"C{i}_2")
    roles, edges = dict(gadget.roles), list(gadget.edges)
    for j, target in enumerate(targets):
        u1, u2 = f"U{i}_{2 * j + 1}", f"U{i}_{2 * j + 2}"
        roles |= {u1: Role.UPSTREAM_CAUSE, u2: Role.UPSTREAM_CAUSE}
        edges += [(u1, target), (u2, target)]
    return _Gadget(roles=roles, edges=edges, candidate=gadget.candidate)


def procedural_scaffold(seed: SeedLike, n_candidates: int = 5) -> Scaffold:
    """Compose the motif graph ``G0`` deterministically from ``seed``.

    Two driver slots are reserved: one confounded gadget left unresolved (its effect depends on an
    orientation only names can settle) and one resolved (confounded but identified from data). The other
    slots draw an archetype from ``FREE_ARCHETYPES`` and are resolved with ``RESOLVE_PROBABILITY``. Decoys
    attach to a random confounder. ``n_candidates`` counts drivers plus decoys.
    """
    minimum = DECOY_COUNT_RANGE[1] + 2
    if n_candidates < minimum:
        raise ValueError(f"n_candidates must be at least {minimum}, got {n_candidates}")
    rng = np.random.default_rng(seed)
    n_decoys = int(rng.integers(DECOY_COUNT_RANGE[0], DECOY_COUNT_RANGE[1] + 1))
    n_drivers = n_candidates - n_decoys
    resolved_slot, ambiguous_slot = (int(x) for x in rng.choice(n_drivers, size=2, replace=False))

    gadgets = []
    for i in range(n_drivers):
        if i in (resolved_slot, ambiguous_slot):
            archetype = str(rng.choice(_CONFOUNDED_ARCHETYPES))
            resolve = i == resolved_slot
        else:
            archetype = str(rng.choice(FREE_ARCHETYPES))
            resolve = archetype in _CONFOUNDED_ARCHETYPES and rng.random() < RESOLVE_PROBABILITY
        gadget = _BUILDERS[archetype](i)
        gadgets.append(_resolve(gadget, archetype, i) if resolve else gadget)

    confounders = [name for g in gadgets for name, role in g.roles.items() if role is Role.CONFOUNDER]
    for j in range(n_decoys):
        gadgets.append(_decoy(j, confounder=str(rng.choice(confounders))))

    roles: dict[str, Role] = {OUTCOME: Role.OUTCOME}
    edges: list[tuple[str, str]] = []
    for gadget in gadgets:
        roles.update(gadget.roles)
        edges.extend(gadget.edges)
    roles[OUTCOME_CHILD] = Role.OUTCOME_CHILD
    edges.append((OUTCOME, OUTCOME_CHILD))
    return Scaffold(
        node_names=tuple(roles),
        roles=roles,
        edges=tuple(edges),
        drivers=tuple(g.candidate for g in gadgets),
        outcome=OUTCOME,
    )

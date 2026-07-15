"""Concrete causal-graph structures.

Stage 0 defines one, ``CANONICAL``, a 6-node graph::

    C -> A    C -> Y    C -> Dz      (C confounds A; induces a spurious Dz-Y correlation)
    A -> M    M -> Y                   (M mediates part of A's effect: the bad-control trap)
    A -> Y                             (A's direct effect)
    B -> Y                             (B is a clean, unconfounded driver)

Key consequences the generator and gates rely on:

* ``Dz`` has no outgoing edge, so its total effect on ``Y`` is exactly zero.
* ``A``'s only backdoor path ``A <- C -> Y`` is blocked by ``{C}``; its causal paths ``A -> Y`` and
  ``A -> M -> Y`` stay open, so ``M`` must be excluded from ``A``'s adjustment set.
* ``B`` is exogenous and unconfounded, so its adjustment set is empty.

Later structures (colliders, instruments, latents, ...) are added here as further ``Scaffold`` values.
"""

from __future__ import annotations

from causalagentbench.types import Role, Scaffold

CONFOUNDER = "C"
DRIVER_CONFOUNDED = "A"
DRIVER_CLEAN = "B"
MEDIATOR = "M"
DECOY = "Dz"
OUTCOME = "Y"

CANONICAL = Scaffold(
    node_names=(CONFOUNDER, DRIVER_CONFOUNDED, DRIVER_CLEAN, MEDIATOR, DECOY, OUTCOME),
    roles={
        CONFOUNDER: Role.CONFOUNDER,
        DRIVER_CONFOUNDED: Role.DRIVER_CONFOUNDED,
        DRIVER_CLEAN: Role.DRIVER_CLEAN,
        MEDIATOR: Role.MEDIATOR,
        DECOY: Role.DECOY,
        OUTCOME: Role.OUTCOME,
    },
    edges=(
        (CONFOUNDER, DRIVER_CONFOUNDED),
        (CONFOUNDER, OUTCOME),
        (CONFOUNDER, DECOY),
        (DRIVER_CONFOUNDED, MEDIATOR),
        (DRIVER_CONFOUNDED, OUTCOME),
        (MEDIATOR, OUTCOME),
        (DRIVER_CLEAN, OUTCOME),
    ),
    drivers=(DRIVER_CONFOUNDED, DRIVER_CLEAN, DECOY),
    outcome=OUTCOME,
    correct_adjustment={
        DRIVER_CONFOUNDED: (CONFOUNDER,),
        DRIVER_CLEAN: (),
        DECOY: (CONFOUNDER,),
    },
)

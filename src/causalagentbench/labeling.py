"""Labeler seam: how scaffold nodes get the display labels the agent sees.

A ``Labeler`` maps each node to a column label. Stage 1 ships only ``abstract_labels`` (``X1..Xk``,
role-free), enough to build and test the task interface. The chosen semantic regime - an LLM-delegated
labeler that gives names evoking each node's causal role - and a later misaligned variant are sibling
implementations with the same signature, designed as separate steps.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from causalagentbench.types import Scaffold

Labeler = Callable[[Scaffold, np.random.Generator], dict[str, str]]
"""Maps each node name to a display label. The extensibility seam for naming strategies."""


def abstract_labels(scaffold: Scaffold, rng: np.random.Generator) -> dict[str, str]:
    """Assign role-free labels ``X1..Xk`` to nodes in a randomly permuted order.

    Carries no causal-role information, so it leaks no structure; it also does not make the adjustment
    task well-posed on its own (that is the semantic labeler's job).
    """
    order = rng.permutation(len(scaffold.node_names))
    return {node: f"X{position + 1}" for node, position in zip(scaffold.node_names, order, strict=True)}

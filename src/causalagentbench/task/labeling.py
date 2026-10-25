"""Column naming for public task presentation.

``CauseNetLabeler`` (``generation/causenet.py``) assigns exact CauseNet concepts. ``placeholder_labels``
assigns ``X1..Xk`` for the names ablation. Naming never changes the hidden answer key, and both conditions
produce the same prompt apart from the names themselves.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from causalagentbench.types import SCM


@dataclass(frozen=True)
class Labeling:
    """The output of a labeler: node -> column name, plus where the names came from."""

    node_to_label: Mapping[str, str]
    producer: str  # "placeholder" or "causenet:sha256:<digest>"


Labeler = Callable[[SCM, np.random.Generator], Labeling]
"""Maps a hidden SCM to a ``Labeling``."""


def placeholder_labels(scm: SCM, rng: np.random.Generator) -> Labeling:
    """Assign ``X1..Xk`` in a random order, carrying no domain knowledge."""
    names = scm.scaffold.node_names
    order = rng.permutation(len(names))
    return Labeling(
        node_to_label={node: f"X{pos + 1}" for node, pos in zip(names, order, strict=True)}, producer="placeholder"
    )

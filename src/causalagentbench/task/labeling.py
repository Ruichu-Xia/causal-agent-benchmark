"""Labeler seam: how a hidden SCM becomes the data card the agent sees.

A ``Labeler`` maps an SCM to a ``Labeling`` - a node->name map plus a short domain ``context`` and a
per-column ``schema``. Two implementations exist:

* ``abstract_labels`` - role-free ``X1..Xk``, empty context/schema; the name-blind control and the
  off-axis anchor of the naming axis.
* ``LLMLabeler`` (in ``causalagentbench.generation``) - an LLM authors real names, context, and schema
  for one instance at a target naming alignment ``alpha``.

Whatever the labels, the gold answer is computed on the hidden SCM via ``label_to_node``, so naming is a
pure presentation transform and the reward is alignment-invariant.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from causalagentbench.types import SCM


@dataclass(frozen=True)
class LabelProvenance:
    """How an instance was labeled, for stratified scoring and reproducibility.

    ``node_kind`` (``"aligned"``/``"neutral"``/``"misaligned"`` per node) is what the abstain gold rule
    (``scoring/score.py``) reads: a driver's abstain-correctness depends on whether *its own* name was the
    aligned one, not on the instance's aggregate composition (``n_aligned`` etc., kept for auditing only).
    """

    alignment: float | None  # the constructed grid level in [-1, 1]; None for the name-blind labeler
    n_aligned: int  # composition of the constructed labeling (for auditing)
    n_neutral: int
    n_misaligned: int
    node_kind: Mapping[str, str]  # node -> "aligned"/"neutral"/"misaligned", for per-driver abstain scoring
    domain: str | None
    producer: str  # e.g. "abstract" or "llm:<gen-model>+critic:<critic-model>"


@dataclass(frozen=True)
class Labeling:
    """The output of a labeler: the names plus the agent-facing data card and its provenance."""

    node_to_label: Mapping[str, str]
    context: str  # short domain blurb shown to the agent (may be "")
    column_schema: Mapping[str, str]  # label -> one-line description (may be {})
    provenance: LabelProvenance


Labeler = Callable[[SCM, np.random.Generator], Labeling]
"""Maps a hidden SCM to a ``Labeling``. The extensibility seam for naming strategies."""


def abstract_labels(scm: SCM, rng: np.random.Generator) -> Labeling:
    """Assign role-free labels ``X1..Xk`` in a random order; no context or schema.

    Leaks no structure and carries no naming prior, so it anchors the naming axis off to one side and
    serves as the reasoning-vs-recitation control.
    """
    names = scm.scaffold.node_names
    order = rng.permutation(len(names))
    node_to_label = {node: f"X{pos + 1}" for node, pos in zip(names, order, strict=True)}
    provenance = LabelProvenance(
        alignment=None,
        n_aligned=0,
        n_neutral=0,
        n_misaligned=0,
        node_kind=dict.fromkeys(names, "neutral"),  # no naming manipulation at all - nothing is "aligned"
        domain=None,
        producer="abstract",
    )
    return Labeling(node_to_label=node_to_label, context="", column_schema={}, provenance=provenance)

"""Column naming and provenance for public task presentation.

CauseNetLabeler assigns exact source concepts with edge-preserving matches. Abstract labels are useful
for numeric tests and data-only controls. Naming never changes the hidden numerical answer key.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from causalagentbench.types import SCM


@dataclass(frozen=True)
class LabelProvenance:
    """Label source and optional historical naming metadata.

    ``producer`` records the CauseNet dump's SHA-256 for the default builder. The alignment fields remain
    readable for older dataset artifacts; the default benchmark has no naming manipulation or sweep.
    """

    alignment: float | None  # the constructed grid level in [-1, 1]; None for the name-blind labeler
    n_aligned: int  # composition of the constructed labeling (for auditing)
    n_neutral: int
    n_misaligned: int
    node_kind: Mapping[str, str]  # node -> "aligned"/"neutral"/"misaligned"; provenance, not scoring input
    domain: str | None
    producer: str  # "abstract" or "causenet:sha256:<digest>"


@dataclass(frozen=True)
class Labeling:
    """The output of a labeler: the names plus the agent-facing data card and its provenance."""

    node_to_label: Mapping[str, str]
    context: str  # short domain blurb shown to the agent (may be "")
    column_schema: Mapping[str, str]  # label -> one-line description (may be {})
    provenance: LabelProvenance


Labeler = Callable[[SCM, np.random.Generator], Labeling]
"""Maps a hidden SCM to a ``Labeling``. The extensibility seam for naming strategies."""


#: Context and per-column description used for abstract labels. Deliberately uniform and contentless: the
#: point is that the *prompt shape* matches a named condition (same `Context:` line, same `Columns:` block
#: from `task/instance.py::build_prompt`) while the text carries no domain, role, or structural hint. Without
#: them, a name-blind instance also loses that scaffolding, so it would differ from a named one by layout as
#: well as by naming - and the naming axis is only interpretable if naming is the sole thing that varies.
_ABSTRACT_CONTEXT = "An observational dataset with anonymized columns; all variables are continuous and standardized."
_ABSTRACT_DESCRIPTION = "A standardized continuous measurement; its real-world meaning is not disclosed."


def abstract_labels(scm: SCM, rng: np.random.Generator) -> Labeling:
    """Assign role-free labels ``X1..Xk`` in a random order, with contentless context and descriptions.

    Leaks no structure and carries no naming prior, so it anchors the naming axis off to one side and
    serves as the reasoning-vs-recitation control. Every column gets the *same* description, so the schema
    block adds shape without adding information - see ``_ABSTRACT_CONTEXT`` for why it is there at all.
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
    return Labeling(
        node_to_label=node_to_label,
        context=_ABSTRACT_CONTEXT,
        column_schema=dict.fromkeys(node_to_label.values(), _ABSTRACT_DESCRIPTION),
        provenance=provenance,
    )

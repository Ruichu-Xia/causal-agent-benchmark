"""The "isn't this just a tool call?" control.

Gold identifiability is the population CPDAG: ``scm/identifiability.py`` finds v-structures on the *true*
graph and closes them under Meek's rules 1-3. So an agent handed a perfect causal-discovery output could
read the answer straight off it, and ``identifiability_f1`` would measure tool use rather than reasoning.

This module measures how far that actually gets you on real, finite samples. It runs PC at the same default
alpha the agent's own ``discover`` tool uses and applies a fixed rule with no reasoning in it, so its score
is the floor any agent should clear before its identifiability number means anything. Measured on the built
set (24 instances, 120 drivers): **F1 0.650**, against 0.667-0.769 for a small model - i.e. that model was
barely above a mechanical policy, and the whole gap from 0.650 to 1.0 is finite-sample noise rather than
reasoning depth. That comparison belongs in the results table, which is why this is library code and not a
scratch script.

The rule is deliberately the simple one - a driver is called unidentifiable iff PC leaves any edge incident
to it undirected - so this is a **lower bound** on what a mechanical policy achieves. A sharper rule (check
the specific confounder edge, apply the backdoor criterion over the CPDAG) would score higher and make the
point more strongly, not less.

Needs the optional ``discovery`` extra (causal-learn), imported lazily so importing this module is free.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

import numpy as np

from causalagentbench.scm.identifiability import unidentifiable_drivers
from causalagentbench.task.instance import Instance, InstanceKey

#: The alpha ``agent/tools.py::discover`` defaults to, so the baseline and the agent see the same tool.
DEFAULT_ALPHA = 0.05


class MechanicalF1(NamedTuple):
    """Precision/recall/F1 of the mechanical policy, positive class = *unidentifiable*."""

    precision: float
    recall: float
    f1: float
    n_positive: int
    alpha: float


def mechanical_identifiability_f1(
    instances: Iterable[tuple[Instance, InstanceKey]], *, alpha: float = DEFAULT_ALPHA
) -> MechanicalF1:
    """Run PC on each instance and score the no-reasoning identifiability policy against gold.

    Raises ``ImportError`` if causal-learn is not installed; callers that want the run to proceed without
    the baseline should catch it and report the number as unavailable rather than silently omitting it.
    """
    from causallearn.search.ConstraintBased.PC import pc

    tp = fp = fn = 0
    for instance, key in instances:
        scaffold = key.scm.scaffold
        gold_unidentifiable = set(unidentifiable_drivers(scaffold))
        graph = pc(np.asarray(instance.data), alpha=alpha, show_progress=False).G.graph
        columns = instance.columns
        # causal-learn leaves an edge undirected as graph[i, j] == graph[j, i] == -1.
        touched = {
            columns[a]
            for a in range(len(columns))
            for b in range(len(columns))
            if a != b and graph[a, b] == -1 and graph[b, a] == -1
        }
        for label in instance.candidate_drivers:
            said_unidentifiable = label in touched
            truly_unidentifiable = key.label_to_node[label] in gold_unidentifiable
            tp += said_unidentifiable and truly_unidentifiable
            fp += said_unidentifiable and not truly_unidentifiable
            fn += (not said_unidentifiable) and truly_unidentifiable
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return MechanicalF1(precision, recall, f1, tp + fn, alpha)

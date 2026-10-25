"""Reference baselines, each emitted as an ``Answer`` and scored exactly like a model.

From least to most informed:

* ``zero``: every effect is zero (error 1 by construction).
* ``unadjusted``: regress the outcome on each candidate alone.
* ``adjust_all``: adjust for every other column; biased by mediators, colliders and the outcome child.
* ``pc``: PC discovery, then adjust for each candidate's discovered parents (``scripted_agent.py``).
* ``data_only_best``: the best any name-blind analyst can do. It is given the true CPDAG, regresses on each
  member's parent set, and reports the median of the possible effects, which minimizes expected absolute
  error when every equivalent graph is equally likely (as the generator guarantees).
* ``oracle``: adjust for each candidate's true parents; its error is sampling noise only.

``data_only_best`` and ``oracle`` read the hidden key and are references, not agents.
"""

from __future__ import annotations

import numpy as np

from causalagentbench.scm.identifiability import equivalence_class
from causalagentbench.scoring.metrics import regression_coefficient
from causalagentbench.task.answer import Answer, DriverVerdict
from causalagentbench.task.instance import Instance, InstanceKey

BASELINES = ("zero", "unadjusted", "adjust_all", "pc", "data_only_best", "oracle")


def _regress(instance: Instance, label: str, adjust: list[str]) -> DriverVerdict:
    """A verdict from one adjusted regression on the instance's data."""
    effect = regression_coefficient(instance.data, instance.columns, instance.outcome, label, adjust)
    return DriverVerdict(driver=label, adjustment_set=tuple(adjust), effect=effect)


def _parents(edges: tuple[tuple[str, str], ...], node: str, key: InstanceKey) -> list[str]:
    """Display labels of ``node``'s parents in the DAG ``edges``."""
    return sorted(key.node_to_label[p] for p, c in edges if c == node)


def baseline_answer(name: str, instance: Instance, key: InstanceKey) -> Answer:
    """Return baseline ``name``'s answer for one task."""
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; expected one of {BASELINES}")
    if name == "pc":
        from causalagentbench.scoring.scripted_agent import scripted_answer

        return scripted_answer(instance)
    verdicts = []
    for label in instance.candidate_drivers:
        node = key.label_to_node[label]
        if name == "zero":
            verdicts.append(DriverVerdict(driver=label, adjustment_set=(), effect=0.0))
        elif name == "unadjusted":
            verdicts.append(_regress(instance, label, []))
        elif name == "adjust_all":
            others = [c for c in instance.columns if c not in (label, instance.outcome)]
            verdicts.append(_regress(instance, label, others))
        elif name == "oracle":
            verdicts.append(_regress(instance, label, _parents(key.scm.scaffold.edges, node, key)))
        else:
            fits = []
            for member in equivalence_class(key.scm.scaffold):
                adjust = _parents(member, node, key)
                estimate = regression_coefficient(instance.data, instance.columns, instance.outcome, label, adjust)
                fits.append((estimate, adjust))
            effect = float(np.median([estimate for estimate, _ in fits]))
            # Report the adjustment set of the equivalent graph whose estimate is nearest the median.
            _, adjust = min(fits, key=lambda fit: abs(fit[0] - effect))
            verdicts.append(DriverVerdict(driver=label, adjustment_set=tuple(adjust), effect=effect))
    return Answer(verdicts=tuple(verdicts))

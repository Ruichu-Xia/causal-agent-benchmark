"""A scripted causal-analysis agent: discovery -> adjustment set -> regression, with no LLM.

This is the control that decides what the benchmark is a benchmark *of*. The task decomposes into a
mechanical part - run causal discovery, read an adjustment set off the result, regress - and a
non-mechanical part, which is using the column names on the edges the data provably cannot orient. If a
fifty-line script matches the models, the mechanical part is all anyone is measuring and the paper has to
say so. If the script falls well short, the models are competing against a real baseline.

The policy is the textbook one, and deliberately a *competent* implementation rather than a strawman: in
any DAG the treatment's parent set is a valid backdoor adjustment set, so the script estimates the parents
of each driver from the CPDAG and adjusts for them. Where PC leaves an edge undirected it cannot tell a
parent from a child, and that is exactly the ambiguity the naming axis exists to probe - so the two
admissible readings are both provided:

* ``AmbiguityPolicy.EXCLUDE`` - treat an undirected neighbour as not a parent (do not adjust)
* ``AmbiguityPolicy.INCLUDE`` - treat it as a parent (adjust)

Neither is better a priori without the names, so a name-blind script's expected score is their mean. That
mean is the same quantity ``score.py::reference_scores`` calls ``data-only``, but computed from *estimated*
structure on finite data rather than from the true graph - so the gap between them is the price of
discovery error, which is the thing an oracle baseline cannot show you.

Needs the optional ``discovery`` extra (causal-learn), imported lazily.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from enum import Enum

import numpy as np

from causalagentbench.scoring.metrics import regression_coefficient
from causalagentbench.scoring.score import AnswerScore, score_answer
from causalagentbench.task.answer import Answer, DriverVerdict
from causalagentbench.task.instance import Instance, InstanceKey

#: The alpha ``agent/tools.py::discover`` defaults to, so the script and the agent see the same tool.
DEFAULT_ALPHA = 0.05


class AmbiguityPolicy(Enum):
    """What to do with an edge PC leaves undirected - the choice no data can make for you."""

    EXCLUDE = "exclude"  # assume the ambiguous neighbour is not a parent
    INCLUDE = "include"  # assume it is


class _Cpdag:
    """The parts of a PC result this policy reads: definite parents and undirected neighbours."""

    def __init__(self, graph: np.ndarray, columns: tuple[str, ...]) -> None:
        self.parents: dict[str, set[str]] = {c: set() for c in columns}
        self.undirected: dict[str, set[str]] = {c: set() for c in columns}
        for i, j in itertools.combinations(range(len(columns)), 2):
            # causal-learn's encoding: (-1, 1) is i->j, (1, -1) is j->i, (-1, -1) is undirected.
            head, tail = int(graph[i][j]), int(graph[j][i])
            if head == -1 and tail == -1:
                self.undirected[columns[i]].add(columns[j])
                self.undirected[columns[j]].add(columns[i])
            elif head == -1 and tail == 1:
                self.parents[columns[j]].add(columns[i])
            elif head == 1 and tail == -1:
                self.parents[columns[i]].add(columns[j])


def _fit_cpdag(instance: Instance, alpha: float) -> _Cpdag:
    from causallearn.search.ConstraintBased.PC import pc

    graph = pc(np.asarray(instance.data), alpha=alpha, show_progress=False).G.graph
    return _Cpdag(graph, instance.columns)


def scripted_answer(instance: Instance, *, policy: AmbiguityPolicy, alpha: float = DEFAULT_ALPHA) -> Answer:
    """Run the pipeline and return the answer it produces, in the agent's own contract.

    Emitted as a real ``Answer`` and scored by the same ``score_answer`` the models go through, so the
    comparison cannot drift: any change to the metric moves the script and the models together.
    """
    cpdag = _fit_cpdag(instance, alpha)
    verdicts = []
    for driver in instance.candidate_drivers:
        adjust = set(cpdag.parents[driver])
        if policy is AmbiguityPolicy.INCLUDE:
            adjust |= cpdag.undirected[driver]
        # The outcome can never be an adjustment variable, and neither can the driver itself.
        adjust -= {instance.outcome, driver}
        effect = regression_coefficient(
            np.asarray(instance.data), instance.columns, instance.outcome, driver, sorted(adjust)
        )
        verdicts.append(
            DriverVerdict(
                driver=driver,
                adjustment_set=tuple(sorted(adjust)),
                effect=float(effect),
                # An undirected edge on the driver is precisely "the data did not settle this".
                identifiable_from_data=not cpdag.undirected[driver],
            )
        )
    return Answer(verdicts=tuple(verdicts))


def scripted_scores(
    instances: Iterable[tuple[Instance, InstanceKey]], *, policy: AmbiguityPolicy, alpha: float = DEFAULT_ALPHA
) -> list[AnswerScore]:
    """Score the scripted agent on every instance under one ambiguity policy."""
    return [score_answer(scripted_answer(instance, policy=policy, alpha=alpha), key) for instance, key in instances]

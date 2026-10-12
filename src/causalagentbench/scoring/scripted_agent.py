"""A data-only baseline: PC discovery, candidate-parent adjustment, then OLS.

Undirected neighbors can be included or excluded explicitly. Neither policy is an information-theoretic
ceiling. This baseline uses no CauseNet lookup and no hidden key. The agent and baseline share the PC
configuration and the answer scorer.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from enum import Enum

import numpy as np

from causalagentbench.agent.discovery import DEFAULT_ALPHA, discover_graph
from causalagentbench.scoring.metrics import regression_coefficient
from causalagentbench.scoring.score import AnswerScore, score_answer
from causalagentbench.task.answer import Answer, DriverVerdict
from causalagentbench.task.instance import Instance, InstanceKey


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
    return _Cpdag(discover_graph(instance, alpha), instance.columns)


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

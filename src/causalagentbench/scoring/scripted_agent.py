"""A data-only baseline: PC discovery, adjustment for each candidate's discovered parents, then OLS.

Neighbors PC leaves undirected are not adjusted for. The baseline uses no names and no hidden key, and it
shares the agent's PC configuration and the scorer.
"""

from __future__ import annotations

import itertools

import numpy as np

from causalagentbench.agent.discovery import DEFAULT_ALPHA, discover_graph
from causalagentbench.scoring.metrics import regression_coefficient
from causalagentbench.task.answer import Answer, DriverVerdict
from causalagentbench.task.instance import Instance


class _Cpdag:
    """The definite parents in a PC result; undirected edges are ignored."""

    def __init__(self, graph: np.ndarray, columns: tuple[str, ...]) -> None:
        """Read causal-learn's encoding: (-1, 1) is i->j, (1, -1) is j->i, (-1, -1) is undirected."""
        self.parents: dict[str, set[str]] = {c: set() for c in columns}
        for i, j in itertools.combinations(range(len(columns)), 2):
            head, tail = int(graph[i][j]), int(graph[j][i])
            if head == -1 and tail == 1:
                self.parents[columns[j]].add(columns[i])
            elif head == 1 and tail == -1:
                self.parents[columns[i]].add(columns[j])


def _fit_cpdag(instance: Instance, alpha: float) -> _Cpdag:
    """Run PC on the instance's data."""
    return _Cpdag(discover_graph(instance, alpha), instance.columns)


def scripted_answer(instance: Instance, *, alpha: float = DEFAULT_ALPHA) -> Answer:
    """Run the pipeline and return the answer it produces, in the agent's own contract.

    Emitted as a real ``Answer`` and scored by the same ``score_answer`` the models go through, so the
    comparison cannot drift: any change to the metric moves the script and the models together.
    """
    cpdag = _fit_cpdag(instance, alpha)
    verdicts = []
    for driver in instance.candidate_drivers:
        adjust = set(cpdag.parents[driver])
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
            )
        )
    return Answer(verdicts=tuple(verdicts))

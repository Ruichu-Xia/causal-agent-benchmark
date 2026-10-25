"""Exact scoring against the hidden linear SCM.

The headline metric pools capped absolute errors over every candidate of every task:

    error = sum_i min(|e_i - t_i|, 1) / sum_i |t_i|

where ``e_i`` is the estimate and ``t_i`` the true total effect in standardized units. Predicting zero
everywhere scores exactly 1 because every true effect is below 1 (``scoring/gates.py``). The cap bounds any
single blunder, which keeps task-clustered confidence intervals valid. Absolute error makes the best answer
the median of the agent's beliefs, so committing to the likelier orientation beats averaging two.

The RL reward applies the same metric to one task: ``1 - min(error, 2)``, and ``-1`` for an invalid or
missing answer, so failing on purpose never beats answering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causalagentbench.scm.identifiability import name_critical_candidates
from causalagentbench.task.answer import Answer, AnswerParseError, parse_answer_for_instance
from causalagentbench.task.instance import InstanceKey

SCORING_VERSION = "capped-error-v1"

ERROR_CAP = 1.0
"""Per-candidate cap on absolute error, in standardized effect units."""

REWARD_CLIP = 2.0
"""Task errors above this all earn the same reward, ``1 - REWARD_CLIP``."""

INVALID_REWARD = -1.0
"""Reward for an invalid, incomplete, or missing answer: the floor of the valid range."""


@dataclass(frozen=True)
class AnswerScore:
    """Per-candidate results for one task, keyed by hidden node name."""

    parse_ok: bool
    estimate: dict[str, float]  # zero for every candidate when the answer is invalid
    truth: dict[str, float]
    name_critical: dict[str, bool]
    adjustment_valid: dict[str, bool]  # diagnostic: the reported set removes all population bias

    def capped_error(self, candidate: str) -> float:
        """``min(|estimate - truth|, ERROR_CAP)`` for one candidate."""
        return min(abs(self.estimate[candidate] - self.truth[candidate]), ERROR_CAP)

    @property
    def task_error(self) -> float:
        """The headline metric restricted to this task."""
        return sum(self.capped_error(d) for d in self.truth) / sum(abs(t) for t in self.truth.values())

    @property
    def reward(self) -> float:
        """Bounded per-task RL reward in ``[-1, 1]``."""
        if not self.parse_ok:
            return INVALID_REWARD
        return 1.0 - min(self.task_error, REWARD_CLIP)


def _population_coefficient(key: InstanceKey, driver: str, adjust: list[str]) -> float:
    """Population OLS coefficient of ``driver`` on the outcome given ``adjust``."""
    sigma, index = key.scm.sigma, key.scm.scaffold.index
    rows = [index(driver), *(index(z) for z in adjust)]
    return float(np.linalg.solve(sigma[np.ix_(rows, rows)], sigma[rows, index(key.scm.scaffold.outcome)])[0])


def score_answer(answer: Answer | None, key: InstanceKey) -> AnswerScore:
    """Score a contract-valid ``answer`` (or ``None`` for invalid output) against ``key``."""
    drivers = key.scm.scaffold.drivers
    truth = {d: key.scm.total_effect_on_outcome(d) for d in drivers}
    critical = set(name_critical_candidates(key.scm))
    verdicts = {} if answer is None else {key.label_to_node[v.driver]: v for v in answer.verdicts}
    estimate, valid = {}, {}
    for d in drivers:
        verdict = verdicts.get(d)
        estimate[d] = 0.0 if verdict is None or verdict.effect is None else verdict.effect
        if verdict is None:
            valid[d] = False
        else:
            adjust = [key.label_to_node[z] for z in verdict.adjustment_set]
            valid[d] = abs(_population_coefficient(key, d, adjust) - truth[d]) < 1e-6
    return AnswerScore(
        parse_ok=answer is not None,
        estimate=estimate,
        truth=truth,
        name_critical={d: d in critical for d in drivers},
        adjustment_valid=valid,
    )


def score_output(text: str, key: InstanceKey) -> AnswerScore:
    """Parse model output and score it; invalid output scores as all-zero estimates."""
    labels = tuple(key.node_to_label[d] for d in key.scm.scaffold.drivers)
    try:
        answer = parse_answer_for_instance(
            text,
            labels,
            columns=tuple(key.node_to_label.values()),
            outcome=key.node_to_label[key.scm.scaffold.outcome],
        )
    except AnswerParseError:
        answer = None
    return score_answer(answer, key)

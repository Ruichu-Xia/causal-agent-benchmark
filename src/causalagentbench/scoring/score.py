"""Exact, bounded scoring against the hidden linear SCM.

Reward is effect accuracy: one minus mean normalized, clipped absolute estimation error.
Every candidate is scored. Unclipped mean absolute error, adjustment quality and ranking are diagnostics,
not reward terms. Historical identifiability/naming metadata is retained for loading earlier results.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from causalagentbench.scm.identifiability import unidentifiable_drivers
from causalagentbench.scoring.baselines import effect_scale, population_coefficient
from causalagentbench.task.answer import Answer, AnswerParseError, DriverVerdict, parse_answer_for_instance
from causalagentbench.task.instance import InstanceKey

SCORING_VERSION = "effect-v1"
"""Distinguishes effect-only rewards from the previous adjustment/effect average."""


@dataclass(frozen=True)
class AnswerScore:
    """Bounded scores plus unclipped effect MAE (None for invalid output)."""

    parse_ok: bool
    overall: float  # effect accuracy only; identical to estimation
    identification: float  # 1 - mean induced adjustment bias over every driver
    estimation: float  # 1 - mean normalized effect error over every driver
    effect_mae: float | None  # absolute error in standardized effect units, without clipping
    ranking_tau: float  # Kendall tau vs the true |effect| order over all candidates; secondary
    adjustment_bias: Mapping[str, float]  # per-driver induced bias on [0, 1]; never None, nothing ungraded
    effect_error: Mapping[str, float]  # per-driver normalized error on [0, 1]; never None
    identifiability_judgment: Mapping[str, bool | None]  # what the agent said; None = it did not say
    identifiable: Mapping[str, bool]  # truth, from the population CPDAG; a stratification axis
    alignment: float | None  # naming-alignment level (None = name-blind); the other stratification axis


def _identifiable_drivers(key: InstanceKey) -> Mapping[str, bool]:
    """Every candidate driver, flagged with whether its effect is identified from data alone.

    Resolved once per answer rather than per driver - ``unidentifiable_drivers`` rebuilds the whole CPDAG on
    each call. Answer-independent, so it is filled the same way on a parse failure.
    """
    unidentifiable = set(unidentifiable_drivers(key.scm.scaffold))
    return {driver: driver not in unidentifiable for driver in key.scm.scaffold.drivers}


def _verdict_for(answer: Answer, key: InstanceKey, driver: str) -> DriverVerdict | None:
    """The agent's verdict for ``driver`` (by its display label), or ``None`` if it gave none."""
    label = key.node_to_label[driver]
    return next((v for v in answer.verdicts if v.driver == label), None)


def _estimate(verdict: DriverVerdict | None) -> float:
    """``verdict``'s signed effect estimate, treating a missing verdict as an estimate of zero.

    Signed, because the error metric must be: reporting +0.3 where the truth is -0.3 is a wrong answer, and
    ``scoring/gates.py`` measures its blind policies the same way. The ranking takes the magnitude
    separately, since the task ranks by effect size.
    """
    if verdict is None or verdict.effect is None:
        return 0.0
    return verdict.effect


def _effect_error(answer: Answer, key: InstanceKey, driver: str, scale: float) -> float:
    """Normalized estimation error for ``driver`` on ``[0, 1]``; defined for every driver, no exemptions.

    The per-driver term of ``baselines.py::normalized_effect_error``, which is what the causal-load gate
    certifies no blind policy can beat - one definition, so what is certified and what is scored cannot
    drift.
    """
    verdict = _verdict_for(answer, key, driver)
    error = abs(_estimate(verdict) - key.scm.total_effect_on_outcome(driver)) / scale
    return min(error, 1.0)


def _adjustment_bias(answer: Answer, key: InstanceKey, driver: str, scale: float) -> float:
    """How wrong the agent's *choice of adjustment set* is, on ``[0, 1]``; defined for every driver.

    Every adjustment set implies an exact asymptotic estimand, so the natural measure of a wrong set is the
    bias that set induces, in the same normalized units the estimate is graded in. See the module docstring
    for the four qualitatively different mistakes a validity boolean collapsed to an identical 0.
    """
    verdict = _verdict_for(answer, key, driver)
    if verdict is None:
        return 1.0
    mapped = [key.label_to_node.get(z) for z in verdict.adjustment_set]
    if any(node is None for node in mapped):
        return 1.0  # references a column that does not exist
    nodes = [node for node in mapped if node is not None]
    # Conditioning on the treatment or the outcome is not an estimand at all, so there is no bias to read
    # off - a malformed request rather than a mis-chosen set, and it takes the maximum penalty.
    if driver in nodes or key.scm.scaffold.outcome in nodes:
        return 1.0
    try:
        induced = population_coefficient(key.scm, driver, nodes)
    except np.linalg.LinAlgError:
        return 1.0  # degenerate regressor set; no estimand to compare against
    return min(abs(induced - key.scm.total_effect_on_outcome(driver)) / scale, 1.0)


def _ranking_tau(answer: Answer, key: InstanceKey) -> float:
    """Pairwise ranking agreement over all candidates, without arbitrary tie breaking."""
    drivers = key.scm.scaffold.drivers
    verdicts = {d: _verdict_for(answer, key, d) for d in drivers}
    if any(verdict is None for verdict in verdicts.values()):
        return 0.0  # a candidate with no verdict has no estimate, so no position
    # Preserve submitted ties: guessing the same effect for every driver must not earn an arbitrary
    # ranking determined by hidden node order. True ties (e.g. two zero-effect decoys) are not ranked.
    true = [abs(key.scm.total_effect_on_outcome(d)) for d in drivers]
    estimated = [abs(_estimate(verdicts[d])) for d in drivers]
    comparable = agreement = 0
    for i in range(len(drivers)):
        for j in range(i + 1, len(drivers)):
            if true[i] == true[j]:
                continue
            comparable += 1
            product = (true[i] - true[j]) * (estimated[i] - estimated[j])
            agreement += int(product > 0) - int(product < 0)
    return agreement / comparable if comparable else 0.0


def score_answer(answer: Answer, key: InstanceKey) -> AnswerScore:
    """Score a parsed ``answer`` against the true SCM in ``key``."""
    drivers = key.scm.scaffold.drivers
    scale = effect_scale(key.scm)
    biases = {driver: _adjustment_bias(answer, key, driver, scale) for driver in drivers}
    errors = {driver: _effect_error(answer, key, driver, scale) for driver in drivers}
    identification = 1.0 - sum(biases.values()) / len(biases)
    estimation = 1.0 - sum(errors.values()) / len(errors)
    effect_mae = sum(
        abs(_estimate(_verdict_for(answer, key, driver)) - key.scm.total_effect_on_outcome(driver)) / len(drivers)
        for driver in drivers
    )
    judgments = {
        driver: (v.identifiable_from_data if (v := _verdict_for(answer, key, driver)) is not None else None)
        for driver in drivers
    }
    return AnswerScore(
        parse_ok=True,
        overall=estimation,
        identification=identification,
        estimation=estimation,
        effect_mae=effect_mae,
        ranking_tau=_ranking_tau(answer, key),
        adjustment_bias=biases,
        effect_error=errors,
        identifiability_judgment=judgments,
        identifiable=_identifiable_drivers(key),
        alignment=key.provenance.alignment,
    )


def score_output(text: str, key: InstanceKey) -> AnswerScore:
    """Parse ``text`` and score it; an unparseable answer scores zero rather than raising."""
    try:
        candidate_labels = tuple(key.node_to_label[driver] for driver in key.scm.scaffold.drivers)
        answer = parse_answer_for_instance(
            text,
            candidate_labels,
            columns=tuple(key.node_to_label.values()),
            outcome=key.node_to_label[key.scm.scaffold.outcome],
        )
    except AnswerParseError:
        drivers = key.scm.scaffold.drivers
        return AnswerScore(
            parse_ok=False,
            overall=0.0,
            identification=0.0,
            estimation=0.0,
            effect_mae=None,
            ranking_tau=0.0,
            # No set and no estimate for any candidate: maximum induced bias and maximum error, the same
            # meaning an explicit "no effect, adjust for nothing" would carry.
            adjustment_bias=dict.fromkeys(drivers, 1.0),
            effect_error=dict.fromkeys(drivers, 1.0),
            identifiability_judgment=dict.fromkeys(drivers, None),
            identifiable=_identifiable_drivers(key),  # a property of the instance, not of the answer
            alignment=key.provenance.alignment,
        )
    return score_answer(answer, key)

"""Score an agent's answer against the true SCM.

The task: for **every** candidate driver, give an adjustment set, an estimate of its total causal effect,
and a judgement of whether the data alone settles that effect. There is no opt-out, and that is the design's
load-bearing property - every metric below has a denominator fixed by the instance, never chosen by the
agent. The previous version let the agent abstain, which excluded drivers from three metrics at once and
made the denominators move with the answer; a blanket-abstainer was scored on 4.2 drivers against 7.0 for a
model that always answered, and one cell scored a perfect 1.000 for answering nothing.

All metrics are computed on the hidden SCM via ``key.label_to_node``, so scoring is invariant to how the
columns were named - which is what lets naming be an experimental condition rather than part of the key.

* ``identification`` - ``1 - `` mean induced adjustment bias. Every proposed adjustment set implies an exact
  asymptotic estimand (``baselines.py::population_coefficient``, read off the SCM's covariance with no
  sampling), so the natural cost of a wrong set is the bias it induces, in the same normalized units the
  estimate is graded in. Continuous, not a validity boolean: measured on one driver, "adjusted for one of
  two confounders" (bias 0.972), "adjusted for neither" (0.833) and "gold set plus a mediator" (0.292) all
  scored an identical 0 under the boolean, which also marked a harmless extra collider as a failure.
* ``estimation`` - ``1 - `` mean normalized estimation error, ``|estimate - true| / scale`` clipped at 1.
  The component no shortcut reaches: blind policies rank drivers at tau 0.68-0.78 but estimate them 21-34%
  of a typical effect away. This is what carries the naming result (see ``agent/eval.py`` Gain/Harm).
* ``overall`` - the mean of those two. One number to rank models; not a product, so failing one step costs
  half rather than everything.
* ``identifiability_judgment`` - what the agent said about "does the data alone settle this?", stored raw
  (``None`` = it did not say) so ``agent/eval.py`` can pool precision/recall/F1 across instances. F1 rather
  than accuracy, following AbstentionBench: accuracy rewards a blanket answer, and ours had a 0.605 floor.
  Reported as a **diagnostic, never in** ``overall`` - gold identifiability *is* the population CPDAG
  (``scm/identifiability.py`` runs Meek 1-3 on the true graph), so a perfect discovery tool reads it
  straight off. A mechanical PC policy scores F1 0.650 on the built set; that baseline belongs beside this
  number so it is not mistaken for reasoning.
* ``ranking_tau`` - Kendall agreement with the true ``|effect|`` order, over **all** candidates. Kept for one
  reason: it is invariant to any monotone rescaling of the estimates, so a model that understands the
  structure but reports its numbers on the wrong scale shows up as high tau with low ``estimation`` rather
  than as a causal failure. Secondary - too coarse at 5 candidates to carry a headline.

``identifiable`` and ``alignment`` are the two stratification axes (``agent/eval.py::driver_cells``): naming
is expected to move ``estimation`` only where the data cannot settle the effect, and the identifiable half
doubles as a placebo - a nuisance effect would move both.

A parse failure scores zero rather than raising. New metrics are added by extending ``AnswerScore`` and
``score_answer``; nothing else needs to change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from causalagentbench.scm.identifiability import unidentifiable_drivers
from causalagentbench.scoring.baselines import effect_scale, population_coefficient
from causalagentbench.scoring.metrics import kendall_tau
from causalagentbench.task.answer import Answer, AnswerParseError, DriverVerdict, parse_answer_for_instance
from causalagentbench.task.instance import InstanceKey


@dataclass(frozen=True)
class AnswerScore:
    """The score of one answer. Metrics are on [0, 1] except ``ranking_tau`` which is on [-1, 1]."""

    parse_ok: bool
    overall: float  # mean of identification and estimation; the leaderboard number
    identification: float  # 1 - mean induced adjustment bias over every driver
    estimation: float  # 1 - mean normalized effect error over every driver
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
    """Kendall agreement of the agent's ranking with the true order, over every candidate.

    Derived from the estimates rather than stated separately (``task/answer.py::Answer``), so ranking credit
    is only reachable through them - a separately stated order could be produced by ranking on raw
    correlation, the measured tau=+0.627 shortcut, independently of the numbers.

    Nothing is excluded, so ``n_compared`` is the candidate count on every instance and in every naming
    condition. It used to drop correctly-abstained drivers, which made the population differ systematically
    by condition (measured: 5.00 candidates ranked at alpha=+1 against 3.38 at alpha=-1, the only contrast
    in that run whose CI excluded zero) and left every cross-condition ranking comparison confounded.

    Compares actual ``|true effect|`` magnitudes, not rank positions - two decoys always share exactly the
    same true effect (0), and only the raw magnitude preserves that tie for ``kendall_tau``'s tie handling
    to act on. Rank positions collapse to distinct integers even when the effects are exactly equal, turning
    a genuine tie into an arbitrary, gradeable coin flip.
    """
    drivers = key.scm.scaffold.drivers
    verdicts = {d: _verdict_for(answer, key, d) for d in drivers}
    if any(verdict is None for verdict in verdicts.values()):
        return 0.0  # a candidate with no verdict has no estimate, so no position
    # `sorted` is stable, so drivers the agent estimated equally keep canonical scaffold order rather than
    # an arbitrary one; true-side ties are counted for neither side regardless.
    agent_nodes = sorted(drivers, key=lambda d: -abs(_estimate(verdicts[d])))
    # Negated so "smaller = ranked first" matches agent_position's convention (index 0 = strongest);
    # negation preserves exact ties, which is the property being protected.
    true_magnitude = {node: -abs(key.scm.total_effect_on_outcome(node)) for node in drivers}
    agent_position = {node: i for i, node in enumerate(agent_nodes)}
    return kendall_tau([true_magnitude[d] for d in drivers], [agent_position[d] for d in drivers])


def score_answer(answer: Answer, key: InstanceKey) -> AnswerScore:
    """Score a parsed ``answer`` against the true SCM in ``key``."""
    drivers = key.scm.scaffold.drivers
    scale = effect_scale(key.scm)
    biases = {driver: _adjustment_bias(answer, key, driver, scale) for driver in drivers}
    errors = {driver: _effect_error(answer, key, driver, scale) for driver in drivers}
    identification = 1.0 - sum(biases.values()) / len(biases)
    estimation = 1.0 - sum(errors.values()) / len(errors)
    judgments = {
        driver: (v.identifiable_from_data if (v := _verdict_for(answer, key, driver)) is not None else None)
        for driver in drivers
    }
    return AnswerScore(
        parse_ok=True,
        overall=(identification + estimation) / 2.0,
        identification=identification,
        estimation=estimation,
        ranking_tau=_ranking_tau(answer, key),
        adjustment_bias=biases,
        effect_error=errors,
        identifiability_judgment=judgments,
        identifiable=_identifiable_drivers(key),
        alignment=key.provenance.alignment,
    )


def reference_scores(keys: Sequence[InstanceKey]) -> dict[str, float]:
    """``overall`` for fixed reference policies, on the same instances.

    A headline is uninterpretable alone, and these move with the built set - they depend on the realized
    identifiable/unidentifiable mix - so they are recomputed per set rather than quoted.

    ``knows the orientation`` is the ceiling: it is handed the true structure including the edges the data
    cannot orient, so it scores 1.0 by construction.

    ``data-only`` is the number that matters - **the ceiling for any agent that ignores column names**, and
    therefore the bar the naming axis has to clear. On a driver the data settles it adjusts correctly. On a
    driver it does not, the ambiguity is a single unoriented edge with exactly two admissible readings
    (adjust for the ambiguous parent, or do not), so the best a name-blind agent can do is pick one: this
    scores the **average of both branches**, which is that guess's expected value.

    Averaging matters and is not a detail. An earlier version always took the give-up branch and scored
    0.605 on the built set against 0.803 for the coin flip - which would have credited the naming axis with
    0.395 of headroom when the honest figure is 0.197, overstating the entire claim by a factor of two.
    ``data-only (gives up)`` is kept as the pessimistic branch so the two stay visible side by side.
    """
    totals: dict[str, list[float]] = {}
    for key in keys:
        scaffold = key.scm.scaffold
        label = key.node_to_label
        unidentifiable = set(unidentifiable_drivers(scaffold))

        def gold(driver: str, *, _key: InstanceKey = key, _label: Mapping[str, str] = label) -> DriverVerdict:
            adjust = _key.scm.scaffold.correct_adjustment.get(driver, ())
            return DriverVerdict(
                driver=_label[driver],
                adjustment_set=tuple(_label[z] for z in adjust),
                effect=_key.scm.total_effect_on_outcome(driver),
            )

        def naive(driver: str, *, _key: InstanceKey = key, _label: Mapping[str, str] = label) -> DriverVerdict:
            return DriverVerdict(
                driver=_label[driver], adjustment_set=(), effect=population_coefficient(_key.scm, driver)
            )

        def score(verdicts: list[DriverVerdict], *, _key: InstanceKey = key) -> float:
            return score_answer(Answer(verdicts=tuple(verdicts)), _key).overall

        # The two admissible readings of every unoriented edge; a name-blind agent can only guess between
        # them, so its expected score is their mean.
        guessed_right = score([gold(d) for d in scaffold.drivers])
        gave_up = score([naive(d) if d in unidentifiable else gold(d) for d in scaffold.drivers])
        for name, value in (
            ("knows the orientation", guessed_right),
            ("data-only", (guessed_right + gave_up) / 2.0),
            ("data-only (gives up)", gave_up),
            ("correlation only", score([naive(d) for d in scaffold.drivers])),
        ):
            totals.setdefault(name, []).append(value)
    return {name: sum(values) / len(values) for name, values in totals.items() if values}


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

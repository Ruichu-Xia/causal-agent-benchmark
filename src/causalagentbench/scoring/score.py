"""Score an agent's answer against the true SCM.

The task is: estimate each candidate driver's total causal effect, give a valid adjustment set for each,
and abstain instead of committing where the effect is genuinely not identified. All metrics are computed
on the hidden SCM via ``key.label_to_node``, so the score is invariant to how the columns were named - the
whole point of the naming axis.

* ``mean_driver_score`` - **the headline**. Every candidate has exactly one correct deliverable, set by its
  ``quadrant``, and this asks whether the agent produced it. Scored on *every* driver, so unlike the three
  metrics below the denominator is never the agent's to choose. See ``_driver_score``. Interpret it against
  ``reference_scores``, which recomputes the reference points on the same set - in particular the
  data-only oracle, the ceiling for any agent that ignores column names.
* ``effect_accuracy`` - estimation error alone, and the component no shortcut reaches. Measured on the
  generator, blind policies (marginal correlation, adjust-for-everything) rank the drivers at tau 0.68-0.78
  but estimate them 21-34% of a typical effect away, while correct adjustment recovers the truth exactly.
  So an order is cheap and a number is not; ``scoring/gates.py::SHORTCUT_ERROR_THRESHOLD`` certifies that
  gap per instance, and this measures the same quantity the same way.
* ``ranking_tau`` - Kendall rank agreement between the agent's order and the true ``|effect|`` order. That
  order is *derived* from the estimates (``task/answer.py::Answer``) rather than stated separately, so it
  cannot be earned by ranking on raw correlation independently of them. Drivers correctly abstained on are
  excluded from the comparison entirely - ``abstention_accuracy`` alone carries credit for those.
* ``n_compared`` / ``ranking_tau_z`` - task-size context for ``ranking_tau`` and its standardized form.
  Raw tau isn't comparable across instances with different candidate counts (fewer candidates means a
  coarser achievable tau and higher variance under pure chance); ``n_compared`` is the actual candidate
  count behind ``ranking_tau`` (not ``n_drivers`` - decoys are candidates too, and correctly-abstained
  drivers are excluded), and ``ranking_tau_z`` standardizes ``ranking_tau`` against chance at that size
  (``scoring/metrics.py::kendall_tau_zscore``) so scores across differently-sized tasks are comparable.
* ``adjustment_accuracy`` - fraction of *graded* drivers whose proposed set is a valid backdoor adjustment
  set, judged by the backdoor criterion against the true graph. Only *correctly* abstained drivers are
  ungraded - a wrong abstention scores as invalid, the same exclusion rule ``ranking_tau`` uses. Otherwise
  the denominator would be the agent's to choose, and abstaining would raise this score for free.
* ``abstention_accuracy`` - fraction of drivers where the agent's abstain/commit choice matches the gold
  rule (``_gold_abstain``): abstaining is correct iff the data cannot orient the driver's confounder edge
  *and* its own displayed name isn't the aligned one. Read it per cell, or via
  ``agent/eval.py::abstention_rates``: as one pooled number it has a floor set by class balance, since a
  model that never abstains scores 0.605 without deciding anything.

``Quadrant``/``quadrant()`` classify a driver into one of §6's four naming-vs-identifiability cells, and
those cells are the capability under test rather than a reporting convenience: trust data where data
decides, fall back on a name only where data cannot and the name has proven consistent with the structure
data *did* pin down, abstain when neither settles it. ``_gold_abstain`` is the last of those four cells -
single source of truth, not two independently maintained checks of the same fact.

Three per-driver maps (``adjustment_valid``, ``abstention_correct``, ``ranking_pairs``) plus
``identifiable`` exist so results can be stratified *below* instance grain - an instance-level mean hides
whether a failure fell on a driver whose effect the data can actually resolve. ``identifiable`` and
``alignment`` are the two stratification axes (``agent/eval.py::aggregate_by_driver_cell``); the naming
axis is deliberately carried by ``alignment`` rather than a stored ``Quadrant``, because
``construct_labeling`` applies a level uniformly to every non-outcome node, making a driver's "aligned"
bit exactly ``alignment > 0`` (``procedural_scaffold_plan.md`` §11.8).

``alignment`` carries the naming-alignment level for stratified analysis. A parse failure or a malformed
ranking scores zero rather than raising. New metrics are added by extending ``AnswerScore`` and
``score_answer``; nothing else needs to change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import networkx as nx

from causalagentbench.scm.identifiability import unidentifiable_drivers
from causalagentbench.scoring.baselines import effect_scale, population_coefficient
from causalagentbench.scoring.metrics import kendall_tau, kendall_tau_zscore
from causalagentbench.task.answer import Answer, AnswerParseError, DriverVerdict, parse_answer_for_instance
from causalagentbench.task.instance import InstanceKey
from causalagentbench.types import Scaffold


class RankingPairs(NamedTuple):
    """Concordant/discordant pair counts involving one driver (ties excluded, as in ``kendall_tau``).

    Pooling these counts over every driver of an answer recovers that answer's ``ranking_tau`` exactly:
    each pair has two endpoints, so ``sum(concordant) == 2C`` and ``sum(discordant) == 2D``, and
    ``(2C - 2D) / (2C + 2D)`` is ``kendall_tau``'s own ``(C - D) / (C + D)``. That makes a per-driver
    ranking score a true *decomposition* of the reported tau rather than a second, differently-defined
    metric - and it keeps cross-cell pairs (an identifiable driver compared against an unidentifiable one)
    contributing to both drivers, which restricting the ranking to one cell's drivers would delete.

    Counts, not a ratio: the unweighted mean of per-driver ``(c - d) / (c + d)`` does *not* recover tau
    once drivers differ in how many tied pairs they have, so only the raw counts pool correctly.
    """

    concordant: int
    discordant: int


@dataclass(frozen=True)
class AnswerScore:
    """The score of one answer. Metrics are on [0, 1] except ``ranking_tau`` which is on [-1, 1]."""

    parse_ok: bool
    mean_driver_score: float  # THE headline; mean of driver_score over every candidate, nothing ungraded
    driver_score: Mapping[str, float]  # per driver on [0, 1]; see _driver_score
    gold_abstain: Mapping[str, bool]  # what the quadrant rule required, for sensitivity/specificity
    ranking_tau: float  # Kendall tau vs the true order; 1.0 perfect, 0.0 malformed/unparsed
    n_compared: int  # candidates actually fed into ranking_tau (drivers minus correctly-abstained ones)
    ranking_tau_z: float | None  # ranking_tau standardized against chance at this n_compared; None if < 2
    ranking_pairs: Mapping[str, RankingPairs]  # per-driver pair counts; pooling them reproduces ranking_tau
    effect_accuracy: float  # 1 - mean normalized estimation error over graded drivers; the causal-load metric
    effect_error: Mapping[str, float | None]  # per-driver error on [0, 1]; None only if abstaining was correct
    adjustment_accuracy: float  # fraction of graded (non-abstained) drivers with a valid backdoor set
    adjustment_valid: Mapping[str, bool | None]  # per-driver validity; None only if abstaining was correct
    abstention_accuracy: float  # fraction of drivers whose abstain/commit choice matches the gold rule
    abstention_correct: Mapping[str, bool]  # per-driver abstain/commit match; not derivable from the above
    identifiable: Mapping[str, bool]  # per-driver: effect identified from data alone, for stratification
    alignment: float | None  # naming-alignment level, for stratification


def _valid_backdoor(scaffold: Scaffold, driver: str, adjust: set[str]) -> bool:
    """True if ``adjust`` is a valid backdoor adjustment set for ``driver -> outcome`` in the true graph."""
    graph = scaffold.digraph()
    outcome = scaffold.outcome
    if driver in adjust or outcome in adjust:
        return False
    if adjust & (nx.descendants(graph, driver) | {driver}):  # no descendant of the treatment (e.g. a mediator)
        return False
    backdoor = graph.copy()
    backdoor.remove_edges_from(list(graph.out_edges(driver)))
    return bool(nx.is_d_separator(backdoor, {driver}, {outcome}, adjust))


class Quadrant(Enum):
    """Which of §6's four cells a driver falls into: data-identifiability x naming alignment."""

    IDENTIFIABLE_ALIGNED = "identifiable_aligned"  # trivial - both signals agree
    IDENTIFIABLE_NOT_ALIGNED = "identifiable_not_aligned"  # trust data
    UNIDENTIFIABLE_ALIGNED = "unidentifiable_aligned"  # trust naming
    UNIDENTIFIABLE_NOT_ALIGNED = "unidentifiable_not_aligned"  # abstain is correct


def quadrant(key: InstanceKey, driver: str) -> Quadrant:
    """Classify ``driver`` into one of the four naming-vs-identifiability cells (§6's table).

    Identifiable iff not in ``unidentifiable_drivers`` (topology alone, discovered not declared - §4/§11).
    Aligned iff ``driver``'s own displayed name is the "aligned" kind (``node_kind``, §6/§7); neutral and
    misaligned both collapse into "not aligned" for v1.

    These four cells are the capability being tested, not a reporting convenience. Each demands a different
    decision: trust the data where data decides (``IDENTIFIABLE_*``, whatever the name says), fall back on
    the name only where data cannot decide *and* the name has proven trustworthy
    (``UNIDENTIFIABLE_ALIGNED``), and abstain when neither source settles it
    (``UNIDENTIFIABLE_NOT_ALIGNED``). ``_gold_abstain`` is that last cell - single source of truth, not two
    independently maintained checks of the same fact.
    """
    identifiable = driver not in unidentifiable_drivers(key.scm.scaffold)
    aligned = key.provenance.node_kind.get(driver) == "aligned"
    if identifiable:
        return Quadrant.IDENTIFIABLE_ALIGNED if aligned else Quadrant.IDENTIFIABLE_NOT_ALIGNED
    return Quadrant.UNIDENTIFIABLE_ALIGNED if aligned else Quadrant.UNIDENTIFIABLE_NOT_ALIGNED


def _gold_abstain(key: InstanceKey, driver: str) -> bool:
    """True iff abstaining on ``driver`` is the correct call - `quadrant`'s "abstain is correct" cell.

    Abstaining is right only when *both* sources fail: the data cannot orient the driver's confounder edge,
    and its own displayed name is not the aligned one. Where the name *is* aligned, it settles the
    orientation the data left open, so committing is the correct call and abstaining is over-caution.

    This is answerable from evidence, which is the property it has to have. Naming is applied uniformly to
    every node, and ~29 of an instance's ~35 edges are orientable from data - so an aligned skin is
    consistent with the structure the agent can recover, and a misaligned one contradicts it. The agent
    establishes which world it is in by checking names against what discovery pins down, then relies on them
    only for the ~6 edges data cannot settle. ``task/instance.py::build_prompt`` states that evidence model
    explicitly; without it the task said "abstain when the data cannot resolve it" while the gold expected
    the name to be used, and a model that abstained was following the instructions and being marked wrong.
    """
    return quadrant(key, driver) == Quadrant.UNIDENTIFIABLE_NOT_ALIGNED


def _identifiable_drivers(key: InstanceKey) -> Mapping[str, bool]:
    """Every candidate driver, flagged with whether its effect is identified from data alone.

    The topology half of what ``quadrant`` reads, recorded per driver so results can be stratified by it.
    Resolves ``unidentifiable_drivers`` once rather than per driver as ``quadrant`` does - that call rebuilds
    the whole CPDAG each time. Answer-independent, so it is filled the same way on a parse failure.
    """
    unidentifiable = set(unidentifiable_drivers(key.scm.scaffold))
    return {driver: driver not in unidentifiable for driver in key.scm.scaffold.drivers}


def _verdict_for(answer: Answer, key: InstanceKey, driver: str) -> DriverVerdict | None:
    """The agent's verdict for ``driver`` (by its display label), or ``None`` if it gave none."""
    label = key.node_to_label[driver]
    return next((v for v in answer.verdicts if v.driver == label), None)


def _estimate(verdict: DriverVerdict | None) -> float:
    """``verdict``'s signed effect estimate, treating "no number given" as an estimate of zero.

    An abstention or a missing verdict is a refusal to commit to a magnitude, and zero is what that refusal
    asserts: no detectable effect. Scoring it as such keeps both the ranking and the error metric defined for
    every driver without inventing a value.

    Signed, because the error metric must be: reporting +0.3 where the truth is -0.3 is a wrong answer, not a
    right one, and ``scoring/gates.py`` measures its blind policies the same way. The ranking takes the
    magnitude of this separately, since the task ranks by effect size.
    """
    if verdict is None or verdict.effect is None:
        return 0.0
    return verdict.effect


def _effect_error(answer: Answer, key: InstanceKey, driver: str, scale: float) -> float | None:
    """Normalized estimation error for ``driver``; ``None`` only when abstaining was the *correct* call.

    On ``[0, 1]``: ``|estimate - true| / scale``, clipped at 1 - the per-driver term of
    ``scoring/baselines.py::normalized_effect_error``, which is what the causal-load gate certifies no blind
    policy can beat. One definition, so what is certified and what is scored cannot drift. Ungraded is
    reserved for a correct abstention, the same exclusion
    rule ``_adjustment_valid`` and ``_ranking_tau`` use: where the effect is not point-identified there is no
    true value to have hit, but a *wrong* abstention is still a failure to estimate, not an exemption.
    """
    verdict = _verdict_for(answer, key, driver)
    if verdict is not None and verdict.abstain and _gold_abstain(key, driver):
        return None
    error = abs(_estimate(verdict) - key.scm.total_effect_on_outcome(driver)) / scale
    return min(error, 1.0)


def _ranking_tau(answer: Answer, key: InstanceKey) -> tuple[float, int, Mapping[str, RankingPairs]]:
    """Kendall agreement of the agent's ranking with the true order; 0.0 unless it is a clean permutation.

    Returns ``(tau, n_compared, pairs)`` - ``n_compared`` is the real denominator behind ``tau``, needed to
    standardize it (``kendall_tau_zscore``): raw tau isn't comparable across instances of different
    ``n_compared``, since both its achievable granularity and its variance under chance depend on it.

    ``pairs`` attributes each comparison to both of its endpoints (``RankingPairs``), so the ranking metric
    can be stratified per driver. It is counted here, off the same magnitudes and positions ``kendall_tau``
    is handed, so the concordance convention cannot drift between the two; ``tau`` itself is still taken
    from ``kendall_tau`` rather than derived from these counts, leaving the two independently computed for
    the test that pins their agreement. Drivers with no comparison - correctly abstained, or any driver at
    all when the ranking is malformed - get ``RankingPairs(0, 0)``, which contributes to neither side of a
    pooled ratio.

    Drivers correctly abstained on are dropped from the pairwise comparison before scoring - ``kendall_tau``
    already returns 1.0 (vacuously) if that leaves fewer than two drivers to compare.

    Compares actual ``|true effect|`` magnitudes, not rank positions - two decoys always share exactly the
    same true effect (0), and only the raw magnitude preserves that tie for ``kendall_tau``'s own tie
    handling (a tied pair counts as neither concordant nor discordant) to act on. Rank positions collapse
    to always-distinct integers even when the underlying effects are exactly equal, which silently turns a
    genuine tie - no correct order exists - into an arbitrary, gradeable coin flip.
    """
    drivers = key.scm.scaffold.drivers
    empty = dict.fromkeys(drivers, RankingPairs(0, 0))
    verdicts = {d: _verdict_for(answer, key, d) for d in drivers}
    if any(verdict is None for verdict in verdicts.values()):
        return 0.0, len(drivers), empty  # a candidate with no verdict has no estimate, so no position
    # Derived from the estimates, not stated separately (`task/answer.py::Answer`): ranking credit is then
    # only reachable through the estimates. `sorted` is stable, so drivers the agent estimated equally keep
    # canonical scaffold order rather than an arbitrary one - deterministic, and true-side ties are counted
    # for neither side below regardless.
    agent_nodes = sorted(drivers, key=lambda d: -abs(_estimate(verdicts[d])))
    correctly_abstained = {
        d for d in drivers if (v := _verdict_for(answer, key, d)) is not None and v.abstain and _gold_abstain(key, d)
    }
    compared = [d for d in drivers if d not in correctly_abstained]
    # Negated so "smaller = ranked first" matches agent_position's convention (index 0 = the agent's pick
    # for strongest); negation preserves exact ties, which is the property being fixed here.
    true_magnitude = {node: -abs(key.scm.total_effect_on_outcome(node)) for node in drivers}
    agent_position = {node: i for i, node in enumerate(agent_nodes)}
    tau = kendall_tau([true_magnitude[d] for d in compared], [agent_position[d] for d in compared])

    concordant = dict.fromkeys(drivers, 0)
    discordant = dict.fromkeys(drivers, 0)
    for i, first in enumerate(compared):
        for second in compared[i + 1 :]:
            # Same product test kendall_tau applies, so a tied pair (product 0) is counted for neither.
            product = (true_magnitude[first] - true_magnitude[second]) * (
                agent_position[first] - agent_position[second]
            )
            side = concordant if product > 0 else discordant if product < 0 else None
            if side is not None:
                side[first] += 1
                side[second] += 1
    pairs = {d: RankingPairs(concordant[d], discordant[d]) for d in drivers}
    return tau, len(compared), pairs


def _adjustment_valid(answer: Answer, key: InstanceKey, driver: str) -> bool | None:
    """Valid backdoor set for ``driver``; ``None`` only when abstaining was the *correct* call.

    Ungraded is reserved for a *correct* abstention, matching ``_ranking_tau``'s exclusion rule exactly - a
    wrong abstention is a failure to supply a valid set, not an exemption from being asked. Excluding every
    abstention (the previous behavior) made abstaining free on this metric and let the denominator be chosen
    by the agent: measured live, one model abstained on 22 of 48 identifiable drivers - gold commits on all
    of them - and scored 1.000 in a cell where it abstained on all 12, i.e. a perfect score for answering
    nothing, while the one cell it answered in full scored 0.583. Two models that looked far apart were
    identical (0.583) wherever both actually answered.
    """
    verdict = _verdict_for(answer, key, driver)
    if verdict is None:
        return False
    if verdict.abstain:
        return None if _gold_abstain(key, driver) else False
    mapped = [key.label_to_node.get(z) for z in verdict.adjustment_set]
    if any(node is None for node in mapped):
        return False  # references a column that does not exist
    return _valid_backdoor(key.scm.scaffold, driver, {node for node in mapped if node is not None})


def _abstention_correct(answer: Answer, key: InstanceKey, driver: str) -> bool:
    """True iff the agent's abstain/commit choice for ``driver`` matches ``_gold_abstain``.

    A missing verdict is never correct (mirrors ``_adjustment_valid``'s treatment of an absent verdict as a
    failure, not an implicit, credit-worthy "did not abstain").
    """
    verdict = _verdict_for(answer, key, driver)
    if verdict is None:
        return False
    return verdict.abstain == _gold_abstain(key, driver)


def _driver_score(answer: Answer, key: InstanceKey, driver: str, scale: float) -> float:
    """The one number per driver: did the agent produce the correct deliverable for it?

    Every candidate has exactly one correct action, set by its ``quadrant``, so every candidate is scored -
    there is no ungraded case and the denominator is never the agent's to choose. That is the defect this
    replaces: ``adjustment_accuracy`` and ``effect_accuracy`` both exclude correctly-abstained drivers, so a
    model that abstains more is scored on a smaller population (measured: 4.2 drivers graded for a
    blanket-abstainer against 7.0 for one that always commits).

    Gold abstain -> 1.0 for abstaining, else 0.0. Committing scores zero even when the number happens to be
    right: the effect was not identified, so being close was luck rather than calibration.

    Gold commit -> 0.0 for abstaining (the failure this benchmark exists to catch: declining a question the
    evidence did answer), else the mean of adjustment validity and estimate accuracy. Equal weights because
    that is the only split with no free parameter to justify; the mean rather than a product so a right
    number from a wrong set scores 0.5 instead of 0, which is both more informative and a denser signal.
    """
    verdict = _verdict_for(answer, key, driver)
    abstained = verdict is not None and verdict.abstain
    if _gold_abstain(key, driver):
        return 1.0 if abstained else 0.0
    if abstained:
        return 0.0
    valid = _adjustment_valid(answer, key, driver)
    error = _effect_error(answer, key, driver, scale)
    # Neither is None here: both return None only for a *correct* abstention, handled above.
    assert valid is not None and error is not None
    return (float(valid) + (1.0 - error)) / 2.0


def score_answer(answer: Answer, key: InstanceKey) -> AnswerScore:
    """Score a parsed ``answer`` against the true SCM in ``key``."""
    drivers = key.scm.scaffold.drivers
    scale = effect_scale(key.scm)
    errors = {driver: _effect_error(answer, key, driver, scale) for driver in drivers}
    graded_errors = [e for e in errors.values() if e is not None]
    valid = {driver: _adjustment_valid(answer, key, driver) for driver in drivers}
    graded = [v for v in valid.values() if v is not None]
    # Every driver abstained on: nothing to grade, not a failure - vacuously accurate, same convention
    # kendall_tau already uses for "fewer than two items to compare" (scoring/metrics.py).
    adjustment_accuracy = sum(graded) / len(graded) if graded else 1.0
    abstention_correct = {driver: _abstention_correct(answer, key, driver) for driver in drivers}
    ranking_tau, n_compared, ranking_pairs = _ranking_tau(answer, key)
    scores = {driver: _driver_score(answer, key, driver, scale) for driver in drivers}
    return AnswerScore(
        parse_ok=True,
        mean_driver_score=sum(scores.values()) / len(scores),
        driver_score=scores,
        gold_abstain={driver: _gold_abstain(key, driver) for driver in drivers},
        ranking_tau=ranking_tau,
        n_compared=n_compared,
        ranking_tau_z=kendall_tau_zscore(ranking_tau, n_compared),
        ranking_pairs=ranking_pairs,
        # Same vacuous-when-nothing-graded convention as `adjustment_accuracy` above.
        effect_accuracy=1.0 - sum(graded_errors) / len(graded_errors) if graded_errors else 1.0,
        effect_error=errors,
        adjustment_accuracy=adjustment_accuracy,
        adjustment_valid=valid,
        abstention_accuracy=sum(abstention_correct.values()) / len(abstention_correct),
        abstention_correct=abstention_correct,
        identifiable=_identifiable_drivers(key),
        alignment=key.provenance.alignment,
    )


def reference_scores(keys: Sequence[InstanceKey]) -> dict[str, float]:
    """Headline ``mean_driver_score`` for fixed reference strategies, on the same instances.

    A headline is uninterpretable alone, and these reference points *move with the built set* - they depend
    on the realized identifiable/unidentifiable mix - so they have to be recomputed per set rather than
    quoted from anywhere.

    The two that carry the benchmark's claim are the *fixed-policy* oracles. Both know the structure
    perfectly; they differ only in what they do where the data runs out. ``never trust names`` abstains
    there, ``always trust names`` commits using the name. Measured over the paired conditions they score
    **identically**, and the reason is the design rather than a coincidence::

        policy                alpha=+1   alpha=-1   paired
        never trust names       0.500      1.000     0.750
        always trust names      1.000      0.500     0.750

    Each is perfect in the condition its fixed assumption happens to match and loses every unidentifiable
    driver in the other. So **0.750 is the ceiling for any policy that does not decide, per instance, whether
    the names can be trusted** - which is the capability under test. Only an agent that checks the names
    against the structure the data does settle can exceed it, and the ~25 points between 0.750 and the full
    oracle is exactly what that calibration is worth.

    Reported per condition as well as paired: neither condition alone separates the two policies, since
    ``never trust names`` is optimal at alpha=-1 and ``always trust names`` is optimal at alpha=+1. Only the
    pair does.
    """
    totals: dict[str, list[float]] = {}
    for key in keys:
        scaffold = key.scm.scaffold
        label = key.node_to_label
        unidentifiable = set(unidentifiable_drivers(scaffold))
        commit = {
            d: DriverVerdict(
                driver=label[d],
                adjustment_set=tuple(label[z] for z in scaffold.correct_adjustment.get(d, ())),
                effect=key.scm.total_effect_on_outcome(d),
            )
            for d in scaffold.drivers
        }
        abstain = {d: DriverVerdict(driver=label[d], adjustment_set=(), abstain=True) for d in scaffold.drivers}
        blind = {
            d: DriverVerdict(driver=label[d], adjustment_set=(), effect=population_coefficient(key.scm, d))
            for d in scaffold.drivers
        }
        for name, verdicts in (
            # Calibrates: uses the name exactly where the data ran out and the name is trustworthy.
            ("oracle (calibrates)", [abstain[d] if _gold_abstain(key, d) else commit[d] for d in scaffold.drivers]),
            # The two fixed policies. Both know the structure; neither decides whether to believe the names.
            ("never trust names", [abstain[d] if d in unidentifiable else commit[d] for d in scaffold.drivers]),
            ("always trust names", [commit[d] for d in scaffold.drivers]),
            ("always abstain", [abstain[d] for d in scaffold.drivers]),
            ("correlation, never abstain", [blind[d] for d in scaffold.drivers]),
        ):
            totals.setdefault(name, []).append(score_answer(Answer(verdicts=tuple(verdicts)), key).mean_driver_score)
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
            # No deliverable at all for any candidate - zero on every one, no exemptions.
            mean_driver_score=0.0,
            driver_score=dict.fromkeys(drivers, 0.0),
            gold_abstain={driver: _gold_abstain(key, driver) for driver in drivers},
            ranking_tau=0.0,
            n_compared=len(drivers),
            ranking_tau_z=kendall_tau_zscore(0.0, len(drivers)),
            # No ranking was produced, so no pair was observed - not a sweep of discordant ones, which would
            # invent data. Consequence: parse failures are absent from pooled per-driver ranking tallies
            # while still dragging the instance-level mean tau toward 0; `parse_rate` is where to see them.
            ranking_pairs=dict.fromkeys(drivers, RankingPairs(0, 0)),
            # No estimate was produced for any driver, which the metric already has a meaning for: maximum
            # error, the same score an explicit "no effect" would get.
            effect_accuracy=0.0,
            effect_error=dict.fromkeys(drivers, 1.0),
            adjustment_accuracy=0.0,
            adjustment_valid=dict.fromkeys(drivers, False),
            abstention_accuracy=0.0,
            abstention_correct=dict.fromkeys(drivers, False),
            identifiable=_identifiable_drivers(key),  # a property of the instance, not of the answer
            alignment=key.provenance.alignment,
        )
    return score_answer(answer, key)

"""Score an agent's answer against the true SCM.

The task is: rank the candidate drivers by causal effect, give a valid adjustment set for each, and
abstain instead of committing where the driver's role is genuinely undetermined. All metrics are computed
on the hidden SCM via ``key.label_to_node``, so the score is invariant to how the columns were named - the
whole point of the naming axis.

* ``ranking_tau`` - Kendall rank agreement between the agent's order and the true ``|effect|`` order (the
  headline: a reasoner stays flat across naming alignment, a reciter degrades). Drivers correctly abstained
  on are excluded from the comparison entirely - ``abstention_accuracy`` alone carries credit for those.
* ``n_compared`` / ``ranking_tau_z`` - task-size context for ``ranking_tau`` and its standardized form.
  Raw tau isn't comparable across instances with different candidate counts (fewer candidates means a
  coarser achievable tau and higher variance under pure chance); ``n_compared`` is the actual candidate
  count behind ``ranking_tau`` (not ``n_drivers`` - decoys are candidates too, and correctly-abstained
  drivers are excluded), and ``ranking_tau_z`` standardizes ``ranking_tau`` against chance at that size
  (``scoring/metrics.py::kendall_tau_zscore``) so scores across differently-sized tasks are comparable.
* ``adjustment_accuracy`` - fraction of *graded* drivers (excludes any the agent abstained on) whose
  proposed set is a valid backdoor adjustment set, judged by the backdoor criterion against the true graph.
* ``abstention_accuracy`` - fraction of drivers where the agent's abstain/commit choice matches the gold
  rule (``_gold_abstain``): abstaining is correct iff the driver is genuinely unidentifiable from data
  alone *and* its own displayed name isn't the aligned one (the collapsed v1 rule, `neutral` and
  `misaligned` treated the same - see ``procedural_scaffold_plan.md`` §6).

``Quadrant``/``quadrant()`` classify a driver into one of §6's four naming-vs-identifiability cells;
``_gold_abstain`` is just its "abstain is correct" cell - single source of truth, not two independently
maintained checks of the same underlying fact.

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

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import networkx as nx

from causalagentbench.scm.identifiability import unidentifiable_drivers
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
    ranking_tau: float  # Kendall tau vs the true order; 1.0 perfect, 0.0 malformed/unparsed
    n_compared: int  # candidates actually fed into ranking_tau (drivers minus correctly-abstained ones)
    ranking_tau_z: float | None  # ranking_tau standardized against chance at this n_compared; None if < 2
    ranking_pairs: Mapping[str, RankingPairs]  # per-driver pair counts; pooling them reproduces ranking_tau
    adjustment_accuracy: float  # fraction of graded (non-abstained) drivers with a valid backdoor set
    adjustment_valid: Mapping[str, bool | None]  # per-driver validity; None if the agent abstained
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
    misaligned both collapse into "not aligned" for v1 (the collapsed rule already used for abstain scoring,
    procedural_scaffold_plan.md §6) - single source of truth for that collapse, not duplicated logic.
    """
    identifiable = driver not in unidentifiable_drivers(key.scm.scaffold)
    aligned = key.provenance.node_kind.get(driver) == "aligned"
    if identifiable:
        return Quadrant.IDENTIFIABLE_ALIGNED if aligned else Quadrant.IDENTIFIABLE_NOT_ALIGNED
    return Quadrant.UNIDENTIFIABLE_ALIGNED if aligned else Quadrant.UNIDENTIFIABLE_NOT_ALIGNED


def _gold_abstain(key: InstanceKey, driver: str) -> bool:
    """True iff abstaining on ``driver`` is the correct call - the "abstain is correct" cell of `quadrant`."""
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
    agent_nodes = [key.label_to_node.get(label) for label in answer.ranking]
    if len(agent_nodes) != len(drivers) or set(agent_nodes) != set(drivers):
        return 0.0, len(drivers), empty  # not a well-formed ranking of exactly the candidate drivers
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
    """Valid backdoor set for ``driver``, or ``None`` if the agent abstained (not graded on content)."""
    verdict = _verdict_for(answer, key, driver)
    if verdict is None:
        return False
    if verdict.abstain:
        return None
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


def score_answer(answer: Answer, key: InstanceKey) -> AnswerScore:
    """Score a parsed ``answer`` against the true SCM in ``key``."""
    drivers = key.scm.scaffold.drivers
    valid = {driver: _adjustment_valid(answer, key, driver) for driver in drivers}
    graded = [v for v in valid.values() if v is not None]
    # Every driver abstained on: nothing to grade, not a failure - vacuously accurate, same convention
    # kendall_tau already uses for "fewer than two items to compare" (scoring/metrics.py).
    adjustment_accuracy = sum(graded) / len(graded) if graded else 1.0
    abstention_correct = {driver: _abstention_correct(answer, key, driver) for driver in drivers}
    ranking_tau, n_compared, ranking_pairs = _ranking_tau(answer, key)
    return AnswerScore(
        parse_ok=True,
        ranking_tau=ranking_tau,
        n_compared=n_compared,
        ranking_tau_z=kendall_tau_zscore(ranking_tau, n_compared),
        ranking_pairs=ranking_pairs,
        adjustment_accuracy=adjustment_accuracy,
        adjustment_valid=valid,
        abstention_accuracy=sum(abstention_correct.values()) / len(abstention_correct),
        abstention_correct=abstention_correct,
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
            ranking_tau=0.0,
            n_compared=len(drivers),
            ranking_tau_z=kendall_tau_zscore(0.0, len(drivers)),
            # No ranking was produced, so no pair was observed - not a sweep of discordant ones, which would
            # invent data. Consequence: parse failures are absent from pooled per-driver ranking tallies
            # while still dragging the instance-level mean tau toward 0; `parse_rate` is where to see them.
            ranking_pairs=dict.fromkeys(drivers, RankingPairs(0, 0)),
            adjustment_accuracy=0.0,
            adjustment_valid=dict.fromkeys(drivers, False),
            abstention_accuracy=0.0,
            abstention_correct=dict.fromkeys(drivers, False),
            identifiable=_identifiable_drivers(key),  # a property of the instance, not of the answer
            alignment=key.provenance.alignment,
        )
    return score_answer(answer, key)

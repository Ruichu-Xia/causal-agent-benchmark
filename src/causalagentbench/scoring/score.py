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

``alignment`` carries the naming-alignment level for stratified analysis. A parse failure or a malformed
ranking scores zero rather than raising. New metrics are added by extending ``AnswerScore`` and
``score_answer``; nothing else needs to change.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import networkx as nx

from causalagentbench.scm.identifiability import unidentifiable_drivers
from causalagentbench.scoring.metrics import kendall_tau, kendall_tau_zscore
from causalagentbench.task.answer import Answer, AnswerParseError, DriverVerdict, parse_answer_for_instance
from causalagentbench.task.instance import InstanceKey
from causalagentbench.types import Scaffold


@dataclass(frozen=True)
class AnswerScore:
    """The score of one answer. Metrics are on [0, 1] except ``ranking_tau`` which is on [-1, 1]."""

    parse_ok: bool
    ranking_tau: float  # Kendall tau vs the true order; 1.0 perfect, 0.0 malformed/unparsed
    n_compared: int  # candidates actually fed into ranking_tau (drivers minus correctly-abstained ones)
    ranking_tau_z: float | None  # ranking_tau standardized against chance at this n_compared; None if < 2
    adjustment_accuracy: float  # fraction of graded (non-abstained) drivers with a valid backdoor set
    adjustment_valid: Mapping[str, bool | None]  # per-driver validity; None if the agent abstained
    abstention_accuracy: float  # fraction of drivers whose abstain/commit choice matches the gold rule
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


def _verdict_for(answer: Answer, key: InstanceKey, driver: str) -> DriverVerdict | None:
    """The agent's verdict for ``driver`` (by its display label), or ``None`` if it gave none."""
    label = key.node_to_label[driver]
    return next((v for v in answer.verdicts if v.driver == label), None)


def _ranking_tau(answer: Answer, key: InstanceKey) -> tuple[float, int]:
    """Kendall agreement of the agent's ranking with the true order; 0.0 unless it is a clean permutation.

    Returns ``(tau, n_compared)`` - ``n_compared`` is the real denominator behind ``tau``, needed to
    standardize it (``kendall_tau_zscore``): raw tau isn't comparable across instances of different
    ``n_compared``, since both its achievable granularity and its variance under chance depend on it.

    Drivers correctly abstained on are dropped from the pairwise comparison before scoring - ``kendall_tau``
    already returns 1.0 (vacuously) if that leaves fewer than two drivers to compare.

    Compares actual ``|true effect|`` magnitudes, not rank positions - two decoys always share exactly the
    same true effect (0), and only the raw magnitude preserves that tie for ``kendall_tau``'s own tie
    handling (a tied pair counts as neither concordant nor discordant) to act on. Rank positions collapse
    to always-distinct integers even when the underlying effects are exactly equal, which silently turns a
    genuine tie - no correct order exists - into an arbitrary, gradeable coin flip.
    """
    drivers = key.scm.scaffold.drivers
    agent_nodes = [key.label_to_node.get(label) for label in answer.ranking]
    if len(agent_nodes) != len(drivers) or set(agent_nodes) != set(drivers):
        return 0.0, len(drivers)  # not a well-formed ranking of exactly the candidate drivers
    correctly_abstained = {
        d for d in drivers if (v := _verdict_for(answer, key, d)) is not None and v.abstain and _gold_abstain(key, d)
    }
    compared = [d for d in drivers if d not in correctly_abstained]
    # Negated so "smaller = ranked first" matches agent_position's convention (index 0 = the agent's pick
    # for strongest); negation preserves exact ties, which is the property being fixed here.
    true_magnitude = {node: -abs(key.scm.total_effect_on_outcome(node)) for node in drivers}
    agent_position = {node: i for i, node in enumerate(agent_nodes)}
    tau = kendall_tau([true_magnitude[d] for d in compared], [agent_position[d] for d in compared])
    return tau, len(compared)


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
    abstention_correct = [_abstention_correct(answer, key, driver) for driver in drivers]
    ranking_tau, n_compared = _ranking_tau(answer, key)
    return AnswerScore(
        parse_ok=True,
        ranking_tau=ranking_tau,
        n_compared=n_compared,
        ranking_tau_z=kendall_tau_zscore(ranking_tau, n_compared),
        adjustment_accuracy=adjustment_accuracy,
        adjustment_valid=valid,
        abstention_accuracy=sum(abstention_correct) / len(abstention_correct),
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
            adjustment_accuracy=0.0,
            adjustment_valid=dict.fromkeys(drivers, False),
            abstention_accuracy=0.0,
            alignment=key.provenance.alignment,
        )
    return score_answer(answer, key)

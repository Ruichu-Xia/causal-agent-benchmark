import json

from causalagentbench import build_instance, sample
from causalagentbench.agent.eval import aggregate_by_alignment, aggregate_by_driver_cell, run_evaluation
from causalagentbench.agent.loop import ModelTurn
from causalagentbench.scoring.gates import generate_gated_procedural
from causalagentbench.scoring.score import AnswerScore, RankingPairs, score_output
from causalagentbench.task.instance import Instance, InstanceKey


def _instance(seed: int) -> tuple[Instance, InstanceKey]:
    scm, _ = generate_gated_procedural(seed)
    data = sample(scm, 200, seed=seed)
    return build_instance(scm, data, seed=seed)


class _FixedReplyClient:
    """Answers immediately every call, ignoring the message history (no tool use)."""

    def __init__(self, reply: str):
        self._reply = reply

    def complete(self, messages, tools, *, force_final: bool) -> ModelTurn:
        return ModelTurn(content=self._reply)


def _perfect_answer_text(key: InstanceKey) -> str:
    """A well-formed answer that matches gold exactly (true ranking, gold adjustment sets)."""
    label = dict(key.node_to_label)
    nodes = key.scm.true_ranking()
    verdicts = [
        {"driver": label[node], "adjustment_set": [label[z] for z in key.scm.scaffold.correct_adjustment.get(node, ())]}
        for node in nodes
    ]
    ranking = [label[node] for node in nodes]
    return f"<answer>{json.dumps({'verdicts': verdicts, 'ranking': ranking})}</answer>"


class _RaisesOnFirstCallClient:
    """Simulates a client/provider error on the first instance only.

    A stand-in for anything the tool loop itself can't turn into text (e.g. a network failure).
    """

    def __init__(self, reply: str):
        self._reply = reply
        self._calls = 0

    def complete(self, messages, tools, *, force_final: bool) -> ModelTurn:
        self._calls += 1
        if self._calls == 1:
            raise RuntimeError("simulated provider failure")
        return ModelTurn(content=self._reply)


def test_run_evaluation_isolates_a_crashing_instance_instead_of_losing_the_batch():
    second = _instance(1)
    instances = [_instance(0), second]
    errors: list[tuple[int, Exception]] = []
    scores = run_evaluation(
        instances,
        _RaisesOnFirstCallClient(_perfect_answer_text(second[1])),
        on_instance_error=lambda index, error: errors.append((index, error)),
    )
    assert len(scores) == 2
    assert not scores[0].parse_ok  # the crashed instance scores as a parse failure, not an exception
    assert scores[1].parse_ok  # the second instance still ran normally
    assert len(errors) == 1
    assert errors[0][0] == 0 and isinstance(errors[0][1], RuntimeError)


def test_run_evaluation_returns_one_score_per_instance():
    instances = [_instance(0), _instance(1)]
    scores = run_evaluation(instances, _FixedReplyClient("not an answer"))
    assert len(scores) == 2
    assert all(not s.parse_ok for s in scores)  # no <answer> block in the fixed reply


def test_aggregate_groups_by_alignment_and_sorts_none_last():
    instances = [_instance(0), _instance(1), _instance(2)]  # all abstract-labeled -> alignment=None
    scores = run_evaluation(instances, _FixedReplyClient("no answer"))
    summaries = aggregate_by_alignment(scores)
    assert len(summaries) == 1
    assert summaries[0].alignment is None
    assert summaries[0].n == 3
    assert summaries[0].parse_rate == 0.0  # none parsed


def test_aggregate_computes_means_correctly():
    _instance_obj, key = _instance(0)
    good = score_output(_perfect_answer_text(key), key)
    bad = score_output("nothing", key)
    assert good.parse_ok and good.ranking_tau == 1.0
    assert not bad.parse_ok

    summaries = aggregate_by_alignment([good, bad])
    assert len(summaries) == 1
    assert summaries[0].n == 2
    assert summaries[0].parse_rate == 0.5
    assert summaries[0].mean_ranking_tau == (good.ranking_tau + bad.ranking_tau) / 2
    assert summaries[0].mean_adjustment_accuracy == (good.adjustment_accuracy + bad.adjustment_accuracy) / 2


# ---- driver-grain cells -------------------------------------------------------------------------------
# Built directly rather than through a model, so identifiability and alignment can be varied independently
# (the real pipeline ties "aligned" to alignment > 0, and abstract labels always give alignment=None).


def _score(alignment: float | None, drivers: dict[str, tuple[bool, bool | None, bool, RankingPairs]]) -> AnswerScore:
    """One score from ``{driver: (identifiable, adjustment_valid, abstention_correct, pairs)}``."""
    return AnswerScore(
        parse_ok=True,
        ranking_tau=0.0,
        n_compared=len(drivers),
        ranking_tau_z=None,
        ranking_pairs={d: v[3] for d, v in drivers.items()},
        adjustment_accuracy=0.0,
        adjustment_valid={d: v[1] for d, v in drivers.items()},
        abstention_accuracy=0.0,
        abstention_correct={d: v[2] for d, v in drivers.items()},
        identifiable={d: v[0] for d, v in drivers.items()},
        alignment=alignment,
    )


def test_driver_cells_split_identifiable_from_unidentifiable_within_each_alignment():
    scores = [
        _score(0.0, {"a": (True, True, True, RankingPairs(2, 0)), "u": (False, False, False, RankingPairs(0, 2))}),
        _score(1.0, {"a": (True, False, True, RankingPairs(1, 1)), "u": (False, True, False, RankingPairs(2, 0))}),
    ]
    cells = {(c.identifiable, c.alignment): c for c in aggregate_by_driver_cell(scores)}
    assert len(cells) == 4
    assert cells[(True, 0.0)].mean_adjustment_accuracy == 1.0
    assert cells[(False, 0.0)].mean_adjustment_accuracy == 0.0
    assert cells[(True, 0.0)].ranking_tau == 1.0  # (2-0)/(2+0)
    assert cells[(False, 0.0)].ranking_tau == -1.0  # (0-2)/(0+2)
    assert cells[(True, 1.0)].ranking_tau == 0.0  # (1-1)/(1+1)


def test_driver_cells_pool_across_instances_rather_than_averaging_instance_means():
    # Two instances at the same alignment, different driver counts: one identifiable driver right, two
    # wrong. Pooled gives 1/3; a mean of per-instance means would give (1.0 + 0.0)/2 = 0.5.
    scores = [
        _score(0.0, {"a": (True, True, True, RankingPairs(0, 0))}),
        _score(0.0, {"a": (True, False, True, RankingPairs(0, 0)), "b": (True, False, True, RankingPairs(0, 0))}),
    ]
    (cell,) = aggregate_by_driver_cell(scores)
    assert cell.n == 3
    assert cell.mean_adjustment_accuracy == 1 / 3


def test_driver_cells_exclude_abstained_drivers_from_the_graded_denominator():
    scores = [_score(0.0, {"a": (True, True, True, RankingPairs(0, 0)), "b": (True, None, True, RankingPairs(0, 0))})]
    (cell,) = aggregate_by_driver_cell(scores)
    assert cell.n == 2  # both drivers are observed...
    assert cell.n_graded == 1  # ...but only the non-abstained one is graded on adjustment
    assert cell.mean_adjustment_accuracy == 1.0


def test_driver_cells_are_vacuously_accurate_when_nothing_was_graded_or_compared():
    scores = [_score(0.0, {"a": (True, None, True, RankingPairs(0, 0))})]
    (cell,) = aggregate_by_driver_cell(scores)
    assert cell.n_graded == 0
    assert cell.mean_adjustment_accuracy == 1.0  # not 0.0 - a failure the agent never had a chance to make
    assert cell.n_ranking_pairs == 0
    assert cell.ranking_tau == 1.0


def test_driver_cells_sort_identifiable_first_and_name_blind_last():
    """Name-blind puts ``None`` in *two* cells here, not one - pin the order that falls out.

    The ``alignment is None`` component of the key is what keeps that safe: it splits None from float
    before any ``<`` is applied, so None is only ever compared for equality. (Dropping it *would* break -
    ``None < 1.0`` raises - which is why this asserts a mixed name-blind/aligned set rather than one level.)
    """
    scores = [
        _score(None, {"a": (True, True, True, RankingPairs(0, 0)), "u": (False, True, True, RankingPairs(0, 0))}),
        _score(1.0, {"a": (True, True, True, RankingPairs(0, 0)), "u": (False, True, True, RankingPairs(0, 0))}),
    ]
    cells = aggregate_by_driver_cell(scores)
    assert [(c.identifiable, c.alignment) for c in cells] == [(True, 1.0), (True, None), (False, 1.0), (False, None)]


def test_driver_cells_cover_every_candidate_driver_end_to_end():
    _instance_obj, key = _instance(0)
    score = score_output(_perfect_answer_text(key), key)
    cells = aggregate_by_driver_cell([score])
    assert sum(c.n for c in cells) == len(key.scm.scaffold.drivers)


def test_on_turn_tags_every_turn_with_its_own_instance_index():
    instances = [_instance(0), _instance(1)]
    seen: list[int] = []
    run_evaluation(instances, _FixedReplyClient("no tool calls"), on_turn=lambda i, _trace: seen.append(i))
    assert seen == [0, 0, 1, 1]  # invalid replies receive one tagged formatting repair per instance


def test_on_turn_is_never_called_when_omitted():
    instances = [_instance(0)]
    run_evaluation(instances, _FixedReplyClient("no tool calls"))  # would raise if on_turn were required

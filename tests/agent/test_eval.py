import json

from causalagentbench import build_instance, sample
from causalagentbench.agent.eval import aggregate_by_alignment, run_evaluation
from causalagentbench.agent.loop import ModelTurn
from causalagentbench.scoring.gates import generate_gated_procedural
from causalagentbench.scoring.score import score_output
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
    scores = run_evaluation(instances, _RaisesOnFirstCallClient(_perfect_answer_text(second[1])))
    assert len(scores) == 2
    assert not scores[0].parse_ok  # the crashed instance scores as a parse failure, not an exception
    assert scores[1].parse_ok  # the second instance still ran normally


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

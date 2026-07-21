import json

from causalagentbench import build_instance, generate_gated, sample
from causalagentbench.agent.eval import aggregate_by_alignment, run_evaluation
from causalagentbench.agent.loop import ModelTurn
from causalagentbench.scoring.score import score_output
from causalagentbench.task.instance import Instance, InstanceKey


def _instance(seed: int) -> tuple[Instance, InstanceKey]:
    scm, _ = generate_gated(seed)
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

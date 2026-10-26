import json

import numpy as np
import pytest

from causalagentbench.agent.eval import episode_record, evaluate_task
from causalagentbench.scm.sample import sample
from causalagentbench.scoring.gates import generate_scm
from causalagentbench.scoring.score import SCORING_VERSION
from causalagentbench.task.instance import build_instance


@pytest.fixture
def task():
    """A placeholder-named accepted task with a small table."""
    scm, _ = generate_scm(np.random.SeedSequence(0))
    return build_instance(scm, sample(scm, 300, 0), 0)


def _exact_answer(key) -> str:
    """The true effect of every candidate, as model output."""
    verdicts = [
        {"driver": key.node_to_label[d], "adjustment_set": [], "effect": key.scm.total_effect_on_outcome(d)}
        for d in key.scm.scaffold.drivers
    ]
    return f"<answer>{json.dumps({'verdicts': verdicts})}</answer>"


def test_episode_record_lists_every_candidate_with_its_score(task):
    _, key = task
    record = episode_record(3, "m", _exact_answer(key), key, seconds=1.5)
    assert record["type"] == "episode" and record["index"] == 3 and record["model"] == "m"
    assert record["scoring_version"] == SCORING_VERSION
    assert record["parse_ok"] and record["task_error"] == 0.0 and record["reward"] == 1.0
    assert record["seconds"] == 1.5
    assert [c["node"] for c in record["candidates"]] == list(key.scm.scaffold.drivers)
    for candidate in record["candidates"]:
        assert candidate["label"] == key.node_to_label[candidate["node"]]
        assert candidate["estimate"] == candidate["truth"] == key.scm.total_effect_on_outcome(candidate["node"])
        assert candidate["error"] == candidate["capped_error"] == 0.0
    assert any(c["name_critical"] for c in record["candidates"])
    json.dumps(record, allow_nan=False)


def test_episode_record_of_invalid_output_scores_zero_estimates(task):
    _, key = task
    record = episode_record(0, "m", "no answer", key)
    assert not record["parse_ok"] and record["reward"] == -1.0
    for candidate in record["candidates"]:
        assert candidate["estimate"] == 0.0
        assert candidate["capped_error"] == pytest.approx(abs(candidate["truth"]))


def test_a_baseline_episode_has_no_transcript(task):
    record, transcript = evaluate_task(1, task, model=None, baseline="oracle", max_tool_calls=5, max_turns=5)
    assert transcript is None
    assert record["type"] == "episode" and record["model"] == "oracle" and record["index"] == 1
    assert record["parse_ok"]
    assert record["tool_calls"] == 0 and record["served_models"] == []


def test_a_model_episode_records_usage_and_its_transcript(task, fake_llm):
    record, transcript = evaluate_task(0, task, model="fake", baseline=None, max_tool_calls=5, max_turns=5)
    assert record["type"] == "episode" and record["model"] == "fake"
    assert record["parse_ok"] and record["task_error"] == 1.0  # the fake answers zero everywhere
    assert record["turns"] == 2 and record["tool_calls"] == 1
    assert record["prompt_tokens"] == 200 and record["completion_tokens"] == 20
    assert record["estimated_cost_usd"] == pytest.approx(0.002)
    assert record["served_models"] == ["fake-served"]
    assert [t["tool_calls"] for t in record["turn_usage"]] == [1, 0]
    assert [t["prompt_tokens"] for t in record["turn_usage"]] == [100, 100]
    assert record["cached_tokens"] == 0 and record["reasoning_tokens"] == 0  # the fake reports neither
    assert transcript["type"] == "transcript" and transcript["final_answer"] == record["answer"]
    assert transcript["error"] is None
    roles = [m["role"] for m in transcript["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert transcript["messages"][-1]["content"] == record["answer"]


def test_a_provider_failure_is_an_error_record_not_a_score(task, fake_llm, monkeypatch):
    monkeypatch.setattr(fake_llm, "fail", True)
    record, transcript = evaluate_task(0, task, model="fake", baseline=None, max_tool_calls=5, max_turns=5)
    assert record == {"type": "error", "index": 0, "model": "fake", "error": "RuntimeError: provider unavailable"}
    assert transcript["error"] == "RuntimeError: provider unavailable"
    assert transcript["final_answer"] is None

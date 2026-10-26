import inspect
import json

import numpy as np
import pytest

from causalagentbench.benchmark import make_task
from causalagentbench.environment import CausalEnv, final_reply, reward_func, training_labeler
from causalagentbench.generation.causenet import CauseNetLabeler
from causalagentbench.scm.sample import sample
from causalagentbench.scoring.gates import generate_scm
from causalagentbench.scoring.score import INVALID_REWARD, score_output
from causalagentbench.task.dataset import concept_pairs, save_dataset
from causalagentbench.task.instance import build_instance


@pytest.fixture
def task():
    """A placeholder-named accepted task with a small table."""
    scm, _ = generate_scm(np.random.SeedSequence(0))
    return build_instance(scm, sample(scm, 500, 0), 0)


def _answer(task, shift: float = 0.0) -> str:
    """An answer block with every true effect moved by ``shift``."""
    _, key = task
    verdicts = [
        {"driver": key.node_to_label[d], "adjustment_set": [], "effect": key.scm.total_effect_on_outcome(d) + shift}
        for d in key.scm.scaffold.drivers
    ]
    return f"<answer>{json.dumps({'verdicts': verdicts})}</answer>"


def _any_tool(env: CausalEnv, task) -> str:
    """Spend one tool call: the marginal regression of the first candidate."""
    return env.regress(task[0].candidate_drivers[0])


def _rollout(final: str) -> list[dict]:
    """A conversational completion: one tool call, its result, then ``final``."""
    return [
        {"role": "assistant", "content": "", "tool_calls": [{"name": "discover", "arguments": {}}]},
        {"role": "tool", "name": "discover", "content": "PC ..."},
        {"role": "assistant", "content": final},
    ]


def test_the_public_methods_are_exactly_the_evaluation_tools():
    """TRL exposes every public method except ``reset`` as a tool; they must match the evaluation harness."""
    public = {name for name, _ in inspect.getmembers(CausalEnv, inspect.isfunction) if not name.startswith("_")}
    assert public == {"reset", "correlate", "regress", "discover"}


def test_reset_on_a_frozen_task_returns_its_prompt(task):
    env = CausalEnv()
    assert env.reset(task=task) == task[0].prompt
    assert not env.done and env.reward == INVALID_REWARD


def test_reset_with_a_seed_serves_that_pre_built_task(task):
    env = CausalEnv(tasks=[None, task])
    assert env.reset(seed=1, prompt="ignored dataset column") == task[0].prompt


def test_tools_report_the_remaining_budget_and_then_raise(task):
    instance, _ = task
    driver = instance.candidate_drivers[0]
    env = CausalEnv(max_tool_calls=3)
    env.reset(task=task)
    first = env.correlate(driver, instance.outcome)
    assert first.startswith(f"corr({driver}, {instance.outcome} | {{}})")
    assert first.endswith("[budget: 2 of 3 tool calls remaining]")
    assert env.regress(driver, []).startswith(f"regress({driver} | {{}})")
    assert env.regress(driver).endswith("[budget: 0 of 3 tool calls remaining]")
    with pytest.raises(ValueError, match="budget exhausted"):
        env.discover()
    assert not env.done  # an exhausted budget still allows the final answer


def test_reward_scores_the_final_reply_exactly_as_evaluation_does(task):
    env = CausalEnv()
    env.reset(task=task)
    _any_tool(env, task)
    expected = score_output(_answer(task, shift=0.05), task[1]).reward
    assert reward_func([_rollout(_answer(task, shift=0.05))], [env]) == [pytest.approx(expected)]
    assert env.done and 0.0 < env.reward < 1.0
    assert env.info["parse_ok"] and env.info["tool_calls"] == 1


def test_an_exact_answer_earns_the_top_reward(task):
    env = CausalEnv()
    env.reset(task=task)
    assert reward_func([_answer(task)], [env]) == [1.0]
    assert env.info == {"parse_ok": True, "task_error": 0.0, "tool_calls": 0}


def test_a_reply_without_an_answer_block_earns_the_floor_reward(task):
    envs = [CausalEnv(), CausalEnv()]
    for env in envs:
        env.reset(task=task)
    ended_on_a_tool_call = _rollout("")[:1]
    assert reward_func([_rollout("I am not sure."), ended_on_a_tool_call], envs) == [INVALID_REWARD] * 2
    assert not envs[0].info["parse_ok"]


def test_final_reply_reads_the_last_assistant_text():
    parts = [{"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    assert final_reply("plain text") == "plain text"
    assert final_reply(_rollout("done")) == "done"
    assert final_reply(parts) == "ab"
    assert final_reply([]) == ""


def test_the_episode_is_over_after_scoring(task):
    env = CausalEnv()
    env.reset(task=task)
    reward_func([_answer(task)], [env])
    with pytest.raises(ValueError, match="episode is over"):
        _any_tool(env, task)
    with pytest.raises(ValueError, match="episode is over"):
        reward_func([_answer(task)], [env])


def test_reset_starts_a_fresh_episode(task):
    env = CausalEnv(max_tool_calls=1)
    env.reset(task=task)
    _any_tool(env, task)
    reward_func([_answer(task)], [env])
    env.reset(task=task)
    assert not env.done and env.reward == INVALID_REWARD and env.info == {}
    _any_tool(env, task)  # the budget is restored


def test_tools_need_a_started_episode():
    with pytest.raises(ValueError, match="episode is over"):
        CausalEnv().discover()


def test_reset_needs_a_frozen_task_or_a_seed_with_a_task_source():
    with pytest.raises(ValueError, match="frozen task"):
        CausalEnv().reset()
    with pytest.raises(ValueError, match="frozen task"):
        CausalEnv().reset(seed=0)


def test_reset_with_a_seed_and_a_labeler_builds_that_training_task(causenet_graph):
    labeler = CauseNetLabeler(causenet_graph)
    env = CausalEnv(labeler=labeler)
    prompt = env.reset(seed=1, prompt="ignored dataset column")
    assert prompt == make_task("train", 1, labeler=labeler)[0].prompt


def test_training_labeler_avoids_every_evaluation_concept_pair(tmp_path, causenet_file):
    evaluation = [make_task("dev", i, labeler=CauseNetLabeler.from_path(causenet_file), n_rows=50) for i in range(2)]
    save_dataset(tmp_path / "dev", evaluation, split="dev")
    labeler = training_labeler(str(causenet_file), [str(tmp_path / "dev")])
    pairs = concept_pairs(evaluation)
    assert labeler.source.startswith("causenet:sha256:")
    assert not any(frozenset(edge) in pairs for edge in labeler.one_way.edges)
    assert any(frozenset(edge) in pairs for edge in labeler.graph.edges)  # still known for the order rule

import json

import numpy as np
import pytest

from causalagentbench.benchmark import make_task
from causalagentbench.environment import CausalEnv, reward_func, training_labeler
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
    """JSON verdicts with every true effect moved by ``shift``."""
    _, key = task
    verdicts = [
        {"driver": key.node_to_label[d], "adjustment_set": [], "effect": key.scm.total_effect_on_outcome(d) + shift}
        for d in key.scm.scaffold.drivers
    ]
    return json.dumps({"verdicts": verdicts})


def _any_tool(env: CausalEnv, task) -> str:
    """Spend one tool call: the marginal regression of the first candidate."""
    return env.regress(task[0].candidate_drivers[0])


def test_reset_on_a_frozen_task_returns_its_prompt(task):
    env = CausalEnv()
    assert env.reset(task=task) == task[0].prompt
    assert not env.done and env.reward == INVALID_REWARD


def test_tools_spend_the_budget_and_then_raise(task):
    instance, _ = task
    driver = instance.candidate_drivers[0]
    env = CausalEnv(max_tool_calls=3)
    env.reset(task=task)
    assert env.correlate(driver, instance.outcome).startswith(f"corr({driver}, {instance.outcome} | {{}})")
    assert env.regress(driver, []).startswith(f"regress({driver} | {{}})")
    assert env.regress(driver).startswith(f"regress({driver} | {{}})")
    with pytest.raises(ValueError, match="budget exhausted"):
        env.discover()
    assert not env.done  # an exhausted budget still allows the submission


def test_submit_rewards_the_answer_by_the_benchmark_score(task):
    env = CausalEnv()
    env.reset(task=task)
    _any_tool(env, task)
    assert env.submit(f"<answer>{_answer(task)}</answer>") == "Answer recorded."
    assert env.done and env.reward == 1.0
    assert env.info == {"parse_ok": True, "task_error": 0.0, "tool_calls": 1}


def test_submit_accepts_bare_json_and_matches_the_scorer(task):
    env = CausalEnv()
    env.reset(task=task)
    env.submit(_answer(task, shift=0.05))
    expected = score_output(f"<answer>{_answer(task, shift=0.05)}</answer>", task[1]).reward
    assert env.reward == pytest.approx(expected)
    assert 0.0 < env.reward < 1.0


def test_an_invalid_submission_earns_the_floor_reward(task):
    env = CausalEnv()
    env.reset(task=task)
    assert env.submit("I am not sure.").startswith("Invalid answer")
    assert env.done and env.reward == INVALID_REWARD
    assert not env.info["parse_ok"]


def test_an_episode_without_a_submission_keeps_the_floor_reward(task):
    env = CausalEnv(max_tool_calls=1)
    env.reset(task=task)
    _any_tool(env, task)
    assert not env.done
    assert reward_func([env]) == [INVALID_REWARD]


def test_the_episode_is_over_after_submission(task):
    env = CausalEnv()
    env.reset(task=task)
    env.submit(_answer(task))
    with pytest.raises(ValueError, match="episode is over"):
        _any_tool(env, task)
    with pytest.raises(ValueError, match="episode is over"):
        env.submit(_answer(task))


def test_reset_starts_a_fresh_episode(task):
    env = CausalEnv(max_tool_calls=1)
    env.reset(task=task)
    _any_tool(env, task)
    env.submit(_answer(task))
    env.reset(task=task)
    assert not env.done and env.reward == INVALID_REWARD and env.info == {}
    _any_tool(env, task)  # the budget is restored


def test_tools_need_a_started_episode():
    with pytest.raises(ValueError, match="episode is over"):
        CausalEnv().discover()


def test_reward_func_returns_each_environment_reward(task):
    envs = [CausalEnv(), CausalEnv(), CausalEnv()]
    for env in envs:
        env.reset(task=task)
    envs[0].submit(_answer(task))
    envs[1].submit("nonsense")
    assert reward_func(envs, prompts=["unused"]) == [1.0, INVALID_REWARD, INVALID_REWARD]


def test_reset_needs_a_frozen_task_or_a_seed_and_a_labeler():
    with pytest.raises(ValueError, match="frozen task"):
        CausalEnv().reset()
    with pytest.raises(ValueError, match="frozen task"):
        CausalEnv().reset(seed=0)


def test_reset_with_a_seed_builds_that_training_task(causenet_graph):
    labeler = CauseNetLabeler(causenet_graph)
    env = CausalEnv(labeler)
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

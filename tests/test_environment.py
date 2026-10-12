import math

import pytest

from causalagentbench import CausalEnv, build_instance, generate_gated_procedural


@pytest.fixture
def task():
    scm, report = generate_gated_procedural(0, n_candidates=5)
    return build_instance(scm, report.data, 0)


def gold(task):
    _, key = task
    return {
        "verdicts": [
            {
                "driver": key.node_to_label[d],
                "adjustment_set": [key.node_to_label[z] for z in key.scm.scaffold.correct_adjustment[d]],
                "effect": key.scm.total_effect_on_outcome(d),
            }
            for d in key.scm.scaffold.drivers
        ]
    }


def test_tools_then_submission_use_the_exact_benchmark_reward(task):
    env = CausalEnv(max_tool_calls=1)
    observation, info = env.reset(task=task)
    assert info == {}
    assert "scm" not in observation and "data" not in observation and "key" not in observation
    observation, reward, terminated, truncated, info = env.step({"type": "tool", "name": "preview"})
    assert reward == 0 and not terminated and not truncated
    assert observation["must_submit"] and observation["tool_calls_remaining"] == 0
    _, reward, terminated, truncated, info = env.step({"type": "submit", "answer": gold(task)})
    assert reward == pytest.approx(1)
    assert terminated and not truncated and info["parse_ok"]
    with pytest.raises(RuntimeError, match="reset"):
        env.step({"type": "submit", "answer": gold(task)})


def test_invalid_tool_attempts_spend_budget_and_cannot_loop_forever(task):
    env = CausalEnv(max_tool_calls=1)
    env.reset(task=task)
    observation, *_ = env.step({"type": "unknown"})
    assert observation["content"].startswith("Error:")
    _, reward, terminated, truncated, _ = env.step({"type": "tool", "name": "preview"})
    assert reward == 0 and truncated and not terminated


def test_submission_reward_is_effect_accuracy_even_with_biased_adjustment(task):
    env = CausalEnv()
    env.reset(task=task)
    answer = gold(task)
    for verdict in answer["verdicts"]:
        verdict["adjustment_set"] = []
    _, reward, terminated, truncated, info = env.step({"type": "submit", "answer": answer})
    assert reward == info["effect_score"] == 1.0
    assert info["adjustment_score"] < 1.0
    assert info["effect_mae"] == 0.0
    assert info["scoring_version"] == "effect-v1"
    assert terminated and not truncated


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "NaN"])
def test_nonfinite_submission_terminates_with_finite_zero_reward(task, value):
    env = CausalEnv()
    env.reset(task=task)
    answer = gold(task)
    answer["verdicts"][0]["effect"] = value
    _, reward, terminated, truncated, info = env.step({"type": "submit", "answer": answer})
    assert math.isfinite(reward) and reward == 0
    assert terminated and not truncated and not info["parse_ok"]


def test_reset_starts_fresh_and_does_not_share_mutable_tool_schemas(task):
    env = CausalEnv()
    first, _ = env.reset(task=task)
    env.step({"type": "tool", "name": "preview"})
    first["tools"].clear()
    second, _ = env.reset(task=task)
    assert second["tool_calls_remaining"] == env.max_tool_calls
    assert len(second["tools"]) == 4


def test_reset_requires_a_dataset_or_grounded_generator():
    with pytest.raises(ValueError, match="frozen task"):
        CausalEnv().reset()

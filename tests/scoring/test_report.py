import math

import numpy as np
import pytest

from causalagentbench.scoring.report import Z_95, compare, names_ablation, summarize


def _candidate(capped: float, truth: float, critical: bool, *, error: float | None = None, valid: bool = True):
    """One scored candidate as it appears in an episode record."""
    return {
        "capped_error": capped,
        "error": capped if error is None else error,
        "truth": truth,
        "name_critical": critical,
        "adjustment_valid": valid,
    }


def _episode(index: int, candidates, *, parse_ok=True, reward=0.0, tool_calls=0, tokens=(0, 0), cost=0.0):
    """A scored episode record with the fields the report reads."""
    return {
        "index": index,
        "candidates": candidates,
        "parse_ok": parse_ok,
        "reward": reward,
        "tool_calls": tool_calls,
        "prompt_tokens": tokens[0],
        "completion_tokens": tokens[1],
        "estimated_cost_usd": cost,
    }


def _single(index: int, capped: float, truth: float):
    """An episode with one name-critical candidate and one exactly estimated data-identified candidate."""
    return _episode(index, [_candidate(capped, truth, True), _candidate(0.0, 1.0, False)])


_EPISODES = [
    _episode(
        0,
        [_candidate(0.2, 0.5, True), _candidate(1.0, -0.5, False, error=3.0, valid=False)],
        reward=0.3,
        tool_calls=4,
        tokens=(100, 20),
        cost=0.01,
    ),
    _episode(
        1,
        [_candidate(0.1, 1.0, False), _candidate(0.3, 0.2, True, valid=False)],
        parse_ok=False,
        reward=-1.0,
        tool_calls=2,
        tokens=(50, 10),
        cost=0.03,
    ),
]


def test_summarize_matches_hand_computation():
    summary = summarize(_EPISODES)
    ratio = 1.6 / 2.2  # task sums: errors (1.2, 0.4), |truth| (1.0, 1.2)
    se = math.sqrt(((1.2 - ratio * 1.0) ** 2 + (0.4 - ratio * 1.2) ** 2) / (2 * 1)) / 1.1
    assert summary["tasks"] == 2
    assert summary["capped_error"]["value"] == pytest.approx(ratio)
    assert summary["capped_error"]["se"] == pytest.approx(se)
    assert summary["uncapped_error"] == pytest.approx(3.6 / 2.2)
    assert summary["capped_candidates"] == 1
    assert summary["name_critical_error"]["value"] == pytest.approx(0.5 / 0.7)
    assert summary["data_identified_error"]["value"] == pytest.approx(1.1 / 1.5)
    assert summary["adjustment_valid_rate"] == 0.5
    assert summary["answer_valid_rate"] == 0.5
    assert summary["mean_reward"] == pytest.approx(-0.35)
    assert summary["mean_tool_calls"] == 3
    assert summary["mean_tokens"] == 90
    assert summary["total_cost_usd"] == pytest.approx(0.04)


def test_delta_method_se_is_the_linearized_task_variance():
    """The delta-method SE is the standard error of the mean residual, divided by mean(b).

    SE(R) = sqrt(sum_t (a_t - R b_t)^2 / (T (T - 1))) / mean(b), and the residuals a_t - R b_t sum to zero.
    """
    a, b = np.array([0.5, 0.2, 0.9]), np.array([1.0, 0.4, 1.5])
    episodes = [_single(i, x, t) for i, (x, t) in enumerate(zip(a, b, strict=True))]
    result = summarize(episodes)["name_critical_error"]
    ratio = a.sum() / b.sum()
    se = np.std(a - ratio * b, ddof=1) / np.sqrt(len(a)) / b.mean()
    assert result["value"] == pytest.approx(ratio)
    assert result["se"] == pytest.approx(se)
    assert result["ci_low"] == pytest.approx(ratio - Z_95 * se)
    assert result["ci_high"] == pytest.approx(ratio + Z_95 * se)


def test_a_single_task_has_no_standard_error():
    result = summarize([_single(0, 0.2, 0.5)])["name_critical_error"]
    assert result["value"] == pytest.approx(0.4)
    assert math.isnan(result["se"]) and math.isnan(result["ci_low"]) and math.isnan(result["ci_high"])


def test_summarize_reports_no_cost_when_any_episode_is_unpriced():
    episodes = [_single(0, 0.2, 0.5), {**_single(1, 0.2, 0.5), "estimated_cost_usd": None}]
    assert summarize(episodes)["total_cost_usd"] is None


def test_summarize_requires_episodes():
    with pytest.raises(ValueError, match="no scored episodes"):
        summarize([])


_FIRST = [_single(0, 0.1, 0.5), _single(1, 0.4, 1.0), _single(2, 0.3, 0.8)]
_SECOND = [_single(0, 0.3, 0.5), _single(1, 0.2, 1.0), _single(2, 0.6, 0.8), _single(9, 0.0, 0.1)]


def test_compare_is_a_paired_difference_over_shared_tasks():
    result = compare(_FIRST, _SECOND)
    assert result["tasks"] == 3
    expected = summarize(_FIRST)["capped_error"]["value"] - summarize(_SECOND[:3])["capped_error"]["value"]
    assert result["difference"]["value"] == pytest.approx(expected)


def test_compare_is_antisymmetric():
    forward, backward = compare(_FIRST, _SECOND)["difference"], compare(_SECOND, _FIRST)["difference"]
    assert forward["value"] == pytest.approx(-backward["value"])
    assert forward["se"] == pytest.approx(backward["se"])
    same = compare(_FIRST, _FIRST)["difference"]
    assert same["value"] == 0.0 and same["se"] == 0.0


def test_compare_needs_two_shared_tasks():
    with pytest.raises(ValueError, match="two shared tasks"):
        compare(_FIRST[:1], _SECOND)


def _two_slices(index: int, critical: tuple[float, float], identified: tuple[float, float]):
    """An episode with one name-critical and one data-identified candidate, each (capped error, truth)."""
    return _episode(index, [_candidate(*critical, True), _candidate(*identified, False)])


def test_names_ablation_point_estimate_matches_hand_computation():
    named = [
        _two_slices(0, (0.1, 0.5), (0.2, 0.4)),
        _two_slices(1, (0.3, 0.5), (0.1, 0.6)),
        _two_slices(2, (0.9, 0.5), (0.9, 0.5)),  # not in the placeholder run, so ignored
    ]
    placeholder = [_two_slices(0, (0.4, 0.5), (0.2, 0.4)), _two_slices(1, (0.5, 0.5), (0.2, 0.6))]
    result = names_ablation(named, placeholder, n_boot=200)
    assert result["tasks"] == 2
    # Critical: 0.9 / 1.0 - 0.4 / 1.0; identified: 0.4 / 1.0 - 0.3 / 1.0.
    assert result["name_critical_increase"]["value"] == pytest.approx(0.5)
    assert result["data_identified_increase"]["value"] == pytest.approx(0.1)
    assert result["difference_in_differences"]["value"] == pytest.approx(0.4)
    for stat in ("name_critical_increase", "data_identified_increase", "difference_in_differences"):
        assert result[stat]["ci_low"] <= result[stat]["ci_high"]
    assert names_ablation(named, placeholder, n_boot=200) == result  # the bootstrap is seeded


def test_names_ablation_needs_two_shared_tasks():
    episode = _two_slices(0, (0.1, 0.5), (0.2, 0.4))
    with pytest.raises(ValueError, match="two shared tasks"):
        names_ablation([episode], [episode])

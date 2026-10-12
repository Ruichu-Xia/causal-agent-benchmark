import json
import sys

import pytest

from causalagentbench import build_instance, generate_gated_procedural, save_dataset
from causalagentbench.agent.eval import evaluate_task
from causalagentbench.cli import _summarize, main


def task():
    scm, report = generate_gated_procedural(0, n_candidates=5)
    return build_instance(scm, report.data, 0)


@pytest.mark.parametrize("baseline", ["oracle", "zero"])
def test_baseline_command_streams_records_and_summary(tmp_path, monkeypatch, baseline):
    dataset = tmp_path / "dataset"
    save_dataset(dataset, [task(), task()])
    output = tmp_path / "results.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "causalagentbench",
            "baseline",
            "--dataset",
            str(dataset),
            "--agent",
            baseline,
            "--out",
            str(output),
            "--workers",
            "2",
        ],
    )
    assert main() == 0
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["type"] for r in records] == ["run", "episode", "episode", "summary"]
    assert len(records[0]["dataset_sha256"]) == 64
    assert records[-1]["scored"] == 2 and records[-1]["errors"] == 0
    assert all(r["scoring_version"] == "effect-v1" for r in records)
    assert all(r["reporting_version"] == "normalized-error-v1" for r in records)
    assert records[-1]["primary_metric"] == "normalized_effect_error"
    assert records[-1]["answer_validity_rate"] == 1.0
    assert records[-1]["reward"] == records[-1]["effect_score"]
    assert records[-1]["effect_mae"] >= 0.0
    if baseline == "oracle":
        assert records[-1]["adjustment_score"] == pytest.approx(1.0)
    else:
        from causalagentbench.task.answer import parse_answer

        assert records[-1]["ranking_tau"] == 0.0
        assert records[-1]["normalized_effect_error"] == 1.0
        for record in records[1:-1]:
            assert all(v.effect == 0.0 and not v.adjustment_set for v in parse_answer(record["answer"]).verdicts)


def test_invalid_answer_counts_as_zero_estimates_in_report(tmp_path, monkeypatch):
    from causalagentbench.agent import clients
    from causalagentbench.agent import eval as evaluator

    monkeypatch.setattr(clients.LiteLLMClient, "validate_tool_capability", lambda self: None)
    monkeypatch.setattr(evaluator, "run_agent", lambda *args, **kwargs: "invalid")
    dataset = tmp_path / "dataset"
    save_dataset(dataset, [task()])
    output = tmp_path / "results.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["causalagentbench", "eval", "--dataset", str(dataset), "--model", "stub", "--out", str(output)]
    )
    assert main() == 0
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[-1]["parse_failures"] == 1
    assert records[-1]["reward"] == 0.0
    assert records[-1]["normalized_effect_error"] == 1.0
    assert records[-1]["effect_mae"] == records[-1]["zero_effect_mae"]
    assert records[-1]["answer_validity_rate"] == 0.0


def test_provider_failure_is_an_error_not_a_zero_score(monkeypatch):
    from causalagentbench.agent import clients
    from causalagentbench.agent import eval as evaluator

    monkeypatch.setattr(clients, "LiteLLMClient", lambda **kwargs: object())

    def fail(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(evaluator, "run_agent", fail)
    result = evaluate_task(0, task(), model="stub", baseline=None, max_tool_calls=10, max_turns=5)
    assert result["type"] == "error"
    assert "reward" not in result
    assert "provider unavailable" in result["error"]


def _row(n, mae, zero_mae, *, valid=True):
    return {
        "type": "episode",
        "n_candidates": n,
        "effect_mae": mae,
        "zero_effect_mae": zero_mae,
        "parse_ok": valid,
        "reward": 0.0,
        "effect_score": 0.0,
        "adjustment_score": 0.0,
        "ranking_tau": 0.0,
        "estimated_cost_usd": 0.0,
    }


def test_headline_pools_candidates_before_normalizing():
    # Three errors of 1 and one error of 2; truth magnitudes 2, 2, 2, 4.
    rows = [_row(3, 1.0, 2.0), _row(1, 2.0, 4.0)]
    summary = _summarize(rows, "stub")
    assert summary["effect_mae"] == 1.25
    assert summary["normalized_effect_error"] == 0.5
    # Different per-task ratios: arithmetic mean would be 0.25, but pooled ratio is 0.3.
    rows[1]["effect_mae"] = 0.0
    assert _summarize(rows, "stub")["normalized_effect_error"] == 0.3


@pytest.mark.parametrize("mae, expected", [(0.0, 0.0), (2.0, 1.0), (6.0, 3.0)])
def test_headline_has_meaningful_reference_points_without_clipping(mae, expected):
    assert _summarize([_row(5, mae, 2.0)], "stub")["normalized_effect_error"] == expected


def test_invalid_answers_remain_in_summary_denominator():
    summary = _summarize([_row(5, 0.0, 2.0), _row(5, 2.0, 2.0, valid=False)], "stub")
    assert summary["normalized_effect_error"] == 0.5
    assert summary["answer_validity_rate"] == 0.5
    assert summary["parse_failures"] == 1


def test_incomplete_run_has_no_official_error_metric():
    summary = _summarize([_row(5, 0.0, 2.0), {"type": "error", "estimated_cost_usd": None}], "stub")
    assert summary["normalized_effect_error"] is None
    assert summary["effect_mae"] is None
    assert summary["answer_validity_rate"] == 0.5
    assert summary["errors"] == 1


def test_all_zero_truth_has_undefined_normalization():
    assert _summarize([_row(5, 0.0, 0.0)], "stub")["normalized_effect_error"] is None

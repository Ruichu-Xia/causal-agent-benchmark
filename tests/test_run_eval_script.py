"""Focused tests for the public LiteLLM evaluation runner."""

import importlib.util
import json
from pathlib import Path

from causalagentbench import build_instance, sample
from causalagentbench.agent.loop import TurnTrace, TurnTranscript
from causalagentbench.scoring.gates import generate_gated_procedural
from causalagentbench.scoring.score import score_output


def _runner_module():
    path = Path(__file__).parents[1] / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _instance(seed: int):
    scm, _ = generate_gated_procedural(seed)
    data = sample(scm, 200, seed=seed)
    return build_instance(scm, data, seed=seed)


def test_spread_across_scms_keeps_paired_conditions_together():
    runner = _runner_module()
    first = _instance(0)
    second = _instance(1)

    selected = runner._spread_across_scms([first, first, second, second], n=1)

    assert len(selected) == 2
    assert {item[1].scm.W.tobytes() for item in selected} == {first[1].scm.W.tobytes()}
    assert runner._spread_across_scms([], n=5) == []


def test_multi_model_sweep_records_litellm_turn_instance_and_summary(monkeypatch, tmp_path):
    runner = _runner_module()
    instance, key = _instance(0)
    score = score_output("not an answer", key)
    calls = []
    monkeypatch.setattr(runner.LiteLLMClient, "validate_tool_capability", lambda self: None)

    def fake_run_evaluation(instances, client, *, max_turns, on_instance, on_turn, on_transcript, on_instance_error):
        calls.append((client.model, client.temperature, client.request_timeout, max_turns, list(instances)))
        trace = TurnTrace(
            turn=0,
            duration_seconds=0.1,
            tool_calls=("preview",),
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.001,
        )
        assert on_turn is not None and on_instance is not None
        on_turn(0, trace)
        if on_transcript is not None:
            on_transcript(
                0,
                TurnTranscript(
                    turn=0,
                    duration_seconds=0.1,
                    force_final=False,
                    messages=({"role": "system", "content": "test"},),
                    tools=(),
                    content="model reply",
                ),
            )
        on_instance(0, score)
        return [score]

    monkeypatch.setattr(runner, "run_evaluation", fake_run_evaluation)
    out = tmp_path / "results.jsonl"
    transcript_out = tmp_path / "transcript.jsonl"
    with out.open("w", encoding="utf-8") as results_file, transcript_out.open("w", encoding="utf-8") as transcript_file:
        runner._run_models(
            models=["openai/model-a", "anthropic/model-b"],
            instances=[(instance, key)],
            n_drivers=[len(key.scm.scaffold.non_decoy_drivers())],
            max_turns=12,
            temperature=0.3,
            sampling_seed=7,
            openrouter_session_prefix=None,
            request_timeout=45.0,
            max_retries=2,
            dataset="outputs/frozen",
            results_file=results_file,
            transcript_file=transcript_file,
        )

    assert [(model, temperature, timeout, max_turns) for model, temperature, timeout, max_turns, _ in calls] == [
        ("openai/model-a", 0.3, 45.0, 12),
        ("anthropic/model-b", 0.3, 45.0, 12),
    ]
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [record["kind"] for record in records] == [
        "turn",
        "instance",
        "model_summary",
        "turn",
        "instance",
        "model_summary",
    ]
    assert all(record["model"] in {"openai/model-a", "anthropic/model-b"} for record in records)
    assert all(record.get("provider") == "litellm" for record in records if record["kind"] == "model_summary")
    assert all(record.get("estimated_cost_usd") == 0.001 for record in records if record["kind"] == "model_summary")
    transcript_records = [json.loads(line) for line in transcript_out.read_text(encoding="utf-8").splitlines()]
    assert len(transcript_records) == 2
    assert all(record["kind"] == "transcript" for record in transcript_records)
    assert all(
        record["request"]["messages"] == [{"role": "system", "content": "test"}] for record in transcript_records
    )
    assert all(record["response"]["content"] == "model reply" for record in transcript_records)


def test_transcript_record_includes_complete_request_and_response():
    runner = _runner_module()
    transcript = TurnTranscript(
        turn=2,
        duration_seconds=0.2,
        force_final=True,
        messages=({"role": "user", "content": "prompt"},),
        tools=({"type": "function", "function": {"name": "preview"}},),
        content="<answer>{}</answer>",
        prompt_tokens=20,
        completion_tokens=10,
        cost_usd=0.003,
    )

    record = runner._transcript_record("openrouter/example", 4, 0.2, 7, "test-session", transcript)

    assert record["kind"] == "transcript"
    assert record["index"] == 4
    assert record["request"] == {
        "messages": transcript.messages,
        "tools": transcript.tools,
        "tool_choice": "none",
        "temperature": 0.2,
        "seed": 7,
        "extra_body": {"session_id": "test-session"},
    }
    assert record["response"] == {
        "content": "<answer>{}</answer>",
        "tool_calls": [],
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "estimated_cost_usd": 0.003,
    }


def test_provider_error_is_recorded_but_excluded_from_model_summary(monkeypatch, tmp_path):
    runner = _runner_module()
    instance, key = _instance(0)
    failed_score = score_output("not an answer", key)

    def fake_run_evaluation(instances, client, *, max_turns, on_instance, on_turn, on_transcript, on_instance_error):
        del client, max_turns, on_turn, on_transcript
        on_instance_error(0, RuntimeError("provider unavailable"))
        on_instance(0, failed_score)
        return [failed_score]

    monkeypatch.setattr(runner, "run_evaluation", fake_run_evaluation)
    out = tmp_path / "results.jsonl"
    with out.open("w", encoding="utf-8") as results_file:
        runner._run_models(
            models=["openrouter/example"],
            instances=[(instance, key)],
            n_drivers=[len(key.scm.scaffold.non_decoy_drivers())],
            max_turns=2,
            temperature=0.2,
            sampling_seed=0,
            openrouter_session_prefix=None,
            request_timeout=45.0,
            max_retries=2,
            dataset=None,
            results_file=results_file,
            transcript_file=None,
        )

    error, summary = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert error["kind"] == "instance_error"
    assert error["error"] == "RuntimeError: provider unavailable"
    assert summary["kind"] == "model_summary"
    assert summary["instances_requested"] == 1
    assert summary["instances_scored"] == 0
    assert summary["provider_error_count"] == 1
    assert summary["per_instance"] == []


def test_openrouter_session_id_is_stable_per_model_and_seed():
    runner = _runner_module()

    first = runner._openrouter_session_id("openrouter/anthropic/claude-haiku-4.5", 7, None)

    assert first == runner._openrouter_session_id("openrouter/anthropic/claude-haiku-4.5", 7, None)
    assert first != runner._openrouter_session_id("openrouter/anthropic/claude-haiku-4.5", 8, None)
    assert first != runner._openrouter_session_id("openrouter/meta-llama/llama-3.3-70b-instruct", 7, None)
    assert runner._openrouter_session_id("anthropic/claude-haiku-4.5", 7, None) is None

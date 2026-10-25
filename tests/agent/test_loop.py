import json

import numpy as np

from causalagentbench import build_instance, sample
from causalagentbench.agent.loop import (
    _FINAL_TURN_REMINDER,
    _FORMAT_REPAIR_REMINDER,
    _SYSTEM_PROMPT,
    _TOOL_SCHEMAS,
    ModelTurn,
    ToolCall,
    TurnTrace,
    TurnTranscript,
    _dispatch,
    run_agent,
)
from causalagentbench.agent.tools import CausalTools
from causalagentbench.scoring.gates import generate_scm


def _instance():
    """A placeholder-named instance of an accepted SCM."""
    scm, _ = generate_scm(np.random.SeedSequence(0))
    data = sample(scm, 200, seed=0)
    instance, _ = build_instance(scm, data, seed=0)
    return instance


def _valid_answer(instance) -> str:
    """A contract-valid answer: a zero estimate for every candidate."""
    verdicts = [{"driver": driver, "adjustment_set": [], "effect": 0.0} for driver in instance.candidate_drivers]
    return f"<answer>{json.dumps({'verdicts': verdicts})}</answer>"


class _ScriptedClient:
    """Pops one scripted ``ModelTurn`` per call, recording every call it received."""

    def __init__(self, turns: list[ModelTurn]):
        """Queue the turns to replay."""
        self._turns = list(turns)
        self.calls: list[tuple[list[dict], list[dict], bool]] = []

    def complete(self, messages, tools, *, force_final: bool) -> ModelTurn:
        """Record the request and return the next scripted turn."""
        self.calls.append((list(messages), tools, force_final))  # snapshot: run_agent mutates in place
        return self._turns.pop(0)


def test_initial_messages_include_instructions_and_prompt_but_no_raw_data():
    instance = _instance()
    client = _ScriptedClient([ModelTurn(content=_valid_answer(instance))])
    run_agent(instance, client)
    messages, _tools, _ = client.calls[0]
    assert messages == [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": instance.prompt},
    ]
    assert not any(instance.to_csv().splitlines()[1] in str(m) for m in messages)  # no raw row pasted in


def test_tool_call_is_dispatched_and_result_fed_back():
    instance = _instance()
    a, b = instance.candidate_drivers[0], instance.outcome
    call = ToolCall(id="1", name="correlate", arguments={"x": a, "y": b})
    answer = _valid_answer(instance)
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=(call,)), ModelTurn(content=answer)])
    result = run_agent(instance, client)
    assert result == answer
    tool_message = client.calls[1][0][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "1"
    assert f"corr({a}, {b})" in tool_message["content"]


def test_replayed_tool_call_arguments_are_a_json_string_on_the_wire():
    """function.arguments must round-trip as a JSON string; chat-completions APIs reject a raw object."""
    instance = _instance()
    a, b = instance.candidate_drivers[0], instance.outcome
    call = ToolCall(id="1", name="correlate", arguments={"x": a, "y": b})
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=(call,)), ModelTurn(content=_valid_answer(instance))])
    run_agent(instance, client)
    assistant_message = client.calls[1][0][-2]
    wire_args = assistant_message["tool_calls"][0]["function"]["arguments"]
    assert isinstance(wire_args, str)
    assert json.loads(wire_args) == {"x": a, "y": b}


def test_unknown_tool_name_returns_error_without_crashing():
    instance = _instance()
    call = ToolCall(id="1", name="nonexistent_tool", arguments={})
    answer = _valid_answer(instance)
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=(call,)), ModelTurn(content=answer)])
    result = run_agent(instance, client)
    assert result == answer
    assert "Error" in client.calls[1][0][-1]["content"]


def test_a_tool_call_that_raises_returns_an_error_string_without_crashing_the_loop():
    # A malformed/hallucinated argument shape (here: an extra kwarg no tool accepts) raises inside the
    # tool itself; the loop must turn that into a tool-result error, not propagate the exception.
    instance = _instance()
    call = ToolCall(id="1", name="preview", arguments={"n": 5, "unexpected_kwarg": True})
    answer = _valid_answer(instance)
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=(call,)), ModelTurn(content=answer)])
    result = run_agent(instance, client)
    assert result == answer
    assert "Error" in client.calls[1][0][-1]["content"]


def test_a_client_side_parse_error_on_a_tool_call_is_reported_without_dispatch():
    # ToolCall.error is set by a client that could not parse the model's raw arguments (e.g. malformed
    # JSON); the loop must report it as a tool-result error and never attempt to dispatch it.
    instance = _instance()
    call = ToolCall(id="1", name="preview", arguments={}, error="malformed arguments JSON: boom")
    answer = _valid_answer(instance)
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=(call,)), ModelTurn(content=answer)])
    result = run_agent(instance, client)
    assert result == answer
    assert "malformed arguments JSON" in client.calls[1][0][-1]["content"]


def test_last_turn_requests_force_final():
    instance = _instance()
    client = _ScriptedClient([ModelTurn(content=_valid_answer(instance))])
    run_agent(instance, client, max_turns=1)
    assert client.calls[0][2] is True  # force_final on the only (last) turn


def test_final_turn_reminder_is_only_added_on_the_forced_final_turn():
    # A model mid-investigation can keep narrating "next I'd check X" instead of answering once tool
    # access is cut off (observed on real models); the explicit reminder tells it this is its last chance.
    instance = _instance()
    call = ToolCall(id="1", name="preview", arguments={})
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=(call,)), ModelTurn(content=_valid_answer(instance))])
    run_agent(instance, client, max_turns=2)

    first_call_messages, _, first_force_final = client.calls[0]
    second_call_messages, _, second_force_final = client.calls[1]
    assert first_force_final is False
    assert not any(m.get("content") == _FINAL_TURN_REMINDER for m in first_call_messages)
    assert second_force_final is True
    assert second_call_messages[-1] == {"role": "user", "content": _FINAL_TURN_REMINDER}


def test_invalid_tool_free_reply_gets_one_tool_free_format_repair():
    instance = _instance()
    answer = _valid_answer(instance)
    client = _ScriptedClient([ModelTurn(content="I cannot access the data."), ModelTurn(content=answer)])
    transcripts: list[TurnTranscript] = []

    result = run_agent(instance, client, on_transcript=transcripts.append)

    assert result == answer
    assert [force_final for _, _, force_final in client.calls] == [False, True]
    repair_messages = client.calls[1][0]
    assert repair_messages[-2] == {"role": "assistant", "content": "I cannot access the data."}
    assert repair_messages[-1] == {"role": "user", "content": _FORMAT_REPAIR_REMINDER}
    assert [transcript.turn for transcript in transcripts] == [0, 1]
    assert [transcript.force_final for transcript in transcripts] == [False, True]


def test_incomplete_json_answer_gets_a_format_repair():
    instance = _instance()
    answer = _valid_answer(instance)
    client = _ScriptedClient([ModelTurn(content='<answer>{"verdicts": []}</answer>'), ModelTurn(content=answer)])

    assert run_agent(instance, client) == answer
    assert client.calls[1][0][-1] == {"role": "user", "content": _FORMAT_REPAIR_REMINDER}


def test_invalid_format_is_repaired_only_once():
    instance = _instance()
    client = _ScriptedClient([ModelTurn(content="first invalid reply"), ModelTurn(content="second invalid reply")])

    assert run_agent(instance, client) == "second invalid reply"
    assert len(client.calls) == 2


def test_misbehaving_model_that_ignores_force_final_yields_empty_string():
    instance = _instance()
    call = ToolCall(id="1", name="preview", arguments={})
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=(call,))])  # keeps requesting tools
    assert run_agent(instance, client, max_turns=1) == ""


def test_valid_tool_free_reply_ends_immediately():
    instance = _instance()
    client = _ScriptedClient([ModelTurn(content=_valid_answer(instance))])
    run_agent(instance, client)
    assert len(client.calls) == 1  # stops at the first non-tool-call reply, well under MAX_TURNS


def test_on_turn_fires_once_per_turn_with_tool_names_and_token_counts():
    instance = _instance()
    call_a = ToolCall(id="1", name="preview", arguments={})
    call_b = ToolCall(id="2", name="correlate", arguments={"x": "a", "y": "b"})
    client = _ScriptedClient(
        [
            ModelTurn(
                content=None, tool_calls=(call_a, call_b), prompt_tokens=100, completion_tokens=20, cost_usd=0.01
            ),
            ModelTurn(content=_valid_answer(instance), prompt_tokens=150, completion_tokens=10, cost_usd=0.02),
        ]
    )
    traces: list[TurnTrace] = []
    run_agent(instance, client, on_turn=traces.append)

    assert [t.turn for t in traces] == [0, 1]
    assert traces[0].tool_calls == ("preview", "correlate")
    assert traces[0].prompt_tokens == 100 and traces[0].completion_tokens == 20
    assert traces[0].cost_usd == 0.01
    assert traces[1].tool_calls == ()  # the final, tool-free turn
    assert traces[1].prompt_tokens == 150 and traces[1].completion_tokens == 10
    assert traces[1].cost_usd == 0.02
    assert all(t.duration_seconds >= 0 for t in traces)


def test_traces_and_transcripts_carry_the_model_the_provider_served():
    instance = _instance()
    call = ToolCall(id="1", name="preview", arguments={})
    client = _ScriptedClient(
        [
            ModelTurn(content=None, tool_calls=(call,), model="provider/model-2026-01"),
            ModelTurn(content=_valid_answer(instance)),  # a provider that reports no model
        ]
    )
    traces: list[TurnTrace] = []
    transcripts: list[TurnTranscript] = []
    run_agent(instance, client, on_turn=traces.append, on_transcript=transcripts.append)

    assert [t.model for t in traces] == ["provider/model-2026-01", None]
    assert [t.model for t in transcripts] == ["provider/model-2026-01", None]


def test_on_turn_is_never_called_when_omitted():
    # Default None must impose zero behavior change on every existing caller.
    instance = _instance()
    client = _ScriptedClient([ModelTurn(content=_valid_answer(instance))])
    run_agent(instance, client)  # would raise if run_agent required on_turn


def test_on_transcript_captures_full_request_and_response():
    instance = _instance()
    answer = _valid_answer(instance)
    client = _ScriptedClient([ModelTurn(content=answer, prompt_tokens=10, completion_tokens=5)])
    transcripts: list[TurnTranscript] = []

    run_agent(instance, client, on_transcript=transcripts.append)

    assert len(transcripts) == 1
    transcript = transcripts[0]
    assert transcript.messages == (
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": instance.prompt},
    )
    assert transcript.tools[0]["function"]["name"] == "preview"
    assert transcript.content == answer
    assert transcript.prompt_tokens == 10
    assert transcript.completion_tokens == 5


def test_discover_is_exposed_to_the_model_and_dispatchable():
    """A tool the model is never told about is a tool it cannot use, so schema and dispatch must agree."""
    instance = _instance()
    schema_names = {schema["function"]["name"] for schema in _TOOL_SCHEMAS}
    assert schema_names == set(_dispatch(CausalTools(instance)))
    assert "discover" in schema_names

    call = ToolCall(id="1", name="discover", arguments={"alpha": 0.01})
    answer = _valid_answer(instance)
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=(call,)), ModelTurn(content=answer)])
    assert run_agent(instance, client) == answer
    assert "Oriented" in client.calls[1][0][-1]["content"]


def test_budget_notice_is_appended_to_the_last_tool_result_of_each_turn():
    """The model is told its remaining budget continuously, so it can pace its own investigation."""
    instance = _instance()
    calls = (ToolCall(id="1", name="preview", arguments={}), ToolCall(id="2", name="preview", arguments={}))
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=calls), ModelTurn(content=_valid_answer(instance))])
    run_agent(instance, client, max_turns=2, max_tool_calls=10)

    tool_messages = [m for m in client.calls[1][0] if m["role"] == "tool"]
    assert len(tool_messages) == 2
    # Only the last result of the turn carries the notice - one notice per turn, not one per call.
    assert "[budget:" not in tool_messages[0]["content"]
    assert "[budget: 8 of 10 tool calls remaining, 1 of 2 turns remaining]" in tool_messages[1]["content"]


def test_exhausting_the_tool_call_budget_forces_the_final_answer():
    """The tool-call budget, not the turn count, is what ends an ordinary investigation."""
    instance = _instance()
    call = ToolCall(id="1", name="preview", arguments={})
    client = _ScriptedClient(
        [ModelTurn(content=None, tool_calls=(call, call)), ModelTurn(content=_valid_answer(instance))]
    )
    run_agent(instance, client, max_turns=20, max_tool_calls=2)

    # Budget spent on turn 0, so turn 1 is forced even though 18 turns remain.
    assert client.calls[0][2] is False
    assert client.calls[1][2] is True
    assert client.calls[1][0][-1] == {"role": "user", "content": _FINAL_TURN_REMINDER}


def test_calls_beyond_the_budget_are_refused_rather_than_executed():
    """A turn may not overshoot the budget: batching many calls must not buy extra investigation."""
    instance = _instance()
    calls = tuple(ToolCall(id=str(i), name="preview", arguments={}) for i in range(4))
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=calls), ModelTurn(content=_valid_answer(instance))])
    run_agent(instance, client, max_turns=2, max_tool_calls=2)

    tool_messages = [m for m in client.calls[1][0] if m["role"] == "tool"]
    assert len(tool_messages) == 4
    for message in tool_messages[:2]:
        assert "budget exhausted" not in message["content"]
    for message in tool_messages[2:]:
        assert "budget exhausted" in message["content"]


def test_turn_trace_reports_the_budget_state_the_model_saw():
    instance = _instance()
    call = ToolCall(id="1", name="preview", arguments={})
    client = _ScriptedClient(
        [ModelTurn(content=None, tool_calls=(call, call)), ModelTurn(content=_valid_answer(instance))]
    )
    traces: list[TurnTrace] = []
    run_agent(instance, client, max_turns=2, max_tool_calls=10, on_turn=traces.append)

    # Pre-turn figure: what the model knew when it chose these calls, not the post-turn total.
    assert (traces[0].budget_used, traces[0].budget_limit) == (0, 10)
    assert (traces[1].budget_used, traces[1].budget_limit) == (2, 10)


def test_exhausting_the_token_budget_forces_the_final_answer():
    """Cost is bounded in a cost unit, so a runaway is cut off even with tool-call budget to spare."""
    instance = _instance()
    call = ToolCall(id="1", name="preview", arguments={})
    expensive = ModelTurn(content=None, tool_calls=(call,), prompt_tokens=400, completion_tokens=200)
    client = _ScriptedClient([expensive, ModelTurn(content=_valid_answer(instance))])
    run_agent(instance, client, max_tool_calls=100, max_token_budget=500)

    # 600 tokens spent on turn 0 exceeds the budget, so turn 1 is forced with 99 tool calls unspent.
    assert client.calls[0][2] is False
    assert client.calls[1][2] is True


def test_a_client_that_reports_no_usage_is_governed_by_the_tool_call_budget_alone():
    """An unenforceable budget must not silently truncate: no usage reported means no token cutoff."""
    instance = _instance()
    call = ToolCall(id="1", name="preview", arguments={})
    silent = ModelTurn(content=None, tool_calls=(call,))  # prompt/completion tokens default to None
    client = _ScriptedClient([silent, silent, ModelTurn(content=_valid_answer(instance))])
    run_agent(instance, client, max_tool_calls=100, max_token_budget=1)

    assert [c[2] for c in client.calls] == [False, False, False]


def test_exhausting_the_turn_budget_forces_the_final_answer_with_calls_to_spare():
    """Turns bind independently of calls: one call per turn runs out of turns long before the call budget."""
    instance = _instance()
    call = ToolCall(id="1", name="preview", arguments={})
    turns = [ModelTurn(content=None, tool_calls=(call,)) for _ in range(2)]
    client = _ScriptedClient([*turns, ModelTurn(content=_valid_answer(instance))])
    run_agent(instance, client, max_tool_calls=100, max_turns=3)

    assert [c[2] for c in client.calls] == [False, False, True]

import json

from causalagentbench import build_instance, sample
from causalagentbench.agent.loop import _FINAL_TURN_REMINDER, MAX_TURNS, ModelTurn, ToolCall, run_agent
from causalagentbench.scoring.gates import generate_gated_procedural


def _instance():
    scm, _ = generate_gated_procedural(0)
    data = sample(scm, 200, seed=0)
    instance, _ = build_instance(scm, data, seed=0)
    return instance


class _ScriptedClient:
    """Pops one scripted ``ModelTurn`` per call, recording every call it received."""

    def __init__(self, turns: list[ModelTurn]):
        self._turns = list(turns)
        self.calls: list[tuple[list[dict], list[dict], bool]] = []

    def complete(self, messages, tools, *, force_final: bool) -> ModelTurn:
        self.calls.append((list(messages), tools, force_final))  # snapshot: run_agent mutates in place
        return self._turns.pop(0)


def test_initial_message_is_the_prompt_only_no_raw_data():
    instance = _instance()
    client = _ScriptedClient([ModelTurn(content="<answer>{}</answer>")])
    run_agent(instance, client)
    messages, _tools, _ = client.calls[0]
    assert messages == [{"role": "user", "content": instance.prompt}]
    assert not any(instance.to_csv().splitlines()[1] in str(m) for m in messages)  # no raw row pasted in


def test_tool_call_is_dispatched_and_result_fed_back():
    instance = _instance()
    a, b = instance.candidate_drivers[0], instance.outcome
    call = ToolCall(id="1", name="correlate", arguments={"x": a, "y": b})
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=(call,)), ModelTurn(content="<answer>done</answer>")])
    result = run_agent(instance, client)
    assert result == "<answer>done</answer>"
    tool_message = client.calls[1][0][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "1"
    assert f"corr({a}, {b})" in tool_message["content"]


def test_replayed_tool_call_arguments_are_a_json_string_on_the_wire():
    """function.arguments must round-trip as a JSON string; chat-completions APIs reject a raw object."""
    instance = _instance()
    a, b = instance.candidate_drivers[0], instance.outcome
    call = ToolCall(id="1", name="correlate", arguments={"x": a, "y": b})
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=(call,)), ModelTurn(content="<answer>done</answer>")])
    run_agent(instance, client)
    assistant_message = client.calls[1][0][-2]
    wire_args = assistant_message["tool_calls"][0]["function"]["arguments"]
    assert isinstance(wire_args, str)
    assert json.loads(wire_args) == {"x": a, "y": b}


def test_unknown_tool_name_returns_error_without_crashing():
    instance = _instance()
    call = ToolCall(id="1", name="nonexistent_tool", arguments={})
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=(call,)), ModelTurn(content="<answer>ok</answer>")])
    result = run_agent(instance, client)
    assert result == "<answer>ok</answer>"
    assert "Error" in client.calls[1][0][-1]["content"]


def test_a_tool_call_that_raises_returns_an_error_string_without_crashing_the_loop():
    # A malformed/hallucinated argument shape (here: an extra kwarg no tool accepts) raises inside the
    # tool itself; the loop must turn that into a tool-result error, not propagate the exception.
    instance = _instance()
    call = ToolCall(id="1", name="preview", arguments={"n": 5, "unexpected_kwarg": True})
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=(call,)), ModelTurn(content="<answer>ok</answer>")])
    result = run_agent(instance, client)
    assert result == "<answer>ok</answer>"
    assert "Error" in client.calls[1][0][-1]["content"]


def test_a_client_side_parse_error_on_a_tool_call_is_reported_without_dispatch():
    # ToolCall.error is set by a client that could not parse the model's raw arguments (e.g. malformed
    # JSON); the loop must report it as a tool-result error and never attempt to dispatch it.
    instance = _instance()
    call = ToolCall(id="1", name="preview", arguments={}, error="malformed arguments JSON: boom")
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=(call,)), ModelTurn(content="<answer>ok</answer>")])
    result = run_agent(instance, client)
    assert result == "<answer>ok</answer>"
    assert "malformed arguments JSON" in client.calls[1][0][-1]["content"]


def test_last_turn_requests_force_final():
    instance = _instance()
    client = _ScriptedClient([ModelTurn(content="<answer>{}</answer>")])
    run_agent(instance, client, max_turns=1)
    assert client.calls[0][2] is True  # force_final on the only (last) turn


def test_final_turn_reminder_is_only_added_on_the_forced_final_turn():
    # A model mid-investigation can keep narrating "next I'd check X" instead of answering once tool
    # access is cut off (observed on real models); the explicit reminder tells it this is its last chance.
    instance = _instance()
    call = ToolCall(id="1", name="preview", arguments={})
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=(call,)), ModelTurn(content="<answer>ok</answer>")])
    run_agent(instance, client, max_turns=2)

    first_call_messages, _, first_force_final = client.calls[0]
    second_call_messages, _, second_force_final = client.calls[1]
    assert first_force_final is False
    assert not any(m.get("content") == _FINAL_TURN_REMINDER for m in first_call_messages)
    assert second_force_final is True
    assert second_call_messages[-1] == {"role": "user", "content": _FINAL_TURN_REMINDER}


def test_misbehaving_model_that_ignores_force_final_yields_empty_string():
    instance = _instance()
    call = ToolCall(id="1", name="preview", arguments={})
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=(call,))])  # keeps requesting tools
    assert run_agent(instance, client, max_turns=1) == ""


def test_default_max_turns_matches_module_constant():
    instance = _instance()
    client = _ScriptedClient([ModelTurn(content="ok")] * MAX_TURNS)
    run_agent(instance, client)
    assert len(client.calls) == 1  # stops at the first non-tool-call reply, well under MAX_TURNS

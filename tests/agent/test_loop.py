from causalagentbench import build_instance, generate_gated, sample
from causalagentbench.agent.loop import MAX_TURNS, ModelTurn, ToolCall, run_agent


def _instance():
    scm, _ = generate_gated(0)
    data = sample(scm, 200, seed=0)
    instance, _ = build_instance(scm, data, seed=0)
    return instance


class _ScriptedClient:
    """Pops one scripted ``ModelTurn`` per call, recording every call it received."""

    def __init__(self, turns: list[ModelTurn]):
        self._turns = list(turns)
        self.calls: list[tuple[list[dict], list[dict], bool]] = []

    def complete(self, messages, tools, *, force_final: bool) -> ModelTurn:
        self.calls.append((messages, tools, force_final))
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


def test_unknown_tool_name_returns_error_without_crashing():
    instance = _instance()
    call = ToolCall(id="1", name="nonexistent_tool", arguments={})
    client = _ScriptedClient([ModelTurn(content=None, tool_calls=(call,)), ModelTurn(content="<answer>ok</answer>")])
    result = run_agent(instance, client)
    assert result == "<answer>ok</answer>"
    assert "Error" in client.calls[1][0][-1]["content"]


def test_last_turn_requests_force_final():
    instance = _instance()
    client = _ScriptedClient([ModelTurn(content="<answer>{}</answer>")])
    run_agent(instance, client, max_turns=1)
    assert client.calls[0][2] is True  # force_final on the only (last) turn


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

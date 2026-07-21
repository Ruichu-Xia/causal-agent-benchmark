from types import SimpleNamespace

from causalagentbench.agent.clients import _parse_tool_call


def _raw(arguments: str, *, id_: str = "1", name: str = "correlate"):
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=arguments))


def test_parse_tool_call_decodes_well_formed_arguments():
    call = _parse_tool_call(_raw('{"x": "A", "y": "B"}'))
    assert call.id == "1"
    assert call.name == "correlate"
    assert call.arguments == {"x": "A", "y": "B"}
    assert call.error is None


def test_parse_tool_call_treats_empty_arguments_as_no_kwargs():
    call = _parse_tool_call(_raw(""))
    assert call.arguments == {}
    assert call.error is None


def test_parse_tool_call_reports_malformed_json_as_an_error_not_a_raised_exception():
    call = _parse_tool_call(_raw("{not valid json"))
    assert call.arguments == {}
    assert call.error is not None
    assert "malformed arguments JSON" in call.error

from types import SimpleNamespace

import pytest

from causalagentbench.agent.clients import _databricks_client, _parse_tool_call


@pytest.fixture(autouse=True)
def _clear_databricks_client_cache():
    _databricks_client.cache_clear()
    yield
    _databricks_client.cache_clear()


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


def test_databricks_client_requires_token(monkeypatch):
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.databricks.net")
    with pytest.raises(RuntimeError, match="DATABRICKS_TOKEN"):
        _databricks_client()


def test_databricks_client_requires_host_when_no_base_url_override(monkeypatch):
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-fake")
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_AI_GATEWAY_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABRICKS_HOST"):
        _databricks_client()


def test_databricks_client_computes_base_url_from_host(monkeypatch):
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-fake")
    monkeypatch.setenv("DATABRICKS_HOST", "https://adb-123.azuredatabricks.net/")
    monkeypatch.delenv("DATABRICKS_AI_GATEWAY_BASE_URL", raising=False)
    client = _databricks_client()
    assert str(client.base_url).rstrip("/") == "https://adb-123.azuredatabricks.net/ai-gateway/mlflow/v1"


def test_databricks_client_respects_base_url_override(monkeypatch):
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-fake")
    monkeypatch.setenv("DATABRICKS_AI_GATEWAY_BASE_URL", "https://custom.gateway/v1/")
    client = _databricks_client()
    assert str(client.base_url).rstrip("/") == "https://custom.gateway/v1"

from types import SimpleNamespace

import pytest

from causalagentbench.agent.clients import _databricks_client, _extract_max_tokens_ceiling, _parse_tool_call


@pytest.fixture(autouse=True)
def _clear_databricks_client_cache(monkeypatch):
    # Also stub load_env to a no-op: python-dotenv's load_dotenv(override=False) fills in any var a
    # test just monkeypatch.delenv'd from a real ambient .env, undoing the test's own setup - these
    # tests must control every DATABRICKS_* var themselves, independent of what's on disk.
    monkeypatch.setattr("causalagentbench.agent.clients.load_env", lambda: None)
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


def test_extract_max_tokens_ceiling_handles_observed_message_formats():
    # Two different real phrasings seen from this gateway for the same underlying rejection.
    llama = "max_new_tokens 32000 cannot be greater than max_output_tokens 8192."
    qwen = "max_tokens (32000) cannot exceed 25000. Please reduce the length of max output tokens generated."
    assert _extract_max_tokens_ceiling(llama, requested=32000) == 8192
    assert _extract_max_tokens_ceiling(qwen, requested=32000) == 25000


def test_extract_max_tokens_ceiling_returns_none_for_an_unrelated_message():
    assert _extract_max_tokens_ceiling("invalid API key", requested=32000) is None


def test_extract_max_tokens_ceiling_returns_none_when_no_number_is_smaller():
    # A number mentioned that's >= requested isn't a ceiling - guards against misreading an unrelated
    # rejection that happens to mention some other large number.
    assert _extract_max_tokens_ceiling("rate limit: retry after 60000ms", requested=32000) is None

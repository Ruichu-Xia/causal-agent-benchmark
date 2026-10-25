import sys
from types import SimpleNamespace

import pytest

from causalagentbench.agent.clients import (
    DatabricksClient,
    LiteLLMClient,
    _databricks_client,
    _extract_max_tokens_ceiling,
    _parse_tool_call,
    _require_litellm_params,
    _validate_litellm_params,
)


@pytest.fixture(autouse=True)
def _clear_databricks_client_cache(monkeypatch):
    """Isolate each test from the cached gateway client and from any ambient .env file."""
    # Also stub load_env to a no-op: python-dotenv's load_dotenv(override=False) fills in any var a
    # test just monkeypatch.delenv'd from a real ambient .env, undoing the test's own setup - these
    # tests must control every DATABRICKS_* var themselves, independent of what's on disk.
    monkeypatch.setattr("causalagentbench.agent.clients.load_env", lambda: None)
    _databricks_client.cache_clear()
    yield
    _databricks_client.cache_clear()


def _raw(arguments: str, *, id_: str = "1", name: str = "correlate"):
    """A wire-format tool call."""
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


def test_litellm_parameter_preflight_accepts_tool_capable_model():
    fake_litellm = SimpleNamespace(get_supported_openai_params=lambda model: ["tools", "tool_choice", "temperature"])

    _require_litellm_params(fake_litellm, "provider/model", {"tools", "tool_choice", "temperature"})


def test_litellm_parameter_preflight_rejects_missing_or_unknown_capability_metadata():
    missing_seed = SimpleNamespace(get_supported_openai_params=lambda model: ["tools", "tool_choice", "temperature"])
    with pytest.raises(RuntimeError, match="seed"):
        _require_litellm_params(missing_seed, "provider/model", {"tools", "tool_choice", "temperature", "seed"})

    unknown = SimpleNamespace(get_supported_openai_params=lambda model: None)
    with pytest.raises(RuntimeError, match="no parameter-support metadata"):
        _require_litellm_params(unknown, "provider/model", {"tools", "tool_choice"})


def _response(model: str = "provider/served-model"):
    """An OpenAI-shaped completion with plain text content and usage."""
    message = SimpleNamespace(content="hello", tool_calls=None)
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=4, cost=0.002)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage, model=model)


@pytest.fixture
def fake_litellm(monkeypatch):
    """Replace the ``litellm`` module with a recorder that supports only the tool parameters."""
    requests = []

    def completion(**kwargs):
        """Record the request and return a fixed response."""
        requests.append(kwargs)
        return _response()

    module = SimpleNamespace(completion=completion, get_supported_openai_params=lambda model: ["tools", "tool_choice"])
    monkeypatch.setitem(sys.modules, "litellm", module)
    _validate_litellm_params.cache_clear()
    yield requests
    _validate_litellm_params.cache_clear()


def test_litellm_client_uses_provider_sampling_defaults(fake_litellm):
    turn = LiteLLMClient(model="provider/model").complete([], [], force_final=False)
    [request] = fake_litellm
    assert "temperature" not in request and "seed" not in request
    assert request["tool_choice"] == "auto"
    assert (turn.content, turn.prompt_tokens, turn.completion_tokens, turn.cost_usd) == ("hello", 3, 4, 0.002)
    assert turn.model == "provider/served-model"


def test_litellm_client_sends_sampling_parameters_only_when_set(fake_litellm, monkeypatch):
    supported = ["tools", "tool_choice", "temperature", "seed"]
    monkeypatch.setattr(sys.modules["litellm"], "get_supported_openai_params", lambda model: supported)
    LiteLLMClient(model="provider/model", temperature=0.0, seed=3).complete([], [], force_final=True)
    [request] = fake_litellm
    assert request["temperature"] == 0.0 and request["seed"] == 3
    assert request["tool_choice"] == "none"


def test_capability_check_requires_temperature_and_seed_only_when_set(fake_litellm):
    LiteLLMClient(model="provider/model").validate_tool_capability()
    with pytest.raises(RuntimeError, match="temperature"):
        LiteLLMClient(model="provider/model", temperature=0.5).validate_tool_capability()
    with pytest.raises(RuntimeError, match="seed"):
        LiteLLMClient(model="provider/model", seed=1).validate_tool_capability()


def test_databricks_client_omits_temperature_by_default_and_reports_the_served_model(monkeypatch):
    pytest.importorskip("openai")
    requests = []

    def create(**kwargs):
        """Record the request and return a fixed response."""
        requests.append(kwargs)
        return _response("served")

    completions = SimpleNamespace(create=create)
    gateway = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr("causalagentbench.agent.clients._databricks_client", lambda: gateway)
    turn = DatabricksClient(model="gateway-model").complete([], [], force_final=False)
    DatabricksClient(model="gateway-model", temperature=0.2).complete([], [], force_final=False)
    assert "temperature" not in requests[0]
    assert requests[1]["temperature"] == 0.2
    assert turn.model == "served"

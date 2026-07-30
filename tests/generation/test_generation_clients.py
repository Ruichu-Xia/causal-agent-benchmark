"""Tests for ``generation/clients.py``.

Named ``test_generation_clients`` rather than ``test_clients`` to avoid colliding with
tests/agent/test_clients.py: this tree has no __init__.py files, so pytest's default import mode requires
every test file to have a globally unique basename.
"""

import pytest

from causalagentbench.generation.clients import _databricks_client


@pytest.fixture(autouse=True)
def _clear_databricks_client_cache(monkeypatch):
    # Also stub load_env to a no-op: python-dotenv's load_dotenv(override=False) fills in any var a
    # test just monkeypatch.delenv'd from a real ambient .env, undoing the test's own setup - these
    # tests must control every DATABRICKS_* var themselves, independent of what's on disk.
    monkeypatch.setattr("causalagentbench.generation.clients.load_env", lambda: None)
    _databricks_client.cache_clear()
    yield
    _databricks_client.cache_clear()


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

import json
import re

import networkx as nx
import pytest

from causalagentbench.agent.loop import ModelTurn, ToolCall

N_CONCEPTS = 40


@pytest.fixture
def causenet_graph() -> nx.DiGraph:
    """A synthetic CauseNet stand-in: concept ``c_i`` causes ``c_j`` for every ``i < j``.

    Any DAG embeds in it by naming nodes in topological order, and no relation can point against that
    order, so it names generated tasks without the real dump.
    """
    names = [f"c{i:02d}" for i in range(N_CONCEPTS)]
    return nx.DiGraph((names[i], names[j]) for i in range(N_CONCEPTS) for j in range(i + 1, N_CONCEPTS))


@pytest.fixture
def causenet_file(tmp_path, causenet_graph):
    """The synthetic CauseNet graph written in CauseNet's JSONL format."""
    path = tmp_path / "causenet.jsonl"
    lines = [
        json.dumps({"causal_relation": {"cause": {"concept": u}, "effect": {"concept": v}}, "support": 1})
        for u, v in causenet_graph.edges
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class FakeLiteLLMClient:
    """Stands in for ``LiteLLMClient``: one ``regress`` call, then a zero estimate for every candidate."""

    fail = False

    def __init__(self, model: str, request_timeout: float = 120.0) -> None:
        """Record the requested model id and request timeout."""
        self.model = model
        self.request_timeout = request_timeout

    def validate_tool_capability(self) -> None:
        """Accept every model, as a provider with full capability metadata would."""

    def complete(self, messages, tools, *, force_final):
        """Return the next scripted turn for the conversation so far."""
        if self.fail:
            raise RuntimeError("provider unavailable")
        candidates = re.search(r"^Candidate drivers: (.*)$", messages[1]["content"], re.MULTILINE).group(1).split(", ")
        usage = {"prompt_tokens": 100, "completion_tokens": 10, "cost_usd": 0.001, "model": "fake-served"}
        if not force_final and not any(m["role"] == "tool" for m in messages):
            call = ToolCall(id="1", name="regress", arguments={"treatment": candidates[0]})
            return ModelTurn(content=None, tool_calls=(call,), **usage)
        verdicts = [{"driver": c, "adjustment_set": [], "effect": 0.0} for c in candidates]
        return ModelTurn(content=f"<answer>{json.dumps({'verdicts': verdicts})}</answer>", **usage)


@pytest.fixture
def fake_llm(monkeypatch):
    """Route model evaluation through ``FakeLiteLLMClient`` and return the class for per-test tweaks."""
    monkeypatch.setattr("causalagentbench.agent.clients.LiteLLMClient", FakeLiteLLMClient)
    return FakeLiteLLMClient

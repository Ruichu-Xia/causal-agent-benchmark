"""Real ``ToolCallingClient`` implementations: any LiteLLM-supported model, or the Databricks AI Gateway.

Kept separate from ``loop.py`` so the loop stays importable (and testable with a stub) without the
``generation`` extra; provider SDKs are imported lazily, same pattern as ``generation.clients``.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from typing import Any

from causalagentbench.agent.loop import ModelTurn, ToolCall
from causalagentbench.generation.clients import load_env

_NUM_RETRIES = 6
_SYSTEM_PROMPT = (
    "You are a careful causal data scientist. Use the provided tools to inspect the data before "
    "answering; never guess an adjustment set or effect without checking it with a tool."
)


@dataclass(frozen=True)
class LiteLLMClient:
    """Default ``ToolCallingClient`` on LiteLLM. Pass ``model`` like ``"anthropic/claude-opus-4-8"``."""

    model: str
    temperature: float = 0.2

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, force_final: bool) -> ModelTurn:
        """Complete one turn with the configured model via LiteLLM."""
        import litellm

        load_env()
        litellm.drop_params = True
        full_messages = [{"role": "system", "content": _SYSTEM_PROMPT}, *messages]
        # _skip_mcp_handler: we pass plain function tools, never litellm's MCP gateway; skipping avoids an
        # eager import of litellm's proxy stack (fastapi/orjson/uvicorn/...), which isn't a dependency here.
        response = litellm.completion(
            model=self.model,
            messages=full_messages,
            tools=tools,
            tool_choice="none" if force_final else "auto",
            temperature=self.temperature,
            num_retries=_NUM_RETRIES,
            _skip_mcp_handler=True,
        )
        message = response.choices[0].message
        raw_calls = getattr(message, "tool_calls", None) or ()
        return ModelTurn(content=message.content, tool_calls=tuple(_parse_tool_call(c) for c in raw_calls))


def _parse_tool_call(raw: Any) -> ToolCall:
    """Parse one wire tool call; malformed arguments JSON becomes a ``ToolCall.error`` the loop can report."""
    try:
        arguments = json.loads(raw.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return ToolCall(id=raw.id, name=raw.function.name, arguments={}, error=f"malformed arguments JSON: {exc}")
    return ToolCall(id=raw.id, name=raw.function.name, arguments=arguments)


_GATEWAY_SUFFIX = "/ai-gateway/mlflow/v1"


@dataclass(frozen=True)
class DatabricksClient:
    """``ToolCallingClient`` over the Databricks AI Gateway.

    The gateway is OpenAI-compatible (native ``tools``/``tool_choice``, same wire shape as
    ``LiteLLMClient``), so this reuses ``_parse_tool_call`` as-is. ``model`` is the gateway-served model
    name, e.g. ``"databricks-claude-opus-4-7"``. Needs ``DATABRICKS_TOKEN`` and ``DATABRICKS_HOST`` in
    ``.env``; ``DATABRICKS_AI_GATEWAY_BASE_URL`` overrides the computed gateway URL if set.
    """

    model: str
    temperature: float = 0.2
    max_tokens: int = 4096

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, force_final: bool) -> ModelTurn:
        """Complete one turn with the configured model via the Databricks AI Gateway."""
        full_messages = [{"role": "system", "content": _SYSTEM_PROMPT}, *messages]
        response = _databricks_client().chat.completions.create(
            model=self.model,
            messages=full_messages,
            tools=tools,
            tool_choice="none" if force_final else "auto",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        message = response.choices[0].message
        raw_calls = getattr(message, "tool_calls", None) or ()
        return ModelTurn(content=message.content, tool_calls=tuple(_parse_tool_call(c) for c in raw_calls))


@functools.lru_cache(maxsize=1)
def _databricks_client() -> Any:
    """Build (once) the OpenAI SDK client pointed at the Databricks AI Gateway; token/host from ``.env``."""
    from openai import OpenAI

    load_env()
    token = os.environ.get("DATABRICKS_TOKEN")
    if not token:
        raise RuntimeError("DATABRICKS_TOKEN not found; add it to the gitignored repo-root .env")
    base_url = os.environ.get("DATABRICKS_AI_GATEWAY_BASE_URL")
    if not base_url:
        host = os.environ.get("DATABRICKS_HOST")
        if not host:
            raise RuntimeError("DATABRICKS_HOST not found; add it to the gitignored repo-root .env")
        base_url = f"{host.rstrip('/')}{_GATEWAY_SUFFIX}"
    return OpenAI(api_key=token, base_url=base_url.rstrip("/"), max_retries=_NUM_RETRIES)

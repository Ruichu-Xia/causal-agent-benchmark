"""The real ``ToolCallingClient``: any LiteLLM-supported model via native function calling.

Kept separate from ``loop.py`` so the loop stays importable (and testable with a stub) without the
``generation`` extra; ``litellm`` is imported lazily, same pattern as ``generation.clients``.
"""

from __future__ import annotations

import json
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

"""Real ``ToolCallingClient`` implementations: any LiteLLM-supported model, or the Databricks AI Gateway.

Kept separate from ``loop.py`` so the loop stays importable (and testable with a stub) without the
``generation`` extra; provider SDKs are imported lazily, same pattern as ``generation.clients``.
"""

from __future__ import annotations

import functools
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

from causalagentbench.agent.loop import ModelTurn, ToolCall
from causalagentbench.generation.clients import load_env

_LITELLM_DEFAULT_MAX_RETRIES = 2
_DATABRICKS_NUM_RETRIES = 6
_REQUIRED_TOOL_PARAMS = frozenset({"tools", "tool_choice", "temperature"})


@dataclass(frozen=True)
class LiteLLMClient:
    """Default ``ToolCallingClient`` on LiteLLM. Pass ``model`` like ``"anthropic/claude-opus-4-8"``."""

    model: str
    temperature: float = 0.0
    request_timeout: float = 120.0
    max_retries: int = _LITELLM_DEFAULT_MAX_RETRIES
    seed: int | None = 0
    openrouter_session_id: str | None = None

    def validate_tool_capability(self) -> None:
        """Fail before an evaluation if LiteLLM cannot verify the required request parameters."""
        load_env()
        if self.openrouter_session_id is not None and not self.model.startswith("openrouter/"):
            raise ValueError("openrouter_session_id requires an openrouter/ model ID")
        required = _REQUIRED_TOOL_PARAMS | ({"seed"} if self.seed is not None else set())
        _validate_litellm_params(self.model, tuple(sorted(required)))

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, force_final: bool) -> ModelTurn:
        """Complete one turn with the configured model via LiteLLM."""
        import litellm

        load_env()
        self.validate_tool_capability()
        # _skip_mcp_handler: we pass plain function tools, never litellm's MCP gateway; skipping avoids an
        # eager import of litellm's proxy stack (fastapi/orjson/uvicorn/...), which isn't a dependency here.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "none" if force_final else "auto",
            "temperature": self.temperature,
            "seed": self.seed,
            "num_retries": self.max_retries,
            "timeout": self.request_timeout,
            "drop_params": False,
            "_skip_mcp_handler": True,
        }
        if self.openrouter_session_id is not None:
            # OpenRouter uses this request-body field for sticky upstream-provider routing. LiteLLM passes
            # OpenRouter-specific body fields through ``extra_body`` rather than dropping them.
            kwargs["extra_body"] = {"session_id": self.openrouter_session_id}
        response = litellm.completion(
            **kwargs,
        )
        message = response.choices[0].message
        raw_calls = getattr(message, "tool_calls", None) or ()
        usage = response.usage
        cost_usd = _response_cost_usd(litellm, response, usage)
        return ModelTurn(
            content=message.content,
            tool_calls=tuple(_parse_tool_call(c) for c in raw_calls),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            cost_usd=cost_usd,
        )


@functools.cache
def _validate_litellm_params(model: str, required: tuple[str, ...]) -> None:
    """Verify that LiteLLM knows the model supports every parameter the benchmark requires."""
    import litellm

    _require_litellm_params(litellm, model, set(required))


def _require_litellm_params(litellm: Any, model: str, required: set[str]) -> None:
    """Raise a clear error when model capability metadata cannot prove a required parameter is supported."""
    try:
        supported = litellm.get_supported_openai_params(model=model)
    except Exception as exc:
        raise RuntimeError(f"could not verify LiteLLM parameter support for {model!r}: {exc}") from exc
    if supported is None:
        raise RuntimeError(f"LiteLLM has no parameter-support metadata for {model!r}; refusing an unverified eval")
    missing = sorted(required - set(supported))
    if missing:
        raise RuntimeError(f"{model!r} does not support required evaluation parameters: {', '.join(missing)}")


def _response_cost_usd(litellm: Any, response: Any, usage: Any) -> float | None:
    """Return LiteLLM's USD estimate for one completion, or ``None`` when it cannot price the response."""
    try:
        cost = getattr(usage, "cost", None)
        if cost is None:
            cost = litellm.completion_cost(completion_response=response)
        cost = float(cost)
    except (TypeError, ValueError, AttributeError, KeyError):
        return None
    except Exception:
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


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
    temperature: float = 0.0
    max_tokens: int = 32_000
    """Generous, not a real constraint: this is a hard external cutoff the model has no awareness of - it
    keeps generating (reasoning tokens count against the same budget as answer tokens) until either it's
    done or the API truncates it wherever it happens to be, with no chance to wrap up. The previous default
    (4096) was an arbitrary pick that turned out too small for an extended-thinking model's internal
    reasoning trace on a forced-final turn - confirmed by direct trace: cut off mid-``reasoning`` block,
    zero answer text, on a turn that needed ~4x that budget. 32000 was confirmed accepted for
    ``claude-opus-5`` (tested up to 128000), but the gateway enforces its own hard ceiling *per model* below
    this in some cases (observed: 8192 for ``meta-llama-3-1-8b-instruct``, 25000 for ``qwen35-122b-a10b``) -
    ``complete`` learns and caches the real ceiling on first rejection (``_max_tokens_ceiling``), so this
    field is really just a starting point/upper bound, not the value every model actually gets."""

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, force_final: bool) -> ModelTurn:
        """Complete one turn with the configured model via the Databricks AI Gateway."""
        import openai

        max_tokens = _max_tokens_ceiling.get(self.model, self.max_tokens)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "none" if force_final else "auto",
            "max_tokens": max_tokens,
        }
        try:
            response = _databricks_client().chat.completions.create(temperature=self.temperature, **kwargs)
        except openai.BadRequestError as exc:
            message_text = str(exc)
            if "temperature" in message_text:
                # Some gateway-served models (observed: claude-opus-4-8) reject "temperature" outright,
                # unlike LiteLLM's drop_params which handles this transparently - retry once without it.
                response = _databricks_client().chat.completions.create(**kwargs)
            elif (ceiling := _extract_max_tokens_ceiling(message_text, max_tokens)) is not None:
                _max_tokens_ceiling[self.model] = ceiling
                kwargs["max_tokens"] = ceiling
                response = _databricks_client().chat.completions.create(temperature=self.temperature, **kwargs)
            else:
                raise
        message = response.choices[0].message
        raw_calls = getattr(message, "tool_calls", None) or ()
        calls = tuple(_parse_tool_call(c) for c in raw_calls)
        usage = response.usage
        return ModelTurn(
            content=_extract_databricks_content(message.content),
            tool_calls=calls,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )


_max_tokens_ceiling: dict[str, int] = {}
"""Learned per-model output-token ceiling (gateway model name -> max accepted ``max_tokens``).

Populated lazily by ``DatabricksClient.complete`` on first rejection, not a hardcoded table - the gateway's
per-model ceilings aren't documented anywhere we can query up front, and a hardcoded table would silently
go stale as the model catalog changes. Once learned for a model, every subsequent call for that model goes
straight to the safe value instead of paying for a failed round trip first."""


def _extract_max_tokens_ceiling(message: str, requested: int) -> int | None:
    """Pull the real ceiling out of a gateway max-tokens rejection, or ``None`` if the message doesn't fit.

    Message wording varies by model/provider behind the gateway (observed: ``"max_new_tokens 32000 cannot
    be greater than max_output_tokens 8192"`` for one model, ``"max_tokens (32000) cannot exceed 25000"``
    for another) - rather than special-casing each phrasing, this takes every integer the message mentions
    and returns the smallest one below what was actually requested. Robust across the observed formats
    because both always state the rejected (larger) value alongside the ceiling (smaller) one; returns
    ``None`` if no number in the message is smaller than ``requested`` (not a max-tokens ceiling message).
    """
    smaller = [n for n in (int(m) for m in re.findall(r"\d+", message)) if n < requested]
    return min(smaller) if smaller else None


def _extract_databricks_content(content: Any) -> str | None:
    """Normalize a Databricks Gateway message's ``content`` field to plain text.

    Databricks-specific, not shared with ``LiteLLMClient``: usually already a string, but some
    gateway-served reasoning models (observed: gpt-oss-120b on this gateway) instead return a list
    mixing a "reasoning" (chain-of-thought) block with the actual "text" answer, which would otherwise
    silently break a downstream ``<answer>`` parse and corrupt the next turn's message-history replay.
    """
    if content is None or isinstance(content, str):
        return content
    parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
    return "".join(parts) or None


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
    return OpenAI(api_key=token, base_url=base_url.rstrip("/"), max_retries=_DATABRICKS_NUM_RETRIES)

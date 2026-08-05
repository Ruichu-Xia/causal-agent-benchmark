"""Provider-agnostic LLM seam, with a LiteLLM default and an optional Databricks AI Gateway path.

One ``LabelProposer`` seam serves both roles in the pipeline - the **generator** (proposes name triples)
and the **critic** (verifies them) - they differ only by prompt. ``LiteLLMProposer`` reaches any provider
by model string (key from ``.env``); ``DatabricksProposer`` talks to the Databricks AI Gateway directly
via the OpenAI SDK - the same gateway ``agent.clients.DatabricksClient`` uses for the agent side, kept as
fully separate code, not a shared helper (same isolation this project already keeps between the Databricks
and LiteLLM paths on the agent side - a Databricks quirk fix must never leak into the LiteLLM path, or
vice versa). Both are lazy-imported so this module stays importable without the optional ``generation``
extra. Transient errors (rate limits, timeouts) are retried with backoff.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Any, Protocol

#: Retries on transient LLM errors (rate limits, timeouts) - LiteLLM honors Retry-After (needs tenacity).
_NUM_RETRIES = 6


class LabelProposer(Protocol):
    """Maps a prompt to a model completion (any vendor, supplied by the caller)."""

    def propose(self, prompt: str) -> str:
        """Return the model completion for ``prompt``."""


@dataclass(frozen=True)
class LiteLLMProposer:
    """Default ``LabelProposer`` on LiteLLM. Pass ``model`` like ``"anthropic/claude-opus-4-8"``."""

    model: str
    temperature: float = 0.7
    max_tokens: int = 32_000
    """Sized for the whole naming payload, for the same reason ``DatabricksProposer`` sets it.

    One naming call emits three names *plus* a description for every node at once, so the response grows
    with graph size, and a truncated response is invalid JSON that fails the schema check and costs the
    whole SCM - silently, and only on the larger graphs, which biases a built dataset toward small ones.
    Left unset, the ceiling is whatever the provider defaults to: unbounded enough for OpenAI, but LiteLLM
    supplies 4096 for Anthropic models, which an 8-driver (28-node) scaffold was measured to overshoot."""

    def propose(self, prompt: str) -> str:
        """Complete ``prompt`` with the configured model via LiteLLM."""
        return _complete(self.model, prompt, self.temperature, self.max_tokens)


@functools.lru_cache(maxsize=1)
def load_env() -> None:
    """Load API keys from a local ``.env`` once (so users never export keys by hand)."""
    from dotenv import load_dotenv

    load_dotenv()


def _complete(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    """Single-turn completion via LiteLLM (lazy imports keep the extra optional).

    ``temperature`` is a preference: LiteLLM honors it where supported and drops it where not (e.g. gpt-5
    accepts only its default), via ``drop_params`` - which equally translates or drops ``max_tokens`` per
    provider (e.g. to ``max_completion_tokens`` for reasoning models). Transient errors are retried.
    """
    import litellm

    load_env()
    litellm.drop_params = True
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        num_retries=_NUM_RETRIES,
    )
    return response.choices[0].message.content or ""


@dataclass(frozen=True)
class DatabricksProposer:
    """``LabelProposer`` over the Databricks AI Gateway.

    The gateway is OpenAI-compatible; ``model`` is the gateway-served model name, e.g.
    ``"databricks-claude-opus-4-7"``. Needs ``DATABRICKS_TOKEN`` and ``DATABRICKS_HOST`` in ``.env``;
    ``DATABRICKS_AI_GATEWAY_BASE_URL`` overrides the computed gateway URL if set - same variables
    ``agent.clients.DatabricksClient`` reads, since both point at the same gateway.
    """

    model: str
    temperature: float = 0.7
    max_tokens: int = 32_000
    """Sized for the whole naming payload, with room to spare.

    One naming call must emit three names *plus* a description for every node at once, so the response
    grows with graph size - and a truncated response is not a partial naming, it is invalid JSON that fails
    the schema check and costs the entire SCM. The previous 4096 was measured to fail on exactly this: an
    8-driver scaffold has 28 nodes and needs 4113 completion tokens, overshooting the cap by 17 and dying
    mid-JSON with ``finish_reason='length'``. That failure is silent and *selective* - it only hits the
    larger graphs - so it quietly biased built datasets toward small SCMs rather than erroring loudly.
    Deliberately duplicated rather than shared with ``agent.clients.DatabricksClient``'s own budget: same
    gateway and same reasoning, but the two paths stay textually separate by this project's convention."""

    def propose(self, prompt: str) -> str:
        """Complete ``prompt`` with the configured model via the Databricks AI Gateway."""
        import openai

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }
        try:
            response = _databricks_client().chat.completions.create(temperature=self.temperature, **kwargs)
        except openai.BadRequestError as exc:
            # Some gateway-served models (observed: claude-opus-4-8) reject "temperature" outright, unlike
            # LiteLLM's drop_params which handles this transparently - retry once without it.
            if "temperature" not in str(exc):
                raise
            response = _databricks_client().chat.completions.create(**kwargs)
        return _extract_databricks_content(response.choices[0].message.content) or ""


def _extract_databricks_content(content: Any) -> str | None:
    """Normalize a Databricks Gateway message's ``content`` field to plain text.

    Usually already a string, but some gateway-served reasoning models (observed: gpt-oss-120b on this
    gateway) instead return a list mixing a "reasoning" (chain-of-thought) block with the actual "text"
    answer, which would otherwise silently corrupt the parsed naming JSON.
    """
    if content is None or isinstance(content, str):
        return content
    parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
    return "".join(parts) or None


_GATEWAY_SUFFIX = "/ai-gateway/mlflow/v1"


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

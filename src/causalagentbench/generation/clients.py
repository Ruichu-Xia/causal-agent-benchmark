"""Provider-agnostic LLM seam, with a LiteLLM default.

One ``LabelProposer`` seam serves both roles in the pipeline - the **generator** (proposes name triples)
and the **critic** (verifies them) - they differ only by prompt. LiteLLM reaches any provider by model
string (key from ``.env``); ``litellm`` is imported lazily so this module stays importable without the
optional ``generation`` extra. Transient errors (rate limits, timeouts) are retried with backoff.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Protocol

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

    def propose(self, prompt: str) -> str:
        """Complete ``prompt`` with the configured model via LiteLLM."""
        return _complete(self.model, prompt, self.temperature)


@functools.lru_cache(maxsize=1)
def load_env() -> None:
    """Load API keys from a local ``.env`` once (so users never export keys by hand)."""
    from dotenv import load_dotenv

    load_dotenv()


def _complete(model: str, prompt: str, temperature: float) -> str:
    """Single-turn completion via LiteLLM (lazy imports keep the extra optional).

    ``temperature`` is a preference: LiteLLM honors it where supported and drops it where not (e.g. gpt-5
    accepts only its default), via ``drop_params``. Transient errors are retried with backoff.
    """
    import litellm

    load_env()
    litellm.drop_params = True
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        num_retries=_NUM_RETRIES,
    )
    return response.choices[0].message.content or ""

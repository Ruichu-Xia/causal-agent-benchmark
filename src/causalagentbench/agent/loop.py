"""A provider-agnostic tool-calling loop over the causal toolset.

``ToolCallingClient`` abstracts one model turn (messages + tool schemas in, a text reply or tool calls
out) behind a tiny ``Protocol``, so ``run_agent`` is fully testable with a stub - no network, no API key -
the same seam pattern as ``generation.clients.LabelProposer``. ``LiteLLMClient`` is the real
implementation, reaching any provider LiteLLM supports via native function calling.

The agent never sees the raw data table: it must call ``preview``/``correlate``/``regress`` to learn
anything about it, which is the point of a tool-using agent rather than a single-shot prompt.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from causalagentbench.agent.tools import CausalTools
from causalagentbench.task.instance import Instance

MAX_TURNS = 20
"""Safety valve: tool-call turns before forcing a final, tool-free answer."""

_FINAL_TURN_REMINDER = (
    "This is your final turn - no further tool calls are possible. You must give your complete final "
    "answer now, as a single <answer>...</answer> JSON block per the instructions above. Your reply has a "
    "generous but finite token budget shared between any reasoning and the answer itself - reach the "
    "<answer> block directly rather than re-deriving everything from scratch, or it may be cut off before "
    "you write it."
)
"""Told to the model only on the forced-final turn: without the first sentence, a model mid-investigation
may keep narrating what it would check next instead of committing to an answer (observed on real models).
The budget clause is a second, independent problem, also observed on a real model (an extended-thinking
model can spend its entire reply on internal reasoning and get cut off before writing any answer text) -
the model has no innate awareness that a token cutoff exists at all, so this is the only way to tell it."""


@dataclass(frozen=True)
class ToolCall:
    """One model-requested tool invocation."""

    id: str
    name: str
    arguments: Mapping[str, Any]
    error: str | None = None  # set by a client that could not parse this call's arguments; skips dispatch


@dataclass(frozen=True)
class ModelTurn:
    """One model response: either a final text answer, or tool calls to execute."""

    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()


class ToolCallingClient(Protocol):
    """One model turn: message history + tool schemas in, a ``ModelTurn`` out."""

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, force_final: bool) -> ModelTurn:
        """Return the model's next turn. If ``force_final``, no tool calls may be returned."""


_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "preview",
            "description": "See the first n rows of the data table (max 20), to sanity-check scale and format.",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer", "description": "number of rows, 1-20", "default": 5}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correlate",
            "description": (
                "Partial correlation between two columns given a conditioning set "
                "(the conditioning set may be empty for the marginal correlation)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                    "given": {"type": "array", "items": {"type": "string"}, "default": []},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "regress",
            "description": (
                "OLS coefficient of a treatment column on the fixed outcome, adjusting for a given set of "
                "columns - the total effect estimate under that adjustment set."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "treatment": {"type": "string"},
                    "adjust": {"type": "array", "items": {"type": "string"}, "default": []},
                },
                "required": ["treatment"],
            },
        },
    },
)


def _dispatch(tools: CausalTools) -> dict[str, Callable[..., str]]:
    """Return the name -> tool-method mapping for executing a model's tool calls."""
    return {"preview": tools.preview, "correlate": tools.correlate, "regress": tools.regress}


def run_agent(instance: Instance, client: ToolCallingClient, *, max_turns: int = MAX_TURNS) -> str:
    """Run the tool-calling loop for one instance; return the model's final reply text.

    The initial message is only ``instance.prompt`` - never the raw table - so every fact the agent uses
    about the data comes from a tool call. Returns whatever text the model last produced (possibly empty
    on a degenerate model), which ``score_output``/``score_answer`` will grade as a parse failure if it
    carries no ``<answer>`` block.
    """
    tools = CausalTools(instance)
    dispatch = _dispatch(tools)
    messages: list[dict[str, Any]] = [{"role": "user", "content": instance.prompt}]

    for turn in range(max_turns):
        force_final = turn == max_turns - 1
        if force_final:
            messages.append({"role": "user", "content": _FINAL_TURN_REMINDER})
        reply = client.complete(messages, list(_TOOL_SCHEMAS), force_final=force_final)
        if not reply.tool_calls:
            return reply.content or ""

        messages.append({"role": "assistant", "content": reply.content, "tool_calls": _as_wire(reply.tool_calls)})
        for call in reply.tool_calls:
            result = _execute(dispatch, call)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # Exhausted max_turns on a tool-call turn; the loop's last force_final request should have produced
    # text, but guard against a model that ignores force_final and keeps requesting tools.
    return ""


def _execute(dispatch: dict[str, Callable[..., str]], call: ToolCall) -> str:
    """Run one tool call, turning any failure into an error string the model can read and retry from.

    Covers three failure modes uniformly: a client-side arguments-parse failure (``call.error``), an
    unknown tool name, and any exception the tool itself raises (a malformed/adversarial argument shape a
    model can produce - e.g. an extra kwarg, or a ``given``/``adjust`` set that makes the underlying
    computation singular) - none of these should ever crash a batch eval over an unknown-quality model.
    """
    if call.error:
        return f"Error: {call.error}"
    handler = dispatch.get(call.name)
    if handler is None:
        return f"Error: unknown tool {call.name!r}"
    try:
        return handler(**call.arguments)
    except Exception as exc:  # deliberately broad: any tool-call shape a model can produce must degrade to text
        return f"Error: {exc}"


def _as_wire(tool_calls: tuple[ToolCall, ...]) -> list[dict[str, Any]]:
    """Render ``ToolCall``s back into the assistant-message wire shape expected by chat APIs.

    ``function.arguments`` must be a JSON-encoded string on the wire (chat-completions APIs reject a raw
    object), even though ``ToolCall.arguments`` is the already-parsed ``Mapping`` for our own dispatch.
    """
    return [
        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(dict(c.arguments))}}
        for c in tool_calls
    ]

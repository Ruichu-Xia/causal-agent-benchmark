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
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from causalagentbench.agent.tools import CausalTools
from causalagentbench.task.answer import AnswerParseError, parse_answer_for_instance
from causalagentbench.task.instance import Instance

MAX_TURNS = 20
"""Safety valve: tool-call turns before forcing a final, tool-free answer."""

_SYSTEM_PROMPT = (
    "You are a careful causal data scientist. Dataset values are not embedded in the messages: call the "
    "provided tools to inspect them, and each tool result will appear in the following conversation turn. "
    "Do not invent tool results or give an <answer> block until you are ready to conclude. Never guess an "
    "adjustment set or effect without checking it with a tool."
)

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

_FORMAT_REPAIR_REMINDER = (
    "Your previous reply was not a valid final answer. Do not call any tools. Reply now with only one "
    "<answer>...</answer> block containing JSON that exactly follows the requested verdicts and ranking shape."
)
"""Sent once after a tool-free reply fails the public answer contract.

This is deliberately a formatting repair rather than a correctness retry: it can see the original prompt and
prior tool results, but never the hidden SCM or score. One bounded, tool-free retry prevents otherwise usable
models from turning ordinary chatty prose into an immediate parse failure without concealing genuine failures.
"""


@dataclass(frozen=True)
class ToolCall:
    """One model-requested tool invocation."""

    id: str
    name: str
    arguments: Mapping[str, Any]
    error: str | None = None  # set by a client that could not parse this call's arguments; skips dispatch


@dataclass(frozen=True)
class ModelTurn:
    """One model response: either a final text answer, or tool calls to execute.

    ``prompt_tokens``/``completion_tokens`` are observability, not scoring input - ``None`` when a client
    doesn't expose usage (not every provider populates it); present for both ``LiteLLMClient`` and
    ``DatabricksClient``, since both providers return standard OpenAI-shaped ``usage``.
    """

    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class TurnTrace:
    """Observability record for one completed turn - not scored, only measured.

    ``tool_calls`` is every tool name requested this turn, in order (repeats allowed - a model can call
    ``correlate`` a dozen times in one turn); a raw tuple, not a count, so a consumer can build whatever
    aggregate it wants (a histogram, a total, per-tool breakdowns) without this module guessing which one
    matters. Per-turn tool-call counts and turn duration are real but not directly comparable across models
    on their own: models differ in *batching strategy* (one model spreads an investigation across many
    single-call turns, another batches the same total work into few many-call turns), so a fair cross-model
    comparison sums these into per-instance totals, not per-turn averages - see the consumer, not this
    module, for that aggregation.
    """

    turn: int
    duration_seconds: float
    tool_calls: tuple[str, ...]
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None = None


@dataclass(frozen=True)
class TurnTranscript:
    """An application-level model request and its response, suitable for opt-in JSONL capture.

    ``messages`` and ``tools`` are snapshots of the exact values supplied to ``client.complete``. A
    transcript is only constructed when requested, since it includes prompts, tool results, and raw model
    text that callers may not want to retain by default. ``error`` is populated when the client raised
    before returning a response; in that case the response fields remain empty.
    """

    turn: int
    duration_seconds: float
    force_final: bool
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    error: str | None = None


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
                "(the conditioning set may be empty for the marginal correlation). Returns sample size "
                "and an approximate 95% confidence interval."
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
                "columns - the total effect estimate under that adjustment set. Returns sample size, standard "
                "error, and an approximate 95% confidence interval."
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
    {
        "type": "function",
        "function": {
            "name": "discover",
            "description": (
                "Run the PC causal-discovery algorithm over every column and return the CPDAG it recovers. "
                "Reports edges it could orient ('A -> B') separately from edges whose direction the data "
                "does not determine ('A -- B'). PC is not exact at finite sample size: it can miss a real "
                "edge or orient one wrongly, so treat the output as evidence to check, not as ground truth."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "alpha": {
                        "type": "number",
                        "description": "significance level for the independence tests, between 0 and 1",
                        "default": 0.01,
                    }
                },
            },
        },
    },
)


def _dispatch(tools: CausalTools) -> dict[str, Callable[..., str]]:
    """Return the name -> tool-method mapping for executing a model's tool calls."""
    return {
        "preview": tools.preview,
        "correlate": tools.correlate,
        "regress": tools.regress,
        "discover": tools.discover,
    }


def run_agent(
    instance: Instance,
    client: ToolCallingClient,
    *,
    max_turns: int = MAX_TURNS,
    on_turn: Callable[[TurnTrace], None] | None = None,
    on_transcript: Callable[[TurnTranscript], None] | None = None,
) -> str:
    """Run the tool-calling loop for one instance; return the model's final reply text.

    The initial request contains fixed system instructions plus ``instance.prompt`` - never the raw table -
    so every fact the agent uses about the data comes from a tool call. Returns whatever text the model
    last produced (possibly empty on a degenerate model). One tool-free reply that fails the public answer
    contract receives a single tool-free formatting repair; a second malformed reply is returned for normal
    scoring as a parse failure.

    ``on_turn``, if given, is called with a ``TurnTrace`` right after each ``client.complete()`` call
    returns - optional so this stays free for every caller that doesn't need turn-level observability.
    ``on_transcript`` is an opt-in companion that receives the complete application-level request and
    response for every model call, including client errors.
    """
    tools = CausalTools(instance)
    dispatch = _dispatch(tools)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": instance.prompt},
    ]

    def complete_turn(turn: int, *, force_final: bool) -> ModelTurn:
        """Make one model call and emit the optional trace/transcript records."""
        turn_start = time.monotonic()
        transcript_messages = tuple(deepcopy(messages)) if on_transcript is not None else ()
        transcript_tools = tuple(deepcopy(_TOOL_SCHEMAS)) if on_transcript is not None else ()
        try:
            reply = client.complete(messages, list(_TOOL_SCHEMAS), force_final=force_final)
        except Exception as exc:
            if on_transcript is not None:
                on_transcript(
                    TurnTranscript(
                        turn=turn,
                        duration_seconds=time.monotonic() - turn_start,
                        force_final=force_final,
                        messages=transcript_messages,
                        tools=transcript_tools,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            raise
        duration_seconds = time.monotonic() - turn_start
        if on_turn is not None:
            on_turn(
                TurnTrace(
                    turn=turn,
                    duration_seconds=duration_seconds,
                    tool_calls=tuple(c.name for c in reply.tool_calls),
                    prompt_tokens=reply.prompt_tokens,
                    completion_tokens=reply.completion_tokens,
                    cost_usd=reply.cost_usd,
                )
            )
        if on_transcript is not None:
            on_transcript(
                TurnTranscript(
                    turn=turn,
                    duration_seconds=duration_seconds,
                    force_final=force_final,
                    messages=transcript_messages,
                    tools=transcript_tools,
                    content=reply.content,
                    tool_calls=reply.tool_calls,
                    prompt_tokens=reply.prompt_tokens,
                    completion_tokens=reply.completion_tokens,
                    cost_usd=reply.cost_usd,
                )
            )
        return reply

    for turn in range(max_turns):
        force_final = turn == max_turns - 1
        if force_final:
            messages.append({"role": "user", "content": _FINAL_TURN_REMINDER})
        reply = complete_turn(turn, force_final=force_final)
        if not reply.tool_calls:
            if _is_valid_final(reply.content, instance.candidate_drivers, instance.columns, instance.outcome):
                return reply.content or ""
            messages.append({"role": "assistant", "content": reply.content or ""})
            messages.append({"role": "user", "content": _FORMAT_REPAIR_REMINDER})
            repaired = complete_turn(turn + 1, force_final=True)
            return repaired.content or ""

        messages.append({"role": "assistant", "content": reply.content, "tool_calls": _as_wire(reply.tool_calls)})
        for call in reply.tool_calls:
            result = _execute(dispatch, call)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # Exhausted max_turns on a tool-call turn; the loop's last force_final request should have produced
    # text, but guard against a model that ignores force_final and keeps requesting tools.
    return ""


def _is_valid_final(
    content: str | None,
    candidate_drivers: tuple[str, ...],
    columns: tuple[str, ...],
    outcome: str,
) -> bool:
    """Return whether ``content`` meets the public answer contract, without checking hidden correctness."""
    try:
        parse_answer_for_instance(content or "", candidate_drivers, columns=columns, outcome=outcome)
    except AnswerParseError:
        return False
    return True


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

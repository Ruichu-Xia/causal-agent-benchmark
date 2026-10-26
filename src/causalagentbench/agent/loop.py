"""A bounded, provider-independent loop over the benchmark's statistical tools.

The model receives a prompt and tool schemas. Tool results are appended to its history until submission
or budget exhaustion. One tool-free format repair is allowed; hidden scores never enter the conversation.
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

PROTOCOL_VERSION = "lean-v2"
"""Names the tools, output formats and budgets below; recorded in every results header.

v2 dropped ``preview``, shortened tool outputs and cut the budgets from 100 calls / 25 turns. In the v1
pilot, the number of calls did not predict error for either model, while re-sent history and per-turn
reasoning drove cost, so the cut targets turns."""

MAX_TOOL_CALLS = 30
"""Maximum executed tool attempts per task, including invalid requests."""

MAX_TURNS = 7
"""Maximum model replies, the last of which must be the answer: six investigation turns plus the answer.

Excludes the optional format repair. The PC-then-regress recipe needs two investigation turns (discover,
then all regressions in parallel), so six leaves room to check and revise."""

MAX_TOKEN_BUDGET = 600_000
"""Prompt-plus-completion token backstop; enforced when the provider reports usage."""

_SYSTEM_PROMPT = (
    "You are a careful causal data scientist. Dataset values are not in the messages - call the provided "
    "tools to inspect them. Never invent a tool result, and never report an adjustment set or effect you "
    "have not checked with a tool. Give an <answer> block only when you are ready to conclude. "
    "You have two budgets: tool calls and turns. A reply spends one turn however many calls it carries, so "
    "batch calls that do not depend on each other's results. Both remaining counts follow each batch of "
    "results; when either runs out you must answer immediately, from whatever you have."
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

_BUDGET_EXHAUSTED_RESULT = "Error: tool-call budget exhausted - this call was not executed. Give your final answer now."
"""Returned in place of a real result for calls a turn requested beyond the remaining budget.

The budget is enforced exactly rather than letting a turn overshoot it: a model that batches 14 calls per
turn must not get to run 13 calls past a limit that stops a one-call-per-turn model dead, or the budget
stops meaning the same amount of investigation per model - which is the entire reason for counting calls
instead of turns."""


def _budget_notice(calls_used: int, call_limit: int, turns_used: int, turn_limit: int) -> str:
    """Render both running budgets, appended to the last tool result of each turn.

    Continuous rather than only at the cutoff: a model told what remains at every step can prioritise what
    is left to check, where one told nothing until the forced-final turn is simply interrupted
    mid-investigation. Both numbers appear together because either can end the run, and because their
    ratio is the thing the agent should act on - turns running out faster than calls means it should be
    batching more per turn.

    Appended to an existing tool result rather than sent as its own message so it costs no extra turn and
    lands exactly where the model reads before choosing its next action. Earlier notices stay in the
    replayed history, which leaves an honest record of the burn rate.
    """
    return (
        f"[budget: {call_limit - calls_used} of {call_limit} tool calls remaining, "
        f"{turn_limit - turns_used} of {turn_limit} turns remaining]"
    )


_FORMAT_REPAIR_REMINDER = (
    "Your previous reply was not a valid final answer. Do not call any tools. Reply now with only one "
    "<answer>...</answer> block containing JSON that exactly follows the requested verdicts shape."
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
    model: str | None = None  # the model identifier the provider reported serving
    cached_tokens: int | None = None  # prompt tokens served from the provider's cache
    reasoning_tokens: int | None = None  # completion tokens spent on hidden reasoning


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
    model: str | None = None
    budget_used: int = 0
    """Tool calls spent *before* this turn - the budget state the model saw when it chose these calls.

    Deliberately the pre-turn figure, not the post-turn one: ``on_turn`` fires the moment the model
    replies, before any of this turn's calls have run, and the question a trace answers is what the model
    knew when it acted. Post-hoc, ``budget_used + len(tool_calls)`` recovers the post-turn total."""
    budget_limit: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None


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
    model: str | None = None
    error: str | None = None


class ToolCallingClient(Protocol):
    """One model turn: message history + tool schemas in, a ``ModelTurn`` out."""

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, force_final: bool) -> ModelTurn:
        """Return the model's next turn. If ``force_final``, no tool calls may be returned."""


_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "correlate",
            "description": (
                "Partial correlation between two columns given a conditioning set "
                "(the conditioning set may be empty for the marginal correlation), with an approximate 95% "
                "confidence interval."
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
                "columns - the total effect estimate under that adjustment set - with its standard error."
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
        "correlate": tools.correlate,
        "regress": tools.regress,
        "discover": tools.discover,
    }


def run_agent(
    instance: Instance,
    client: ToolCallingClient,
    *,
    max_tool_calls: int = MAX_TOOL_CALLS,
    max_turns: int = MAX_TURNS,
    max_token_budget: int | None = MAX_TOKEN_BUDGET,
    on_turn: Callable[[TurnTrace], None] | None = None,
    on_transcript: Callable[[TurnTranscript], None] | None = None,
) -> str:
    """Run the tool-calling loop for one instance; return the model's final reply text.

    The initial request contains fixed system instructions plus ``instance.prompt`` - never the raw table -
    so every fact the agent uses about the data comes from a tool call. Returns whatever text the model
    last produced (possibly empty on a degenerate model). One tool-free reply that fails the public answer
    contract receives a single tool-free formatting repair; a second malformed reply is returned for normal
    scoring as a parse failure.

    The answer is forced as soon as any of ``max_tool_calls`` (investigation), ``max_turns`` (round
    trips, and so cost) or ``max_token_budget`` (backstop) is reached. The first two are reported to the
    model after every batch of results so it can plan against them; the third is not, being set not to
    bind. Calls requested beyond the call budget are refused rather than executed - see ``MAX_TOOL_CALLS``
    and ``_BUDGET_EXHAUSTED_RESULT``. ``max_token_budget=None`` disables the backstop.

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
    tool_calls_used = 0
    tokens_used = 0

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
        nonlocal tokens_used
        tokens_used += (reply.prompt_tokens or 0) + (reply.completion_tokens or 0)
        if on_turn is not None:
            on_turn(
                TurnTrace(
                    turn=turn,
                    duration_seconds=duration_seconds,
                    tool_calls=tuple(c.name for c in reply.tool_calls),
                    prompt_tokens=reply.prompt_tokens,
                    completion_tokens=reply.completion_tokens,
                    cost_usd=reply.cost_usd,
                    model=reply.model,
                    budget_used=tool_calls_used,
                    budget_limit=max_tool_calls,
                    cached_tokens=reply.cached_tokens,
                    reasoning_tokens=reply.reasoning_tokens,
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
                    model=reply.model,
                )
            )
        return reply

    for turn in range(max_turns):
        # Any budget running out ends the investigation; the two the agent was told about bind first.
        over_token_budget = max_token_budget is not None and tokens_used >= max_token_budget
        force_final = tool_calls_used >= max_tool_calls or turn == max_turns - 1 or over_token_budget
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
        for index, call in enumerate(reply.tool_calls):
            if tool_calls_used >= max_tool_calls:
                result = _BUDGET_EXHAUSTED_RESULT
            else:
                result = _execute(dispatch, call)
                tool_calls_used += 1
            if index == len(reply.tool_calls) - 1:
                notice = _budget_notice(tool_calls_used, max_tool_calls, turn + 1, max_turns)
                result = f"{result}\n\n{notice}"
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

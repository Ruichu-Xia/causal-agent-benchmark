"""INTERNAL, not for release: evaluate the tool-using agent against Databricks-served models.

Not part of the public eval path - ``scripts/run_eval.py`` (LiteLLM) is the documented, supported one;
release users won't use Databricks models at all. This script exists only so internal testing against
Databricks-hosted models has a reusable entry point. Do not reference this from README or any user-facing
doc; do not add a Databricks option to the public ``run_eval.py`` to replace it.

Deliberately not sharing code with ``run_eval.py`` beyond the library layer (``run_evaluation``,
``aggregate_by_alignment``, ...) - same isolation this project already keeps between the Databricks and
LiteLLM client paths (see ``agent/clients.py``).

Default ``--dataset`` is the sibling internal script's output (``scripts/internal/build_databricks_testset.py``
-> ``outputs/databricks_internal``); default ``--models`` is a representative spread of size/provider from
the current Databricks gateway catalog that's confirmed to actually work through this gateway's tool-
calling shim. Excluded, with a confirmed reason each (not guessed): embedding-only models (not chat
models); gpt-oss (known reasoning-content-block bug, see ``generation/clients.py``'s
``_extract_databricks_content``); ``llama-4-maverick`` (Llama's native chat-template role tokens -
``ipython``, ``assistant`` - leak as literal text into ``message.content`` instead of being stripped);
``gemma-3-12b`` (Gemma's native function-call text syntax, e.g. ``[correlate(x="X2"...)]``, leaks into
``content`` instead of being translated into the structured ``tool_calls`` field). Both of the last two
were verified directly by tracing a real instance's raw completion, not assumed - and neither is fixed by
raising any budget, since the failure is a format-translation gap, not a resource one.

``claude-opus-5`` was previously excluded from this list on suspicion of a similar unfixable issue
("goes silent" after a growing number of parallel tool calls) - direct tracing found this was actually
two separate, fixable turn-budget problems, not a model/gateway quirk: (1) it genuinely needs more
investigative turns than the other models here (observed: 12, not fixed by more turns alone at the old
``DatabricksClient.max_tokens=4096`` - see below); (2) on the forced-final turn, a growing accumulated
context led it to spend its entire reply on internal reasoning tokens and get cut off (``finish_reason``:
``length``) before writing any answer text, since it has no innate awareness that a token cutoff exists at
all - fixed by raising ``DatabricksClient.max_tokens`` (``agent/clients.py``) and telling it about the
budget on the forced-final turn (``agent/loop.py``'s ``_FINAL_TURN_REMINDER``). Both fixes are provider/
client-level, not internal-script-only, so this is now re-included under the standard tool-call budget.
Needs ``DATABRICKS_TOKEN`` + ``DATABRICKS_HOST`` (or
``DATABRICKS_AI_GATEWAY_BASE_URL``) in ``.env``.

Writes ``--out`` incrementally at three grains, all flushed immediately, all tagged by a ``"kind"`` field
so a reader can tell them apart: one ``"kind": "turn"`` line per model round trip (tool calls requested,
prompt/completion tokens, wall time - raw observability for drilling into *why* one instance was slow, the
same trace-level detail this project reconstructed by hand with one-off scripts before this existed), one
``"kind": "instance"`` line per instance rolling those turns into strategy-agnostic totals (total turns,
total tool calls, a tool-name histogram, total tokens - not per-turn averages, since models differ in
batching strategy: one spreads an investigation across many single-call turns, another batches the same
work into few many-call turns, so only the per-instance total is fairly comparable across models), and one
``"kind": "model_summary"``/``"model_error"`` line once a whole model finishes or fails. The finer grains
aren't optional extras - a model that's killed, hangs, or errors partway through its instances (each up to
the ``--max-tool-calls`` budget against a real, sometimes very slow model - opus-5 observed taking 10-25
minutes per instance) would otherwise lose every instance already scored, even though the work (and money)
to produce them was already spent. Console output is flushed the same way, since Python fully buffers
stdout when it isn't a terminal (piped to a log file, run in the background) - without an explicit flush
nothing appears until the whole process exits, making progress unobservable on a long run.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TextIO

from causalagentbench.agent.clients import DatabricksClient
from causalagentbench.agent.eval import (
    DriverCell,
    NamingResult,
    driver_cells,
    identifiability_f1,
    naming_table,
    run_evaluation,
)
from causalagentbench.agent.loop import (
    MAX_TOKEN_BUDGET,
    MAX_TOOL_CALLS,
    MAX_TURNS,
    ToolCall,
    TurnTrace,
    TurnTranscript,
)
from causalagentbench.generation.clients import load_env
from causalagentbench.scoring.discovery_baseline import mechanical_identifiability_f1
from causalagentbench.scoring.score import AnswerScore, reference_scores
from causalagentbench.task.dataset import load_dataset

#: No default dataset, deliberately. It used to be ``outputs/databricks_internal`` - the pre-firewall pilot
#: whose naming condition was identifiable from word choice alone on 9 of 10 instances, making every
#: alignment number measured on it uninterpretable. Running this script bare therefore re-evaluated a known
#: contaminated set *and* overwrote the previous results file with the new scores. Naming the dataset is now
#: mandatory, and results default beside it, so neither can happen by omission.
#: One representative model per size/provider tier, restricted to models confirmed to work through this
#: gateway's tool-calling shim (see the module docstring for what was excluded and why). Served endpoint
#: names need the "databricks-" prefix (Databricks Foundation Model APIs convention, confirmed empirically
#: - the catalog names alone 404 as "does not exist").
_DEFAULT_MODELS = (
    "databricks-claude-opus-5",  # Anthropic, flagship
    "databricks-claude-haiku-4-5",  # Anthropic, small/fast
    "databricks-qwen35-122b-a10b",  # Alibaba, large MoE
    "databricks-meta-llama-3-1-8b-instruct",  # Meta, small
)


def _print(*args: object) -> None:
    """``print`` that flushes immediately - see the module docstring on why this matters here."""
    print(*args, flush=True)


def _print_scores(model: str, n_drivers: list[int], scores: list[AnswerScore]) -> None:
    _print(f"model={model}  instances={len(scores)}")
    _print("-" * 64)
    for i, (n, score) in enumerate(zip(n_drivers, scores, strict=True)):
        level = "name-blind" if score.alignment is None else f"alpha={score.alignment:+.1f}"
        _print(
            f"  #{i:<3} n_drivers={n:<3} {level:<12} ident={score.identification:.3f} "
            f"est={score.estimation:.3f} overall={score.overall:.3f} "
            f"tau={score.ranking_tau:+.3f} parse={score.parse_ok}"
        )
    _print()


def _print_cells(cells: list[DriverCell]) -> None:
    """The experiment table: naming x identifiability, at driver grain.

    The ``identifiable`` rows are the placebo - the data already settles those, so naming should not move
    them. A nuisance effect travelling with the skin would move both halves; a real semantic effect can
    only move the unidentifiable half.
    """
    _print("-" * 64)
    _print("  driver cells (n counts drivers, not instances)")
    for cell in cells:
        level = "name-blind" if cell.alignment is None else f"alpha={cell.alignment:+.1f}"
        kind = "identifiable" if cell.identifiable else "UNidentifiable"
        _print(f"  {kind:<15}{level:<12} n={cell.n:<4} ident={cell.identification:.3f} est={cell.estimation:.3f}")
    _print()


def _print_headline(
    model: str,
    scores: list[AnswerScore],
    references: dict[str, float],
    pc_f1: float | None,
) -> None:
    """The leaderboard number, then the two contrasts that are the actual result.

    Intervals are over SCMs, never skins or drivers: the three skins of one SCM share every structural
    parameter, and treating them as independent draws shrinks the interval by roughly sqrt(3 x drivers).
    """
    parse_rate = sum(s.parse_ok for s in scores) / len(scores) if scores else float("nan")
    overall = sum(s.overall for s in scores) / len(scores) if scores else float("nan")
    f1 = identifiability_f1(scores)
    _print("-" * 64)
    _print(f"  [{model}] overall = {overall:.3f}   parse_rate = {parse_rate:.2f}")
    for name, value in references.items():
        _print(f"      vs {name:24s} {value:.3f}")
    mechanical = "unavailable (install the discovery extra)" if pc_f1 is None else f"{pc_f1:.3f}"
    _print(
        f"      identifiability F1 = {f1.f1:.3f} (P {f1.precision:.3f} / R {f1.recall:.3f}, "
        f"n_pos={f1.n_positive})   mechanical PC baseline = {mechanical}"
    )
    _print()


def _print_naming(model: str, rows: list[NamingResult]) -> None:
    """Every naming contrast, per metric. Never collapse this into one number.

    ``correctness`` (aligned - misaligned) is printed first and is the one the benchmark's claim rests on:
    it alone holds "the columns carry domain names" fixed and varies only whether they are true.
    """
    _print(f"  [{model}] naming contrasts, paired within SCM (* = 95% CI clears zero)")
    _print(f"      {'metric':<16}{'drivers':<15}{'correctness':>22}{'gain':>22}{'harm':>22}")
    for row in rows:
        cells = []
        for contrast in (row.correctness, row.gain, row.harm):
            star = "*" if contrast.excludes_zero else " "
            cells.append(f"{contrast.delta:+.3f}[{contrast.ci[0]:+.3f},{contrast.ci[1]:+.3f}]{star}")
        _print(f"      {row.metric:<16}{row.subset:<15}{cells[0]:>22}{cells[1]:>22}{cells[2]:>22}")
    _print("      correctness = aligned - misaligned   gain = aligned - blind   harm = blind - misaligned")
    _print()


def _naming_record(row: NamingResult) -> dict[str, object]:
    return {
        "metric": row.metric,
        "subset": row.subset,
        **{
            name: {"delta": c.delta, "ci": list(c.ci), "n_scms": c.n_scms, "excludes_zero": c.excludes_zero}
            for name, c in (("correctness", row.correctness), ("gain", row.gain), ("harm", row.harm))
        },
    }


def _cell_record(cell: DriverCell) -> dict[str, object]:
    return {
        "identifiable": cell.identifiable,
        "alignment": cell.alignment,
        "n": cell.n,
        "identification": cell.identification,
        "estimation": cell.estimation,
    }


def _score_record(n_drivers: int, score: AnswerScore) -> dict[str, object]:
    return {
        "n_drivers": n_drivers,
        "alignment": score.alignment,
        "overall": score.overall,
        "identification": score.identification,
        "estimation": score.estimation,
        "ranking_tau": score.ranking_tau,
        "parse_ok": score.parse_ok,
        # Raw per-driver rows, so a completed run can be re-sliced offline (by archetype, decoy status,
        # anything) without paying for it again. Keyed by hidden node name, which leaks roles - fine in an
        # output artifact, which is never model input.
        "drivers": {
            driver: {
                "identifiable": identifiable,
                "adjustment_bias": score.adjustment_bias[driver],
                "effect_error": score.effect_error[driver],
                "identifiability_judgment": score.identifiability_judgment[driver],
            }
            for driver, identifiable in score.identifiable.items()
        },
    }


def _turn_record(model: str, index: int, trace: TurnTrace) -> dict:
    return {
        "kind": "turn",
        "model": model,
        "index": index,
        "turn": trace.turn,
        "duration_seconds": trace.duration_seconds,
        "tool_calls": list(trace.tool_calls),
        "prompt_tokens": trace.prompt_tokens,
        "completion_tokens": trace.completion_tokens,
        # Budget state the model saw when it chose this turn's calls - lets a finished run show whether the
        # tool-call budget actually bound, and for which models, without re-running anything.
        "budget_used": trace.budget_used,
        "budget_limit": trace.budget_limit,
    }


def _tool_call_record(call: ToolCall) -> dict[str, object]:
    return {"id": call.id, "name": call.name, "arguments": dict(call.arguments), "error": call.error}


def _transcript_record(model: str, index: int, transcript: TurnTranscript) -> dict:
    """Render one complete application-level model exchange - full messages, tool results, raw reply.

    Same shape as ``run_eval.py``'s own ``_transcript_record``, minus the LiteLLM-only fields
    (``seed``/``openrouter_session_id``) that don't apply to the Databricks Gateway path.
    """
    return {
        "kind": "transcript",
        "model": model,
        "index": index,
        "turn": transcript.turn,
        "duration_seconds": transcript.duration_seconds,
        "request": {
            "messages": transcript.messages,
            "tools": transcript.tools,
            "tool_choice": "none" if transcript.force_final else "auto",
        },
        "response": None
        if transcript.error is not None
        else {
            "content": transcript.content,
            "tool_calls": [_tool_call_record(call) for call in transcript.tool_calls],
            "prompt_tokens": transcript.prompt_tokens,
            "completion_tokens": transcript.completion_tokens,
        },
        "error": transcript.error,
    }


def _instance_record(
    model: str, index: int, duration_seconds: float, n_drivers: int, score: AnswerScore, turns: list[TurnTrace]
) -> dict:
    """One instance's score plus rolled-up totals across its turns.

    Totals, not per-turn averages - two models that batch tool calls into turns differently (many calls in
    few turns vs. one call per turn across many turns) aren't fairly compared by a per-turn average, but
    the total tool calls / tokens spent investigating the same instance is a strategy-agnostic number.
    """
    prompt_tokens = [t.prompt_tokens for t in turns if t.prompt_tokens is not None]
    completion_tokens = [t.completion_tokens for t in turns if t.completion_tokens is not None]
    tool_call_histogram: dict[str, int] = {}
    for trace in turns:
        for name in trace.tool_calls:
            tool_call_histogram[name] = tool_call_histogram.get(name, 0) + 1
    return {
        "kind": "instance",
        "model": model,
        "index": index,
        "duration_seconds": duration_seconds,
        "total_turns": len(turns),
        "total_tool_calls": sum(tool_call_histogram.values()),
        "tool_call_histogram": tool_call_histogram,
        "total_prompt_tokens": sum(prompt_tokens) if prompt_tokens else None,
        "total_completion_tokens": sum(completion_tokens) if completion_tokens else None,
        **_score_record(n_drivers, score),
    }


def _spread_across_scms(instances: list, n: int) -> list:
    """Take ~``n`` instances spread across the dataset's SCMs rather than its first ``n``.

    Whole SCMs are kept together: each contributes one instance per naming condition, and comparing those
    conditions is only meaningful within a fixed SCM (paired skins),
    so a subset that splits one is not a smaller version of the experiment. Groups are keyed by the SCM's
    weight matrix, not by position, so this does not depend on the build script's write order.

    A prefix is biased by whatever order the SCMs happened to be generated in, and measurably so on the
    current build: the first 20 instances are 12x ``n_drivers=3`` and contain no 5- or 6-driver SCM at all,
    while the full set spans 3-7. Since driver count is the axis acceptance and ranking difficulty both vary
    on, that turns a partial run into a systematically easy one.
    """
    groups: dict[bytes, list] = {}
    for item in instances:
        groups.setdefault(item[1].scm.W.tobytes(), []).append(item)
    ordered = list(groups.values())
    # Largest evenly-spaced set of whole groups that still fits in n.
    for count in range(len(ordered), 0, -1):
        picked = sorted({round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)} if count > 1 else {0})
        chosen = [inst for i in picked for inst in ordered[i]]
        if len(chosen) <= n:
            return chosen
    return instances[:n]


def _require_discovery(parser: argparse.ArgumentParser) -> None:
    """Refuse to start if the ``discovery`` extra is missing, rather than degrade silently.

    Mirrors ``scripts/run_eval.py::_require_discovery``, duplicated rather than imported to keep this
    internal path textually separate from the public one. ``agent/tools.py::discover`` imports causal-learn
    lazily and returns an error *string* when it is absent - correct for that module, whose contract is that
    tool failures come back as readable text - but it means an eval missing the extra still completes and
    still writes scores, depressed by an environment defect, with nothing in the output recording the
    difference. This is the last point at which that is still distinguishable from a capability result, and
    it runs before any gateway call is billed.
    """
    try:
        import causallearn  # noqa: F401
    except ImportError:
        parser.error(
            "the `discover` tool needs the optional discovery extra (causal-learn), which is not installed; "
            "run `uv sync --extra databricks --extra discovery`"
        )


def main() -> None:
    """Run the agent over a pre-built dataset for each model, in turn; print + persist scores as they land."""
    load_env()
    parser = argparse.ArgumentParser(description="INTERNAL: evaluate the agent against Databricks-served models.")
    parser.add_argument("--models", default=",".join(_DEFAULT_MODELS), help="comma-separated gateway model names")
    parser.add_argument("--dataset", required=True, help="path to a build_testset(.py)-shaped output")
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=MAX_TOOL_CALLS,
        help="investigation budget: tool calls per instance before the answer is forced",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=MAX_TURNS,
        help="round-trip budget: model replies per instance before the answer is forced",
    )
    parser.add_argument(
        "--max-token-budget",
        type=int,
        default=MAX_TOKEN_BUDGET,
        help="cost guardrail: prompt+completion tokens per instance before the answer is forced (0 disables)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="incremental results (.jsonl, one line per model); defaults to <dataset>_eval.jsonl",
    )
    parser.add_argument(
        "-n",
        "--n-instances",
        type=int,
        default=None,
        help="evaluate about N instances - whole SCMs spread across the dataset, not the first N (default all)",
    )
    parser.add_argument(
        "--transcript-out",
        help=(
            "write full JSONL request/response transcripts - complete message history, tool call arguments, "
            "tool results, and raw model text - for tracing whether a model's reasoning and tool use are "
            "actually sound rather than just looking at the final score (existing file is overwritten)"
        ),
    )
    args = parser.parse_args()
    _require_discovery(parser)

    instances = load_dataset(args.dataset)
    if args.n_instances is not None:
        instances = _spread_across_scms(instances, args.n_instances)
    n_drivers = [len(key.scm.scaffold.non_decoy_drivers()) for _, key in instances]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    # Results land beside their dataset unless told otherwise, so two datasets cannot share one file.
    out = Path(args.out) if args.out else Path(f"{Path(args.dataset).with_suffix('')}_eval.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    transcript_out = Path(args.transcript_out) if args.transcript_out else None
    if transcript_out is not None:
        if transcript_out.resolve() == out.resolve():
            parser.error("--out and --transcript-out must be different files")
        transcript_out.parent.mkdir(parents=True, exist_ok=True)

    _print(f"dataset={args.dataset}  instances={len(instances)}  models={models}")
    _print(f"writing incrementally to {out} - one line per instance, plus one per model, each flushed as it lands")
    if transcript_out is not None:
        _print(f"writing full request/response transcripts to {transcript_out}")
    _print()

    transcript_context = transcript_out.open("w", encoding="utf-8") if transcript_out is not None else nullcontext(None)
    with out.open("w", encoding="utf-8") as results_file, transcript_context as transcript_file:
        for model in models:
            record: dict = {"model": model, "dataset": args.dataset}
            start = time.monotonic()
            last = [start]  # mutable box: closures can't rebind an outer float, only mutate a container
            durations: list[float] = []
            turns_by_instance: dict[int, list[TurnTrace]] = {}
            errors_by_instance: dict[int, str] = {}

            def _on_turn(
                i: int, trace: TurnTrace, *, model: str = model, turns_by_instance: dict = turns_by_instance
            ) -> None:
                turns_by_instance.setdefault(i, []).append(trace)
                results_file.write(json.dumps(_turn_record(model, i, trace)) + "\n")
                results_file.flush()

            def _progress(
                i: int,
                score: AnswerScore,
                *,
                model: str = model,
                start: float = start,
                last: list[float] = last,
                durations: list[float] = durations,
                turns_by_instance: dict = turns_by_instance,
                errors_by_instance: dict = errors_by_instance,
            ) -> None:
                now = time.monotonic()
                instance_secs = now - last[0]
                last[0] = now
                durations.append(instance_secs)
                turns = turns_by_instance.pop(i, [])
                total_tools = sum(len(t.tool_calls) for t in turns)
                error = errors_by_instance.get(i)
                if error is not None:
                    _print(
                        f"  [{model}] instance {i + 1}/{len(instances)}  (+{instance_secs:5.0f}s, "
                        f"{now - start:6.0f}s total)  ERROR: {error}"
                    )
                    results_file.write(
                        json.dumps(
                            {
                                "kind": "instance_error",
                                "model": model,
                                "index": i,
                                "duration_seconds": instance_secs,
                                "n_drivers": n_drivers[i],
                                "error": error,
                                "total_turns": len(turns),
                                "total_tool_calls": total_tools,
                            }
                        )
                        + "\n"
                    )
                    results_file.flush()
                    return
                _print(
                    f"  [{model}] instance {i + 1}/{len(instances)}  (+{instance_secs:5.0f}s, "
                    f"{now - start:6.0f}s total)  turns={len(turns):<3} tools={total_tools:<3} "
                    f"parse={score.parse_ok}  overall={score.overall:.3f} tau={score.ranking_tau:+.3f}"
                )
                results_file.write(
                    json.dumps(_instance_record(model, i, instance_secs, n_drivers[i], score, turns)) + "\n"
                )
                results_file.flush()

            def _on_instance_error(i: int, error: Exception, *, errors_by_instance: dict = errors_by_instance) -> None:
                errors_by_instance[i] = f"{type(error).__name__}: {error}"

            def _on_transcript(
                i: int,
                transcript: TurnTranscript,
                *,
                model: str = model,
                transcript_file: TextIO = transcript_file,
            ) -> None:
                if transcript_file is not None:
                    transcript_file.write(json.dumps(_transcript_record(model, i, transcript)) + "\n")
                    transcript_file.flush()

            try:
                scores = run_evaluation(
                    instances,
                    DatabricksClient(model=model),
                    max_tool_calls=args.max_tool_calls,
                    max_turns=args.max_turns,
                    max_token_budget=args.max_token_budget or None,
                    on_instance=_progress,
                    on_turn=_on_turn,
                    on_transcript=_on_transcript if transcript_file is not None else None,
                    on_instance_error=_on_instance_error,
                )
            except Exception as exc:  # one unavailable/misnamed model shouldn't abort the rest of the sweep
                _print(f"model={model}  FAILED: {exc}\n")
                record["kind"] = "model_error"
                record["error"] = str(exc)
                results_file.write(json.dumps(record) + "\n")
                results_file.flush()
                continue

            record["kind"] = "model_summary"
            record["total_seconds"] = time.monotonic() - start
            record["instance_seconds"] = durations
            # A provider failure is scored as a parse failure by run_evaluation (all zeros) and would
            # otherwise be averaged in as if the model had answered badly, deflating it for a network blip.
            indexed = enumerate(zip(n_drivers, scores, strict=True))
            scored = [(i, n, s) for i, (n, s) in indexed if i not in errors_by_instance]
            scored_n_drivers = [n for _, n, _ in scored]
            scored_scores = [s for _, _, s in scored]
            cells = driver_cells(scored_scores)
            keyed = [(instances[i][1], s) for i, _, s in scored]
            # Every contrast, per metric: `overall` averages two axes that respond differently, so a
            # naming effect read off it alone can be hidden by its own components.
            naming = naming_table(keyed)
            references = reference_scores([k for k, s in keyed if s.alignment is not None])
            try:
                pc_f1: float | None = mechanical_identifiability_f1([instances[i] for i, _, _ in scored]).f1
            except ImportError:
                pc_f1 = None
            _print_scores(model, scored_n_drivers, scored_scores)
            _print_cells(cells)
            _print_headline(model, scored_scores, references, pc_f1)
            _print_naming(model, naming)
            record["instances_requested"] = len(instances)
            record["instances_scored"] = len(scored)
            record["provider_error_count"] = len(errors_by_instance)
            record["failed_instances"] = [{"index": i, "error": e} for i, e in sorted(errors_by_instance.items())]
            record["per_instance"] = [_score_record(n, s) for _, n, s in scored]
            record["driver_cells"] = [_cell_record(c) for c in cells]
            record["naming_contrasts"] = [_naming_record(row) for row in naming]
            record["overall"] = sum(s.overall for s in scored_scores) / len(scored_scores) if scored_scores else None
            record["parse_rate"] = (
                sum(s.parse_ok for s in scored_scores) / len(scored_scores) if scored_scores else None
            )
            f1 = identifiability_f1(scored_scores)
            record["identifiability_f1"] = {
                "precision": f1.precision,
                "recall": f1.recall,
                "f1": f1.f1,
                "n_positive": f1.n_positive,
                "mechanical_pc_baseline": pc_f1,
            }
            record["reference_scores"] = references
            results_file.write(json.dumps(record) + "\n")
            results_file.flush()

    _print(f"done - wrote {len(models)} model result(s) to {out}")


if __name__ == "__main__":
    main()

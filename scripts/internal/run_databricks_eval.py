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
raising ``--max-turns``, since the failure is a format-translation gap, not a turn-budget one.

``claude-opus-5`` was previously excluded from this list on suspicion of a similar unfixable issue
("goes silent" after a growing number of parallel tool calls) - direct tracing found this was actually
two separate, fixable turn-budget problems, not a model/gateway quirk: (1) it genuinely needs more
investigative turns than the other models here (observed: 12, not fixed by more turns alone at the old
``DatabricksClient.max_tokens=4096`` - see below); (2) on the forced-final turn, a growing accumulated
context led it to spend its entire reply on internal reasoning tokens and get cut off (``finish_reason``:
``length``) before writing any answer text, since it has no innate awareness that a token cutoff exists at
all - fixed by raising ``DatabricksClient.max_tokens`` (``agent/clients.py``) and telling it about the
budget on the forced-final turn (``agent/loop.py``'s ``_FINAL_TURN_REMINDER``). Both fixes are provider/
client-level, not internal-script-only, so this is now re-included with a default ``--max-turns`` well
above what it was observed to need. Needs ``DATABRICKS_TOKEN`` + ``DATABRICKS_HOST`` (or
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
``--max-turns`` model round trips against a real, sometimes very slow model - opus-5 observed taking 10-25
minutes per instance) would otherwise lose every instance already scored, even though the work (and money)
to produce them was already spent. Console output is flushed the same way, since Python fully buffers
stdout when it isn't a terminal (piped to a log file, run in the background) - without an explicit flush
nothing appears until the whole process exits, making progress unobservable on a long run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from causalagentbench.agent.clients import DatabricksClient
from causalagentbench.agent.eval import (
    AlignmentSummary,
    DriverCellSummary,
    aggregate_by_alignment,
    aggregate_by_driver_cell,
    run_evaluation,
)
from causalagentbench.agent.loop import MAX_TURNS, TurnTrace
from causalagentbench.generation.clients import load_env
from causalagentbench.scoring.score import AnswerScore
from causalagentbench.task.dataset import load_dataset

_DEFAULT_DATASET = "outputs/databricks_internal"
_DEFAULT_OUT = "outputs/databricks_eval_results.jsonl"
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
#: opus-5 was directly observed needing 12 turns to finish investigating even a modest (4-driver)
#: instance - default to loop.py's own considered safety-valve value, not a second guessed number.
_DEFAULT_MAX_TURNS = MAX_TURNS


def _print(*args: object) -> None:
    """``print`` that flushes immediately - see the module docstring on why this matters here."""
    print(*args, flush=True)


def _fmt_z(z: float | None) -> str:
    return f"{z:+.2f}" if z is not None else " n/a"


def _print_scores(model: str, n_drivers: list[int], scores: list[AnswerScore]) -> None:
    _print(f"model={model}  instances={len(scores)}")
    _print("-" * 64)
    for i, (n, score) in enumerate(zip(n_drivers, scores, strict=True)):
        level = "name-blind" if score.alignment is None else f"alpha={score.alignment:+.1f}"
        _print(
            f"  #{i:<3} n_drivers={n:<3} {level:<12} ranking_tau={score.ranking_tau:+.3f} "
            f"(n={score.n_compared:<2} z={_fmt_z(score.ranking_tau_z)})  "
            f"score={score.mean_driver_score:.3f} eff_acc={score.effect_accuracy:.3f} "
            f"adj_acc={score.adjustment_accuracy:.3f} abstain_acc={score.abstention_accuracy:.3f} "
            f"parse={score.parse_ok}"
        )


def _print_summaries(summaries: list[AlignmentSummary]) -> None:
    _print("-" * 64)
    for summary in summaries:
        level = "name-blind" if summary.alignment is None else f"alpha={summary.alignment:+.1f}"
        _print(
            f"  {level:<12} n={summary.n:<4} ranking_tau={summary.mean_ranking_tau:+.3f} "
            f"adj_acc={summary.mean_adjustment_accuracy:.3f} abstain_acc={summary.mean_abstention_accuracy:.3f} "
            f"parse_rate={summary.parse_rate:.2f}"
        )
    _print()


def _print_cells(cells: list[DriverCellSummary]) -> None:
    """Print the driver-grain breakdown; `n` counts drivers here, not instances (unlike _print_summaries)."""
    _print("-" * 64)
    for cell in cells:
        level = "name-blind" if cell.alignment is None else f"alpha={cell.alignment:+.1f}"
        kind = "identifiable" if cell.identifiable else "unidentifiable"
        _print(
            f"  {kind:<15}{level:<12} n={cell.n:<4} graded={cell.n_graded:<4} "
            f"score={cell.mean_driver_score:.3f} eff_acc={cell.mean_effect_accuracy:.3f} "
            f"adj_acc={cell.mean_adjustment_accuracy:.3f} abstain_acc={cell.mean_abstention_accuracy:.3f} "
            f"ranking_tau={cell.ranking_tau:+.3f} (pairs={cell.n_ranking_pairs})"
        )
    _print()


def _score_record(n_drivers: int, score: AnswerScore) -> dict:
    return {
        "n_drivers": n_drivers,
        "alignment": score.alignment,
        "ranking_tau": score.ranking_tau,
        "n_compared": score.n_compared,
        "ranking_tau_z": score.ranking_tau_z,
        "mean_driver_score": score.mean_driver_score,
        "effect_accuracy": score.effect_accuracy,
        "adjustment_accuracy": score.adjustment_accuracy,
        "abstention_accuracy": score.abstention_accuracy,
        "parse_ok": score.parse_ok,
        # Raw per-driver rows, so a completed run can be re-sliced offline without paying for it again.
        "drivers": {
            driver: {
                "identifiable": identifiable,
                "gold_abstain": score.gold_abstain[driver],
                "driver_score": score.driver_score[driver],
                "effect_error": score.effect_error[driver],
                "adjustment_valid": score.adjustment_valid[driver],
                "abstention_correct": score.abstention_correct[driver],
                "concordant": score.ranking_pairs[driver].concordant,
                "discordant": score.ranking_pairs[driver].discordant,
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


def _summary_record(summary: AlignmentSummary) -> dict:
    return {
        "alignment": summary.alignment,
        "n": summary.n,
        "mean_ranking_tau": summary.mean_ranking_tau,
        "mean_adjustment_accuracy": summary.mean_adjustment_accuracy,
        "mean_abstention_accuracy": summary.mean_abstention_accuracy,
        "parse_rate": summary.parse_rate,
    }


def _cell_record(cell: DriverCellSummary) -> dict:
    return {
        "identifiable": cell.identifiable,
        "alignment": cell.alignment,
        "n": cell.n,
        "n_graded": cell.n_graded,
        "mean_driver_score": cell.mean_driver_score,
        "mean_effect_accuracy": cell.mean_effect_accuracy,
        "mean_adjustment_accuracy": cell.mean_adjustment_accuracy,
        "mean_abstention_accuracy": cell.mean_abstention_accuracy,
        "n_ranking_pairs": cell.n_ranking_pairs,
        "ranking_tau": cell.ranking_tau,
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


def main() -> None:
    """Run the agent over a pre-built dataset for each model, in turn; print + persist scores as they land."""
    load_env()
    parser = argparse.ArgumentParser(description="INTERNAL: evaluate the agent against Databricks-served models.")
    parser.add_argument("--models", default=",".join(_DEFAULT_MODELS), help="comma-separated gateway model names")
    parser.add_argument("--dataset", default=_DEFAULT_DATASET, help="path to a build_testset(.py)-shaped output")
    parser.add_argument(
        "--max-turns", type=int, default=_DEFAULT_MAX_TURNS, help="tool-call turns before forcing a final answer"
    )
    parser.add_argument("--out", default=_DEFAULT_OUT, help="incremental results (.jsonl, one line per model)")
    parser.add_argument(
        "-n",
        "--n-instances",
        type=int,
        default=None,
        help="evaluate about N instances - whole SCMs spread across the dataset, not the first N (default all)",
    )
    args = parser.parse_args()

    instances = load_dataset(args.dataset)
    if args.n_instances is not None:
        instances = _spread_across_scms(instances, args.n_instances)
    n_drivers = [len(key.scm.scaffold.non_decoy_drivers()) for _, key in instances]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    _print(f"dataset={args.dataset}  instances={len(instances)}  models={models}")
    _print(f"writing incrementally to {out} - one line per instance, plus one per model, each flushed as it lands")
    _print()

    with out.open("w", encoding="utf-8") as results_file:
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
                    f"parse={score.parse_ok}  ranking_tau={score.ranking_tau:+.3f} "
                    f"(n={score.n_compared} z={_fmt_z(score.ranking_tau_z)})"
                )
                results_file.write(
                    json.dumps(_instance_record(model, i, instance_secs, n_drivers[i], score, turns)) + "\n"
                )
                results_file.flush()

            def _on_instance_error(i: int, error: Exception, *, errors_by_instance: dict = errors_by_instance) -> None:
                errors_by_instance[i] = f"{type(error).__name__}: {error}"

            try:
                scores = run_evaluation(
                    instances,
                    DatabricksClient(model=model),
                    max_turns=args.max_turns,
                    on_instance=_progress,
                    on_turn=_on_turn,
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
            paired = enumerate(zip(n_drivers, scores, strict=True))
            scored = [(n, s) for i, (n, s) in paired if i not in errors_by_instance]
            scored_n_drivers = [n for n, _ in scored]
            scored_scores = [s for _, s in scored]
            summaries = aggregate_by_alignment(scored_scores)
            cells = aggregate_by_driver_cell(scored_scores)
            _print_scores(model, scored_n_drivers, scored_scores)
            _print_summaries(summaries)
            _print_cells(cells)
            record["instances_requested"] = len(instances)
            record["instances_scored"] = len(scored)
            record["provider_error_count"] = len(errors_by_instance)
            record["failed_instances"] = [{"index": i, "error": e} for i, e in sorted(errors_by_instance.items())]
            record["per_instance"] = [_score_record(n, s) for n, s in scored]
            record["summaries"] = [_summary_record(s) for s in summaries]
            record["driver_cells"] = [_cell_record(c) for c in cells]
            results_file.write(json.dumps(record) + "\n")
            results_file.flush()

    _print(f"done - wrote {len(models)} model result(s) to {out}")


if __name__ == "__main__":
    main()

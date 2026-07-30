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

Writes ``--out`` incrementally, one JSON line per model flushed the moment that model finishes (success
or failure) - a long multi-model sweep over real network calls (each instance up to ``--max-turns`` model
round trips) is exactly the kind of run that shouldn't lose every already-completed model's scores just
because a later one hangs, errors, or the run gets interrupted. Console output is flushed the same way,
since Python fully buffers stdout when it isn't a terminal (piped to a log file, run in the background) -
without an explicit flush nothing appears until the whole process exits, making progress unobservable on
a long run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalagentbench.agent.clients import DatabricksClient
from causalagentbench.agent.eval import AlignmentSummary, aggregate_by_alignment, run_evaluation
from causalagentbench.agent.loop import MAX_TURNS
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


def _print_scores(model: str, n_drivers: list[int], scores: list[AnswerScore]) -> None:
    _print(f"model={model}  instances={len(scores)}")
    _print("-" * 64)
    for i, (n, score) in enumerate(zip(n_drivers, scores, strict=True)):
        level = "name-blind" if score.alignment is None else f"alpha={score.alignment:+.1f}"
        _print(
            f"  #{i:<3} n_drivers={n:<3} {level:<12} ranking_tau={score.ranking_tau:+.3f} "
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


def _score_record(n_drivers: int, score: AnswerScore) -> dict:
    return {
        "n_drivers": n_drivers,
        "alignment": score.alignment,
        "ranking_tau": score.ranking_tau,
        "adjustment_accuracy": score.adjustment_accuracy,
        "abstention_accuracy": score.abstention_accuracy,
        "parse_ok": score.parse_ok,
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
    args = parser.parse_args()

    instances = load_dataset(args.dataset)
    n_drivers = [len(key.scm.scaffold.non_decoy_drivers()) for _, key in instances]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    _print(f"dataset={args.dataset}  instances={len(instances)}  models={models}")
    _print(f"writing incrementally to {out} - one line per model, flushed as each completes")
    _print()

    with out.open("w", encoding="utf-8") as results_file:
        for model in models:
            record: dict = {"model": model, "dataset": args.dataset}
            try:
                scores = run_evaluation(instances, DatabricksClient(model=model), max_turns=args.max_turns)
            except Exception as exc:  # one unavailable/misnamed model shouldn't abort the rest of the sweep
                _print(f"model={model}  FAILED: {exc}\n")
                record["error"] = str(exc)
                results_file.write(json.dumps(record) + "\n")
                results_file.flush()
                continue

            summaries = aggregate_by_alignment(scores)
            _print_scores(model, n_drivers, scores)
            _print_summaries(summaries)
            record["per_instance"] = [_score_record(n, s) for n, s in zip(n_drivers, scores, strict=True)]
            record["summaries"] = [_summary_record(s) for s in summaries]
            results_file.write(json.dumps(record) + "\n")
            results_file.flush()

    _print(f"done - wrote {len(models)} model result(s) to {out}")


if __name__ == "__main__":
    main()

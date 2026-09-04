"""Run one or more LiteLLM models against the benchmark.

The default path is generated, abstract-labeled instances. Pass ``--dataset`` to evaluate a frozen
``scripts/build_testset.py`` output instead; this is the appropriate path for a model comparison because
every model then receives identical data and naming conditions. ``--models`` runs a comma-separated list
sequentially and, with ``--out``, writes JSONL traces incrementally so an interrupted sweep retains all
completed turns and instances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TextIO

from causalagentbench.agent.clients import LiteLLMClient
from causalagentbench.agent.eval import (
    DriverCell,
    NamingResult,
    driver_cells,
    identifiability_f1,
    naming_table,
    run_evaluation,
)
from causalagentbench.agent.loop import MAX_TOKEN_BUDGET, MAX_TOOL_CALLS, MAX_TURNS, ToolCall, TurnTrace, TurnTranscript
from causalagentbench.generation.clients import load_env
from causalagentbench.scm.sample import sample
from causalagentbench.scoring.discovery_baseline import mechanical_identifiability_f1
from causalagentbench.scoring.gates import generate_gated_procedural
from causalagentbench.scoring.score import AnswerScore, reference_scores
from causalagentbench.task.dataset import load_dataset
from causalagentbench.task.instance import Instance, InstanceKey, build_instance


def _print(*args: object) -> None:
    """Print progress immediately, including when stdout is redirected to a log file."""
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


def _turn_record(model: str, index: int, trace: TurnTrace) -> dict[str, object]:
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


def _tool_call_record(call: ToolCall) -> dict[str, object]:
    return {
        "id": call.id,
        "name": call.name,
        "arguments": dict(call.arguments),
        "error": call.error,
    }


def _transcript_record(
    model: str,
    index: int,
    temperature: float,
    sampling_seed: int | None,
    openrouter_session_id: str | None,
    transcript: TurnTranscript,
) -> dict[str, object]:
    """Render one complete application-level model exchange for a separate JSONL transcript."""
    request: dict[str, object] = {
        "messages": transcript.messages,
        "tools": transcript.tools,
        "tool_choice": "none" if transcript.force_final else "auto",
        "temperature": temperature,
        "seed": sampling_seed,
    }
    if openrouter_session_id is not None:
        request["extra_body"] = {"session_id": openrouter_session_id}
    return {
        "kind": "transcript",
        "model": model,
        "index": index,
        "turn": transcript.turn,
        "duration_seconds": transcript.duration_seconds,
        "request": request,
        "response": None
        if transcript.error is not None
        else {
            "content": transcript.content,
            "tool_calls": [_tool_call_record(call) for call in transcript.tool_calls],
            "prompt_tokens": transcript.prompt_tokens,
            "completion_tokens": transcript.completion_tokens,
            "estimated_cost_usd": transcript.cost_usd,
        },
        "error": transcript.error,
    }


def _instance_record(
    model: str, index: int, duration_seconds: float, n_drivers: int, score: AnswerScore, turns: list[TurnTrace]
) -> dict[str, object]:
    """Record an instance score and its strategy-agnostic turn totals."""
    prompt_tokens = [trace.prompt_tokens for trace in turns if trace.prompt_tokens is not None]
    completion_tokens = [trace.completion_tokens for trace in turns if trace.completion_tokens is not None]
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


def _instance_error_record(
    model: str,
    index: int,
    duration_seconds: float,
    n_drivers: int,
    error: str,
    turns: list[TurnTrace],
) -> dict[str, object]:
    """Record a provider/client failure without treating it as a model score."""
    prompt_tokens = [trace.prompt_tokens for trace in turns if trace.prompt_tokens is not None]
    completion_tokens = [trace.completion_tokens for trace in turns if trace.completion_tokens is not None]
    tool_call_histogram: dict[str, int] = {}
    for trace in turns:
        for name in trace.tool_calls:
            tool_call_histogram[name] = tool_call_histogram.get(name, 0) + 1
    return {
        "kind": "instance_error",
        "model": model,
        "index": index,
        "duration_seconds": duration_seconds,
        "n_drivers": n_drivers,
        "error": error,
        "total_turns": len(turns),
        "total_tool_calls": sum(tool_call_histogram.values()),
        "tool_call_histogram": tool_call_histogram,
        "total_prompt_tokens": sum(prompt_tokens) if prompt_tokens else None,
        "total_completion_tokens": sum(completion_tokens) if completion_tokens else None,
    }


def _spread_across_scms(instances: list[tuple[Instance, InstanceKey]], n: int) -> list[tuple[Instance, InstanceKey]]:
    """Take about ``n`` instances, retaining every naming condition for each selected SCM."""
    if n < 1:
        raise ValueError("n must be positive")

    groups: dict[bytes, list[tuple[Instance, InstanceKey]]] = {}
    for item in instances:
        groups.setdefault(item[1].scm.W.tobytes(), []).append(item)
    ordered = list(groups.values())
    if not ordered:
        return []
    for count in range(len(ordered), 0, -1):
        picked = sorted({round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)} if count > 1 else {0})
        selected = [instance for i in picked for instance in ordered[i]]
        if len(selected) <= n:
            return selected

    # A requested budget below one full SCM still evaluates one paired condition set; splitting it would
    # make the alignment comparison invalid.
    return ordered[0]


def _generated_instances(
    n_instances: int, n_samples: int | None, n_candidates: int | None
) -> list[tuple[Instance, InstanceKey]]:
    """Build fresh abstract-labeled instances, reusing each gate-certified table by default."""
    instances = []
    for seed in range(n_instances):
        scm, report = generate_gated_procedural(seed, n_candidates=n_candidates)
        if n_samples is None:
            assert report.data is not None
            data = report.data
        else:
            data = sample(scm, n_samples, seed=seed)
        instances.append(build_instance(scm, data, seed=seed))
    return instances


def _models(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    """Return validated model IDs from either the retained single-model flag or the sweep flag."""
    value = args.models if args.models is not None else args.model
    models = [model.strip() for model in value.split(",") if model.strip()]
    if not models:
        parser.error("provide at least one non-empty LiteLLM model ID")
    return models


def _require_discovery(parser: argparse.ArgumentParser) -> None:
    """Refuse to start if the ``discovery`` extra is missing, rather than degrade silently.

    ``agent/tools.py::discover`` imports causal-learn lazily and returns an error *string* when it is
    absent - correct for that module, whose contract is that every tool failure comes back as readable text
    the agent can act on, but it means an evaluation run missing the extra still completes and still writes
    scores. Those scores are depressed by an environment defect and are not comparable to a run that had
    PC available, with nothing in the output JSONL recording the difference. The failure has to surface
    here, before any model is billed, because this is the last point at which it is still distinguishable
    from a capability result.
    """
    try:
        import causallearn  # noqa: F401
    except ImportError:
        parser.error(
            "the `discover` tool needs the optional discovery extra (causal-learn), which is not installed; "
            "run `uv sync --extra generation --extra discovery`"
        )


def _openrouter_session_id(model: str, sampling_seed: int | None, prefix: str | None) -> str | None:
    """Return a stable per-model OpenRouter routing session, or ``None`` for other providers."""
    if not model.startswith("openrouter/"):
        return None
    digest = hashlib.sha256(f"{model}\0{sampling_seed}".encode()).hexdigest()[:16]
    return f"{prefix or 'causalagentbench'}-{digest}"


def main() -> None:
    """Run a LiteLLM-backed tool-using agent over generated or frozen instances."""
    load_env()
    parser = argparse.ArgumentParser(description="Evaluate one or more LiteLLM models on CausalAgentBench.")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="one LiteLLM model ID, e.g. anthropic/claude-opus-4-8")
    model_group.add_argument("--models", help="comma-separated LiteLLM model IDs for a sequential sweep")
    parser.add_argument("--dataset", help="path to a scripts/build_testset.py output stem")
    parser.add_argument(
        "-n",
        "--n-instances",
        type=int,
        default=5,
        help="generated-instance count, or approximate frozen-dataset count while retaining whole SCM groups",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="rows per generated instance; by default reuse the table validated by the oracle gate",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=None,
        help="fix generated-instance total candidate count (drivers + decoys)",
    )
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
    parser.add_argument("--temperature", type=float, default=0.0, help="shared LiteLLM sampling temperature")
    parser.add_argument("--sampling-seed", type=int, default=0, help="shared provider sampling seed")
    parser.add_argument(
        "--openrouter-session-prefix",
        help="prefix for stable per-model OpenRouter routing sessions (default: causalagentbench)",
    )
    parser.add_argument("--request-timeout", type=float, default=120.0, help="per-request-attempt timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=2, help="retry attempts after a failed model request")
    parser.add_argument(
        "--out",
        help="write flushed JSONL turn, instance, and model-summary records (existing file is overwritten)",
    )
    parser.add_argument(
        "--transcript-out",
        help=(
            "write full JSONL request/response transcripts; includes prompts, tool results, and raw model output "
            "(existing file is overwritten)"
        ),
    )
    args = parser.parse_args()

    if args.n_instances < 1:
        parser.error("--n-instances must be positive")
    if args.max_tool_calls < 1:
        parser.error("--max-tool-calls must be positive")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries must be non-negative")
    _require_discovery(parser)
    models = _models(args, parser)
    if args.dataset:
        instances = _spread_across_scms(load_dataset(args.dataset), args.n_instances)
        dataset = args.dataset
    else:
        instances = _generated_instances(args.n_instances, args.n_samples, args.n_candidates)
        dataset = None
    if not instances:
        parser.error("the selected dataset contains no instances")

    n_drivers = [len(key.scm.scaffold.non_decoy_drivers()) for _, key in instances]
    out = Path(args.out) if args.out else None
    transcript_out = Path(args.transcript_out) if args.transcript_out else None
    if out is not None and transcript_out is not None and out.resolve() == transcript_out.resolve():
        parser.error("--out and --transcript-out must be different files")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
    if transcript_out is not None:
        transcript_out.parent.mkdir(parents=True, exist_ok=True)

    location = f"dataset={dataset}" if dataset is not None else "dataset=generated"
    _print(f"{location}  instances={len(instances)}  models={models}")
    if out is not None:
        _print(f"writing incrementally to {out} - one line per turn, instance, and model")
    if transcript_out is not None:
        _print(f"writing full request/response transcripts to {transcript_out}")
    _print()

    context = out.open("w", encoding="utf-8") if out is not None else nullcontext(None)
    transcript_context = transcript_out.open("w", encoding="utf-8") if transcript_out is not None else nullcontext(None)
    with context as results_file, transcript_context as transcript_file:
        _run_models(
            models=models,
            instances=instances,
            n_drivers=n_drivers,
            max_tool_calls=args.max_tool_calls,
            max_turns=args.max_turns,
            max_token_budget=args.max_token_budget or None,
            temperature=args.temperature,
            sampling_seed=args.sampling_seed,
            openrouter_session_prefix=args.openrouter_session_prefix,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
            dataset=dataset,
            results_file=results_file,
            transcript_file=transcript_file,
        )


def _run_models(
    *,
    models: list[str],
    instances: list[tuple[Instance, InstanceKey]],
    n_drivers: list[int],
    max_tool_calls: int,
    max_turns: int,
    max_token_budget: int | None,
    temperature: float,
    sampling_seed: int | None,
    openrouter_session_prefix: str | None,
    request_timeout: float,
    max_retries: int,
    dataset: str | None,
    results_file: TextIO | None,
    transcript_file: TextIO | None,
) -> None:
    """Evaluate every model in sequence and write each completed result before continuing."""
    for model in models:
        openrouter_session_id = _openrouter_session_id(model, sampling_seed, openrouter_session_prefix)
        record: dict[str, object] = {"model": model, "dataset": dataset, "provider": "litellm"}
        start = time.monotonic()
        last = [start]
        durations: list[float] = []
        costs: list[float] = []
        turns_by_instance: dict[int, list[TurnTrace]] = {}
        errors_by_instance: dict[int, str] = {}

        def on_turn(
            index: int,
            trace: TurnTrace,
            *,
            model: str = model,
            costs: list[float] = costs,
            turns_by_instance: dict[int, list[TurnTrace]] = turns_by_instance,
            results_file: TextIO | None = results_file,
        ) -> None:
            turns_by_instance.setdefault(index, []).append(trace)
            if trace.cost_usd is not None:
                costs.append(trace.cost_usd)
            if results_file is not None:
                results_file.write(json.dumps(_turn_record(model, index, trace)) + "\n")
                results_file.flush()

        def on_instance(
            index: int,
            score: AnswerScore,
            *,
            model: str = model,
            start: float = start,
            last: list[float] = last,
            durations: list[float] = durations,
            n_drivers: list[int] = n_drivers,
            turns_by_instance: dict[int, list[TurnTrace]] = turns_by_instance,
            errors_by_instance: dict[int, str] = errors_by_instance,
            results_file: TextIO | None = results_file,
        ) -> None:
            now = time.monotonic()
            instance_seconds = now - last[0]
            last[0] = now
            durations.append(instance_seconds)
            turns = turns_by_instance.pop(index, [])
            total_tools = sum(len(trace.tool_calls) for trace in turns)
            error = errors_by_instance.get(index)
            if error is not None:
                _print(
                    f"  [{model}] instance {index + 1}/{len(instances)}  (+{instance_seconds:5.0f}s, "
                    f"{now - start:6.0f}s total)  ERROR: {error}"
                )
                if results_file is not None:
                    results_file.write(
                        json.dumps(
                            _instance_error_record(model, index, instance_seconds, n_drivers[index], error, turns)
                        )
                        + "\n"
                    )
                    results_file.flush()
                return
            _print(
                f"  [{model}] instance {index + 1}/{len(instances)}  (+{instance_seconds:5.0f}s, "
                f"{now - start:6.0f}s total)  turns={len(turns):<3} tools={total_tools:<3} "
                f"parse={score.parse_ok}  overall={score.overall:.3f} tau={score.ranking_tau:+.3f}"
            )
            if results_file is not None:
                results_file.write(
                    json.dumps(_instance_record(model, index, instance_seconds, n_drivers[index], score, turns)) + "\n"
                )
                results_file.flush()

        def on_instance_error(
            index: int,
            error: Exception,
            *,
            errors_by_instance: dict[int, str] = errors_by_instance,
        ) -> None:
            errors_by_instance[index] = f"{type(error).__name__}: {error}"

        def on_transcript(
            index: int,
            transcript: TurnTranscript,
            *,
            model: str = model,
            temperature: float = temperature,
            sampling_seed: int | None = sampling_seed,
            openrouter_session_id: str | None = openrouter_session_id,
            transcript_file: TextIO | None = transcript_file,
        ) -> None:
            if transcript_file is not None:
                transcript_file.write(
                    json.dumps(
                        _transcript_record(model, index, temperature, sampling_seed, openrouter_session_id, transcript)
                    )
                    + "\n"
                )
                transcript_file.flush()

        try:
            client = LiteLLMClient(
                model=model,
                temperature=temperature,
                seed=sampling_seed,
                openrouter_session_id=openrouter_session_id,
                request_timeout=request_timeout,
                max_retries=max_retries,
            )
            client.validate_tool_capability()
            scores = run_evaluation(
                instances,
                client,
                max_tool_calls=max_tool_calls,
                max_turns=max_turns,
                max_token_budget=max_token_budget,
                on_instance=on_instance,
                on_turn=on_turn,
                on_transcript=on_transcript if transcript_file is not None else None,
                on_instance_error=on_instance_error,
            )
        except Exception as exc:
            _print(f"model={model}  FAILED: {exc}\n")
            record["kind"] = "model_error"
            record["error"] = str(exc)
            if results_file is not None:
                results_file.write(json.dumps(record) + "\n")
                results_file.flush()
            continue

        scored = [
            (index, n_drivers[index], score) for index, score in enumerate(scores) if index not in errors_by_instance
        ]
        scored_n_drivers = [n_drivers for _, n_drivers, _ in scored]
        scored_scores = [score for _, _, score in scored]
        cells = driver_cells(scored_scores)
        keyed = [(instances[index][1], score) for index, _, score in scored]
        # Every contrast, per metric: `overall` averages two axes that respond differently, so a
        # naming effect read off it alone can be hidden by its own components.
        naming = naming_table(keyed)
        scored_indices = {index for index, _, _ in scored}
        references = reference_scores(
            [
                key
                for index, (_instance, key) in enumerate(instances)
                if index in scored_indices and key.provenance.alignment is not None
            ]
        )
        # The "isn't this just a discovery call?" control, on exactly the instances that were scored.
        try:
            pc_f1: float | None = mechanical_identifiability_f1([instances[index] for index, _, _ in scored]).f1
        except ImportError:
            pc_f1 = None
        _print_scores(model, scored_n_drivers, scored_scores)
        _print_cells(cells)
        _print_headline(model, scored_scores, references, pc_f1)
        _print_naming(model, naming)
        record["kind"] = "model_summary"
        record["total_seconds"] = time.monotonic() - start
        record["evaluation_config"] = {
            "max_tool_calls": max_tool_calls,
            "max_turns": max_turns,
            "max_token_budget": max_token_budget,
            "temperature": temperature,
            "sampling_seed": sampling_seed,
            "openrouter_session_id": openrouter_session_id,
            "request_timeout_seconds": request_timeout,
            "max_retries": max_retries,
            "max_format_repairs": 1,
        }
        record["instance_seconds"] = durations
        record["estimated_cost_usd"] = sum(costs) if costs else None
        record["costed_turns"] = len(costs)
        record["instances_requested"] = len(instances)
        record["instances_scored"] = len(scored)
        record["provider_error_count"] = len(errors_by_instance)
        record["failed_instances"] = [
            {"index": index, "error": error} for index, error in sorted(errors_by_instance.items())
        ]
        record["per_instance"] = [_score_record(n_drivers, score) for _, n_drivers, score in scored]
        record["driver_cells"] = [_cell_record(cell) for cell in cells]
        record["naming_contrasts"] = [_naming_record(row) for row in naming]
        record["overall"] = sum(s.overall for s in scored_scores) / len(scored_scores) if scored_scores else None
        record["parse_rate"] = sum(s.parse_ok for s in scored_scores) / len(scored_scores) if scored_scores else None
        f1 = identifiability_f1(scored_scores)
        record["identifiability_f1"] = {
            "precision": f1.precision,
            "recall": f1.recall,
            "f1": f1.f1,
            "n_positive": f1.n_positive,
            "mechanical_pc_baseline": pc_f1,
        }
        record["reference_scores"] = references
        if costs:
            _print(f"  [{model}] estimated_cost_usd=${sum(costs):.6f} across {len(costs)} model call(s)")
        else:
            _print(f"  [{model}] estimated_cost_usd=unavailable (LiteLLM could not price the model calls)")
        if results_file is not None:
            results_file.write(json.dumps(record) + "\n")
            results_file.flush()


if __name__ == "__main__":
    main()

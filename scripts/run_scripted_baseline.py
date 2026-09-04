"""Run the no-LLM scripted agent over a built test set and print the same report the models get.

Free (no API calls) and decisive: it answers "is this benchmark just running the method?" with a number.
Read the script's ``overall`` against ``data-only`` from ``reference_scores`` - ``data-only`` is that same
policy given the *true* structure, so the difference between them is the cost of discovery error on finite
samples, and the difference between the script and a model is what the model adds on top.

    python scripts/run_scripted_baseline.py --dataset outputs/testset_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalagentbench.agent.eval import driver_cells, identifiability_f1, naming_result
from causalagentbench.scoring.score import reference_scores
from causalagentbench.scoring.scripted_agent import DEFAULT_ALPHA, AmbiguityPolicy, scripted_scores
from causalagentbench.task.dataset import load_dataset


def main() -> None:
    """Score the scripted agent under both ambiguity policies and report."""
    parser = argparse.ArgumentParser(description="Run the scripted (no-LLM) causal-analysis baseline.")
    parser.add_argument("--dataset", required=True, help="path to a build_testset(.py)-shaped output")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="PC significance level")
    parser.add_argument("--out", default=None, help="write the summary as JSON here")
    args = parser.parse_args()

    instances = load_dataset(args.dataset)
    named = [key for _instance, key in instances if key.provenance.alignment is not None]
    print(f"dataset={args.dataset}  instances={len(instances)}  alpha={args.alpha}\n")

    print("  oracle references (computed from the TRUE structure, not from discovery)")
    references = reference_scores(named)
    for name, value in references.items():
        print(f"      {name:24s} {value:.3f}")
    print()

    summary: dict[str, object] = {"dataset": args.dataset, "alpha": args.alpha, "reference_scores": references}
    overalls = {}
    for policy in AmbiguityPolicy:
        scores = scripted_scores(instances, policy=policy, alpha=args.alpha)
        overall = sum(s.overall for s in scores) / len(scores)
        identification = sum(s.identification for s in scores) / len(scores)
        estimation = sum(s.estimation for s in scores) / len(scores)
        tau = sum(s.ranking_tau for s in scores) / len(scores)
        f1 = identifiability_f1(scores)
        overalls[policy.value] = overall
        print(f"  scripted agent, ambiguous edges: {policy.value.upper()}")
        print(f"      overall={overall:.3f}  ident={identification:.3f}  est={estimation:.3f}  tau={tau:+.3f}")
        print(f"      identifiability F1={f1.f1:.3f} (P {f1.precision:.3f} / R {f1.recall:.3f})")
        for cell in driver_cells(scores):
            level = "name-blind" if cell.alignment is None else f"alpha={cell.alignment:+.1f}"
            kind = "identifiable" if cell.identifiable else "UNidentifiable"
            print(
                f"        {kind:<15}{level:<12} n={cell.n:<4} ident={cell.identification:.3f} est={cell.estimation:.3f}"
            )
        # The script cannot read column names, so every naming contrast must come out at zero. This is the
        # design's own negative control: a non-zero value here would mean the *pipeline* leaks naming into
        # the score, and would invalidate any Gain/Harm measured on a model.
        keyed = list(zip([k for _i, k in instances], scores, strict=True))
        control = naming_result(keyed, "estimation", identifiable=False)
        print(
            f"        naming control (must be ~0): correctness={control.correctness.delta:+.4f} "
            f"gain={control.gain.delta:+.4f} harm={control.harm.delta:+.4f}"
        )
        summary[f"scripted_{policy.value}"] = {
            "overall": overall,
            "identification": identification,
            "estimation": estimation,
            "ranking_tau": tau,
            "identifiability_f1": f1.f1,
            "naming_control_correctness": control.correctness.delta,
            "naming_control_gain": control.gain.delta,
            "naming_control_harm": control.harm.delta,
        }
        print()

    expected = sum(overalls.values()) / len(overalls)
    summary["scripted_coin_flip"] = expected
    print(f"  scripted agent, expected score of a coin flip on ambiguous edges: {expected:.3f}")
    print(f"      vs oracle data-only (same policy, true structure):            {references['data-only']:.3f}")
    print(
        f"      difference = the cost of finite-sample discovery error:       {references['data-only'] - expected:+.3f}"
    )

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

"""Run the tool-using agent over instances and report scores by naming alignment.

Uses abstract labels (X1..Xk) by default - no CauseNet dump or naming LLM needed, only the agent's own
model. Pass ``--dataset`` to instead evaluate a pre-built dataset (``scripts/build_testset.py``'s
output) - the natural choice when comparing several models, so every model sees identical instances.
Needs a ``.env`` with that model's provider API key (see ``.env.example``).
"""

from __future__ import annotations

import argparse

from causalagentbench.agent.clients import LiteLLMClient
from causalagentbench.agent.eval import aggregate_by_alignment, run_evaluation
from causalagentbench.generation.clients import load_env
from causalagentbench.scm.sample import sample
from causalagentbench.scoring.gates import generate_gated_procedural
from causalagentbench.task.dataset import load_dataset
from causalagentbench.task.instance import build_instance


def main() -> None:
    """Run the agent over gated (or pre-built) instances and print scores by alignment."""
    load_env()
    parser = argparse.ArgumentParser(description="Evaluate a tool-using agent on the benchmark.")
    parser.add_argument("--model", required=True, help="LiteLLM model id, e.g. anthropic/claude-opus-4-8")
    parser.add_argument("--dataset", help="path to a scripts/build_testset.py output stem; overrides -n/--n-samples")
    parser.add_argument("-n", "--n-instances", type=int, default=5, help="number of gated SCM seeds to evaluate")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="rows per instance; default reuses the exact table the oracle gate already validated "
        "(report.data) - pass a value to instead draw a fresh, independent sample at that size, which is "
        "not guaranteed to clear the same oracle bar the gate certified",
    )
    parser.add_argument("--n-drivers", type=int, default=None, help="fix the driver count; default varies per seed")
    parser.add_argument("--max-turns", type=int, default=8, help="tool-call turns before forcing a final answer")
    args = parser.parse_args()

    if args.dataset:
        instances = load_dataset(args.dataset)
    else:
        instances = []
        for seed in range(args.n_instances):
            scm, report = generate_gated_procedural(seed, n_drivers=args.n_drivers)
            if args.n_samples is None:
                assert report.data is not None  # guaranteed: generate_gated_procedural only returns a passing report
                data = report.data
            else:
                data = sample(scm, args.n_samples, seed=seed)
            instances.append(build_instance(scm, data, seed=seed))

    n_drivers = [len(key.scm.scaffold.non_decoy_drivers()) for _, key in instances]

    client = LiteLLMClient(model=args.model)
    scores = run_evaluation(instances, client, max_turns=args.max_turns)

    print(f"model={args.model}  instances={len(instances)}")
    print("-" * 64)
    for i, (n, score) in enumerate(zip(n_drivers, scores, strict=True)):
        level = "name-blind" if score.alignment is None else f"alpha={score.alignment:+.1f}"
        z = f"{score.ranking_tau_z:+.2f}" if score.ranking_tau_z is not None else " n/a"
        print(
            f"  #{i:<3} n_drivers={n:<3} {level:<12} ranking_tau={score.ranking_tau:+.3f} "
            f"(n={score.n_compared:<2} z={z})  "
            f"adj_acc={score.adjustment_accuracy:.3f} abstain_acc={score.abstention_accuracy:.3f} "
            f"parse={score.parse_ok}"
        )
    print("-" * 64)
    for summary in aggregate_by_alignment(scores):
        level = "name-blind" if summary.alignment is None else f"alpha={summary.alignment:+.1f}"
        print(
            f"  {level:<12} n={summary.n:<4} ranking_tau={summary.mean_ranking_tau:+.3f} "
            f"adj_acc={summary.mean_adjustment_accuracy:.3f} abstain_acc={summary.mean_abstention_accuracy:.3f} "
            f"parse_rate={summary.parse_rate:.2f}"
        )


if __name__ == "__main__":
    main()

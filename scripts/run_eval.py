"""Run the tool-using agent over gated instances and report scores by naming alignment.

Uses abstract labels (X1..Xk) by default - no CauseNet dump or naming LLM needed, only the agent's own
model. Needs a ``.env`` with that model's provider API key (see ``.env.example``).
"""

from __future__ import annotations

import argparse

from causalagentbench.agent.clients import LiteLLMClient
from causalagentbench.agent.eval import aggregate_by_alignment, run_evaluation
from causalagentbench.generation.clients import load_env
from causalagentbench.scm.sample import sample
from causalagentbench.scoring.gates import generate_gated
from causalagentbench.task.instance import build_instance


def main() -> None:
    """Generate N gated instances, run the agent on each, and print scores by alignment."""
    load_env()
    parser = argparse.ArgumentParser(description="Evaluate a tool-using agent on the benchmark.")
    parser.add_argument("--model", required=True, help="LiteLLM model id, e.g. anthropic/claude-opus-4-8")
    parser.add_argument("-n", "--n-instances", type=int, default=5, help="number of gated SCM seeds to evaluate")
    parser.add_argument("--n-samples", type=int, default=1000, help="rows per instance")
    parser.add_argument("--max-turns", type=int, default=8, help="tool-call turns before forcing a final answer")
    args = parser.parse_args()

    instances = []
    for seed in range(args.n_instances):
        scm, _ = generate_gated(seed)
        data = sample(scm, args.n_samples, seed=seed)
        instances.append(build_instance(scm, data, seed=seed))

    client = LiteLLMClient(model=args.model)
    scores = run_evaluation(instances, client, max_turns=args.max_turns)

    print(f"model={args.model}  instances={len(instances)}")
    print("-" * 64)
    for summary in aggregate_by_alignment(scores):
        level = "name-blind" if summary.alignment is None else f"alpha={summary.alignment:+.1f}"
        print(
            f"  {level:<12} n={summary.n:<4} ranking_tau={summary.mean_ranking_tau:+.3f} "
            f"adj_acc={summary.mean_adjustment_accuracy:.3f} parse_rate={summary.parse_rate:.2f}"
        )


if __name__ == "__main__":
    main()

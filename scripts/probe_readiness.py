"""Gate G1: is a base model ready for GRPO on this environment? Reads repeated eval runs on the same tasks.

    python scripts/probe_readiness.py outputs/probe/qwen3_4b_s*.jsonl

GRPO learns only from groups whose rewards differ, so the gate needs answers that are valid often enough
and rewards that vary across samples of the same task. Thresholds are the plan's: at least 50% valid, at
most half the tasks with identical rewards across samples, and a median final context under about 6k tokens.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def main(paths: list[str]) -> None:
    """Print the readiness statistics and the gate verdict."""
    rewards: dict[int, list[float]] = defaultdict(list)
    episodes = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["type"] == "episode":
                    episodes.append(record)
                    rewards[record["index"]].append(record["reward"])
    valid = [e for e in episodes if e["parse_ok"]]
    # The final context (last request plus its reply) is what an RL rollout holds; summed prompt tokens
    # would count the re-sent history once per turn.
    tokens = [e["turn_usage"][-1]["prompt_tokens"] + e["turn_usage"][-1]["completion_tokens"] for e in episodes]
    grouped = [r for r in rewards.values() if len(r) > 1]
    tied = sum(1 for r in grouped if max(r) == min(r)) / len(grouped)
    within_sd = statistics.mean(statistics.pstdev(r) for r in grouped)
    valid_rate = len(valid) / len(episodes)
    print(f"episodes {len(episodes)} over {len(rewards)} tasks ({len(paths)} samples each)")
    print(f"valid rate {valid_rate:.2f}")
    print(f"mean reward {statistics.mean(e['reward'] for e in episodes):+.3f}")
    print(f"mean task error on valid answers {statistics.mean(e['task_error'] for e in valid):.3f}")
    print(f"tasks with identical rewards across samples {tied:.2f}; mean within-task reward SD {within_sd:.3f}")
    calls, turns = statistics.mean(e["tool_calls"] for e in episodes), statistics.mean(e["turns"] for e in episodes)
    print(f"median final context {statistics.median(tokens):.0f} tokens; tool calls {calls:.1f}; turns {turns:.1f}")
    passed = valid_rate >= 0.5 and tied <= 0.5 and statistics.median(tokens) < 6000
    print("G1", "PASS" if passed else "FAIL", "(warm start needed)" if valid_rate < 0.2 else "")


if __name__ == "__main__":
    main(sys.argv[1:])

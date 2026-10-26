"""Summarize a GRPO run: training curves and reward-hacking checks from its metrics and rollouts.

    python scripts/rl_report.py outputs/rl/main   # reads metrics.jsonl and rollouts.jsonl, writes curves.png

The checks look for ways to raise the reward without estimating effects: answering zero everywhere, and
reporting effects the agent never obtained from a tool (provenance). Every reported effect should equal, at
the tools' three decimals, a coefficient that a ``regress`` call returned during the same episode.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

_REGRESS = re.compile(r"regress\(.*?\) = ([+-]\d+\.\d+)")


def _windows(values: list[float], size: int) -> list[float]:
    """Trailing moving average, so noisy per-step curves stay readable."""
    return [statistics.mean(values[max(0, i - size + 1) : i + 1]) for i in range(len(values))]


def curves(run: Path) -> list[dict]:
    """Per-step metrics, plotted to ``curves.png``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    rows = [r for r in rows if "reward" in r]
    steps = [r["step"] for r in rows]
    panels = [
        ("reward", "mean reward"),
        ("valid_rate", "valid-answer rate"),
        ("valid_task_error", "task error on valid answers"),
        ("tools/call_frequency", "tool calls per rollout"),
    ]
    figure, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.2))
    for axis, (key, title) in zip(axes, panels, strict=True):
        values = [r.get(key, float("nan")) for r in rows]
        axis.plot(steps, values, color="#9db4d0", linewidth=1)
        axis.plot(steps, _windows(values, 10), color="#1f4e8c", linewidth=2)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("step")
    figure.tight_layout()
    figure.savefig(run / "curves.png", dpi=150)
    return rows


def hacking_checks(run: Path) -> dict[str, float]:
    """Shares of rollouts that answer zero everywhere, or report effects no regress call returned."""
    zero = untraced = valid = total = 0
    for line in (run / "rollouts.jsonl").read_text().splitlines():
        record = json.loads(line)
        total += 1
        if not record.get("parse_ok"):
            continue
        valid += 1
        messages = record["completion"] if isinstance(record["completion"], list) else []
        returned = {
            float(value)
            for message in messages
            if message.get("role") == "tool"
            for value in _REGRESS.findall(str(message.get("content", "")))
        }
        answer = _last_answer(messages)
        effects = [float(v["effect"]) for v in answer.get("verdicts", [])] if answer else []
        if effects and all(abs(e) < 1e-9 for e in effects):
            zero += 1
        if any(all(abs(e - r) > 5e-4 for r in returned) for e in effects):
            untraced += 1
    return {
        "rollouts": total,
        "valid_rate": valid / total,
        "all_zero_share_of_valid": zero / max(valid, 1),
        "untraced_effect_share_of_valid": untraced / max(valid, 1),
    }


def _last_answer(messages: list[dict]) -> dict | None:
    """The JSON inside the final assistant message's last <answer> block, if any."""
    for message in reversed(messages):
        if message.get("role") == "assistant":
            blocks = re.findall(r"<answer>(.*?)</answer>", str(message.get("content") or ""), re.DOTALL)
            try:
                return json.loads(blocks[-1]) if blocks else None
            except json.JSONDecodeError:
                return None
    return None


def main(run: Path) -> None:
    """Print first-versus-last-window metrics and the reward-hacking checks."""
    rows = curves(run)
    for key in ("reward", "valid_rate", "valid_task_error", "tools/call_frequency", "frac_reward_zero_std"):
        values = [r[key] for r in rows if key in r and r[key] == r[key]]
        head, tail = values[:10], values[-10:]
        print(f"{key:24s} first 10 steps {statistics.mean(head):+.3f}  last 10 steps {statistics.mean(tail):+.3f}")
    print(json.dumps(hacking_checks(run), indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

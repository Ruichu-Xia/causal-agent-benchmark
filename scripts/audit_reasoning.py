r"""Audit the text causal-reasoning generator on every evaluation split, with the thresholds of the v2 design.

    python scripts/audit_reasoning.py --n 10000 --seed 11

Checks (``docs/environment_v2.md``): every answer re-derives from its problem; on yes/no questions every
named shortcut and the best constant answer score within [0.45, 0.55] (discovery: the best constant
within each statement type and mode at most 0.56); on choice questions the best constant and the
shortcut score at most 0.35; on numeric questions the best constant scores at most 0.15. Exits non-zero
if any check fails.
"""

from __future__ import annotations

import argparse
import sys

from causalagentbench.reasoning.audit import audit
from causalagentbench.reasoning.dataset import SPLITS, build, load_concepts


def failures(report: dict) -> list[str]:
    """Every threshold the audit report breaks."""
    out = [f"label mismatch {i}" for i in report["label_mismatches"]]
    for name, group in report["groups"].items():
        kind = name.split("/")[1]
        scores = {"best constant": group["best_constant"], "shortcut": group["shortcut"]}
        for label, value in scores.items():
            if value is None:
                continue
            if kind == "yesno" and not 0.45 <= value <= 0.55:
                out.append(f"{name}: {label} {value}")
            if kind == "choice" and value > 0.35:
                out.append(f"{name}: {label} {value}")
            if kind == "number" and label == "best constant" and value > 0.15:
                out.append(f"{name}: {label} {value}")
        if group.get("best_constant_per_cell", 0) > 0.56:
            out.append(f"{name}: best constant per cell {group['best_constant_per_cell']}")
    return out


def main() -> None:
    """Build and audit every evaluation split; print the report and fail on any broken threshold."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    broken = []
    for split in SPLITS[1:]:
        report = audit(build(args.n, args.seed, load_concepts(), split))
        print(f"== {split}: {report['tasks']} tasks, {len(report['label_mismatches'])} label mismatches")
        for name, group in report["groups"].items():
            print(f"  {name:22s} {group}")
        broken += [f"{split} {f}" for f in failures(report)]
    print("FAILED:\n  " + "\n  ".join(broken) if broken else "all checks passed")
    sys.exit(1 if broken else 0)


if __name__ == "__main__":
    main()

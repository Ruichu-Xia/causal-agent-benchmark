"""Assemble the final results table: models against the non-LLM baselines, plus the naming verdict.

The eval runner reports one model at a time and knows nothing about the scripted agent. This joins the two
so the numbers that settle the paper's claims sit in one table:

* does the model beat the **scripted agent**? Below it, the model loses to fifty lines of causal-learn and
  no naming claim is interesting yet.
* does it beat **data-only**? No name-blind policy can, so exceeding it is direct evidence of name use.
* does **correctness** (aligned - misaligned) clear zero? That is the only contrast holding "these columns
  carry domain names" fixed while varying whether the names are *true*, and it is the one the benchmark's
  central claim rests on. ``gain`` (aligned - blind) conflates it with "any names help" and measured live
  the two disagreed.

Contrasts are recomputed here from the raw per-driver rows rather than read from the record, so one
analysis path serves every eval file regardless of which reporting fields the runner wrote at the time.

    python scripts/report_results.py --eval outputs/haiku_full_v2.jsonl --scripted outputs/scripted.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalagentbench.agent.eval import CONTRAST_METRICS, paired_bootstrap_ci, scm_group_key
from causalagentbench.task.dataset import load_dataset

#: How much larger a contrast must be on unidentifiable drivers to call the effect concentrated rather than
#: general. Fixed in advance, and deliberately modest: the claim is "tracks identifiability", which a factor
#: of two supports and a factor of 1.1 does not.
CONCENTRATION_RATIO = 2.0

_PAIRS = (("correctness", 1.0, -1.0), ("gain", 1.0, None), ("harm", None, -1.0))


def _summaries(path: Path) -> list[dict]:
    lines = (line for line in path.read_text().splitlines() if line.strip())
    return [r for r in (json.loads(line) for line in lines) if r.get("kind") == "model_summary"]


def _value(row: dict, metric: str, subset: str) -> float | None:
    """One instance's score on ``metric``, optionally restricted to an identifiability class."""
    if subset == "all":
        return row.get(metric)
    if metric not in ("identification", "estimation"):
        return None
    want = subset == "identifiable"
    field = "adjustment_bias" if metric == "identification" else "effect_error"
    values = [d[field] for d in row["drivers"].values() if d["identifiable"] is want]
    return 1.0 - sum(values) / len(values) if values else None


def _contrasts(record: dict, metric: str, subset: str) -> dict[str, tuple[float, tuple[float, float], int]]:
    """Every naming contrast for one metric/subset, paired within SCM."""
    keys = {i: k for i, (_x, k) in enumerate(load_dataset(record["dataset"]))}
    by_scm: dict[str, dict[float | None, float]] = {}
    for index, row in enumerate(record["per_instance"]):
        value = _value(row, metric, subset)
        if value is not None:
            by_scm.setdefault(scm_group_key(keys[index]), {})[row["alignment"]] = value
    out = {}
    for name, a, b in _PAIRS:
        diffs = [c[a] - c[b] for c in by_scm.values() if a in c and b in c]
        if not diffs:
            out[name] = (float("nan"), (float("nan"), float("nan")), 0)
        else:
            out[name] = (sum(diffs) / len(diffs), paired_bootstrap_ci([[d] for d in diffs], seed=0), len(diffs))
    return out


def _clears_zero(ci: tuple[float, float]) -> bool:
    return ci[0] == ci[0] and not (ci[0] <= 0.0 <= ci[1])


def main() -> None:
    """Print the joined results table and the pre-registered naming verdict."""
    parser = argparse.ArgumentParser(description="Assemble the final results table.")
    parser.add_argument("--eval", required=True, nargs="+", help="one or more eval JSONL outputs")
    parser.add_argument("--scripted", required=True, help="run_scripted_baseline.py --out JSON")
    args = parser.parse_args()

    scripted = json.loads(Path(args.scripted).read_text())
    refs = scripted["reference_scores"]
    script_rows = {k: v for k, v in scripted.items() if k.startswith("scripted_") and isinstance(v, dict)}
    best_script = max(v["overall"] for v in script_rows.values())

    print(f"{'policy / model':<34}{'overall':>9}{'ident':>8}{'est':>8}{'tau':>8}")
    print("-" * 67)
    print(f"{'knows the orientation (ceiling)':<34}{refs['knows the orientation']:>9.3f}")
    print(f"{'data-only (name-blind ceiling)':<34}{refs['data-only']:>9.3f}")
    for key, row in sorted(script_rows.items()):
        label = f"scripted agent ({key.removeprefix('scripted_')})"
        print(
            f"{label:<34}{row['overall']:>9.3f}{row['identification']:>8.3f}"
            f"{row['estimation']:>8.3f}{row['ranking_tau']:>+8.3f}"
        )
    print(f"{'correlation only (floor)':<34}{refs['correlation only']:>9.3f}")
    print("-" * 67)

    records = [r for path in args.eval for r in _summaries(Path(path))]
    for record in records:
        print(f"{record['model']:<34}{record['overall']:>9.3f}{'':>8}{'':>8}{'':>8}   (n={record['instances_scored']})")

    for record in records:
        model = record["model"]
        print(f"\n=== {model} ===")
        parse = record.get("parse_rate")
        print(f"  overall {record['overall']:.3f}" + ("" if parse is None else f"   parse_rate {parse:.2f}"))
        print(f"    vs scripted agent {best_script:.3f}: {'BEATS' if record['overall'] > best_script else 'below'}")
        print(
            f"    vs data-only {refs['data-only']:.3f}: "
            f"{'BEATS - direct evidence of name use' if record['overall'] > refs['data-only'] else 'below'}"
        )
        f1 = record.get("identifiability_f1") or {}
        if f1:
            pc = f1.get("mechanical_pc_baseline")
            print(f"  identifiability F1 {f1['f1']:.3f} vs mechanical PC {'n/a' if pc is None else f'{pc:.3f}'}")

        print("\n  naming contrasts, paired within SCM (* = 95% CI clears zero)")
        print(f"    {'metric':<16}{'drivers':<16}{'correctness':>24}{'gain':>24}{'harm':>24}")
        table = {}
        for subset in ("all", "unidentifiable", "identifiable"):
            for metric in CONTRAST_METRICS:
                if subset != "all" and metric not in ("identification", "estimation"):
                    continue
                got = _contrasts(record, metric, subset)
                table[(metric, subset)] = got
                cells = []
                for name, _a, _b in _PAIRS:
                    delta, ci, _n = got[name]
                    cells.append(f"{delta:+.3f}[{ci[0]:+.3f},{ci[1]:+.3f}]{'*' if _clears_zero(ci) else ' '}")
                print(f"    {metric:<16}{subset:<16}{cells[0]:>24}{cells[1]:>24}{cells[2]:>24}")
        print("    correctness = aligned - misaligned   gain = aligned - blind   harm = blind - misaligned")
        print(f"\n  -> {_verdict(table)}")


def _verdict(table: dict) -> str:
    """Apply the pre-registered rule in ``agent/eval.py::NamingResult``, without reinterpretation.

    Keyed on ``correctness``, because that is the contrast the claim is about. An earlier version read only
    ``harm``, and returned "no effect" on a result whose signal was elsewhere entirely.
    """
    moved = [
        (metric, table[(metric, "all")]["correctness"])
        for metric in CONTRAST_METRICS
        if _clears_zero(table[(metric, "all")]["correctness"][1])
    ]
    if not moved:
        return (
            "NO NAMING EFFECT: correctness (aligned - misaligned) does not clear zero on any metric. "
            "Report the null against the 0.197 headroom."
        )

    named = ", ".join(f"{m} {c[0]:+.3f}" for m, c in moved)
    notes = [f"CORRECT NAMES HELP: correctness clears zero on {named}."]

    # The negative control, per metric that moved and can be split by identifiability.
    for metric, _contrast in moved:
        if metric not in ("identification", "estimation"):
            continue
        here = table[(metric, "unidentifiable")]["correctness"][0]
        there = table[(metric, "identifiable")]["correctness"][0]
        if here != here or there != there:
            continue
        if there > here:
            notes.append(
                f"SUSPECT on {metric}: the effect is larger where the data already settles the "
                f"answer ({there:+.3f} vs {here:+.3f}); something travels with the skin."
            )
        elif here >= CONCENTRATION_RATIO * max(there, 1e-9):
            notes.append(
                f"CONCENTRATED on {metric}: {here:+.3f} on unidentifiable vs {there:+.3f} on "
                "identifiable - the effect tracks identifiability."
            )
        else:
            notes.append(
                f"NOT CONCENTRATED on {metric}: {here:+.3f} vs {there:+.3f} - comparable in both "
                "halves, so this is general semantic contamination or a nuisance confound, and "
                "this design cannot separate them."
            )
    notes.append(
        "Subset contrasts have ~2.4 drivers per instance against ~5 overall, so a null there is "
        "weak evidence about concentration rather than evidence against it."
    )
    return " ".join(notes)


if __name__ == "__main__":
    main()

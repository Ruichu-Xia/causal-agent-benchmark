r"""The pre-registered confirmatory analysis of the environment-v2 training study (``docs/preregistration_v2.md``).

    python scripts/confirm_v2.py                          # before the control exists: trained vs base
    python scripts/confirm_v2.py --control v2-control     # the full analysis

For each endpoint: the difference trained - comparator in accuracy (balanced accuracy for Corr2Cause), its 95%
paired cluster-bootstrap interval, and a one-sided bootstrap p-value (20,000 resamples, seed 0). An endpoint's
p-value is the larger of its p-values against base and against control (an intersection-union test), and Holm's
procedure over the six endpoints holds the family-wise error at one-sided 0.025. Without a control, Holm runs on
the p-values against base; that gates training the control, since the larger p-value can only reject less.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from transfer_eval import OUT, _load, interval, items, p_value, paired_draws, scores

ENDPOINTS = {
    "ood_wording_test": False,
    "ood_vocab_test": False,
    "ood_size_test": False,
    "dsep": False,
    "c2c_test": True,
    "cladder_medcol": False,
}
"""Confirmatory endpoints and whether each uses balanced accuracy."""
MANIPULATION_CHECK = "iid_test"
SAMPLES = {
    "iid_test": 1,
    "ood_wording_test": 1,
    "ood_vocab_test": 1,
    "ood_size_test": 1,
    "dsep": 1,
    "c2c_test": 2,
    "cladder_medcol": 4,
}
"""Samples per item on each set."""
SEED = 1000
"""Sample j of every item uses seed SEED + j."""
ALPHA = 0.025
RESAMPLES = 20_000


def holm(p: dict[str, float], alpha: float) -> dict[str, bool]:
    """Holm's step-down procedure: whether each hypothesis is rejected at family-wise level ``alpha``."""
    rejected = {}
    going = True
    for k, name in enumerate(sorted(p, key=lambda key: p[key])):
        going = going and p[name] <= alpha / (len(p) - k)
        rejected[name] = going
    return rejected


def complete(name: str, model: str) -> dict[str, dict[str, float]]:
    """A model's per-item scores on a set, after checking it has exactly the pre-registered replies."""
    expected = {(item["id"], SEED + j) for item in items(name) for j in range(SAMPLES[name])}
    found = Counter((r["id"], r["seed"]) for r in _load(OUT / f"{name}.jsonl") if r["model"] == model)
    if set(found) != expected or max(found.values()) != 1:
        raise ValueError(f"{model} does not have exactly the pre-registered replies on {name}")
    return scores(name, model)


def compare(name: str, balanced: bool, trained: str, comparator: str) -> dict[str, Any]:
    """Difference trained - comparator on one set: point, 95% interval and one-sided p-value."""
    point, draws = paired_draws(name, complete(name, trained), complete(name, comparator), balanced, RESAMPLES)
    low, high = interval(draws, 0.95)
    return {"delta": point, "low": low, "high": high, "p": p_value(draws)}


def main() -> None:
    """Print the manipulation check and every endpoint's comparisons, p-value and Holm decision."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trained", default="v2-trained")
    parser.add_argument("--base", default="v2-base")
    parser.add_argument("--control")
    args = parser.parse_args()
    comparators = [args.base] + ([args.control] if args.control else [])
    check = compare(MANIPULATION_CHECK, False, args.trained, args.base)
    print(f"manipulation check {MANIPULATION_CHECK} vs {args.base}: {_format(check)}")
    results = {
        name: {c: compare(name, balanced, args.trained, c) for c in comparators} for name, balanced in ENDPOINTS.items()
    }
    p = {name: max(r["p"] for r in by_comparator.values()) for name, by_comparator in results.items()}
    rejected = holm(p, ALPHA)
    for name, by_comparator in results.items():
        cells = "; ".join(f"vs {c}: {_format(r)}" for c, r in by_comparator.items())
        print(f"{name}: {cells}; p = {p[name]:.5f}; {'CONFIRMED' if rejected[name] else 'not confirmed'}")
    if not args.control:
        print("no control: decisions use p against base only and gate the control run, not a claim")


def _format(result: dict[str, Any]) -> str:
    """One comparison as 'delta [low, high], p' in percentage points."""
    delta, low, high = (100 * result[key] for key in ("delta", "low", "high"))
    return f"{delta:+.1f} [{low:+.1f}, {high:+.1f}], p = {result['p']:.5f}"


if __name__ == "__main__":
    main()

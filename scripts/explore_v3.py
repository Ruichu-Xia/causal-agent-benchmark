r"""Scope conditions and exploratory analyses of the environment-v3 study (``docs/preregistration_v3.md``).

    python scripts/explore_v3.py                      # trained vs base
    python scripts/explore_v3.py --control v3-control

Every reply set is first checked by ``confirm_v3.scores`` (registration, regrade, manifest). The scope
conditions recompute an endpoint's difference on a subset of items or under the strict QRData reading,
with the registered weights restricted to that subset and the registered bootstrap; nothing here confirms an
endpoint.
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from confirm_v3 import (
    ENDPOINTS,
    MANIPULATION_CHECK,
    SETS,
    _replies,
    bootstrap_interval,
    cluster,
    items,
    scores,
    weights,
)

from causalagentbench.dataenv import qrdata
from causalagentbench.dataenv.dataset import world_from_dict
from causalagentbench.dataenv.families import TOLERANCE, _population_regression


def replies(name: str, arm: str) -> dict[str, list[dict[str, Any]]]:
    """An arm's checked replies on a set, by item id."""
    scores(name, arm)
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in _replies(name, arm):
        out[r["id"]].append(r)
    return out


def answered(r: dict[str, Any]) -> bool:
    """Whether a reply ended with an answer line within the budget."""
    return bool(r["answered"]) and not r["truncated"]


def interval_on(name: str, ids: set[str], a: dict[str, float], b: dict[str, float]) -> tuple[float, float, float, int]:
    """Difference a - b on the given items, with the registered weights recomputed on them, and its interval."""
    subset = [item for item in items(name) if item["id"] in ids]
    w = weights(name, subset)
    sums: dict[str, float] = defaultdict(float)
    for item in subset:
        sums[cluster(name, item)] += w[item["id"]] * (a[item["id"]] - b[item["id"]])
    differences = np.array([sums[key] for key in sorted(sums)])
    low, high = bootstrap_interval(differences)
    return float(differences.sum()), low, high, len(subset)


def strict_scores(name: str, by_id: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """Per-item QRData scores with ``strict_choice`` for multiple-choice items; numbers keep the official rule."""
    tasks = {item["id"]: item for item in items(name)}
    out = {}
    for i, rs in by_id.items():
        if tasks[i]["kind"] != "qrdata_choice":
            out[i] = statistics.mean(float(r["reward"]) for r in rs)
            continue
        values = []
        for r in rs:
            reply = "" if r["truncated"] else str(r["messages"][-1].get("content") or "")
            values.append(qrdata.strict_choice(tasks[i]["answer"], reply))
        out[i] = statistics.mean(values)
    return out


def _fmt(delta: float, low: float, high: float, n: int) -> str:
    """A difference and interval in points, with the item count."""
    return f"{100 * delta:+.1f} [{100 * low:+.1f}, {100 * high:+.1f}] on {n} items"


def scope(trained: str, comparators: list[str]) -> None:
    """The registered scope conditions for every endpoint (and the manipulation check)."""
    print("== scope conditions (95% bootstrap intervals; described as skill only if above zero) ==")
    for name in [MANIPULATION_CHECK, *ENDPOINTS]:
        mine, theirs = replies(name, trained), {arm: replies(name, arm) for arm in comparators}
        everyone = [mine, *theirs.values()]
        ids = {i for i in mine if all(answered(r) for arm in everyone for r in arm[i])}
        for arm in comparators:
            result = interval_on(name, ids, scores(name, trained), scores(name, arm))
            print(f"{name} vs {arm}, answered by all: {_fmt(*result)}")
            if name == "qrdata_causal":
                a, b = strict_scores(name, mine), strict_scores(name, theirs[arm])
                print(f"{name} vs {arm}, strict choices: {_fmt(*interval_on(name, set(a), a, b))}")


def rates(arms: list[str]) -> None:
    """Truncation, no-answer, tool-failure rates, tool calls and tokens per arm and set."""
    print("\n== rates per arm and set ==")
    for name in SETS:
        for arm in arms:
            rs = [r for group in replies(name, arm).values() for r in group]
            tool = Counter()
            for r in rs:
                for m in r["messages"]:
                    if m["role"] == "tool":
                        text = str(m["content"])
                        tool["timeout"] += "did not finish within" in text
                        tool["startup"] += "failed to start" in text
                        tool["exited"] += "had exited" in text
            print(
                f"{name:18} {arm:11} n={len(rs):5} truncated {statistics.mean(r['truncated'] for r in rs):.3f} "
                f"no answer {statistics.mean(not answered(r) for r in rs):.3f} tool calls "
                f"{statistics.mean(r['tool_calls'] for r in rs):.2f} "
                f"tokens {statistics.mean(r['completion_tokens'] for r in rs):.0f} "
                f"timeouts {tool['timeout']} startup failures {tool['startup']} exits {tool['exited']}"
            )


def _mean(values: list[float]) -> str:
    """A mean with its count, or a dash."""
    return f"{statistics.mean(values):.3f} (n={len(values)})" if values else "-"


def by_cell(name: str, arms: list[str], key: Any) -> None:
    """Accuracy per arm within the cells ``key(item)`` defines (None leaves an item out)."""
    cells: dict[Any, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_arm = {arm: scores(name, arm) for arm in arms}
    for item in items(name):
        cell = key(item)
        if cell is not None:
            for arm in arms:
                cells[cell][arm].append(per_arm[arm][item["id"]])
    for cell in sorted(cells, key=str):
        print(f"  {name} {cell}: " + "; ".join(f"{arm} {_mean(cells[cell][arm])}" for arm in arms))


def shortcut_cells(item: dict[str, Any]) -> list[str]:
    """The family's shortcut cells an item falls in, as on dev (phase C)."""
    m, family = item["meta"], item["family"]
    if family == "independence":
        out = [f"marginal {'right' if m['marginal_right'] else 'wrong'}"]
        if "names_right" in m:
            out.append(f"names {'right' if m['names_right'] else 'wrong'}")
        return out
    if family == "structure":
        if "cell" in m:
            return [f"cell {m['cell']}"]
        return [f"determined {m['determined']}"]
    if family == "adjustment":
        return [f"confounder rule {'right' if all(c['confounder_right'] for c in m['cells']) else 'wrong'}"]
    both = "both shortcuts wrong" if not m["unadjusted_right"] and not m["adjust_all_right"] else "a shortcut right"
    return [f"unadjusted {'right' if m['unadjusted_right'] else 'wrong'}", both]


def column_order(item: dict[str, Any]) -> str | None:
    """For a related structure item, whether 'the earlier column causes the later' is right."""
    m = item["meta"]
    if item["family"] != "structure" or m["truth"] == "none":
        return None
    return f"column-order rule {'right' if (m['truth'] == 'ab') == (m['x'] < m['y']) else 'wrong'}"


def effect_landing(name: str, arm: str) -> Counter:
    """Where effect answers land on items whose two regression shortcuts are both wrong."""
    landing: Counter = Counter()
    by_id = replies(name, arm)
    for item in items(name):
        m = item["meta"]
        if item["family"] != "effect" or m["unadjusted_right"] or m["adjust_all_right"]:
            continue
        world = world_from_dict(item["world"])
        x, y = m["x"], m["y"]
        others = [v for v in range(world.n) if v not in (x, y)]
        for r in by_id[item["id"]]:
            found = re.search(
                r"Answer:\s*\**\s*(-?\d+(?:\.\d+)?)",
                "" if r["truncated"] else str(r["messages"][-1].get("content") or ""),
            )
            if not found:
                landing["no number"] += 1
                continue
            value = float(found.group(1))
            near = [
                label
                for label, target in (
                    ("truth", m["effect"]),
                    ("unadjusted", world.sigma[x, y]),
                    ("adjust_all", _population_regression(world, x, y, others)),
                )
                if abs(value - target) <= TOLERANCE
            ]
            landing["/".join(near) or "none"] += 1
    return landing


def qrdata_details(arms: list[str]) -> None:
    """Official accuracy, by question type, 'no relationship' rate, prefix-only rights, rounding misses."""
    print("\n== QRData ==")
    for name in ("qrdata_causal", "qrdata_statistics"):
        tasks = {item["id"]: item for item in items(name)}
        for arm in arms:
            by_id = replies(name, arm)
            rs = [r for group in by_id.values() for r in group]
            official = statistics.mean(float(r["reward"]) for r in rs)
            prefix_only = none_said = none_offered = rounding = 0
            for r in rs:
                task = tasks[r["id"]]
                reply = "" if r["truncated"] else str(r["messages"][-1].get("content") or "")
                if task["kind"] == "qrdata_choice":
                    prefix_only += r["reward"] == 1.0 and qrdata.strict_choice(task["answer"], reply) == 0.0
                    options = {
                        line[0].lower(): line for line in task["prompt"].splitlines() if re.match(r"[A-D]\. ", line)
                    }
                    none_letter = next((k for k, v in options.items() if "no causal" in v.lower()), None)
                    if none_letter:
                        none_offered += 1
                        none_said += qrdata.strict_choice(none_letter, reply) == 1.0
                elif r["reward"] == 0.0:
                    text = qrdata.final_answer(reply) or ""
                    number = qrdata._first_number(text)
                    gold = task["answer"]
                    decimals = len(number.split(".")[1]) if number and "." in number else 0
                    if number and not gold.endswith("%") and not number.endswith("%"):
                        rounding += round(float(gold), decimals) == round(float(number), decimals) and decimals < 3
            print(
                f"{name} {arm}: official accuracy {official:.3f} over {len(rs)} replies; right only under the prefix "
                f"rule {prefix_only}; says 'no causal relationship' {none_said}/{none_offered}; numerical misses "
                f"that match gold at the reply's own rounding {rounding}"
            )
        for arm in arms:
            per = scores(name, arm)
            kinds: dict[str, list[float]] = defaultdict(list)
            for item in items(name):
                keywords = item["meta"]["keywords"]
                kinds[f"{keywords[1] if len(keywords) > 1 else keywords[0]} / {item['kind']}"].append(per[item["id"]])
            print(f"  {name} {arm} by type: " + "; ".join(f"{k} {_mean(v)}" for k, v in sorted(kinds.items())))


def bnrep_details(arms: list[str]) -> None:
    """Placeholder names, the real-minus-placeholder gap per arm, and per-network results."""
    print("\n== bnRep ==")
    for arm in arms:
        real, placeholder = scores("bnrep_real", arm), scores("bnrep_placeholder", arm)
        gap = statistics.mean(real[i] for i in real) - statistics.mean(
            placeholder[i.replace("-real", "-placeholder")] for i in real
        )
        mean_real, mean_placeholder = statistics.mean(real.values()), statistics.mean(placeholder.values())
        print(f"{arm}: real {mean_real:.3f}, placeholder {mean_placeholder:.3f}, gap {gap:+.3f}")
    for name in ("bnrep_real", "bnrep_placeholder"):
        by_cell(name, arms, lambda item: f"network {item['network']}")


def main() -> None:
    """Print the scope conditions and every pre-registered exploratory analysis."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trained", default="v3-trained")
    parser.add_argument("--base", default="v3-base")
    parser.add_argument("--control")
    args = parser.parse_args()
    comparators = [args.base] + ([args.control] if args.control else [])
    arms = [args.trained, *comparators]
    scope(args.trained, comparators)
    rates(arms)
    print("\n== by family, shortcut cell and column order ==")
    for name in SETS:
        if name.startswith("qrdata"):
            continue
        by_cell(name, arms, lambda item: f"family {item['family']} / {item['kind']}")
        for k in range(2):
            by_cell(
                name,
                arms,
                lambda item, k=k: f"{item['family']}: {cells[k]}" if len(cells := shortcut_cells(item)) > k else None,
            )
        if name.endswith("_test"):
            by_cell(name, arms, column_order)
    print("\n== effect items where both regression shortcuts are wrong: where answers land ==")
    for name in SETS:
        if not name.startswith("qrdata"):
            for arm in arms:
                print(f"{name} {arm}: {dict(effect_landing(name, arm).most_common())}")
    qrdata_details(arms)
    bnrep_details(arms)
    print("\n== gap to iid_test per arm ==")
    for arm in arms:
        iid = statistics.mean(scores(MANIPULATION_CHECK, arm).values())
        print(
            f"{arm}: "
            + "; ".join(
                f"{s} {statistics.mean(scores(s, arm).values()) - iid:+.3f}"
                for s in ("ood_wording_test", "ood_size_test", "ood_vocab_test")
            )
        )


if __name__ == "__main__":
    main()

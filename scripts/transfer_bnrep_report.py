r"""Near-transfer report: paired bnRep comparisons overall, on tasks every model answered validly, and per network.

    python scripts/transfer_bnrep_report.py trained base control

Reads ``outputs/transfer/bnrep_<model>.jsonl`` (real names) and ``bnrep_<model>_x.jsonl`` (placeholders) and
compares the first model with each other one by the paired difference in pooled capped error. The overall
comparison uses the pre-registered 97.5% interval; the validity split and per-network rows are exploratory
and use 95%.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from causalagentbench.scoring.report import compare, summarize

OUT = Path("outputs/transfer")
GRAPHS = ("expenditure", "lexical", "algal2", "liquefaction")
Z = {0.95: 1.959964, 0.975: 2.241403}


def episodes(name: str) -> dict[int, dict[str, Any]]:
    """Scored episodes of one results file, by task index."""
    rows = [json.loads(line) for line in (OUT / f"bnrep_{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    return {r["index"]: r for r in rows if r["type"] == "episode"}


def difference(a: dict[int, dict[str, Any]], b: dict[int, dict[str, Any]], ids: list[int], level: float) -> str:
    """Paired difference a - b in pooled capped error on ``ids``, with a normal interval at ``level``."""
    d = compare([a[i] for i in ids], [b[i] for i in ids])["difference"]
    half = Z[level] * d["se"]
    return f"n={len(ids)} {d['value']:+.3f} [{d['value'] - half:+.3f}, {d['value'] + half:+.3f}] ({level:.1%})"


def main() -> None:
    """Print the near-transfer comparisons for both naming conditions."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", help="the first model is compared with each of the others")
    models = parser.parse_args().models
    records = json.loads((OUT / "bnrep.json").read_text(encoding="utf-8"))["records"]
    network = [record["producer"].removeprefix("bnrep:") for record in records]
    first, *others = models
    for suffix, condition in (("", "real names"), ("_x", "placeholders")):
        runs = {m: episodes(m + suffix) for m in models}
        print(f"== {condition}")
        for m, run in runs.items():
            s = summarize(list(run.values()))
            print(f"  {m}: error {s['capped_error']['value']:.3f}, valid {s['answer_valid_rate']:.3f}")
        shared = sorted(set.intersection(*[set(run) for run in runs.values()]))
        valid = [i for i in shared if all(run[i]["parse_ok"] for run in runs.values())]
        for other in others:
            print(f"  {first} - {other}, all: {difference(runs[first], runs[other], shared, 0.975)}")
            print(f"  {first} - {other}, all valid: {difference(runs[first], runs[other], valid, 0.95)}")
            for graph in GRAPHS:
                ids = [i for i in valid if network[i] == graph]
                print(f"    {graph}: {difference(runs[first], runs[other], ids, 0.95)}")


if __name__ == "__main__":
    main()

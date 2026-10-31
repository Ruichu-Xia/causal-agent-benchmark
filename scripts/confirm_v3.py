r"""The pre-registered confirmatory analysis of environment v3 (``docs/preregistration_v3.md``).

    python scripts/confirm_v3.py hash                     # the registered item-list hashes
    python scripts/confirm_v3.py record v3-base iid_test  # right after collecting a set's replies, then commit
    python scripts/confirm_v3.py analyze                  # before the control exists: the gate
    python scripts/confirm_v3.py analyze --control v3-control

The trained arm is compared with the base model and, once it exists, the shuffled-reward control. Each
comparison reports the difference in score, its 95% paired cluster-bootstrap interval (descriptive) and a
one-sided p-value from a cluster sign-flip randomization test (20,000 resamples). An endpoint's p-value is the
largest over its comparators, and Holm's procedure over the five endpoints holds the family-wise error at
one-sided 0.025. Before the control exists the same procedure decides the gate.

Scores are weighted means of per-item scores (each item's mean reward over its samples). On every set but
``qrdata_causal`` each item weighs 1/n. On ``qrdata_causal``, multiple-choice items are balanced over gold
answers within each group of questions sharing a type and choice set: every gold answer's items weigh the same
in total, so no answering habit that ignores the item scores more than 1/K of a group (K its number of distinct
gold answers), however it prefers options. Groups weigh by their size and each numerical item weighs 1/n.

Every input is checked first: item lists (``REGISTERED``), the prompt each reply answered, each arm's model,
revision, adapter digest, token limit, vLLM version and sampling (one configuration per arm), and each arm's
replies against the manifest committed when they were collected (``RECORDS``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from causalagentbench.dataenv.environment import grade_reply

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "outputs" / "dataenv" / "eval"
RECORDS = ROOT / "docs" / "preregistration_v3_records.json"
SETS = {
    "iid_test": ("outputs/dataenv/v3/iid_test.jsonl", 1),
    "ood_wording_test": ("outputs/dataenv/v3/ood_wording_test.jsonl", 1),
    "ood_size_test": ("outputs/dataenv/v3/ood_size_test.jsonl", 1),
    "ood_vocab_test": ("outputs/dataenv/v3/ood_vocab_test.jsonl", 1),
    "qrdata_causal": ("outputs/dataenv/transfer/qrdata_causal.jsonl", 4),
    "bnrep_real": ("outputs/dataenv/transfer/bnrep_real.jsonl", 2),
    "qrdata_statistics": ("outputs/dataenv/transfer/qrdata_statistics.jsonl", 4),
    "bnrep_placeholder": ("outputs/dataenv/transfer/bnrep_placeholder.jsonl", 2),
}
"""Set name -> (task file, samples per item). Sample j of every item uses seed SEED + j."""
MANIPULATION_CHECK = "iid_test"
ENDPOINTS = ("ood_wording_test", "ood_size_test", "ood_vocab_test", "qrdata_causal", "bnrep_real")
REGISTERED = {
    "iid_test": "7d761a225299c6caadd9e2efbf72bbee8d147185f356b94f56097b85f8c1116e",
    "ood_wording_test": "28ac5e146b57af787082be7ab76b4398e9c65fc5b30ae2727907b617f44027ba",
    "ood_size_test": "83a539bb87d20f800dbfcb4085dc8fa5462d96e9091838101aa07b0abd6b6fb7",
    "ood_vocab_test": "1e5717b94f20777880bc431b091356872bae4e303626189709197cce119b57d1",
    "qrdata_causal": "d8c36a59640a2534742fb0158d396dcc020d9715783465d3479fb11b6ee919b7",
    "bnrep_real": "45bef4f8664f8ebc195638cd727ce1df887381d4f3eee47b31230835b68c4849",
    "qrdata_statistics": "6ddd936604bf048492df4d0914c2dff478e64a3836b9c1b3a65b6383d29eeabe",
    "bnrep_placeholder": "ccc1dcd6ec0e89655de0d715912b629a9f21911f0efd8d8ea25c8833f8e8c02a",
}
"""SHA-256 of each set's items (``hash``)."""
SEED = 1000
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MAX_TOKENS = 8192
VLLM = "0.31.0"
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
ADAPTERS = {"v3-base": "", "v3-trained": "data-pilot/adapter", "v3-control": "data-control/adapter"}
ADAPTER_SHA256 = {
    "v3-base": None,
    "v3-trained": "722501e468587c49e50e314bca62b6a62b1ed9d0d1502d6fafcfbb68bc514547",
    "v3-control": "80c8fa815f66f9f55753e1ca21577a31b44e524622770f5623388e228ac70e2e",
}
"""The control's digest is committed right after it is trained, before it is evaluated."""
RESAMPLES = 20_000
ALPHA = 0.025


def items(name: str) -> list[dict[str, Any]]:
    """A set's task records, in id order."""
    path = ROOT / SETS[name][0]
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return sorted(records, key=lambda record: record["id"])


def fingerprint(name: str) -> str:
    """SHA-256 of each item's id, prompt, answer and answer format, in id order."""
    rows = [[item["id"], item["prompt"], item["answer"], item["kind"]] for item in items(name)]
    return hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest()


def cluster(name: str, item: dict[str, Any]) -> str:
    """The resampling unit: the data files on QRData (several questions share a file), else the item."""
    return "+".join(item["files"]) if name.startswith("qrdata") else item["id"]


def weights(name: str, item_list: list[dict[str, Any]]) -> dict[str, float]:
    """Each item's weight in the set's score (summing to 1); see the module docstring."""
    n = len(item_list)
    if name != "qrdata_causal":
        return {item["id"]: 1 / n for item in item_list}
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for item in item_list:
        if item["kind"] == "qrdata_choice":
            source = json.dumps([item["meta"]["keywords"], _choices(item)])
            groups[source].append(item)
    out = {item["id"]: 1 / n for item in item_list if item["kind"] != "qrdata_choice"}
    for group in groups.values():
        cells = Counter(item["answer"] for item in group)
        for item in group:
            out[item["id"]] = len(group) / n / (len(cells) * cells[item["answer"]])
    return out


def _choices(item: dict[str, Any]) -> str:
    """The answer instruction's choice list, which names the item's choice set."""
    return item["prompt"].rsplit("where <answer> is ", 1)[1]


def _replies(name: str, arm: str) -> list[dict[str, Any]]:
    """An arm's reply records on a set, in (id, seed) order."""
    path = EVAL / f"{arm}-{name}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return sorted(records, key=lambda record: (record["id"], record["seed"]))


def _digest(records: list[dict[str, Any]]) -> str:
    """SHA-256 of reply records in canonical JSON."""
    return hashlib.sha256("\n".join(json.dumps(r, sort_keys=True) for r in records).encode("utf-8")).hexdigest()


SHARED = ("base_model", "revision", "max_completion_tokens", "vllm", "sampling", "eval_code", "packages")
"""Settings every arm must share: the same model, generation, evaluation code and packages."""


def configuration(record: dict[str, Any]) -> str:
    """The generation settings of one reply, as a comparable string."""
    keys = ("adapter", "adapter_sha256", *SHARED)
    return json.dumps({key: record.get(key) for key in keys}, sort_keys=True)


def check(name: str, arm: str) -> str:
    """Check a set's items and one arm's replies against the registration; return the arm's configuration."""
    if fingerprint(name) != REGISTERED.get(name):
        raise ValueError(f"the items of {name} differ from the registered list")
    records = _replies(name, arm)
    expected = {(item["id"], SEED + j) for item in items(name) for j in range(SETS[name][1])}
    found = Counter((r["id"], r["seed"]) for r in records)
    if set(found) != expected or max(found.values()) != 1:
        raise ValueError(f"{arm} does not have exactly the registered replies on {name}")
    prompts = {item["id"]: item["prompt"] for item in items(name)}
    if any(r["messages"][0]["content"] != prompts[r["id"]] for r in records):
        raise ValueError(f"some {arm} replies on {name} answered a prompt other than the registered one")
    settings = {configuration(r) for r in records}
    if len(settings) != 1:
        raise ValueError(f"{arm} replies on {name} come from more than one configuration")
    setting = json.loads(next(iter(settings)))
    wanted = {
        "base_model": MODEL,
        "revision": REVISION,
        "adapter": ADAPTERS[arm],
        "adapter_sha256": ADAPTER_SHA256[arm],
        "max_completion_tokens": MAX_TOKENS,
        "vllm": VLLM,
        "sampling": SAMPLING,
    }
    if any(setting[key] != value for key, value in wanted.items()):
        raise ValueError(f"{arm} on {name} was generated with {setting}, not {wanted}")
    if not setting["eval_code"] or not setting["packages"]:
        raise ValueError(f"{arm} replies on {name} do not record the evaluation code and packages")
    by_id = {item["id"]: item for item in items(name)}
    for r in records:
        reply = "" if r["truncated"] else str(r["messages"][-1].get("content") or "")
        if grade_reply(by_id[r["id"]], reply) != r["reward"]:
            raise ValueError(f"{arm} reply {r['id']} seed {r['seed']} on {name}: stored reward differs from a regrade")
    return next(iter(settings))


def scores(name: str, arm: str) -> dict[str, float]:
    """An arm's per-item mean reward on a set, after checking it against the registration and the manifest."""
    check(name, arm)
    manifest = json.loads(RECORDS.read_text(encoding="utf-8")) if RECORDS.exists() else {}
    records = _replies(name, arm)
    if manifest.get(f"{name}/{arm}") != _digest(records):
        raise ValueError(f"{arm} replies on {name} do not match the committed manifest {RECORDS.name}")
    rewards: dict[str, list[float]] = defaultdict(list)
    for r in records:
        rewards[r["id"]].append(float(r["reward"]))
    return {i: float(np.mean(values)) for i, values in rewards.items()}


def record(arm: str, names: list[str]) -> None:
    """Add an arm's reply digests on the given sets to the manifest; refuses to overwrite an entry.

    Each set is recorded right after its replies are collected, before any score is computed.
    """
    for name in names:
        check(name, arm)
    manifest = json.loads(RECORDS.read_text(encoding="utf-8")) if RECORDS.exists() else {}
    for name in names:
        key = f"{name}/{arm}"
        if key in manifest:
            raise ValueError(f"{key} is already recorded")
        manifest[key] = _digest(_replies(name, arm))
    RECORDS.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"recorded {arm} on {', '.join(names)} in {RECORDS.name}; commit it before any analysis")


def cluster_differences(name: str, a: dict[str, float], b: dict[str, float]) -> np.ndarray:
    """Each cluster's weighted score difference a - b, in cluster order; they sum to the score difference."""
    item_list = items(name)
    w = weights(name, item_list)
    sums: dict[str, float] = defaultdict(float)
    for item in item_list:
        sums[cluster(name, item)] += w[item["id"]] * (a[item["id"]] - b[item["id"]])
    return np.array([sums[key] for key in sorted(sums)])


def sign_flip_p(differences: np.ndarray) -> float:
    """One-sided p for a > b: every cluster's difference flips sign with probability 1/2 (numpy seed 0).

    p = (1 + number of resampled sums >= observed) / (1 + resamples); a small tolerance keeps resamples equal
    to the observed sum from being lost to rounding.
    """
    signs = np.random.default_rng(0).choice([-1.0, 1.0], size=(RESAMPLES, len(differences)))
    exceed = int(np.sum(signs @ differences >= differences.sum() - 1e-12))
    return (1 + exceed) / (1 + RESAMPLES)


def bootstrap_interval(differences: np.ndarray) -> tuple[float, float]:
    """95% percentile interval of the score difference from resampled clusters (numpy seed 0), descriptive.

    A resample of k clusters sums their differences and rescales by (number of clusters) / k = 1, so weights
    stay fixed at their full-set values.
    """
    rng = np.random.default_rng(0)
    draws = np.sort(
        [differences[rng.integers(len(differences), size=len(differences))].sum() for _ in range(RESAMPLES)]
    )
    return float(draws[int(0.025 * RESAMPLES)]), float(draws[int(0.975 * RESAMPLES) - 1])


def compare(name: str, a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    """Difference a - b on one set: point, 95% bootstrap interval and sign-flip p-value."""
    differences = cluster_differences(name, a, b)
    low, high = bootstrap_interval(differences)
    return {"delta": float(differences.sum()), "low": low, "high": high, "p": sign_flip_p(differences)}


def holm(p: dict[str, float], alpha: float) -> dict[str, bool]:
    """Holm's step-down procedure: whether each hypothesis is rejected at family-wise level ``alpha``."""
    rejected = {}
    going = True
    for k, name in enumerate(sorted(p, key=lambda key: p[key])):
        going = going and p[name] <= alpha / (len(p) - k)
        rejected[name] = going
    return rejected


def _format(result: dict[str, float]) -> str:
    """One comparison as 'delta [low, high], p' in percentage points."""
    delta, low, high = (100 * result[key] for key in ("delta", "low", "high"))
    return f"{delta:+.1f} [{low:+.1f}, {high:+.1f}], p = {result['p']:.5f}"


def analyze(trained: str, base: str, control: str | None) -> None:
    """Print the manipulation check, every endpoint's comparisons, p-value and Holm decision."""
    arms = [base] + ([control] if control else [])
    shared = set()
    for arm in [trained, *arms]:
        settings = {check(name, arm) for name in SETS}
        if len(settings) != 1:
            raise ValueError(f"{arm} used more than one configuration across sets")
        setting = json.loads(next(iter(settings)))
        shared.add(json.dumps({key: setting[key] for key in SHARED}, sort_keys=True))
    if len(shared) != 1:
        raise ValueError("the arms were evaluated with different code, packages or settings")
    result = compare(MANIPULATION_CHECK, scores(MANIPULATION_CHECK, trained), scores(MANIPULATION_CHECK, base))
    worked = "passes" if result["p"] <= ALPHA else "fails: a null elsewhere says nothing about generalization"
    print(f"manipulation check {MANIPULATION_CHECK} vs {base}: {_format(result)}; {worked}")
    results = {}
    for name in ENDPOINTS:
        mine = scores(name, trained)
        results[name] = {arm: compare(name, mine, scores(name, arm)) for arm in arms}
    p = {name: max(r["p"] for r in by_arm.values()) for name, by_arm in results.items()}
    rejected = holm(p, ALPHA)
    verdict = ("CONFIRMED", "not confirmed") if control else ("passes the gate", "does not pass the gate")
    for name, by_arm in results.items():
        cells = "; ".join(f"vs {arm}: {_format(r)}" for arm, r in by_arm.items())
        print(f"{name}: {cells}; p = {p[name]:.5f}; {verdict[0] if rejected[name] else verdict[1]}")


def main() -> None:
    """Print item hashes, record an arm's replies, or run the analysis."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hash")
    recorder = sub.add_parser("record")
    recorder.add_argument("arm", choices=sorted(ADAPTERS))
    recorder.add_argument("sets", nargs="+", choices=list(SETS))
    analyzer = sub.add_parser("analyze")
    analyzer.add_argument("--trained", default="v3-trained")
    analyzer.add_argument("--base", default="v3-base")
    analyzer.add_argument("--control")
    args = parser.parse_args()
    if args.command == "hash":
        for name in SETS:
            print(f"    {name!r}: {fingerprint(name)!r},")
    elif args.command == "record":
        record(args.arm, args.sets)
    else:
        analyze(args.trained, args.base, args.control)


if __name__ == "__main__":
    main()

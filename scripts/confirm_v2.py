r"""The pre-registered confirmatory analysis of the environment-v2 training study (``docs/preregistration_v2.md``).

    python scripts/confirm_v2.py --record v2-base          # once, right after collecting an arm's replies
    python scripts/confirm_v2.py                           # before the control exists: the gate
    python scripts/confirm_v2.py --control v2-control      # the full analysis

For each endpoint, the trained arm is compared with every comparator: the base model, the control (when it
exists), and on the yes/no transfer sets chance (a score of 0.5 on every item). Each comparison reports the
difference in accuracy (balanced accuracy on the transfer sets), its 95% paired cluster-bootstrap interval, and a
one-sided p-value from a cluster sign-flip randomization test (20,000 resamples). An endpoint's p-value is the
largest over its comparators (an intersection-union test), and Holm's procedure over the six endpoints holds the
family-wise error at one-sided 0.025. Before the control exists the same procedure decides the gate: the control
is trained only if some endpoint passes, since adding a comparator can only raise an endpoint's p-value.

Every input is checked against the registration first: item lists (``REGISTERED``), the prompt each reply
answered, each arm's model revision, adapter, token limit, vLLM version and sampling (one configuration per arm
across all sets), and each arm's replies against the manifest ``--record`` committed when they were collected
(``RECORDS``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from typing import Any

from transfer_eval import OUT, ROOT, _load, fingerprint, interval, items, paired_draws, permutation_p, scores

ENDPOINTS = {
    "ood_wording_test": False,
    "ood_vocab_test": False,
    "ood_size_test": False,
    "dsep": True,
    "c2c_test": True,
    "cladder_adjust": True,
}
"""Confirmatory endpoints and whether each uses balanced accuracy over its cells (``transfer_eval.stratum``)."""
AGAINST_CHANCE = ("dsep", "c2c_test", "cladder_adjust")
"""Yes/no sets on which the trained arm must also beat chance: a below-chance base can be beaten by guessing."""
MANIPULATION_CHECK = "iid_test"
SAMPLES = {
    "iid_test": 1,
    "ood_wording_test": 1,
    "ood_vocab_test": 1,
    "ood_size_test": 1,
    "dsep": 1,
    "c2c_test": 2,
    "cladder_adjust": 4,
}
"""Samples per item on each set."""
REGISTERED = {
    "iid_test": "9ca960158c0202b519ee0a10b0ff2918585428d51341951f5b502efd2f06e96c",
    "ood_wording_test": "f53cf292d94578d220ea6ed6b3f8d91209d5ef089cb02fcc856e1b02238e1a06",
    "ood_vocab_test": "57ede41b5f52608cfc84f075df4c2d089d2ca63233ad071a570f4f94cbffe615",
    "ood_size_test": "d81bee267be3679ae0e8246bd23f35d25bf9ef477694cfacfa7c5f17eb7ad335",
    "dsep": "daa0fc225333938fc4e3a4a8c87bfd834582157024b6495b9c6858dddad551b4",
    "c2c_test": "b81bdce10ce4a115ed382588969549f08af8c8baa9b4481aa6792cd3f99eb817",
    "cladder_adjust": "b84502525ddc570f19a6deec222906468c1f60ae02852de345f179d98b8ec452",
}
"""Item-list fingerprints (``transfer_eval.py hash``) fixed by the pre-registration."""
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
VLLM = "0.31.0"
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
"""The model's generation-config defaults, which every arm must have sampled with."""
ADAPTERS = {"v2-base": "", "v2-trained": "reason-v2/adapter", "v2-control": "control-v2/adapter"}
"""The adapter each arm must have been evaluated with."""
ADAPTER_SHA256 = {"v2-base": "", "v2-trained": "", "v2-control": ""}
"""Each adapter's ``modal_eval.adapter_digest`` (config and weights), committed right after its training run and
before any evaluation."""
RECORDS = ROOT / "docs" / "preregistration_v2_records.json"
"""Committed manifest: SHA-256 of each arm's replies on each set, written by ``--record`` right after collection."""
SEED = 1000
"""Sample j of every item uses seed SEED + j."""
MAX_TOKENS = 8192
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


def _replies(name: str, model: str) -> list[dict[str, Any]]:
    """An arm's reply records on a set, in a canonical order."""
    records = [r for r in _load(OUT / f"{name}.jsonl") if r["model"] == model]
    return sorted(records, key=lambda r: (r["id"], r["seed"]))


def _digest(records: list[dict[str, Any]]) -> str:
    """SHA-256 of reply records in canonical JSON."""
    return hashlib.sha256("\n".join(json.dumps(r, sort_keys=True) for r in records).encode("utf-8")).hexdigest()


def configuration(record: dict[str, Any]) -> str:
    """The generation settings a reply was produced with, as one comparable string."""
    keys = ("base_model", "revision", "adapter", "adapter_sha256", "max_tokens", "vllm", "sampling")
    return json.dumps({key: record.get(key) for key in keys}, sort_keys=True)


def check(name: str, model: str) -> str:
    """Check a set's items and one arm's replies against the registration; return the arm's configuration."""
    if fingerprint(name) != REGISTERED[name]:
        raise ValueError(f"the items of {name} differ from the registered list")
    records = _replies(name, model)
    expected = {(item["id"], SEED + j) for item in items(name) for j in range(SAMPLES[name])}
    found = Counter((r["id"], r["seed"]) for r in records)
    if set(found) != expected or max(found.values()) != 1:
        raise ValueError(f"{model} does not have exactly the pre-registered replies on {name}")
    prompts = {item["id"]: hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest() for item in items(name)}
    if any(r.get("prompt_sha256") != prompts[r["id"]] for r in records):
        raise ValueError(f"some {model} replies on {name} answered a prompt other than the registered one")
    settings = {configuration(r) for r in records}
    if len(settings) != 1:
        raise ValueError(f"{model} replies on {name} come from more than one configuration")
    setting = json.loads(next(iter(settings)))
    wanted = {
        "base_model": MODEL,
        "revision": REVISION,
        "adapter": ADAPTERS[model],
        "max_tokens": MAX_TOKENS,
        "vllm": VLLM,
        "sampling": SAMPLING,
    }
    if any(setting[key] != value for key, value in wanted.items()):
        raise ValueError(f"{model} on {name} was generated with {setting}, not {wanted}")
    if setting["adapter_sha256"] != ADAPTER_SHA256[model]:
        raise ValueError(f"{model} on {name} used adapter weights {setting['adapter_sha256']}, not the registered ones")
    return next(iter(settings))


def complete(name: str, model: str) -> dict[str, dict[str, float]]:
    """An arm's per-item scores on a set, after checking it against the registration and the reply manifest."""
    check(name, model)
    manifest = json.loads(RECORDS.read_text(encoding="utf-8")) if RECORDS.exists() else {}
    if manifest.get(f"{name}/{model}") != _digest(_replies(name, model)):
        raise ValueError(f"{model} replies on {name} do not match the committed manifest {RECORDS.name}")
    return scores(name, model)


def record(model: str) -> None:
    """Add an arm's reply digests on every set to the manifest; refuses to overwrite an entry."""
    if len({check(name, model) for name in SAMPLES}) != 1:
        raise ValueError(f"{model} used more than one configuration across sets")
    manifest = json.loads(RECORDS.read_text(encoding="utf-8")) if RECORDS.exists() else {}
    for name in SAMPLES:
        key = f"{name}/{model}"
        if key in manifest:
            raise ValueError(f"{key} is already recorded")
        manifest[key] = _digest(_replies(name, model))
    RECORDS.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"recorded {model} on {len(SAMPLES)} sets in {RECORDS}; commit it before any analysis")


def compare(name: str, balanced: bool, trained: dict, comparator: dict) -> dict[str, Any]:
    """Difference trained - comparator on one set: point, 95% bootstrap interval and sign-flip p-value."""
    point, draws = paired_draws(name, trained, comparator, balanced, RESAMPLES)
    low, high = interval(draws, 0.95)
    p = permutation_p(name, trained, comparator, balanced, RESAMPLES)
    return {"delta": point, "low": low, "high": high, "p": p}


def analyze(trained_arm: str, base_arm: str, control_arm: str | None) -> None:
    """Print the manipulation check and every endpoint's comparisons, p-value and Holm decision."""
    arms = [base_arm] + ([control_arm] if control_arm else [])
    for arm in [trained_arm, *arms]:
        if len({check(name, arm) for name in SAMPLES}) != 1:
            raise ValueError(f"{arm} used more than one configuration across sets")
    trained_check = complete(MANIPULATION_CHECK, trained_arm)
    result = compare(MANIPULATION_CHECK, False, trained_check, complete(MANIPULATION_CHECK, base_arm))
    worked = "passes" if result["p"] <= ALPHA else "fails: a null elsewhere says nothing about generalization"
    print(f"manipulation check {MANIPULATION_CHECK} vs {base_arm}: {_format(result)}; {worked}")
    results = {}
    for name, balanced in ENDPOINTS.items():
        trained = complete(name, trained_arm)
        comparators = {arm: complete(name, arm) for arm in arms}
        if name in AGAINST_CHANCE:
            comparators["chance"] = {i: {"accuracy": 0.5} for i in trained}
        results[name] = {c: compare(name, balanced, trained, scored) for c, scored in comparators.items()}
    p = {name: max(r["p"] for r in by_comparator.values()) for name, by_comparator in results.items()}
    rejected = holm(p, ALPHA)
    verdict = ("CONFIRMED", "not confirmed") if control_arm else ("passes the gate", "does not pass the gate")
    for name, by_comparator in results.items():
        cells = "; ".join(f"vs {c}: {_format(r)}" for c, r in by_comparator.items())
        print(f"{name}: {cells}; p = {p[name]:.5f}; {verdict[0] if rejected[name] else verdict[1]}")


def _format(result: dict[str, Any]) -> str:
    """One comparison as 'delta [low, high], p' in percentage points."""
    delta, low, high = (100 * result[key] for key in ("delta", "low", "high"))
    return f"{delta:+.1f} [{low:+.1f}, {high:+.1f}], p = {result['p']:.5f}"


def main() -> None:
    """Record an arm's replies, or run the analysis."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", metavar="ARM", help="add this arm's reply digests to the manifest")
    parser.add_argument("--trained", default="v2-trained")
    parser.add_argument("--base", default="v2-base")
    parser.add_argument("--control")
    args = parser.parse_args()
    if args.record:
        record(args.record)
    else:
        analyze(args.trained, args.base, args.control)


if __name__ == "__main__":
    main()

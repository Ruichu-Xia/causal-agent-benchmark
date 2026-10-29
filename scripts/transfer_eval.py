r"""Far-transfer evaluation: yes/no causal questions from CLadder and Corr2Cause, in their own text format.

    python scripts/transfer_eval.py run primary --model base --model trained --model control --samples 4
    python scripts/transfer_eval.py report primary --model trained --against base --against control

Models are the ``modal_serve.py`` endpoint's served names (``HOSTED_VLLM_API_BASE`` and
``HOSTED_VLLM_API_KEY`` in .env). Every model gets the same single user message (no system prompt, no tools)
and the server's default sampling from the model's generation config; sample ``j`` of every item uses
seed ``j``. Replies go to ``outputs/transfer/<set>.jsonl``; ``run`` skips (item, model) pairs already there.
The pre-registered analysis is in ``docs/transfer_preregistration.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
import statistics
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "transfer"
CLADDER = "https://huggingface.co/datasets/causal-nlp/CLadder/resolve/18d684e444ed07cc57ae0f3a91ee983a79b8119a/data"
CORR2CAUSE = "https://huggingface.co/datasets/causal-nlp/corr2cause/resolve/42ba12c769e11ff6427c9f52d7db58e3f9bf3e53"
SOURCES = {
    "cladder": f"{CLADDER}/full_v1.5_default.csv",
    "c2c_test": f"{CORR2CAUSE}/test.csv",
    "c2c_dev": f"{CORR2CAUSE}/dev.csv",
    "c2c_refactor": f"{CORR2CAUSE}/perturbation_by_refactorization_test.csv",
}
INSTRUCTION = 'Think it through, then end your reply with a final line that is exactly "Answer: Yes" or "Answer: No".'
C2C_QUESTION = "Given the premise, is the hypothesis necessarily true?"
MAX_TOKENS = 4096
PILOT_QUERIES = ("nde", "nie", "ett", "det-counterfactual")  # rung 3, outside every scored CLadder set
ALL_QUERIES = (
    "marginal",
    "correlation",
    "exp_away",
    "ate",
    "backadj",
    "collider_bias",
    "det-counterfactual",
    "ett",
    "nde",
    "nie",
)
DEV_STORIES = 10
"""CLadder stories drawn (seed 0) for development of the text-reasoning environment; the rest stay untouched."""
MEDIATOR_COLLIDER_GRAPHS = ("chain", "mediation", "arrowhead", "diamond", "collision")
"""CLadder graphs whose adjustment set Z holds a mediator or a collider: the misconception the environment targets."""
DSEP = OUT / "candidates" / "axiomatic_dsep"
"""d-separation evaluation files of Vashishtha et al. (2024), ``eval_dsep/dsep_eval_branching_factor14`` of
github.com/AniketVashishtha/Causal_Axioms at commit ``fbddd14``. The repository has no license, so the files are
used locally and never redistributed."""
_DSEP_HEADER = "Answer the following question with ONLY 'Yes' or 'No'.\n\n"

# The pre-registered answer rule: the last "answer ... yes/no", else a final line that is only yes or no.
_ANSWER = re.compile(r"answer\W*(?:is\W*)?(yes|no)\b", re.IGNORECASE)
_BARE = re.compile(r"^\W*(yes|no)\W*$", re.IGNORECASE)


def _rows(source: str) -> list[dict[str, str]]:
    """Rows of one source CSV, downloaded once at its pinned revision."""
    path = OUT / "data" / f"{source}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCES[source], path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _cladder(
    query_types: tuple[str, ...], stories: set[str] | None = None, graphs: tuple[str, ...] | None = None
) -> list[dict[str, Any]]:
    """CLadder v1.5 items of the given query types (and stories and graphs, if given); story is the cluster."""
    return [
        {
            "id": f"cladder-{row['id']}",
            "prompt": f"{row['prompt']}\n\n{INSTRUCTION}",
            "label": row["label"] == "yes",
            "cluster": row["story_id"],
            "group": row["query_type"],
            "property": row["question_property"],
        }
        for row in _rows("cladder")
        if row["query_type"] in query_types
        and (stories is None or row["story_id"] in stories)
        and (graphs is None or row["graph_id"] in graphs)
    ]


def cladder_dev_stories() -> set[str]:
    """The CLadder stories reserved for developing the text-reasoning environment."""
    stories = sorted({row["story_id"] for row in _rows("cladder")})
    return set(random.Random(0).sample(stories, DEV_STORIES))


def cladder_held_out_stories() -> set[str]:
    """Every CLadder story outside the development stories."""
    return {row["story_id"] for row in _rows("cladder")} - cladder_dev_stories()


def _dsep(nodes: tuple[int, ...]) -> list[dict[str, Any]]:
    """Axiomatic-training d-separation items for the given graph sizes; the graph is the resampling cluster.

    The files' "answer with only Yes or No" header is replaced by the final-line instruction, as for the other
    yes/no sets. A question repeated within the files is kept once.
    """
    items: list[dict[str, Any]] = []
    labels: dict[str, str] = {}
    for n in nodes:
        lines = (DSEP / f"dsep_instances_{n}_nodes_updated.jsonl").read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            row = json.loads(line)
            if not row["prompt"].startswith(_DSEP_HEADER) or row["completion"] not in ("Yes", "No"):
                raise ValueError(f"unexpected d-separation item {n}-{i}")
            question = row["prompt"].removeprefix(_DSEP_HEADER)
            if question in labels:
                if labels[question] != row["completion"]:
                    raise ValueError(f"d-separation item {n}-{i} repeats a question with the other label")
                continue
            labels[question] = row["completion"]
            items.append(
                {
                    "id": f"dsep{n}-{i}",
                    "prompt": f"{question}\n\n{INSTRUCTION}",
                    "label": row["completion"] == "Yes",
                    "cluster": f"{n}: {question.split(' Are ')[0]}",
                    "group": f"{n} nodes",
                    "property": "",
                }
            )
    return items


def _reasoning(path: Path) -> list[dict[str, Any]]:
    """Generated text-reasoning tasks (``build_reasoning.py``), graded by their own answer format."""
    from causalagentbench.reasoning.dataset import load

    return [
        {
            "id": task["id"],
            "prompt": task["prompt"],
            "label": task["answer"],
            "kind": task["kind"],
            "cluster": task["id"],
            "group": task["family"],
            "property": f"{task['kind']}/{task['naming']}",
        }
        for task in load(path)
    ]


def _corr2cause(source: str) -> list[dict[str, Any]]:
    """Corr2Cause items; each item is its own cluster."""
    items = []
    for i, row in enumerate(_rows(source)):
        premise, hypothesis = row["input"].split("Hypothesis:")
        items.append(
            {
                "id": f"{source}-{i}",
                "prompt": f"{premise.strip()}\nHypothesis:{hypothesis}\n\n{C2C_QUESTION} {INSTRUCTION}",
                "label": row["label"] == "1",
                "cluster": f"{source}-{i}",
                "group": row.get("template", "refactorization"),
                "property": row.get("num_variables", ""),
            }
        )
    return items


def _sample(items: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """A fixed random subset, for pilots."""
    return random.Random(0).sample(items, n)


def items(name: str) -> list[dict[str, Any]]:
    """The items of one named evaluation set."""
    sets = {
        "primary": lambda: _cladder(("backadj", "collider_bias")),
        "cladder_ate": lambda: _cladder(("ate",)),
        "cladder_correlation": lambda: _cladder(("correlation",)),
        "c2c_test": lambda: _corr2cause("c2c_test"),
        "c2c_refactor": lambda: _corr2cause("c2c_refactor"),
        "pilot_cladder": lambda: _sample(_cladder(PILOT_QUERIES), 100),
        "pilot_c2c": lambda: _sample(_corr2cause("c2c_dev"), 100),
        "cladder_dev": lambda: _cladder(ALL_QUERIES, cladder_dev_stories()),
        "cladder_medcol": lambda: _cladder(("backadj",), cladder_held_out_stories(), MEDIATOR_COLLIDER_GRAPHS),
        "dsep": lambda: _dsep((5, 8)),
        "reasoning_dev": lambda: _reasoning(ROOT / "outputs" / "reasoning" / "dev.jsonl"),
    }
    v2 = ROOT / "outputs" / "reasoning" / "v2"
    for split in ("iid", "ood_wording", "ood_size", "ood_vocab"):
        for role in ("dev", "test"):
            sets[f"{split}_{role}"] = lambda path=v2 / f"{split}_{role}.jsonl": _reasoning(path)
    return sets[name]()


def fingerprint(name: str) -> str:
    """SHA-256 of a set's items (id, prompt and label of each, in id order), to pin item lists."""
    lines = sorted(json.dumps([item["id"], item["prompt"], item["label"]]) for item in items(name))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def extract(reply: str) -> bool | None:
    """The reply's yes/no answer under the pre-registered rule, or None if it has none."""
    matches = _ANSWER.findall(reply)
    if matches:
        return matches[-1].lower() == "yes"
    lines = [line for line in reply.strip().splitlines() if line.strip()]
    bare = _BARE.match(lines[-1]) if lines else None
    return bare.group(1).lower() == "yes" if bare else None


async def _ask(client: Any, model: str, item: dict[str, Any], samples: int, gate: asyncio.Semaphore) -> list[dict]:
    """All samples of one item from one model, one request per seed."""
    records = []
    async with gate:
        for seed in range(samples):
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": item["prompt"]}],
                max_tokens=MAX_TOKENS,
                seed=seed,
            )
            choice = response.choices[0]
            records.append(
                {
                    "id": item["id"],
                    "model": model,
                    "seed": seed,
                    "reply": choice.message.content or "",
                    "finish_reason": choice.finish_reason,
                    "completion_tokens": response.usage.completion_tokens,
                }
            )
    return records


async def run(name: str, models: list[str], samples: int, workers: int) -> None:
    """Query every model on every item of set ``name``, appending to its results file."""
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    load_dotenv(ROOT / ".env", override=False)
    client = AsyncOpenAI(
        base_url=os.environ["HOSTED_VLLM_API_BASE"],
        api_key=os.environ["HOSTED_VLLM_API_KEY"],
        timeout=600,
        max_retries=8,
    )
    path = OUT / f"{name}.jsonl"
    done = {(r["id"], r["model"]) for r in _load(path)}
    todo = [(item, model) for model in models for item in items(name) if (item["id"], model) not in done]
    gate = asyncio.Semaphore(workers)
    tasks = [asyncio.create_task(_ask(client, model, item, samples, gate)) for item, model in todo]
    with path.open("a", encoding="utf-8") as handle:
        for finished, task in enumerate(asyncio.as_completed(tasks), 1):
            for record in await task:
                handle.write(json.dumps(record) + "\n")
            handle.flush()
            if finished % 100 == 0 or finished == len(tasks):
                print(f"{name}: {finished}/{len(tasks)}", flush=True)


def _load(path: Path) -> list[dict[str, Any]]:
    """Records of one results file (empty if absent)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def judge(item: dict[str, Any], reply: str) -> tuple[bool, bool, bool]:
    """(correct, answered yes, gave no answer) for one reply to one item."""
    if "kind" in item:  # generated text-reasoning task
        from causalagentbench.reasoning.grading import final_answer, grade

        text = final_answer(reply)
        return grade(item["kind"], item["label"], reply) == 1.0, bool(text and text.startswith("yes")), text is None
    answer = extract(reply)
    return answer is item["label"], answer is True, answer is None


def scores(name: str, model: str) -> dict[str, dict[str, float]]:
    """Per item: accuracy over samples (no answer counts as wrong), yes-rate, and no-answer rate."""
    by_id = {item["id"]: item for item in items(name)}
    judged: dict[str, list[tuple[bool, bool, bool]]] = defaultdict(list)
    for record in _load(OUT / f"{name}.jsonl"):
        if record["model"] == model:
            judged[record["id"]].append(judge(by_id[record["id"]], record["reply"]))
    return {
        i: {
            "accuracy": statistics.mean(j[0] for j in judged[i]),
            "yes": statistics.mean(j[1] for j in judged[i]),
            "missing": statistics.mean(j[2] for j in judged[i]),
        }
        for i in by_id
        if i in judged
    }


def _metric(per_item: dict[str, dict[str, float]], ids: list[str], labels: dict[str, bool], balanced: bool) -> float:
    """Accuracy over ``ids``, or balanced accuracy (mean of the accuracies on true and on false items)."""
    if not balanced:
        return statistics.mean(per_item[i]["accuracy"] for i in ids)
    by_label = [[per_item[i]["accuracy"] for i in ids if labels[i] is value] for value in (True, False)]
    return statistics.mean(statistics.mean(values) for values in by_label)


def paired_draws(name: str, a: dict, b: dict, balanced: bool, resamples: int) -> tuple[float, list[float]]:
    """Difference a - b in (balanced) accuracy, and its sorted paired bootstrap draws over clusters (seed 0)."""
    item_list = [item for item in items(name) if item["id"] in a and item["id"] in b]
    labels = {item["id"]: item["label"] for item in item_list}
    clusters: dict[str, list[str]] = defaultdict(list)
    for item in item_list:
        clusters[item["cluster"]].append(item["id"])
    keys = sorted(clusters)
    ids = [i for key in keys for i in clusters[key]]
    point = _metric(a, ids, labels, balanced) - _metric(b, ids, labels, balanced)
    rng = random.Random(0)
    draws = []
    for _ in range(resamples):
        sample = [i for key in rng.choices(keys, k=len(keys)) for i in clusters[key]]
        draws.append(_metric(a, sample, labels, balanced) - _metric(b, sample, labels, balanced))
    return point, sorted(draws)


def interval(draws: list[float], level: float) -> tuple[float, float]:
    """Percentile interval at ``level`` from sorted bootstrap draws."""
    tail = (1 - level) / 2
    return draws[int(tail * len(draws))], draws[int((1 - tail) * len(draws)) - 1]


def p_value(draws: list[float]) -> float:
    """One-sided bootstrap p-value against a difference of zero or less: (1 + #draws <= 0) / (1 + #draws)."""
    return (1 + sum(d <= 0 for d in draws)) / (1 + len(draws))


def paired_interval(
    name: str, a: dict, b: dict, balanced: bool, level: float = 0.95, resamples: int = 10_000
) -> tuple[float, float, float]:
    """Difference a - b in (balanced) accuracy, with a paired bootstrap interval over clusters at ``level``."""
    point, draws = paired_draws(name, a, b, balanced, resamples)
    return point, *interval(draws, level)


def report(name: str, model: str, against: list[str], balanced: bool, level: float) -> dict[str, Any]:
    """Each model's accuracy, yes-rate and no-answer rate overall and by group, and paired differences."""
    item_list = items(name)
    labels = {item["id"]: item["label"] for item in item_list}
    out: dict[str, Any] = {"set": name, "items": len(item_list), "balanced_accuracy": balanced, "level": level}
    out["models"] = {}
    per_model = {m: scores(name, m) for m in [model, *against]}
    for m, per_item in per_model.items():
        groups: dict[str, list[str]] = defaultdict(list)
        for item in item_list:
            if item["id"] in per_item:
                groups["all"].append(item["id"])
                groups[f"group={item['group']}"].append(item["id"])
                groups[f"property={item['property']}"].append(item["id"])
        out["models"][m] = {
            key: {
                "n": len(ids),
                "accuracy": round(_metric(per_item, ids, labels, balanced and key == "all"), 4),
                "yes_rate": round(statistics.mean(per_item[i]["yes"] for i in ids), 4),
                "missing_rate": round(statistics.mean(per_item[i]["missing"] for i in ids), 4),
            }
            for key, ids in sorted(groups.items())
        }
    out["differences"] = {
        f"{model} - {b}": [round(x, 4) for x in paired_interval(name, per_model[model], per_model[b], balanced, level)]
        for b in against
    }
    return out


def main() -> None:
    """Parse arguments and run, report, or print item-list hashes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("set")
    run_parser.add_argument("--model", action="append", required=True)
    run_parser.add_argument("--samples", type=int, default=1)
    run_parser.add_argument("--workers", type=int, default=48)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("set")
    report_parser.add_argument("--model", required=True)
    report_parser.add_argument("--against", action="append", default=[])
    report_parser.add_argument("--balanced", action="store_true", help="balanced accuracy (Corr2Cause)")
    report_parser.add_argument("--level", type=float, default=0.95, help="interval level (0.975 for secondaries)")
    hash_parser = sub.add_parser("hash")
    hash_parser.add_argument("set", nargs="+")
    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(run(args.set, args.model, args.samples, args.workers))
    elif args.command == "hash":
        for name in args.set:
            print(name, len(items(name)), fingerprint(name))
    else:
        print(json.dumps(report(args.set, args.model, args.against, args.balanced, args.level), indent=1))


if __name__ == "__main__":
    main()

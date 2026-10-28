r"""Build the text causal-reasoning datasets: a training set, and dev and test sets of every evaluation split.

    python scripts/build_reasoning.py --out-dir outputs/reasoning/v2 --train 4000 --eval 400

Writes ``train.jsonl`` and ``{split}_{role}.jsonl`` for role dev (development decisions) and test (the
pre-registered evaluation only). Within a role, ``iid``, ``ood_wording`` and ``ood_vocab`` share a seed,
so they pose the same problems item by item. Any item whose prompt also appears in train, or in the other
role, is dropped from every split of its role, which keeps the pairing. Concept names are the packaged
pool; ``--write-concepts DUMP`` rebuilds it from the CauseNet-Precision dump.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from causalagentbench.reasoning.dataset import CONCEPTS, SPLITS, build, concept_pool, load_concepts, save

SEEDS = {"train": 0, "dev": 100, "test": 200}


def write_concepts(dump: str) -> None:
    """Rebuild the packaged concept pool from the CauseNet-Precision dump."""
    from causalagentbench.generation.grounding import load_causenet

    CONCEPTS.write_text("\n".join(concept_pool(load_causenet(dump))) + "\n", encoding="utf-8")


def main() -> None:
    """Build, de-duplicate and save every set, and print each one's composition."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--eval", type=int, required=True, help="items per evaluation split and role")
    parser.add_argument("--write-concepts", metavar="DUMP", help="first rebuild the concept pool from CauseNet")
    args = parser.parse_args()
    if args.write_concepts:
        write_concepts(args.write_concepts)
    out, pool = Path(args.out_dir), load_concepts()
    train = build(args.train, SEEDS["train"], pool, "train")
    sets = {
        role: {split: build(args.eval, SEEDS[role], pool, split) for split in SPLITS[1:]} for role in ("dev", "test")
    }
    seen = {t["prompt"] for t in train}
    for role in ("dev", "test"):
        clash = {i for tasks in sets[role].values() for i, t in enumerate(tasks) if t["prompt"] in seen}
        sets[role] = {s: [t for i, t in enumerate(tasks) if i not in clash] for s, tasks in sets[role].items()}
        seen |= {t["prompt"] for tasks in sets[role].values() for t in tasks}
        print(f"{role}: dropped {len(clash)} items whose prompt appeared earlier")
    save(train, out / "train.jsonl")
    print("train", len(train), Counter((t["family"], t["kind"]) for t in train))
    for role, by_split in sets.items():
        for split, tasks in by_split.items():
            save(tasks, out / f"{split}_{role}.jsonl")
            print(role, split, len(tasks))


if __name__ == "__main__":
    main()

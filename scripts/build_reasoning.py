r"""Build text causal-reasoning datasets for training and development.

    python scripts/build_reasoning.py --n 4000 --seed 0 --out outputs/reasoning/train.jsonl
    python scripts/build_reasoning.py --n 400 --seed 1 --out outputs/reasoning/dev.jsonl

Concept names come from ``outputs/reasoning/concepts.txt``, written from the CauseNet dump on first use.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from causalagentbench.reasoning.dataset import build, concept_pool, save

ROOT = Path(__file__).resolve().parent.parent
CONCEPTS = ROOT / "outputs" / "reasoning" / "concepts.txt"
CAUSENET = Path.home() / "Desktop/causal-agent-benchmark/causenet-precision.jsonl.bz2"


def concepts() -> list[str]:
    """The concept-name pool, extracted from CauseNet once."""
    if not CONCEPTS.exists():
        from causalagentbench.generation.grounding import load_causenet

        CONCEPTS.parent.mkdir(parents=True, exist_ok=True)
        CONCEPTS.write_text("\n".join(concept_pool(load_causenet(str(CAUSENET)))) + "\n", encoding="utf-8")
    return CONCEPTS.read_text(encoding="utf-8").splitlines()


def main() -> None:
    """Build and save one dataset, and print its composition."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tasks = build(args.n, args.seed, concepts())
    save(tasks, args.out)
    print(Counter((t["family"], t["kind"]) for t in tasks))


if __name__ == "__main__":
    main()

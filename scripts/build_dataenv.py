r"""Build the environment-v3 task splits (``docs/environment_v3.md``).

    python scripts/build_dataenv.py --out-dir outputs/dataenv --train 4000 --dev 200 --test 500 --workers 8

Writes ``train.jsonl`` and, for each evaluation split, ``<split>_dev.jsonl`` and ``<split>_test.jsonl``. A dev
item whose prompt appears in train, or a test item whose prompt appears in train or dev, is dropped.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from causalagentbench.dataenv.dataset import SPLITS, build, drop_seen, save

CAUSENET = str(Path.home() / "Desktop/causal-agent-benchmark/causenet-precision.jsonl.bz2")


def main() -> None:
    """Build train, then every evaluation split's dev and test sets."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--dev", type=int, default=200)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--causenet", default=CAUSENET)
    args = parser.parse_args()
    out = Path(args.out_dir)
    seen: set[str] = set()
    for role, n in (("train", args.train), ("dev", args.dev), ("test", args.test)):
        splits = ["train"] if role == "train" else [split for split in SPLITS if split != "train"]
        prompts: set[str] = set()
        for split in splits:
            if n == 0:
                continue
            start = time.time()
            records = drop_seen(build(args.causenet, split, role, n, args.workers), seen)
            name = "train" if role == "train" else f"{split}_{role}"
            save(records, out / f"{name}.jsonl")
            prompts |= {record["prompt"] for record in records}
            print(f"{name}: {len(records)} of {n} kept, {time.time() - start:.0f} s", flush=True)
        seen |= prompts


if __name__ == "__main__":
    main()

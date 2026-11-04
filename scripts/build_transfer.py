r"""Build environment v3's transfer sets: bnRep networks in the four families, and QRData.

    python scripts/build_transfer.py --out-dir outputs/dataenv/transfer --per-family 12 \
        --qrdata outputs/dataenv/transfer/qrdata_source

Writes ``bnrep_real.jsonl`` and ``bnrep_placeholder.jsonl`` (item ``i`` of one is item ``i`` of the other
under placeholder names), ``qrdata_causal.jsonl`` and ``qrdata_statistics.jsonl``. QRData (CC BY-NC 4.0)
stays local.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from causalagentbench import qrdata
from causalagentbench.bnrep import FINDERS, NETWORKS, SEED, items, network_world, placeholders
from causalagentbench.dataset import save, to_record

ROOT = Path(__file__).resolve().parent.parent


def bnrep(per_family: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``per_family`` items of every family on every network, under real and placeholder names."""
    real, placeholder = [], []
    for network in NETWORKS:
        world = network_world(ROOT / "data" / "bnrep" / f"{network}.json")
        names = placeholders(world, np.random.default_rng([SEED, NETWORKS.index(network), 99]))
        for family in FINDERS:
            used: set[Any] = set()
            for k in range(per_family):
                tasks = items(network, world, family, k, names, used)
                for naming, task, out in (("real", tasks[0], real), ("placeholder", tasks[1], placeholder)):
                    record = to_record(task, f"bnrep-{network}-{family}-{k}-{naming}", f"bnrep_{naming}")
                    out.append({**record, "network": network, "pair": f"bnrep-{network}-{family}-{k}"})
        print(f"{network}: {len(real)} items so far", flush=True)
    return real, placeholder


def main() -> None:
    """Build and save the transfer sets."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-family", type=int, default=12)
    parser.add_argument("--qrdata", required=True, help="the benchmark/ directory of QRData, with data/ unzipped")
    args = parser.parse_args()
    out = Path(args.out_dir)
    real, placeholder = bnrep(args.per_family)
    save(real, out / "bnrep_real.jsonl")
    save(placeholder, out / "bnrep_placeholder.jsonl")
    benchmark = Path(args.qrdata)
    questions = qrdata.records(benchmark)
    missing = {name for record in questions for name in record["files"] if not (benchmark / "data" / name).is_file()}
    if missing:
        raise FileNotFoundError(f"QRData files missing from {benchmark / 'data'}: {sorted(missing)}")
    for subset in ("qrdata_causal", "qrdata_statistics"):
        save([record for record in questions if record["split"] == subset], out / f"{subset}.jsonl")


if __name__ == "__main__":
    main()

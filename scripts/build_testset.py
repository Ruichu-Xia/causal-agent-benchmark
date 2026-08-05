"""Offline dataset producer: propose per-SCM naming triples, construct all grid levels.

Driven by a versioned build config (``configs/*.yaml``); secrets and the CauseNet path come from ``.env``.
Per SCM it runs ONE generator proposal (+ optional correctness critic + revise loop), then *constructs*
every alignment level ``{-1,0,+1}`` plus the name-blind control - paired skins over one SCM, sharing the
exact data table ``generate_gated_procedural`` already validated the oracle gate against (``report.data``)
- never redrawn independently, which would only be guaranteed to match the gate's own certified oracle
recovery by coincidence, not by construction (``procedural_scaffold_plan.md`` §12).
Writes ``<out>.json`` (metadata + SCM params + naming card, one record per instance: ``id``, ``seed``,
``scaffold``, ``W``/``noise_var``/``sigma``/``tau``, ``node_to_label``, ``columns``, ``context``,
``column_schema``, ``provenance``) + ``<out>.npz`` (one data table per record id).

Writes incrementally to ``<out>.partial/`` (crash-safe, inspectable mid-run); the final files are
assembled at the end. Never runs in CI; needs the ``generation`` extra, an LLM key, and a CauseNet dump.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np

from causalagentbench.generation.clients import LiteLLMProposer, load_env
from causalagentbench.generation.config import BuildConfig
from causalagentbench.generation.labeler import GRID, construct_labeling, grounding_fn, propose_triples
from causalagentbench.scoring.gates import generate_gated_procedural
from causalagentbench.task.dataset import serialize_scaffold
from causalagentbench.task.instance import Instance, InstanceKey, build_instance
from causalagentbench.task.labeling import Labeling, abstract_labels

_OUT = "outputs/testset_v1"
log = logging.getLogger("build_testset")


def _record(rid: str, seed: int, instance: Instance, key: InstanceKey) -> dict:
    """Serialize one instance's SCM params + naming card (everything but the data table)."""
    prov = key.provenance
    return {
        "id": rid,
        "seed": seed,
        "scaffold": serialize_scaffold(key.scm.scaffold),
        "W": key.scm.W.tolist(),
        "noise_var": key.scm.noise_var.tolist(),
        "sigma": key.scm.sigma.tolist(),
        "tau": key.scm.tau.tolist(),
        "node_to_label": dict(key.node_to_label),
        "columns": list(instance.columns),
        "context": instance.context,
        "column_schema": dict(instance.column_schema),
        "provenance": {
            "alignment": prov.alignment,
            "n_aligned": prov.n_aligned,
            "n_neutral": prov.n_neutral,
            "n_misaligned": prov.n_misaligned,
            "node_kind": dict(prov.node_kind),
            "domain": prov.domain,
            "producer": prov.producer,
        },
    }


def main() -> None:
    """Materialize the fixed test set from the live pipeline, driven by a build config."""
    load_env()
    parser = argparse.ArgumentParser(description="Build the CausalAgentBench test set (offline).")
    parser.add_argument("--config", default="configs/testset_v1.yaml", help="path to the build config YAML")
    parser.add_argument("--dump", default=os.environ.get("CAUSENET_DUMP"), help="CauseNet dump (or $CAUSENET_DUMP)")
    parser.add_argument("--out", default=_OUT, help="output path stem (.json + .npz)")
    parser.add_argument("-n", "--limit", type=int, default=None, help="override n_scms (for a quick probe)")
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=None,
        help="override the config's n_candidates: total candidates (drivers + decoys) per instance",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log per-call detail (DEBUG)")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.dump:
        parser.error("provide --dump or set CAUSENET_DUMP (e.g. in .env)")
    config = BuildConfig.from_yaml(args.config)
    n_scms = args.limit if args.limit is not None else config.n_scms
    n_candidates = args.n_candidates if args.n_candidates is not None else config.n_candidates
    out = Path(args.out)
    partial = out.with_name(out.name + ".partial")
    (partial / "data").mkdir(parents=True, exist_ok=True)
    producer = f"llm:{config.gen_model}" + (f"+critic:{config.critic_model}" if config.critic_model else "")
    proposer = LiteLLMProposer(config.gen_model)
    critic = LiteLLMProposer(config.critic_model) if config.critic_model else None
    grounding = grounding_fn(args.dump)
    log.info(
        "config: gen=%s critic=%s n_scms=%d n_candidates=%s -> levels %s + name-blind",
        config.gen_model,
        config.critic_model,
        n_scms,
        n_candidates if n_candidates is not None else "free draw",
        GRID,
    )
    log.info("writing incrementally to %s; CauseNet loads on first use...", partial)

    with (partial / "records.jsonl").open("w", encoding="utf-8") as records_file:
        for seed in range(n_scms):
            scm, report = generate_gated_procedural(seed, n_candidates=n_candidates)
            assert report.data is not None  # guaranteed: generate_gated_procedural only returns a passing report
            data = report.data
            try:
                triples = propose_triples(
                    scm,
                    np.random.default_rng(seed),
                    proposer=proposer,
                    grounding=grounding,
                    critic=critic,
                    max_revisions=config.max_revisions,
                )
            except Exception as exc:  # skip an SCM the generator/critic cannot satisfy
                log.warning("[scm %d/%d] SKIPPED: %s", seed + 1, n_scms, exc)
                continue

            conditions: list[tuple[float | None, Labeling]] = [
                (level, construct_labeling(triples, scm, level, producer)) for level in GRID
            ]
            conditions.append((None, abstract_labels(scm, np.random.default_rng(seed))))  # name-blind control
            domain = triples.domain
            for level, labeling in conditions:
                tag = "blind" if level is None else f"a{level:+.2f}"
                rid = f"s{seed:04d}_{tag}"
                instance, key = build_instance(scm, data, seed, labeler=lambda _s, _r, _l=labeling: _l)
                np.save(partial / "data" / f"{rid}.npy", np.ascontiguousarray(instance.data))
                records_file.write(json.dumps(_record(rid, seed, instance, key)) + "\n")
                records_file.flush()
            log.info("[scm %d/%d] domain=%s -> %d instances", seed + 1, n_scms, domain, len(conditions))

    _assemble(out, partial)
    shutil.rmtree(partial)
    log.info("done - REVIEW %s.json before committing.", out)


def _assemble(out: Path, partial: Path) -> None:
    """Build the final single-file ``<out>.json`` + ``<out>.npz`` from the incremental partial dir."""
    records = [json.loads(line) for line in (partial / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    tables = {r["id"]: np.load(partial / "data" / f"{r['id']}.npy") for r in records}
    n_samples = next(iter(tables.values())).shape[0] if tables else 0  # from the actual data, not a duplicated constant
    blob = {"version": out.name, "scaffold": "procedural", "n_samples": n_samples, "records": records}
    out.with_suffix(".json").write_text(json.dumps(blob, indent=2), encoding="utf-8")
    with out.with_suffix(".npz").open("wb") as handle:
        np.savez_compressed(handle, **tables)
    log.info("assembled %d instances into %s.json (+ .npz)", len(records), out)


if __name__ == "__main__":
    main()

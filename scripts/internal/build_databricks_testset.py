"""INTERNAL, not for release: build a small test set via a Databricks-served model.

Not part of the public build path - ``scripts/build_testset.py`` (LiteLLM) is the documented, supported
one; ``BuildConfig``/its YAML deliberately have no Databricks option, since release users won't use
Databricks models at all. This script exists only so internal testing against Databricks-hosted models
has a reusable entry point instead of throwaway one-off code. Do not reference this from README or any
user-facing doc; do not add a Databricks option to the public ``BuildConfig``/``build_testset.py`` to
replace it.

Deliberately not sharing code with ``build_testset.py`` beyond the library layer (``propose_triples``,
``construct_labeling``, ...) - same reasoning as this project's existing rule against sharing helpers
between the Databricks and LiteLLM client paths: keeping the two write-loops textually separate means a
Databricks-specific fix here can never accidentally change the public LiteLLM path's behavior.

Needs ``DATABRICKS_TOKEN`` + ``DATABRICKS_HOST`` (or ``DATABRICKS_AI_GATEWAY_BASE_URL``) and
``CAUSENET_DUMP`` in ``.env``. Writes ``<out>.json`` + ``<out>.npz`` in the same shape
``task/dataset.py::load_dataset`` reads, incrementally to ``<out>.partial/`` (crash-safe - a Databricks
call failing partway through a run shouldn't lose everything already paid for).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from causalagentbench.generation.clients import DatabricksProposer, load_env
from causalagentbench.generation.labeler import GRID, construct_labeling, grounding_fn, propose_triples
from causalagentbench.scoring.gates import generate_gated_procedural
from causalagentbench.task.dataset import serialize_scaffold
from causalagentbench.task.instance import Instance, InstanceKey, build_instance
from causalagentbench.task.labeling import Labeling, abstract_labels

_OUT = "outputs/databricks_internal"
log = logging.getLogger("build_databricks_testset")


def _record(rid: str, seed: int, instance: Instance, key: InstanceKey) -> dict:
    """Serialize one instance's SCM params + naming card (everything but the data table).

    Identical shape to ``build_testset.py``'s own ``_record`` - both must match what
    ``task/dataset.py::load_dataset`` expects, so this is duplicated deliberately, not drifted.
    """
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
    """Materialize a small test set from the live pipeline, driven by CLI flags (no YAML build config)."""
    load_env()
    parser = argparse.ArgumentParser(description="INTERNAL: build a test set via a Databricks-served model.")
    parser.add_argument("--model", default="databricks-claude-opus-4-8", help="Databricks gateway model")
    parser.add_argument("--critic-model", default="databricks-claude-opus-4-8", help="critic model; '' to skip")
    parser.add_argument("--dump", default=os.environ.get("CAUSENET_DUMP"), help="CauseNet dump (or $CAUSENET_DUMP)")
    parser.add_argument("--out", default=_OUT, help="output path stem (.json + .npz)")
    parser.add_argument("-n", "--n-scms", type=int, default=3, help="number of SCM seeds")
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=None,
        help="fix the total candidate count, drivers + decoys (default: varies per seed over CANDIDATE_COUNT_RANGE)",
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
    critic_model = args.critic_model or None
    out = Path(args.out)
    partial = out.with_name(out.name + ".partial")
    (partial / "data").mkdir(parents=True, exist_ok=True)
    producer = f"databricks:{args.model}" + (f"+critic:{critic_model}" if critic_model else "")
    proposer = DatabricksProposer(args.model)
    critic = DatabricksProposer(critic_model) if critic_model else None
    grounding = grounding_fn(args.dump)
    log.info("model=%s critic=%s n_scms=%d -> levels %s + name-blind", args.model, critic_model, args.n_scms, GRID)
    log.info("writing incrementally to %s; CauseNet loads on first use...", partial)

    with (partial / "records.jsonl").open("w", encoding="utf-8") as records_file:
        for seed in range(args.n_scms):
            scm, report = generate_gated_procedural(seed, n_candidates=args.n_candidates)
            assert report.data is not None  # guaranteed: generate_gated_procedural only returns a passing report
            data = report.data
            try:
                triples = propose_triples(
                    scm, np.random.default_rng(seed), proposer=proposer, grounding=grounding, critic=critic
                )
            except Exception as exc:  # skip an SCM the generator/critic cannot satisfy
                log.warning("[scm %d/%d] SKIPPED: %s", seed + 1, args.n_scms, exc)
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
            log.info("[scm %d/%d] domain=%s -> %d instances", seed + 1, args.n_scms, domain, len(conditions))

    _assemble(
        out,
        partial,
        n_scms_attempted=args.n_scms,
        gen_model=args.model,
        critic_model=critic_model,
        causenet_dump=args.dump,
    )
    shutil.rmtree(partial)
    log.info("done - INTERNAL output, not for publishing: %s.json", out)


def _git_commit() -> str | None:
    """The commit this build ran from, or ``None`` outside a git checkout.

    Suffixed ``-dirty`` when the working tree has uncommitted changes, because a bare hash would be a
    false provenance claim: it names a tree that is not the one that ran. This is not hypothetical - the
    shipped ``testset_v2`` was built from a dirty tree whose uncommitted files included ``labeler.py`` and
    ``grounding.py``, the two modules that decide every column name in it, and its stamp pointed at a
    commit that never produced it. A reader cannot reproduce a dirty build, but they can at least be told
    not to trust the hash.
    """
    root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return None
    if not commit:
        return None
    return f"{commit}-dirty" if dirty else commit


def _file_md5(path: str) -> str | None:
    """MD5 of ``path``, or ``None`` if it cannot be read. Streamed - the CauseNet dump is ~138 MB."""
    digest = hashlib.md5(usedforsecurity=False)
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _assemble(
    out: Path,
    partial: Path,
    *,
    n_scms_attempted: int,
    gen_model: str,
    critic_model: str | None,
    causenet_dump: str,
) -> None:
    """Build the final single-file ``<out>.json`` + ``<out>.npz`` from the incremental partial dir.

    Mirrors ``scripts/build_testset.py::_assemble``, duplicated rather than imported - this module's own
    rule against sharing the write-loop with the public LiteLLM path applies to its assembly step too.

    ``n_scms_attempted`` is what makes set completeness answerable *from the artifact*: seeds are
    ``range(n_scms)`` and every record carries its own, so dropped seeds are derivable only once the
    attempted count is known. A build that lost SCMs to naming failures is otherwise indistinguishable
    from a complete, smaller one - and skipped SCMs are not a random subset, since they are the ones whose
    naming constraints are hardest to satisfy. Measured: a 4-SCM build silently produced 9 records instead
    of 12 and exited 0.

    Deliberately does *not* raise on a short build - discarding the SCMs that did succeed, and the gateway
    spend behind them, is worse than shipping them with the shortfall recorded.
    """
    records = [json.loads(line) for line in (partial / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    tables = {r["id"]: np.load(partial / "data" / f"{r['id']}.npy") for r in records}
    n_samples = next(iter(tables.values())).shape[0] if tables else 0  # from the actual data, not a duplicated constant
    accepted = sorted({r["seed"] for r in records})
    log.info("checksumming CauseNet dump (streamed, ~138 MB)...")
    blob = {
        "version": out.name,
        "scaffold": "procedural",
        "n_samples": n_samples,
        "n_scms_attempted": n_scms_attempted,
        "n_scms_accepted": len(accepted),
        "gen_model": gen_model,
        "critic_model": critic_model,
        "causenet_md5": _file_md5(causenet_dump),
        "code_commit": _git_commit(),
        "records": records,
    }
    out.with_suffix(".json").write_text(json.dumps(blob, indent=2), encoding="utf-8")
    with out.with_suffix(".npz").open("wb") as handle:
        np.savez_compressed(handle, **tables)
    log.info("assembled %d instances from %d SCMs into %s.json (+ .npz)", len(records), len(accepted), out)
    if len(accepted) < n_scms_attempted:
        missing = sorted(set(range(n_scms_attempted)) - set(accepted))
        log.warning("=" * 78)
        log.warning(
            "INCOMPLETE SET: %d of %d SCMs produced records. Missing seeds: %s",
            len(accepted),
            n_scms_attempted,
            missing,
        )
        log.warning("Skipped SCMs are not a random subset - do not report this set as %d SCMs.", n_scms_attempted)
        log.warning("=" * 78)


if __name__ == "__main__":
    main()

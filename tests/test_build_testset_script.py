"""Focused tests for the offline test-set builder's assembly step.

Covers the part that decides whether a shipped set can be trusted: whether the artifact says how many
SCMs were *attempted*, so a short build is detectable from the file rather than only from the build log.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np


def _load(relative: str, name: str):
    path = Path(__file__).parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _builder_module():
    return _load("scripts/build_testset.py", "build_testset_script")


def _databricks_module():
    return _load("scripts/internal/build_databricks_testset.py", "build_databricks_testset_script")


def _partial(tmp_path: Path, seeds: list[int], n_rows: int = 8, n_cols: int = 3) -> Path:
    """Write a minimal ``.partial/`` dir with three skins per seed, as the build loop would."""
    partial = tmp_path / "out.partial"
    (partial / "data").mkdir(parents=True)  # tmp_path may be a fresh subdir per caller
    with (partial / "records.jsonl").open("w", encoding="utf-8") as handle:
        for seed in seeds:
            for tag in ("a-1.00", "a+1.00", "blind"):
                rid = f"s{seed:04d}_{tag}"
                np.save(partial / "data" / f"{rid}.npy", np.zeros((n_rows, n_cols)))
                handle.write(json.dumps({"id": rid, "seed": seed}) + "\n")
    return partial


def _assemble(tmp_path: Path, seeds: list[int], attempted: int) -> dict:
    module = _builder_module()
    out = tmp_path / "out"
    module._assemble(
        out,
        _partial(tmp_path, seeds),
        n_scms_attempted=attempted,
        gen_model="test/gen",
        critic_model="test/critic",
        causenet_dump=str(tmp_path / "missing-dump.jsonl.bz2"),
    )
    return json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))


def test_complete_build_records_the_attempted_count(tmp_path):
    blob = _assemble(tmp_path, seeds=[0, 1, 2], attempted=3)
    assert blob["n_scms_attempted"] == 3
    assert blob["n_scms_accepted"] == 3
    assert len(blob["records"]) == 9
    assert blob["gen_model"] == "test/gen" and blob["critic_model"] == "test/critic"
    assert (tmp_path / "out.npz").exists()


def test_short_build_is_recorded_but_not_fatal(tmp_path):
    """A naming failure must not discard the SCMs that did succeed - but must be visible in the artifact.

    Seed 1 is missing, exactly as a skipped SCM would leave it. The build still produces a usable set; the
    artifact carries enough to see it is incomplete, which is impossible without ``n_scms_attempted``
    because seeds are only recorded for records that exist.
    """
    blob = _assemble(tmp_path, seeds=[0, 2], attempted=3)
    assert blob["n_scms_accepted"] == 2
    assert blob["n_scms_attempted"] == 3
    # The dropped seed is derivable only because the attempted count is known.
    assert sorted({r["seed"] for r in blob["records"]}) == [0, 2]
    assert set(range(blob["n_scms_attempted"])) - {r["seed"] for r in blob["records"]} == {1}


def test_missing_causenet_dump_records_none_rather_than_failing(tmp_path):
    """Provenance is documentation; an unreadable dump must not take the build down with it."""
    blob = _assemble(tmp_path, seeds=[0], attempted=1)
    assert blob["causenet_md5"] is None
    assert "code_commit" in blob


def test_databricks_builder_assembles_the_same_shape(tmp_path):
    """The internal Databricks path must not drift from the public one.

    The two write-loops are deliberately duplicated rather than shared (see either module's docstring), so
    the standing risk is that a fix lands in one and not the other - which is exactly how the internal path
    came to be missing ``n_scms_attempted`` in the first place. This pins their assembled output to the
    same shape and the same short-build behaviour.
    """
    public, internal = _builder_module(), _databricks_module()
    blobs = []
    for index, module in enumerate((public, internal)):
        out = tmp_path / f"out{index}"
        module._assemble(
            out,
            _partial(tmp_path / f"w{index}", seeds=[0, 2]),
            n_scms_attempted=3,
            gen_model="m",
            critic_model=None,
            causenet_dump=str(tmp_path / "missing.bz2"),
        )
        blobs.append(json.loads(out.with_suffix(".json").read_text(encoding="utf-8")))

    assert [k for k in blobs[0] if k != "version"] == [k for k in blobs[1] if k != "version"]
    for blob in blobs:
        assert blob["n_scms_attempted"] == 3
        assert blob["n_scms_accepted"] == 2  # neither raises on a short build


def test_commit_stamp_marks_an_uncommitted_tree():
    """A bare hash on a dirty build is a false provenance claim - it names a tree that never ran.

    Checked against this checkout's actual state rather than a fabricated one: the stamp must end in
    ``-dirty`` exactly when ``git status --porcelain`` is non-empty, whichever that happens to be when the
    test runs. Both builders must agree, since the Databricks path deliberately duplicates this helper.
    """
    import subprocess

    root = Path(__file__).parents[1]
    dirty = bool(
        subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout.strip()
    )
    for module in (_builder_module(), _databricks_module()):
        stamp = module._git_commit()
        assert stamp is not None
        assert stamp.endswith("-dirty") is dirty
        assert len(stamp.removesuffix("-dirty")) == 40

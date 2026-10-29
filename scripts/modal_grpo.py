r"""Run GRPO post-training (``scripts/train_grpo.py``) on one Modal GPU, and build training tasks on Modal CPUs.

    uv run --with modal modal run scripts/modal_grpo.py::build_main --n 1000
    uv run --with modal modal run scripts/modal_grpo.py::main --dataset train --run grpo120 --max-steps 120 \
        --extra "--save-steps 20"
    uv run --with modal modal run scripts/modal_grpo.py::reasoning_main --dataset outputs/reasoning/train.jsonl \
        --run reason60 --max-steps 60

Training tasks are uploaded once to the ``causalagentbench-runs`` volume, which also receives each run's
metrics (committed every minute while training) and its LoRA adapter. Model weights are cached in the
``hf-cache`` volume.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
app = modal.App("causalagentbench-grpo")
runs = modal.Volume.from_name("causalagentbench-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "trl[vllm,peft]==1.16.0",
        "vllm==0.31.0",
        "datasets>=4.7",
        "numpy>=2.1",
        "networkx>=3.3",
        "pydantic>=2.9",
        "causal-learn>=0.1.4.8",
    )
    .env({"HF_HOME": "/hf", "TRL_EXPERIMENTAL_SILENCE": "1", "PYTHONPATH": "/root/src"})
    .add_local_dir(ROOT / "src", "/root/src")
    .add_local_file(ROOT / "scripts" / "train_grpo.py", "/root/train_grpo.py")
    .add_local_file(ROOT / "scripts" / "train_reasoning.py", "/root/train_reasoning.py")
)


def _code_digest(root: Path) -> str:
    """SHA-256 over the paths and contents of the training code in the image: ``src`` and the training scripts."""
    files = sorted([*root.glob("src/**/*.py"), root / "train_grpo.py", root / "train_reasoning.py"])
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@app.function(image=image, gpu="H100", volumes={"/runs": runs, "/hf": hf_cache}, timeout=10 * 3600)
def train(stem: str, run: str, args: list[str], script: str = "train_grpo.py") -> int:
    """Train in a subprocess, committing the run directory every minute so progress is visible remotely.

    Modal restarts a preempted container with the same arguments, so training resumes from the run's latest
    checkpoint whenever one exists. The run directory records its training data, arguments and a hash of the
    training code in ``run.json``, and a run is never resumed with different ones.
    """
    record = Path(f"/runs/{run}/run.json")
    setup = {"data": stem, "script": script, "args": args, "code": _code_digest(Path("/root"))}
    if record.exists() and json.loads(record.read_text()) != setup:
        raise ValueError(f"run {run} was started with {record.read_text()}, not {setup}")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(setup))
    command = [sys.executable, f"/root/{script}", "--train", f"/runs/data/{stem}", "--out", f"/runs/{run}", *args]
    if list(Path(f"/runs/{run}").glob("checkpoint-*")) and "--resume" not in args:
        command.append("--resume")
    process = subprocess.Popen(command, env=os.environ.copy())
    while process.poll() is None:
        time.sleep(60)
        runs.commit()
    runs.commit()
    hf_cache.commit()
    return process.returncode


cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy>=2.1", "networkx>=3.3", "pydantic>=2.9", "causal-learn>=0.1.4.8")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(ROOT / "src", "/root/src")
)


@app.function(image=cpu_image, cpu=32, memory=32768, volumes={"/runs": runs}, timeout=2 * 3600)
def build(n: int, causenet: str, check: str) -> None:
    """Build ``n`` training tasks; fail unless the first tasks equal the locally built set ``check``.

    Building is deterministic, so tasks built here must be the tasks a local build produces.
    """
    import numpy as np

    from causalagentbench.cli import main as cli
    from causalagentbench.task.dataset import load_dataset

    sys.argv = ["causalagentbench", "build", "--causenet", f"/runs/data/{causenet}", "--split", "train"]
    sys.argv += ["-n", str(n), "--workers", "32", "--out", "/runs/data/train"]
    sys.argv += ["--exclude", "/runs/data/dev", "/runs/data/test"]
    if cli() != 0:
        raise RuntimeError("build failed")
    built, local = load_dataset("/runs/data/train"), load_dataset(f"/runs/data/{check}")
    for (instance, key), (expected, expected_key) in zip(built, local, strict=False):
        same = instance.prompt == expected.prompt and np.array_equal(instance.data, expected.data)
        if not same or key.scm.scaffold.edges != expected_key.scm.scaffold.edges:
            raise RuntimeError("tasks built on Modal differ from the local build")
    runs.commit()


def _upload(*paths: str) -> None:
    """Upload files to ``/data`` on the runs volume, skipping any already there with the same size."""
    try:
        present = {entry.path: entry.size for entry in runs.listdir("/data")}
    except modal.exception.NotFoundError:  # first upload: the directory does not exist yet
        present = {}
    with runs.batch_upload(force=True) as batch:
        for path in paths:
            local = ROOT / path
            if present.get(f"data/{local.name}") != local.stat().st_size:
                batch.put_file(str(local), f"/data/{local.name}")


def _pair(stem: str) -> tuple[str, str]:
    """A dataset's JSON and NPZ paths."""
    return f"{stem}.json", f"{stem}.npz"


@app.local_entrypoint()
def build_main(n: int, causenet: str = "", check: str = "artifacts/train_probe") -> None:
    """Upload CauseNet and the evaluation sets, then build ``n`` training tasks on Modal and check them."""
    causenet = causenet or str(Path.home() / "Desktop/causal-agent-benchmark/causenet-precision.jsonl.bz2")
    _upload(*_pair("artifacts/dev"), *_pair("artifacts/test"), *_pair(check))
    with runs.batch_upload(force=True) as batch:
        batch.put_file(causenet, f"/data/{Path(causenet).name}")
    build.remote(n, Path(causenet).name, Path(check).name)


@app.local_entrypoint()
def main(dataset: str, run: str, max_steps: int, gpu: str = "H100", extra: str = "") -> None:
    """Train on ``gpu`` with training tasks ``dataset``: a local stem to upload, or a name already on the volume.

    ``extra`` holds further train_grpo.py flags.
    """
    if Path(ROOT / f"{dataset}.json").exists():
        _upload(*_pair(dataset))
    stem = Path(dataset).name
    code = train.with_options(gpu=gpu).remote(stem, run, ["--max-steps", str(max_steps), *extra.split()])
    if code != 0:
        raise SystemExit(f"training exited with status {code}")


@app.local_entrypoint()
def reasoning_main(dataset: str, run: str, max_steps: int, gpu: str = "H100", extra: str = "") -> None:
    """Train ``train_reasoning.py`` on ``gpu`` with the local task file ``dataset`` (JSON lines), uploaded first.

    The file is uploaded as ``<first 16 hex digits of its SHA-256>-<name>``, so the run's command line records
    exactly which data it trained on, and files that share a name never overwrite each other.
    """
    local = ROOT / dataset
    name = f"{hashlib.sha256(local.read_bytes()).hexdigest()[:16]}-{local.name}"
    print(f"training data: /runs/data/{name}", flush=True)
    with runs.batch_upload(force=True) as batch:
        batch.put_file(str(local), f"/data/{name}")
    args = ["--max-steps", str(max_steps), *extra.split()]
    code = train.with_options(gpu=gpu).remote(name, run, args, "train_reasoning.py")
    if code != 0:
        raise SystemExit(f"training exited with status {code}")

r"""Environment v3 on Modal: GRPO training, evaluation and the sandbox isolation check, each in one container.

    uv run --with modal modal deploy scripts/modal_dataenv.py          # after every change to this file
    uv run --with modal modal run scripts/modal_dataenv.py::isolation_main
    uv run --with modal modal run scripts/modal_dataenv.py::eval_main --tasks outputs/dataenv/v3/iid_dev.jsonl \
        --name base --revision <sha> [--adapter data-pilot/adapter] [--samples 1]
    uv run --with modal modal run scripts/modal_dataenv.py::train_main --dataset outputs/dataenv/v3/train.jsonl \
        --run data-pilot --max-steps 60 --extra "--revision <sha>"

The model's code runs only here. Every function first calls ``lock_down``, which makes the runs volume and
the code directory unreadable to the sandbox user (``dataenv/sandbox.py``), and ``isolation_main`` checks that
the sandbox cannot read them.

Evaluation drives the same ``DataEnv`` as training, with offline vLLM: prompts and tool results are rendered
with the model's chat template and tool calls parsed with TRL's ``parse_response``, as TRL does in training.
A completion's budget (model and tool tokens together) and the number of tool rounds are training's.
Evaluation calls are spawned on the deployed app and reattached on rerun, as in ``modal_eval.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).resolve().parent.parent
app = modal.App("causalagentbench-dataenv")
runs = modal.Volume.from_name("causalagentbench-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
PACKAGES = (
    "trl[vllm,peft]==1.16.0",
    "vllm==0.31.0",
    "datasets>=4.7",
    "numpy>=2.1",
    "networkx>=3.3",
    "pydantic>=2.9",
    "causal-learn>=0.1.4.8",
    "pandas>=2.2",
    "scipy>=1.13",
    "statsmodels>=0.14",
)


def _with_code(base: modal.Image) -> modal.Image:
    """``base`` with the packages, settings and code shared by every function."""
    return (
        base.uv_pip_install(*PACKAGES)
        .env({"HF_HOME": "/hf", "TRL_EXPERIMENTAL_SILENCE": "1", "PYTHONPATH": "/root/src"})
        .add_local_dir(ROOT / "src", "/root/src")
        .add_local_file(ROOT / "scripts" / "train_grpo.py", "/root/train_grpo.py")
        .add_local_file(ROOT / "scripts" / "train_dataenv.py", "/root/train_dataenv.py")
    )


image = _with_code(modal.Image.debian_slim(python_version="3.12"))
"""Training image, as in ``modal_grpo.py``."""
eval_image = _with_code(
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12").entrypoint([])
)
"""Evaluation image: vLLM JIT-compiles FlashInfer's top-k/top-p sampling kernels with nvcc (as in ``modal_eval.py``)."""
MAX_COMPLETION_TOKENS = 8192
"""Tokens of one episode after the prompt (model and tool output together), as in training."""
WAVE = 96
"""Episodes run concurrently in evaluation (each holds a Python worker)."""


def lock_down() -> None:
    """Make the runs volume and the code directory unreadable to the sandbox user."""
    for path in ("/runs", "/root"):
        os.chmod(path, 0o700)


def _code_digest(root: Path) -> str:
    """SHA-256 over the paths and contents of the training code in the image."""
    files = sorted([*root.glob("src/**/*.py"), root / "train_grpo.py", root / "train_dataenv.py"])
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


VOLUMES = {"/runs": runs, "/hf": hf_cache}


@app.function(image=image, gpu="H100", cpu=24, memory=65536, volumes=VOLUMES, timeout=12 * 3600)
def train(stem: str, run: str, args: list[str]) -> int:
    """Train in a subprocess, committing the run directory every minute; never resume with other data, args or code."""
    lock_down()
    record = Path(f"/runs/{run}/run.json")
    setup = {"data": stem, "script": "train_dataenv.py", "args": args, "code": _code_digest(Path("/root"))}
    if record.exists() and json.loads(record.read_text()) != setup:
        raise ValueError(f"run {run} was started with {record.read_text()}, not {setup}")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(setup))
    command = [sys.executable, "/root/train_dataenv.py", "--train", f"/runs/data/{stem}", "--out", f"/runs/{run}"]
    command += args
    if list(Path(f"/runs/{run}").glob("checkpoint-*")) and "--resume" not in args:
        command.append("--resume")
    process = subprocess.Popen(command, env=os.environ.copy())
    while process.poll() is None:
        time.sleep(60)
        runs.commit()
    runs.commit()
    hf_cache.commit()
    return process.returncode


ATTACKS = {
    "uid": "import os; print(os.getuid(), os.getgid())",
    "volume": "import os; print(os.listdir('/runs'))",
    "task_file": "print(open('/runs/data/' + __import__('os').listdir('/runs/data')[0]).read()[:100])",
    "code": "import os; print(os.listdir('/root'))",
    "parent_environ": "import os; print(open(f'/proc/{os.getppid()}/environ').read()[:200])",
    "parent_memory": "import os; print(open(f'/proc/{os.getppid()}/mem', 'rb').read(10))",
    "environ": "import os; print(sorted(os.environ))",
}
"""Code the isolation check runs in the sandbox: every read below must fail."""


def _attack() -> dict[str, str]:
    """Run ``ATTACKS`` in a sandbox session, as training does; return each one's output."""
    import numpy as np

    from causalagentbench.dataenv.sandbox import PythonSession

    lock_down()
    session = PythonSession(np.random.default_rng(0).standard_normal((10, 2)), ["a", "b"])
    try:
        return {name: session.run(code) for name, code in ATTACKS.items()}
    finally:
        session.close()


@app.function(image=image, cpu=4, memory=8192, volumes={"/runs": runs}, timeout=1800)
def isolation() -> dict[str, str]:
    """``_attack`` in the training image."""
    return _attack()


@app.function(image=eval_image, cpu=4, memory=8192, volumes={"/runs": runs}, timeout=1800)
def isolation_eval() -> dict[str, str]:
    """``_attack`` in the evaluation image."""
    return _attack()


@app.local_entrypoint()
def isolation_main() -> None:
    """In both images, fail unless the sandbox runs as nobody and cannot read the volume, code or parent process."""
    for check in (isolation, isolation_eval):
        _check_isolation(check.remote())
    print("isolation check passed in both images")


def _check_isolation(results: dict[str, str]) -> None:
    """Fail unless every read in ``ATTACKS`` was denied and the worker ran as nobody with a clean environment."""
    for name, output in results.items():
        print(f"--- {name}\n{output}")
    denied = ("volume", "task_file", "code", "parent_environ", "parent_memory")
    failures = [name for name in denied if "PermissionError" not in results[name]]
    if not results["uid"].startswith("65534 65534"):
        failures.append("uid")
    if any(key in results["environ"] for key in ("MODAL", "HF_", "TOKEN", "KEY")):
        failures.append("environ")
    if failures:
        raise SystemExit(f"isolation check failed: {failures}")


def _regenerate(worlds: list[dict[str, Any]]) -> list[str]:
    """Ids of worlds whose data do not regenerate to their stored moments on this machine."""
    from causalagentbench.dataenv.dataset import world_from_dict

    failed = []
    for record in worlds:
        try:
            world_from_dict(record["world"]).data()
        except RuntimeError:
            failed.append(record["id"])
    return failed


@app.function(image=image, cpu=4, memory=8192, timeout=1800)
def regenerate(worlds: list[dict[str, Any]]) -> list[str]:
    """``_regenerate`` in the training image."""
    return _regenerate(worlds)


@app.function(image=eval_image, cpu=4, memory=8192, timeout=1800)
def regenerate_eval(worlds: list[dict[str, Any]]) -> list[str]:
    """``_regenerate`` in the evaluation image."""
    return _regenerate(worlds)


@app.local_entrypoint()
def regenerate_main(tasks: str, n: int = 300) -> None:
    """Fail unless the first ``n`` tasks' data regenerate to their stored moments in both images."""
    sys.path.insert(0, str(ROOT / "src"))
    from causalagentbench.dataenv.dataset import load

    records = [{"id": r["id"], "world": r["world"]} for r in load(ROOT / tasks)[:n]]
    for label, check in (("training image", regenerate), ("evaluation image", regenerate_eval)):
        failed = check.remote(records)
        if failed:
            raise SystemExit(f"{label}: {len(failed)} of {len(records)} worlds differ, e.g. {failed[:5]}")
    print(f"all {len(records)} worlds regenerate to the build's moments in both images")


def _suffix_ids(tokenizer: Any, tool_messages: list[dict[str, Any]], tools: list[Any]) -> list[int]:
    """Token ids that append tool results and open the next assistant turn (TRL's ``_get_tool_suffix_ids``)."""
    calls = [{"type": "function", "function": {"name": tool_messages[0]["name"], "arguments": {}}}]
    dummy = [{"role": "user", "content": "dummy"}, {"role": "assistant", "content": "", "tool_calls": calls}]
    prefix = tokenizer.apply_chat_template(dummy, tools=tools, add_generation_prompt=False, tokenize=True)
    full = tokenizer.apply_chat_template(dummy + tool_messages, tools=tools, add_generation_prompt=True, tokenize=True)
    if hasattr(prefix, "keys"):
        prefix, full = prefix["input_ids"], full["input_ids"]
    ends = [i for i, token in enumerate(prefix) if token == tokenizer.eos_token_id]
    if ends:
        prefix = prefix[: ends[-1] + 1]
    if full[: len(prefix)] != prefix:
        raise ValueError("unexpected tokenization of tool results")
    return list(full[len(prefix) :])


@app.function(image=eval_image, gpu="L40S", cpu=16, memory=65536, volumes=VOLUMES, timeout=8 * 3600)
def evaluate(
    model: str, revision: str, adapter: str, tasks: list[dict[str, Any]], seeds: list[int]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one episode per (task, seed) with offline vLLM; return the settings and each episode's result."""
    import vllm
    from trl.chat_template_utils import add_response_schema
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    lock_down()
    settings: dict[str, Any] = {"base_model": model, "revision": revision, "adapter": adapter, "vllm": vllm.__version__}
    if adapter:
        directory = Path(f"/runs/{adapter}")
        digest = hashlib.sha256()
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            digest.update(hashlib.sha256((directory / name).read_bytes()).digest())
        settings["adapter_sha256"] = digest.hexdigest()
    lora = {"enable_lora": True, "max_lora_rank": 32} if adapter else {}
    llm = LLM(
        model=model,
        revision=revision or None,
        tokenizer_revision=revision or None,
        max_model_len=MAX_COMPLETION_TOKENS + 4096,
        seed=0,
        **lora,
    )
    hf_cache.commit()
    tokenizer = add_response_schema(llm.get_tokenizer())
    defaults = llm.get_default_sampling_params()
    settings["sampling"] = {key: getattr(defaults, key) for key in ("temperature", "top_p", "top_k", "min_p")}
    settings["max_completion_tokens"] = MAX_COMPLETION_TOKENS
    request = LoRARequest("adapter", 1, str(Path(f"/runs/{adapter}"))) if adapter else None

    results = []
    for start in range(0, len(tasks), WAVE):
        wave = list(zip(tasks[start : start + WAVE], seeds[start : start + WAVE], strict=True))
        results += _run_wave(llm, tokenizer, defaults, request, wave)
    return settings, results


def _run_tools(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the tool calls of an episode's last message; an error becomes the result, as in TRL's tool loop."""
    tool_messages = []
    for call in episode["messages"][-1]["tool_calls"]:
        name = call.get("function", {}).get("name", "unknown")
        try:
            if call.get("type") != "function" or name != "python":
                raise ValueError(f"Tool {name} not found.")
            result: Any = episode["env"].python(**call["function"]["arguments"])
        except Exception as error:  # the model sees the error, as in TRL's tool loop
            result = {"error": str(error)}
        tool_messages.append({"role": "tool", "name": name, "content": str(result)})
    return tool_messages


def _finish(episode: dict[str, Any]) -> None:
    """Grade the episode's last assistant message; a reply cut off by the budget counts as no answer."""
    message = episode["messages"][-1]
    reply = "" if episode["truncated"] else str(message.get("content") or "")
    episode["reward"] = episode["env"]._score(reply)


def _run_wave(llm: Any, tokenizer: Any, defaults: Any, request: Any, wave: list[tuple[dict, int]]) -> list[dict]:
    """Run a wave of episodes to completion, every active episode generating its next turn in one batch.

    As in TRL's tool loop: a turn with tool calls runs them and continues, up to ``MAX_TOOL_CALLS`` rounds and
    while the tool results fit the budget; the episode's final reply is its last assistant message. A reply cut
    off by the token budget counts as no answer.
    """
    from trl.chat_template_utils import parse_response

    from causalagentbench.dataenv.environment import MAX_TOOL_CALLS, DataEnv

    episodes = []
    for task, seed in wave:
        env = DataEnv([])
        messages = [{"role": "user", "content": env.reset(task=task)}]
        ids = tokenizer.apply_chat_template(messages, tools=[env.python], add_generation_prompt=True, tokenize=True)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        episodes.append(
            {"task": task, "env": env, "messages": messages, "prompt_ids": ids, "ids": [], "seed": seed, "rounds": 0}
        )
    active = list(range(len(episodes)))
    while active:
        params = []
        for i in active:
            sample = defaults.clone()
            sample.max_tokens = MAX_COMPLETION_TOKENS - len(episodes[i]["ids"])
            sample.seed = episodes[i]["seed"] * 1000 + len(episodes[i]["messages"])
            params.append(sample)
        outputs = llm.generate(
            [{"prompt_token_ids": episodes[i]["prompt_ids"] + episodes[i]["ids"]} for i in active],
            params,
            lora_request=request,
        )
        to_run = []
        for i, output in zip(active, outputs, strict=True):
            episode, env = episodes[i], episodes[i]["env"]
            new = list(output.outputs[0].token_ids)
            prefix = episode["prompt_ids"] + episode["ids"]
            episode["ids"] += new
            message = parse_response(tokenizer, new, prefix=prefix, tools=[env.python])
            episode["messages"].append(message)
            episode["truncated"] = output.outputs[0].finish_reason == "length"
            if message.get("tool_calls") and not episode["truncated"] and episode["rounds"] < MAX_TOOL_CALLS:
                to_run.append(i)
            else:
                _finish(episode)
        # Tool calls wait on worker processes, so the episodes' calls run concurrently.
        with ThreadPoolExecutor(max(len(to_run), 1)) as pool:
            tool_results = list(pool.map(lambda i: _run_tools(episodes[i]), to_run))
        still = []
        for i, tool_messages in zip(to_run, tool_results, strict=True):
            episode = episodes[i]
            suffix = _suffix_ids(tokenizer, tool_messages, [episode["env"].python])
            if len(episode["ids"]) + len(suffix) <= MAX_COMPLETION_TOKENS:
                episode["ids"] += suffix
                episode["messages"] += tool_messages
                episode["rounds"] += 1
                still.append(i)
            else:
                _finish(episode)
        active = still
    return [
        {
            "id": episode["task"]["id"],
            "seed": episode["seed"],
            "reward": episode["reward"],
            "truncated": episode["truncated"],
            "completion_tokens": len(episode["ids"]),
            **episode["env"].info,
            "messages": episode["messages"],
        }
        for episode in episodes
    ]


@app.local_entrypoint()
def eval_main(tasks: str, name: str, revision: str, adapter: str = "", samples: int = 1, seed: int = 1000) -> None:
    """Evaluate every task of a file ``samples`` times on the deployed app; append to ``eval/<name>.jsonl``."""
    sys.path.insert(0, str(ROOT / "src"))
    from causalagentbench.dataenv.dataset import load

    model = "Qwen/Qwen3-4B-Instruct-2507"
    records = load(ROOT / tasks)
    out = ROOT / "outputs" / "dataenv" / "eval" / f"{name}.jsonl"
    done = {(r["id"], r["seed"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    todo = [(r, seed + j) for r in records for j in range(samples) if (r["id"], seed + j) not in done]
    print(f"{name}: {len(todo)} episodes to run", flush=True)
    if not todo:
        return
    arguments = (model, revision, adapter, [r for r, _ in todo], [s for _, s in todo])
    key = hashlib.sha256(json.dumps([model, revision, adapter, [[r["id"], s] for r, s in todo]]).encode()).hexdigest()
    pending = ROOT / "outputs" / "dataenv" / "pending" / f"{key}.txt"
    if pending.exists():
        call_id = pending.read_text(encoding="utf-8").strip()
        call = modal.FunctionCall.from_id(call_id)
        print(f"{name}: waiting for call {call_id}, spawned earlier", flush=True)
    else:
        call = modal.Function.from_name(app.name, "evaluate").spawn(*arguments)
        pending.parent.mkdir(parents=True, exist_ok=True)
        pending.write_text(call.object_id, encoding="utf-8")
        print(f"{name}: spawned call {call.object_id}", flush=True)
    settings, results = call.get()
    print(f"{name}: {settings}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps({"model": name, **result, **settings}, default=str) + "\n")
    pending.unlink()
    accuracy = sum(r["reward"] for r in results) / len(results)
    print(f"{name}: {len(results)} episodes, accuracy {accuracy:.3f}", flush=True)


@app.local_entrypoint()
def train_main(dataset: str, run: str, max_steps: int, extra: str = "") -> None:
    """Upload the task file as ``<sha16>-<name>`` and train ``run`` on the deployed app's ``train``."""
    local = ROOT / dataset
    stem = f"{hashlib.sha256(local.read_bytes()).hexdigest()[:16]}-{local.name}"
    with runs.batch_upload(force=True) as batch:
        batch.put_file(str(local), f"/data/{stem}")
    print(f"training data: /runs/data/{stem}", flush=True)
    call = modal.Function.from_name(app.name, "train").spawn(stem, run, ["--max-steps", str(max_steps), *extra.split()])
    print(f"spawned training call {call.object_id}", flush=True)
    code = call.get()
    if code != 0:
        raise SystemExit(f"training exited with status {code}")

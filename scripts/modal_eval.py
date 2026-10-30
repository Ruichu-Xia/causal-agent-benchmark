r"""Sample replies to evaluation sets with offline vLLM on one Modal GPU.

    uv run --with modal modal run scripts/modal_eval.py --model Qwen/Qwen3-8B --name qwen3-8b \
        --sets iid_dev,ood_size_dev
    uv run --with modal modal run scripts/modal_eval.py --model Qwen/Qwen3-4B-Instruct-2507 --name trained \
        --adapter reason-v2/adapter --sets iid_test,dsep --samples 2 --seed 1000 --max-tokens 8192

Sets are ``transfer_eval.items`` names; only their prompts go to Modal, and labels stay local. Every prompt is
one user message (no system prompt, no tools), sampled with the model's generation-config defaults and at most
``max_tokens`` new tokens (default ``transfer_eval.MAX_TOKENS``), at Hugging Face revision ``revision`` (default: the
latest); sample j of every item uses seed
``seed + j``. ``adapter`` is a LoRA adapter on the ``causalagentbench-runs`` volume, applied to ``model``.
Hybrid Qwen3 models run with thinking off (``enable_thinking=False``; templates without the switch ignore it).
Replies are appended to ``outputs/transfer/<set>.jsonl`` under ``name``, in ``transfer_eval.py``'s record
format, so ``transfer_eval.py report`` scores them; (item, seed) pairs already answered by ``name`` are skipped.
Each record also carries the model and its revision, the adapter path and ``adapter_digest``, ``max_tokens``, the
sampling parameters, the vLLM version and the SHA-256 of the prompt, so an analysis can check that every reply
came from the registered configuration and the registered prompt. The remote call also saves its settings and
replies to ``/eval/<SHA-256 of the requests>.json`` on the runs volume before returning.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).resolve().parent.parent
app = modal.App("causalagentbench-eval")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("causalagentbench-runs", create_if_missing=True)
# The CUDA toolkit image, as in modal_serve.py: vLLM JIT-compiles FlashInfer sampling kernels with nvcc.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.31.0")
    .env({"HF_HOME": "/hf"})
)


def adapter_digest(directory: Path) -> str:
    """SHA-256 over the SHA-256 digests of a LoRA adapter's config and weights files, in that order."""
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        digest.update(hashlib.sha256((directory / name).read_bytes()).digest())
    return digest.hexdigest()


@app.function(image=image, gpu="L40S", volumes={"/hf": hf_cache, "/runs": runs}, timeout=6 * 3600)
def generate(
    model: str, revision: str, adapter: str, requests: list[tuple[str, int]], max_tokens: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The generation settings, and one reply per (prompt, seed) request with its finish reason and token count."""
    import vllm
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    settings = {
        "base_model": model,
        "revision": revision,
        "adapter": adapter,
        "adapter_sha256": adapter_digest(Path(f"/runs/{adapter}")) if adapter else "",
        "max_tokens": max_tokens,
        "vllm": vllm.__version__,
    }

    lora = {"enable_lora": True, "max_lora_rank": 32} if adapter else {}
    llm = LLM(
        model=model,
        revision=revision or None,
        tokenizer_revision=revision or None,
        max_model_len=max_tokens + 4096,
        seed=0,
        **lora,
    )
    defaults = llm.get_default_sampling_params()
    print(f"{model} {adapter}: {defaults}", flush=True)
    settings["sampling"] = {key: getattr(defaults, key) for key in ("temperature", "top_p", "top_k", "min_p")}
    params = []
    for _, seed in requests:
        sample = defaults.clone()
        sample.max_tokens = max_tokens
        sample.seed = seed
        params.append(sample)
    outputs = llm.chat(
        [[{"role": "user", "content": prompt}] for prompt, _ in requests],
        params,
        chat_template_kwargs={"enable_thinking": False},
        lora_request=LoRARequest("adapter", 1, f"/runs/{adapter}") if adapter else None,
    )
    hf_cache.commit()
    replies = [
        {
            "reply": output.outputs[0].text,
            "finish_reason": output.outputs[0].finish_reason,
            "completion_tokens": len(output.outputs[0].token_ids),
        }
        for output in outputs
    ]
    # A copy on the volume, so a lost local client does not lose the replies.
    request_digest = hashlib.sha256(json.dumps(requests).encode("utf-8")).hexdigest()
    backup = Path(f"/runs/eval/{request_digest}.json")
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text(json.dumps({"settings": settings, "requests": requests, "replies": replies}))
    runs.commit()
    return settings, replies


@app.local_entrypoint()
def main(
    model: str,
    name: str,
    sets: str,
    samples: int = 1,
    seed: int = 0,
    max_tokens: int = 0,
    adapter: str = "",
    gpu: str = "L40S",
    revision: str = "",
) -> None:
    """Answer every item of the comma-separated sets not yet answered by ``name``, in one GPU call.

    ``max_tokens`` 0 means ``transfer_eval.MAX_TOKENS``.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from transfer_eval import MAX_TOKENS, OUT, _load, items

    todo: list[tuple[Path, str, str, int]] = []
    for set_name in sets.split(","):
        out = OUT / f"{set_name}.jsonl"
        done = {(r["id"], r["seed"]) for r in _load(out) if r["model"] == name}
        for item in items(set_name):
            for j in range(samples):
                if (item["id"], seed + j) not in done:
                    todo.append((out, item["id"], item["prompt"], seed + j))
    print(f"{name}: {len(todo)} replies to sample", flush=True)
    if not todo:
        return
    requests = [(prompt, s) for _, _, prompt, s in todo]
    settings, replies = generate.with_options(gpu=gpu).remote(
        model, revision, adapter, requests, max_tokens or MAX_TOKENS
    )
    print(f"{name}: {settings}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for (out, item_id, prompt, s), reply in zip(todo, replies, strict=True):
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        record = {"id": item_id, "model": name, "seed": s, **reply, **settings, "prompt_sha256": prompt_sha256}
        with out.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)  # other evaluations may append to the same file concurrently
            handle.write(json.dumps(record) + "\n")

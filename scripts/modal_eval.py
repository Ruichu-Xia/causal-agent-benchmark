r"""Sample replies to text-reasoning tasks with offline vLLM on one Modal GPU.

    uv run --with modal modal run scripts/modal_eval.py --model Qwen/Qwen3-8B --name qwen3-8b \
        --tasks outputs/reasoning/v2/iid_dev.jsonl,outputs/reasoning/v2/ood_size_dev.jsonl

Only each task's prompt goes to Modal; answers stay local. Every prompt is one user message (no system prompt,
no tools), sampled with the model's generation-config defaults, at most ``max_tokens`` new tokens
(default ``transfer_eval.MAX_TOKENS``) and seed 0, the protocol of the ``modal_serve.py`` endpoint. Hybrid
Qwen3 models run with thinking off (``enable_thinking=False``; templates without the switch ignore it).
Replies are appended to
``outputs/transfer/<task file stem>.jsonl`` under ``name``, in ``transfer_eval.py``'s record format, so
``transfer_eval.py report`` scores them; tasks already answered by ``name`` are skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).resolve().parent.parent
app = modal.App("causalagentbench-eval")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
# The CUDA toolkit image, as in modal_serve.py: vLLM JIT-compiles FlashInfer sampling kernels with nvcc.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.31.0")
    .env({"HF_HOME": "/hf"})
)


@app.function(image=image, gpu="L40S", volumes={"/hf": hf_cache}, timeout=4 * 3600)
def generate(model: str, prompts: list[str], max_tokens: int) -> list[dict[str, Any]]:
    """One reply per prompt, with its finish reason and number of generated tokens."""
    from vllm import LLM

    llm = LLM(model=model, max_model_len=max_tokens + 4096, seed=0)
    params = llm.get_default_sampling_params()
    params.max_tokens = max_tokens
    params.seed = 0
    print(f"{model}: {params}", flush=True)
    outputs = llm.chat(
        [[{"role": "user", "content": prompt}] for prompt in prompts],
        params,
        chat_template_kwargs={"enable_thinking": False},
    )
    hf_cache.commit()
    return [
        {
            "reply": output.outputs[0].text,
            "finish_reason": output.outputs[0].finish_reason,
            "completion_tokens": len(output.outputs[0].token_ids),
        }
        for output in outputs
    ]


@app.local_entrypoint()
def main(model: str, name: str, tasks: str, gpu: str = "L40S", max_tokens: int = 0) -> None:
    """Answer every task in the comma-separated task files not yet answered by ``name``, in one GPU call.

    ``max_tokens`` 0 means ``transfer_eval.MAX_TOKENS``.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from transfer_eval import MAX_TOKENS, OUT, _load

    todo: list[tuple[Path, dict[str, Any]]] = []
    for path in map(Path, tasks.split(",")):
        out = OUT / f"{path.stem}.jsonl"
        done = {r["id"] for r in _load(out) if r["model"] == name}
        for line in path.read_text(encoding="utf-8").splitlines():
            task = json.loads(line)
            if task["id"] not in done:
                todo.append((out, task))
    print(f"{name}: {len(todo)} tasks to answer", flush=True)
    if not todo:
        return
    prompts = [task["prompt"] for _, task in todo]
    replies = generate.with_options(gpu=gpu).remote(model, prompts, max_tokens or MAX_TOKENS)
    OUT.mkdir(parents=True, exist_ok=True)
    for (out, task), reply in zip(todo, replies, strict=True):
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": task["id"], "model": name, "seed": 0, **reply}) + "\n")

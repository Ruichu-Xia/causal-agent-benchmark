r"""Serve the base model and a trained LoRA adapter with vLLM on Modal, for evaluation through the harness.

    ADAPTERS="trained=grpo120/adapter,step40=grpo120/checkpoint-40" \
        uv run --with modal modal deploy scripts/modal_serve.py
    export HOSTED_VLLM_API_BASE=<printed url>/v1   # HOSTED_VLLM_API_KEY is in .env
    causalagentbench eval --dataset artifacts/test --model hosted_vllm/base --model hosted_vllm/trained ...

``ADAPTERS`` names LoRA adapters on the ``causalagentbench-runs`` volume, served beside the base model as
``base``. Every model uses vLLM's defaults from the model's generation config, the same sampling for each.
The endpoint requires the API key in the ``vllm-api-key`` Modal secret (``VLLM_API_KEY``).
"""

from __future__ import annotations

import os
import subprocess

import modal

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTERS = os.environ.get("ADAPTERS", "")
app = modal.App("causalagentbench-serve")
runs = modal.Volume.from_name("causalagentbench-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
# The CUDA toolkit image: vLLM JIT-compiles FlashInfer sampling kernels with nvcc (torch 2.13 uses CUDA 13).
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.31.0")
    .env({"HF_HOME": "/hf", "ADAPTERS": ADAPTERS})
)


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/runs": runs, "/hf": hf_cache},
    secrets=[modal.Secret.from_name("vllm-api-key")],
    timeout=6 * 3600,
    scaledown_window=10 * 60,
)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve() -> None:
    """Start an OpenAI-compatible vLLM server exposing ``base`` and each adapter in ``ADAPTERS``."""
    command = [
        "vllm",
        "serve",
        BASE_MODEL,
        "--served-model-name",
        "base",
        "--port",
        "8000",
        "--api-key",
        os.environ["VLLM_API_KEY"],
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--max-model-len",
        "32768",
    ]
    adapters = [entry.split("=", 1) for entry in os.environ["ADAPTERS"].split(",") if entry]
    if adapters:
        modules = [f"{name}=/runs/{path}" for name, path in adapters]
        command += ["--enable-lora", "--max-lora-rank", "32", "--max-loras", str(len(modules))]
        command += ["--lora-modules", *modules]
    subprocess.Popen(command)

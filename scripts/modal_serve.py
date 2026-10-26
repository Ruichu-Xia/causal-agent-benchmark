"""Serve the base model and a trained LoRA adapter with vLLM on Modal, for evaluation through the harness.

    uv run --with modal modal deploy scripts/modal_serve.py   # prints the endpoint URL
    export HOSTED_VLLM_API_BASE=<url>/v1 HOSTED_VLLM_API_KEY=$VLLM_API_KEY
    causalagentbench eval --dataset artifacts/test --model hosted_vllm/base --model hosted_vllm/trained ...

The adapter is read from the ``causalagentbench-runs`` volume at ``/<ADAPTER_RUN>/adapter``. Both models use
vLLM's defaults from the model's generation config, the same sampling for each. The endpoint requires the
API key in the ``vllm-api-key`` Modal secret (``VLLM_API_KEY``).
"""

from __future__ import annotations

import os
import subprocess

import modal

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_RUN = os.environ.get("ADAPTER_RUN", "main")
app = modal.App("causalagentbench-serve")
runs = modal.Volume.from_name("causalagentbench-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.31.0")
    .env({"HF_HOME": "/hf", "ADAPTER_RUN": ADAPTER_RUN})
)


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/runs": runs, "/hf": hf_cache},
    secrets=[modal.Secret.from_name("vllm-api-key")],
    timeout=6 * 3600,
    scaledown_window=10 * 60,
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve() -> None:
    """Start an OpenAI-compatible vLLM server exposing ``base`` and, once trained, the LoRA ``trained``."""
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
    adapter = f"/runs/{os.environ['ADAPTER_RUN']}/adapter"
    if os.path.isdir(adapter):  # before training there is only the base model to serve
        command += ["--enable-lora", "--max-lora-rank", "32", "--lora-modules", f"trained={adapter}"]
    subprocess.Popen(command)

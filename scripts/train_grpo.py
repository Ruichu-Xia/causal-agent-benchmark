"""GRPO post-training on ``CausalEnv``: LoRA on a small instruct model, TRL with vLLM colocated on one GPU.

Runs on any single CUDA GPU (``scripts/modal_grpo.py`` runs it on Modal)::

    python scripts/train_grpo.py --train artifacts/train --out runs/grpo --max-steps 120

Training tasks come from ``build --split train --exclude <evaluation sets>``. The prompt, tools, budget and
answer format are the evaluation protocol's (``lean-v2``), and the reward is the benchmark's own score of
the final reply. Metrics stream to ``<out>/metrics.jsonl`` and every scored rollout to ``<out>/rollouts.jsonl``;
the LoRA adapter is saved to ``<out>/adapter``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen3-4B-Instruct-2507"

ROLLOUT_LOG: Path | None = None
"""Where ``training_reward`` appends every scored rollout; set by ``main``."""


def keep_lora_base_weights_exact() -> None:
    """Patch TRL's vLLM weight export so merging LoRA for generation never alters the frozen base weights.

    TRL merges the adapter into the bf16 base weights to export them to vLLM, then unmerges by subtracting a
    rounded delta, so the base drifts at every sync (huggingface/trl#7423, unfixed in 1.16.0). The patch
    restores the exact base weights after each export.
    """
    import torch
    from trl.generation import vllm_generation

    original = vllm_generation.VLLMGeneration._iter_named_params

    def exact(self: Any) -> Any:
        """Export as TRL does, then restore every LoRA base weight bit for bit."""
        # Kept in host memory: a GPU copy of every base weight (~7 GB for 4B) contributed to an OOM.
        saved = {
            name: p.detach().to("cpu", copy=True) for name, p in self.model.named_parameters() if ".base_layer." in name
        }
        try:
            yield from original(self)
        finally:
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if name in saved:
                        p.copy_(saved[name])

    vllm_generation.VLLMGeneration._iter_named_params = exact


def check_lora_export_is_exact() -> None:
    """Run the TRL issue's reproduction and fail unless the patched export leaves the base weight unchanged."""
    import torch
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model
    from trl.distributed import DistributedBackend
    from trl.generation.vllm_generation import VLLMGeneration

    model = get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False)), LoraConfig(r=1, lora_alpha=1, target_modules=["0"])
    )
    layer = model.base_model.model[0]
    with torch.no_grad():
        layer.base_layer.weight.fill_(0.1)
        layer.lora_A["default"].weight.fill_(0.2)
        layer.lora_B["default"].weight.fill_(0.2)
    before = layer.base_layer.weight.detach().clone()
    exporter = object.__new__(VLLMGeneration)
    exporter.model = model
    exporter._dist = DistributedBackend(Accelerator(cpu=True))
    for _ in exporter._iter_named_params():
        pass
    if not torch.equal(before, layer.base_layer.weight):
        raise RuntimeError("LoRA export still changes the base weights; the patch did not take effect")


def training_reward(
    completions: list[Any], environments: list[Any], log_metric: Any = None, **kwargs: Any
) -> list[float]:
    """The benchmark reward, plus the validity rate and error on valid answers as training metrics."""
    from causalagentbench.environment import reward_func

    rewards = reward_func(completions, environments)
    if ROLLOUT_LOG is not None:  # full transcripts, for reward-hacking checks (TRL's parquet log fails on them)
        with ROLLOUT_LOG.open("a", encoding="utf-8") as handle:
            for completion, env, reward in zip(completions, environments, rewards, strict=True):
                record = {"time": time.time(), "reward": reward, **env.info, "completion": completion}
                handle.write(json.dumps(record, default=str) + "\n")
    if log_metric is not None:
        valid = [env.info["task_error"] for env in environments if env.info.get("parse_ok")]
        log_metric("valid_rate", len(valid) / len(environments))
        log_metric("valid_task_error", statistics.mean(valid) if valid else float("nan"))
        log_metric("env_tool_calls", statistics.mean(env.info.get("tool_calls", 0) for env in environments))
    return rewards


def main() -> None:
    """Parse arguments, build the trainer and train."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True, help="training dataset stem (built with --split train)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--tasks-per-step", type=int, default=8)
    parser.add_argument("--generations", type=int, default=8, help="rollouts per task (the GRPO group)")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--max-completion-length", type=int, default=8192)
    parser.add_argument("--vllm-memory", type=float, default=0.35, help="GPU memory share for vLLM")
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--micro-batch", type=int, default=4, help="rollouts per backward pass (memory, not math)")
    parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in --out")
    args = parser.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from causalagentbench.agent.loop import _SYSTEM_PROMPT, MAX_TOOL_CALLS, MAX_TURNS
    from causalagentbench.environment import CausalEnv
    from causalagentbench.task.dataset import load_dataset

    keep_lora_base_weights_exact()
    check_lora_export_is_exact()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    global ROLLOUT_LOG
    ROLLOUT_LOG = out / "rollouts.jsonl"
    tasks = load_dataset(args.train)
    # The task prompt is appended to the empty user message by ``CausalEnv.reset``; a row's other columns
    # (``seed``) reach ``reset`` as keyword arguments, so all rollouts of a group share one task.
    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": ""}]] * len(tasks),
            "seed": list(range(len(tasks))),
        }
    )

    class MetricsLog(TrainerCallback):
        """Append every logged metrics dict to ``metrics.jsonl`` as training runs."""

        def on_log(self, _args: Any, state: Any, _control: Any, logs: dict[str, Any] | None = None, **kw: Any) -> None:
            """Write one JSON line per log event, stamped with the step and wall time."""
            if logs:
                with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": state.global_step, "time": time.time(), **logs}) + "\n")

    rollouts_per_step = args.tasks_per_step * args.generations
    if rollouts_per_step % args.micro_batch:
        raise ValueError("--micro-batch must divide the rollouts per step")
    config = GRPOConfig(
        output_dir=str(out),
        max_steps=args.max_steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=rollouts_per_step // args.micro_batch,
        num_generations=args.generations,
        generation_batch_size=rollouts_per_step,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        loss_type="dapo",
        scale_rewards="batch",
        beta=0.0,
        mask_truncated_completions=True,
        max_tool_calling_iterations=MAX_TURNS - 1,  # the evaluation protocol's investigation turns
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_memory,
        vllm_enable_sleep_mode=True,
        vllm_max_model_length=16384,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=args.save_steps,
        report_to="none",
        log_completions=False,  # TRL 1.16 fails to write tool-call completions to parquet; see ROLLOUT_LOG
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=training_reward,
        args=config,
        train_dataset=dataset,
        environment_factory=lambda: CausalEnv(tasks, max_tool_calls=MAX_TOOL_CALLS),
        peft_config=LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, target_modules="all-linear"),
        callbacks=[MetricsLog()],
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(out / "adapter"))
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

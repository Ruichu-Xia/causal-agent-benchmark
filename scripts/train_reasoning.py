"""GRPO on text causal-reasoning tasks: one user message, one reply, reward 1 for a correct final answer.

    python scripts/train_reasoning.py --train outputs/reasoning/train.jsonl --out runs/reason --max-steps 60

Tasks come from ``scripts/build_reasoning.py``. LoRA, the exact-export patch and metric logging are shared
with ``train_grpo.py``. Every scored rollout goes to ``<out>/rollouts.jsonl``; the adapter to ``<out>/adapter``.

``--shuffle-rewards`` is the control run: each step's rewards are permuted across all of its rollouts, with a
generator seeded by the run seed and the step (so a resumed run repeats the same permutations). The reward
distribution is unchanged, but it carries no information about which rollout earned it.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from train_grpo import MODEL, check_lora_export_is_exact, drop_records_after, keep_lora_base_weights_exact, metrics_log

ROLLOUT_LOG: Path | None = None
"""Where ``reward`` appends every scored rollout; set by ``main``."""
SHUFFLE_SEED: int | None = None
"""The run seed when rewards are shuffled (``--shuffle-rewards``), else None; set by ``main``."""
ROLLOUTS_PER_STEP = 0
"""Rollouts in one step; a shuffle must see all of them at once. Set by ``main``."""


def reward(completions: list[Any], answer: list[str], kind: list[str], family: list[str], **kwargs: Any) -> list[float]:
    """1.0 for a correct final answer, else 0.0, permuted in a control run.

    Logs every rollout and the per-family accuracy of the true, unpermuted rewards.
    """
    from causalagentbench.reasoning.grading import grade

    replies = [completion[-1]["content"] for completion in completions]
    rewards = [grade(k, a, r) for k, a, r in zip(kind, answer, replies, strict=True)]
    step = kwargs["trainer_state"].global_step + 1  # the step these rollouts train
    assigned = rewards
    if SHUFFLE_SEED is not None:
        if len(rewards) != ROLLOUTS_PER_STEP:
            raise RuntimeError(f"a shuffle must see the whole step: got {len(rewards)} of {ROLLOUTS_PER_STEP}")
        assigned = random.Random(SHUFFLE_SEED * 1_000_003 + step).sample(rewards, len(rewards))
    if ROLLOUT_LOG is not None:
        with ROLLOUT_LOG.open("a", encoding="utf-8") as handle:
            for i, reply in enumerate(replies):
                record = {"time": time.time(), "step": step, "reward": rewards[i], "assigned_reward": assigned[i]}
                record |= {"family": family[i], "kind": kind[i], "answer": answer[i], "reply": reply}
                handle.write(json.dumps(record) + "\n")
    log_metric = kwargs.get("log_metric")
    if log_metric is not None:
        by_family: dict[str, list[float]] = defaultdict(list)
        for f, r in zip(family, rewards, strict=True):
            by_family[f].append(r)
        for f, values in by_family.items():
            log_metric(f"accuracy/{f}", statistics.mean(values))
    return assigned


def main() -> None:
    """Parse arguments, build the trainer and train."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True, help="tasks from build_reasoning.py (JSON lines)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--tasks-per-step", type=int, default=16)
    parser.add_argument("--generations", type=int, default=8, help="replies per task (the GRPO group)")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--vllm-memory", type=float, default=0.35, help="GPU memory share for vLLM")
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--micro-batch", type=int, default=8, help="replies per backward pass (memory, not math)")
    parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in --out")
    parser.add_argument("--shuffle-rewards", action="store_true", help="control run: permute rewards each step")
    args = parser.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    keep_lora_base_weights_exact()
    check_lora_export_is_exact()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    global ROLLOUT_LOG, SHUFFLE_SEED, ROLLOUTS_PER_STEP
    ROLLOUT_LOG = out / "rollouts.jsonl"
    SHUFFLE_SEED = args.seed if args.shuffle_rewards else None
    ROLLOUTS_PER_STEP = args.tasks_per_step * args.generations
    if args.resume:
        resumed_from = max(int(path.name.split("-")[1]) for path in out.glob("checkpoint-*"))
        drop_records_after(out / "metrics.jsonl", resumed_from)
        drop_records_after(ROLLOUT_LOG, resumed_from)
    tasks = [json.loads(line) for line in Path(args.train).read_text(encoding="utf-8").splitlines()]
    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": t["prompt"]}] for t in tasks],
            "answer": [t["answer"] for t in tasks],
            "kind": [t["kind"] for t in tasks],
            "family": [t["family"] for t in tasks],
        }
    )
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
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_memory,
        vllm_enable_sleep_mode=True,
        vllm_max_model_length=args.max_completion_length + 2048,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=args.save_steps,
        report_to="none",
        log_completions=False,
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward,
        args=config,
        train_dataset=dataset,
        peft_config=LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, target_modules="all-linear"),
        callbacks=[metrics_log(out)],
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(out / "adapter"))
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

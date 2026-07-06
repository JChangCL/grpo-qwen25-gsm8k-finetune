#!/usr/bin/env python3
"""GRPO fine-tuning on the MATH dataset (harder than GSM8K -> real headroom).

Mirrors the GSM8K colocate pipeline but swaps in MATH problems, a \\boxed{}
answer format, and math-equivalence rewards (via math_verify). Run with the
same env as the GSM8K job (CPATH for triton, RANK/WORLD_SIZE for vLLM colocate).
"""
import argparse
from dataclasses import dataclass, fields
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from math_utils import MATH_SYSTEM_PROMPT, is_correct, last_boxed


def str_to_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return v.lower() in {"true", "1", "yes", "y"}


def get_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content", ""))
    return str(completion)


def gold_answer(example: dict) -> str:
    # MATH-500 has 'answer'; raw MATH has the gold inside solution's \boxed{}.
    if example.get("answer"):
        return str(example["answer"])
    return last_boxed(example.get("solution", ""))


def prepare_math(dataset_name: str, split: str, max_samples: int | None, seed: int) -> Dataset:
    ds = load_dataset(dataset_name, split=split)
    # Drop any problem that appears in the MATH-500 eval set (avoid contamination).
    try:
        eval_probs = {p.strip() for p in load_dataset("HuggingFaceH4/MATH-500", split="test")["problem"]}
        before = len(ds)
        ds = ds.filter(lambda ex: ex["problem"].strip() not in eval_probs)
        print(f"[prepare_math] removed {before - len(ds)} eval-overlap problems; {len(ds)} remain")
    except Exception as exc:  # pragma: no cover
        print(f"[prepare_math] WARNING could not filter eval overlap: {exc}")
    if max_samples:
        ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))

    def convert(ex: dict) -> dict[str, Any]:
        return {
            "prompt": [
                {"role": "system", "content": MATH_SYSTEM_PROMPT},
                {"role": "user", "content": ex["problem"]},
            ],
            "ground_truth": gold_answer(ex),
        }

    return ds.map(convert, remove_columns=ds.column_names)


def correctness_reward(completions: list[Any], ground_truth: list[str], **_: Any) -> list[float]:
    return [1.0 if is_correct(get_text(c), g) else 0.0 for c, g in zip(completions, ground_truth)]


def boxed_format_reward(completions: list[Any], **_: Any) -> list[float]:
    return [0.5 if last_boxed(get_text(c)) else 0.0 for c in completions]


@dataclass
class Cfg:
    model_name_or_path: str
    dataset_name: str
    dataset_split: str
    output_dir: str
    max_samples: int
    max_steps: int
    learning_rate: float
    beta: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_generations: int
    max_completion_length: int
    reward_weights: list[float] | None
    lora_r: int
    lora_alpha: int
    vllm_gpu_memory_utilization: float
    save_steps: int
    report_to: str
    run_name: str | None
    seed: int


def parse_args() -> Cfg:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dataset_name", default="hendrycks/competition_math")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--output_dir", default="outputs/qwen2.5-1.5b-math-grpo")
    p.add_argument("--max_samples", type=int, default=4000)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.03)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--max_completion_length", type=int, default=1024)
    p.add_argument("--reward_weights", type=float, nargs=2, default=None,
                   help="Weights for [correctness, boxed_format].")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.35)
    p.add_argument("--save_steps", type=int, default=20)
    p.add_argument("--report_to", default="wandb")
    p.add_argument("--run_name", default=None)
    p.add_argument("--seed", type=int, default=42)
    return Cfg(**vars(p.parse_args()))


def main() -> None:
    args = parse_args()
    import os
    os.environ.setdefault("WANDB_PROJECT", "grpo-gsm8k-simulation")

    train_dataset = prepare_math(args.dataset_name, args.dataset_split, args.max_samples, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    config_kwargs = dict(
        output_dir=args.output_dir, run_name=args.run_name, report_to=args.report_to, seed=args.seed,
        max_steps=args.max_steps, learning_rate=args.learning_rate, lr_scheduler_type="cosine",
        warmup_ratio=0.1, beta=args.beta, reward_weights=args.reward_weights,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations, max_completion_length=args.max_completion_length,
        bf16=True, fp16=False, gradient_checkpointing=True, logging_steps=1,
        save_steps=args.save_steps, save_total_limit=20, log_completions=True, remove_unused_columns=False,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        model_init_kwargs={"torch_dtype": torch.bfloat16, "trust_remote_code": True},
    )
    supported = {f.name for f in fields(GRPOConfig)}
    config_kwargs = {k: v for k, v in config_kwargs.items() if k in supported}
    training_args = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        reward_funcs=[correctness_reward, boxed_format_reward],
        args=training_args, train_dataset=train_dataset, processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

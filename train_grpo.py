import argparse
import os
import re
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


SYSTEM_PROMPT = """You are a careful math reasoning assistant.
Return every response in exactly this XML format:
<reasoning>
step-by-step reasoning here
</reasoning>
<answer>
final numeric answer only
</answer>"""


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def get_text(completion: Any) -> str:
    """TRL can pass either plain strings or chat-message completions."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return str(completion)


def extract_gsm8k_answer(answer: str) -> str:
    if "####" in answer:
        answer = answer.split("####")[-1]
    return normalize_number(answer)


def extract_answer_tag(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return normalize_number(match.group(1))
    return normalize_number(text)


def normalize_number(text: str) -> str:
    text = text.replace(",", "").strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return ""
    value = numbers[-1]
    if value.endswith(".0"):
        value = value[:-2]
    return value


def build_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nQuestion:\n{question}\n\nResponse:"


def prepare_gsm8k(split: str, max_samples: int | None, seed: int) -> Dataset:
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if max_samples:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))

    def convert(example: dict[str, str]) -> dict[str, str]:
        return {
            "prompt": build_prompt(example["question"]),
            "ground_truth": extract_gsm8k_answer(example["answer"]),
            "question": example["question"],
        }

    return dataset.map(convert, remove_columns=dataset.column_names)


def soft_format_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    pattern = re.compile(
        r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    rewards = []
    for completion in completions:
        text = get_text(completion)
        rewards.append(0.5 if pattern.search(text) else 0.0)
    return rewards


def strict_format_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    pattern = re.compile(
        r"^\s*<reasoning>\s*.+?\s*</reasoning>\s*<answer>\s*.+?\s*</answer>\s*$",
        flags=re.DOTALL | re.IGNORECASE,
    )
    return [0.5 if pattern.match(get_text(completion)) else 0.0 for completion in completions]


def numeric_answer_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    rewards = []
    for completion in completions:
        answer = extract_answer_tag(get_text(completion))
        rewards.append(0.25 if answer else 0.0)
    return rewards


def correctness_reward(completions: list[Any], ground_truth: list[str], **kwargs: Any) -> list[float]:
    rewards = []
    for completion, expected in zip(completions, ground_truth):
        predicted = extract_answer_tag(get_text(completion))
        rewards.append(1.0 if predicted and predicted == expected else 0.0)
    return rewards


@dataclass
class TrainConfig:
    model_name_or_path: str
    output_dir: str
    dataset_split: str
    max_samples: int | None
    max_steps: int
    num_train_epochs: float
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_generations: int
    max_prompt_length: int
    max_completion_length: int
    beta: float
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    report_to: str
    run_name: str | None
    seed: int
    bf16: bool
    fp16: bool
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    use_vllm: bool
    vllm_server_host: str
    vllm_server_port: int
    deepspeed: str | None
    push_to_hub: bool
    hub_model_id: str | None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="GRPO fine-tuning on GSM8K, inspired by AMD ROCm GRPO reference.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", default="outputs/qwen2.5-1.5b-gsm8k-grpo")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=192)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--report_to", default="wandb")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", type=str_to_bool, default=False)
    parser.add_argument("--fp16", type=str_to_bool, default=True)
    parser.add_argument("--use_lora", type=str_to_bool, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_vllm", type=str_to_bool, default=False)
    parser.add_argument("--vllm_server_host", default="0.0.0.0")
    parser.add_argument("--vllm_server_port", type=int, default=8000)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--push_to_hub", type=str_to_bool, default=False)
    parser.add_argument("--hub_model_id", default=None)
    return TrainConfig(**vars(parser.parse_args()))


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_PROJECT", "grpo-gsm8k-simulation")

    train_dataset = prepare_gsm8k(args.dataset_split, args.max_samples, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        report_to=args.report_to if args.report_to.lower() != "none" else None,
        seed=args.seed,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        log_completions=True,
        remove_unused_columns=False,
        use_vllm=args.use_vllm,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        deepspeed=args.deepspeed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        model_init_kwargs={
            "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
            "trust_remote_code": True,
        },
    )

    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        reward_funcs=[
            correctness_reward,
            soft_format_reward,
            strict_format_reward,
            numeric_answer_reward,
        ],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()

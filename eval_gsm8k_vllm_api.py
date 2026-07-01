import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from tqdm import tqdm

from train_grpo import SYSTEM_PROMPT, extract_answer_tag, extract_gsm8k_answer


def build_eval_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nQuestion:\n{question}\n\nResponse:"


def completion_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    response = requests.post(
        f"{base_url.rstrip('/')}/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload["choices"][0].get("text", ""))


def chat_request(
    base_url: str,
    model: str,
    question: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question:\n{question}\n\nResponse:"},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload["choices"][0]["message"].get("content", ""))


def generate_with_retry(
    args: argparse.Namespace,
    question: str,
    prompt: str,
) -> str:
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            if args.endpoint == "chat":
                return chat_request(
                    args.base_url,
                    args.model,
                    question,
                    args.max_tokens,
                    args.temperature,
                    args.timeout,
                )
            return completion_request(
                args.base_url,
                args.model,
                prompt,
                args.max_tokens,
                args.temperature,
                args.timeout,
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt < args.retries:
                time.sleep(args.retry_sleep)
    raise RuntimeError(f"vLLM request failed after {args.retries + 1} attempts: {last_error}") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GSM8K exact-match evaluation through a vLLM OpenAI-compatible API.")
    parser.add_argument("--base_url", default="http://10.180.72.205:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry_sleep", type=float, default=3.0)
    parser.add_argument(
        "--endpoint",
        choices=["completions", "chat"],
        default="completions",
        help="Use completions to match eval_gsm8k.py raw prompts, or chat to use the model chat template.",
    )
    parser.add_argument("--output_jsonl", default=None, help="Optional path to save per-example predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    output_path = Path(args.output_jsonl) if args.output_jsonl else None
    output_file = output_path.open("w", encoding="utf-8") if output_path else None

    correct = 0
    try:
        for index, example in enumerate(tqdm(dataset)):
            question = example["question"]
            prompt = build_eval_prompt(question)
            generated = generate_with_retry(args, question, prompt)
            pred = extract_answer_tag(generated)
            gold = extract_gsm8k_answer(example["answer"])
            is_correct = pred == gold
            correct += int(is_correct)

            if output_file:
                record: dict[str, Any] = {
                    "index": index,
                    "question": question,
                    "gold": gold,
                    "prediction": pred,
                    "correct": is_correct,
                    "generated": generated,
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
    finally:
        if output_file:
            output_file.close()

    total = len(dataset)
    print(f"exact_match={correct / total:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()

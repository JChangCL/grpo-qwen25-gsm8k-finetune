#!/usr/bin/env python3
"""Offline vLLM GSM8K exact-match eval using the chat template (greedy).

Loads a model directly with vLLM (no server), applies the chat template, and
computes exact match on the GSM8K test split. Same protocol family as the
GB10B vLLM chat eval, runnable in one process on the H100.

    python scripts/eval_gsm8k_vllm_offline.py --model <path-or-hf-id> --max_samples 500
"""
import argparse
import re

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from train_grpo import SYSTEM_PROMPT, extract_answer_tag, extract_gsm8k_answer, normalize_number


def extract_strict(text: str) -> str:
    """Strict extraction: only accept a number inside <answer>...</answer>.

    No last-number fallback, so a model that does not emit the required format
    scores 0 on that example. This is the headroom-having protocol that reveals
    whether GRPO actually taught the reasoning+answer format (AMD-style)."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    return normalize_number(m.group(1)) if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_samples", type=int, default=500)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--strict", action="store_true", help="Only accept answers inside <answer></answer> (no fallback).")
    ap.add_argument("--plain", action="store_true", help="AMD-style: no XML reasoning system prompt, just the question.")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    extract = extract_strict if args.strict else extract_answer_tag

    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def messages(question: str) -> list[dict[str, str]]:
        if args.plain:  # no reasoning scaffold (AMD-style base condition)
            return [{"role": "user", "content": f"Question:\n{question}"}]
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question:\n{question}\n\nResponse:"},
        ]

    prompts = [
        tok.apply_chat_template(messages(ex["question"]), tokenize=False, add_generation_prompt=True)
        for ex in ds
    ]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        enforce_eager=True,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))

    correct = 0
    for ex, out in zip(ds, outs):
        pred = extract(out.outputs[0].text)
        gold = extract_gsm8k_answer(ex["answer"])
        correct += int(pred == gold)
    total = len(ds)
    print(f"RESULT tag={args.tag} model={args.model} exact_match={correct/total:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline vLLM eval on the MATH-500 benchmark (chat template, greedy).

    PYTHONPATH=math_task python math_task/eval_math.py --model <path-or-id> --tag base
"""
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from math_utils import MATH_SYSTEM_PROMPT, is_correct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_samples", type=int, default=500)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [
        tok.apply_chat_template(
            [
                {"role": "system", "content": MATH_SYSTEM_PROMPT},
                {"role": "user", "content": ex["problem"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ex in ds
    ]

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        enforce_eager=True,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))

    correct = sum(is_correct(o.outputs[0].text, ex["answer"]) for ex, o in zip(ds, outs))
    total = len(ds)
    print(f"RESULT tag={args.tag} model={args.model} MATH exact_match={correct/total:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()

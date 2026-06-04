import argparse
import os

import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_grpo import SYSTEM_PROMPT, extract_gsm8k_answer, extract_answer_tag


def build_eval_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nQuestion:\n{question}\n\nResponse:"


def main() -> None:
    parser = argparse.ArgumentParser(description="Small exact-match evaluation for GSM8K.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path
    adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
    is_peft_adapter = os.path.exists(adapter_config_path)

    if is_peft_adapter:
        peft_config = PeftConfig.from_pretrained(model_name_or_path)
        tokenizer_source = peft_config.base_model_name_or_path
    else:
        peft_config = None
        tokenizer_source = model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path if is_peft_adapter else model_name_or_path,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(args.device)
    if is_peft_adapter:
        model = PeftModel.from_pretrained(model, model_name_or_path).to(args.device)
    model.eval()

    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    correct = 0
    for example in tqdm(dataset):
        prompt = build_eval_prompt(example["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        pred = extract_answer_tag(generated)
        gold = extract_gsm8k_answer(example["answer"])
        correct += int(pred == gold)

    total = len(dataset)
    print(f"exact_match={correct / total:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()

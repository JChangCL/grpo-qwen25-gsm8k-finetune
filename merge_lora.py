import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA adapter into its base causal LM.")
    parser.add_argument("--adapter_path", required=True, help="Path to a PEFT/LoRA adapter directory.")
    parser.add_argument("--output_dir", required=True, help="Directory for the merged full model.")
    parser.add_argument(
        "--base_model_name_or_path",
        default=None,
        help="Optional base model override. Defaults to adapter_config.json base_model_name_or_path.",
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    parser.add_argument("--device_map", default="auto", help='Use "auto" by default, or "cpu" for CPU-only merge.')
    parser.add_argument("--max_shard_size", default="5GB")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter_path)
    output_dir = Path(args.output_dir)

    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"No adapter_config.json found in {adapter_path}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    base_model = args.base_model_name_or_path or peft_config.base_model_name_or_path
    dtype = DTYPES[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path), torch_dtype=dtype)
    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(str(output_dir))
    print(f"Merged model saved to {output_dir}")


if __name__ == "__main__":
    main()

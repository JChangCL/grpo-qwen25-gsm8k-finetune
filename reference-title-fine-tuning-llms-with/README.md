# GRPO Fine-Tuning Simulation on Colab

這個 repo 是仿照 AMD ROCm Blog「Fine-Tuning LLMs with GRPO on AMD MI300X」做的 Colab/GitHub/W&B 版本。

Reference:

- Title: Fine-Tuning LLMs with GRPO on AMD MI300X
- Source: ROCm Blogs / AMD
- Date: June 18, 2025
- URL: https://rocm.blogs.amd.com/software-tools-optimization/llm-grpo-rocm/README.html

AMD 原文使用 `Qwen/Qwen2.5-1.5B-Instruct`、GSM8K、TRL GRPO、vLLM、Accelerate/DeepSpeed，在 8 張 MI300X 上把 1 張 GPU 留給 vLLM generation，其餘 7 張做 GRPO training。

這份 code 預設使用與 AMD reference 相同的 1.5B 模型，並保留較適合 Colab 模擬的 LoRA 小步數設定：

- `Qwen/Qwen2.5-1.5B-Instruct`
- GSM8K
- LoRA
- 50 training steps
- W&B logging
- 不預設啟用 vLLM

## Files

- `train_grpo.py`: GRPO training script
- `eval_gsm8k.py`: 簡單 GSM8K exact-match 評估
- `requirements-colab.txt`: Colab 安裝套件
- `configs/deepspeed_zero2.json`: 多 GPU 或較大模型可用的 DeepSpeed ZeRO-2 config
- `configs/accelerate_config.yaml`: 單 GPU Colab accelerate config
- `notebooks/colab_grpo_gsm8k.ipynb`: Colab 起手 notebook

## Colab Quick Start

在 Colab 選 GPU runtime，然後：

```bash
!git clone https://github.com/YOUR_NAME/YOUR_REPO.git
%cd YOUR_REPO
!pip install -r requirements-colab.txt
```

登入 Hugging Face 和 W&B：

```python
from huggingface_hub import notebook_login
notebook_login()

import wandb
wandb.login()
```

跑 simulation：

```bash
!accelerate launch --config_file configs/accelerate_config.yaml train_grpo.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir outputs/qwen2.5-1.5b-gsm8k-grpo \
  --max_samples 256 \
  --max_steps 50 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_generations 4 \
  --fp16 true \
  --bf16 false \
  --use_lora true \
  --report_to wandb \
  --run_name colab-qwen25-15b-gsm8k-grpo
```

評估：

```bash
!python eval_gsm8k.py \
  --model_name_or_path outputs/qwen2.5-1.5b-gsm8k-grpo \
  --max_samples 100
```

## Closer to the AMD Reference

如果你想跑更接近 AMD 文章的 200-step 設定，可以使用同一個 1.5B 模型放大 steps：

```bash
!accelerate launch train_grpo.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir outputs/qwen2.5-1.5b-gsm8k-grpo-200step \
  --max_samples 2000 \
  --max_steps 200 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_generations 4 \
  --fp16 true \
  --use_lora true \
  --report_to wandb \
  --run_name qwen25-15b-gsm8k-grpo-200step
```

如果環境支援 vLLM，可以另外啟動 vLLM server，並在訓練加上：

```bash
--use_vllm true --vllm_server_host 0.0.0.0 --vllm_server_port 8000
```

## Reward Functions

訓練中會記錄以下 reward：

- `correctness_reward`: `<answer>` 中的數字是否等於 GSM8K ground truth
- `soft_format_reward`: 是否包含 `<reasoning>...</reasoning><answer>...</answer>`
- `strict_format_reward`: 是否完整符合 XML 格式
- `numeric_answer_reward`: `<answer>` 是否有數字

這些 reward 會被 TRL 加總，W&B 上可以看到每條 reward curve。

## Notes

Colab 免費 GPU 跑 GRPO 仍然可能會 OOM。建議先用預設 1.5B LoRA、50 steps 確認 pipeline 可以跑，再逐步增加：

- `max_steps`
- `max_samples`
- `num_generations`
- model size

GRPO 的 global batch size 必須能被 `num_generations` 整除；如果看到相關錯誤，先把 `num_generations` 改成 `2` 或 `4`。

# GRPO Fine-tuning Experiment Report

> Goal: reproduce an AMD-style GRPO fine-tuning workflow with available resources, using Qwen2.5-1.5B-Instruct on GSM8K.  
> Current platform: UTD Juno HPC H100 + W&B + GB10B vLLM evaluation.  
> Current best checkpoint: `outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-150step`  
> Current best same-protocol vLLM result: `68.0%` exact match on 500 GSM8K samples.

[TOC]

---

## 1. Executive Summary

這次實驗已經達成核心目標：**成功用 GRPO 訓練出一個比 base Qwen2.5-1.5B-Instruct 更好的 GSM8K 模型**。

目前結果需要分成兩種 evaluation protocol 來看。

第一組是 Juno 上的 local Hugging Face evaluation：

```text
Base model: 45.4% (227/500)
GRPO model: 54.8% (274/500)
Gain: +9.4 percentage points, +47 correct answers
```

第二組是 GB10B vLLM chat evaluation。這是目前最強、也最適合展示的結果，因為 base model 和 GRPO model 在同一個 vLLM chat protocol 下比較：

```text
Base model + GB10 vLLM chat eval: 54.6% (273/500)
GRPO-trained merged model + GB10 vLLM chat eval: 68.0% (340/500)
Gain: +13.4 percentage points, +67 correct answers
```

重要的是，**vLLM 本身沒有訓練模型，也沒有改變模型權重**。vLLM 在這裡的作用是提供更快且一致的 inference/evaluation backend。真正造成模型能力提升的是前面在 Juno H100 上完成的 GRPO + LoRA fine-tuning。

因此最嚴謹的結論是：

- **Training happened on Juno H100** using TRL GRPO + LoRA.
- **GB10B vLLM was used for inference/evaluation**, not for training.
- **The GRPO gain should be measured within the same evaluation protocol**.
- Under local HF eval, GRPO improved from `45.4%` to `54.8%`.
- Under GB10B vLLM chat eval, GRPO improved from `54.6%` to `68.0%`.
- Do not claim that vLLM alone made the model better; vLLM changed the serving/evaluation backend, while GRPO changed the model.

---

## 2. Project Setup

### Model

```text
Qwen/Qwen2.5-1.5B-Instruct
```

### Dataset

```text
openai/gsm8k main
```

### Training Framework

```text
TRL GRPOTrainer
PEFT LoRA
Accelerate
W&B logging
```

### Main Training Script

```bash
train_grpo.py
```

### Main Eval Script

```bash
eval_gsm8k.py
```

### Main vLLM API Eval Script

```bash
eval_gsm8k_vllm_api.py
```

### UTD Juno Working Directory

```bash
/work/dal717586/grpo-qwen25-gsm8k-finetune
```

---

## 3. How Close Is This to AMD Reference?

這份實作是 **inspired by AMD ROCm GRPO reference**，但不是硬體與執行環境上的完全複製。

### Similarities

- 使用 GRPO fine-tuning。
- 使用 Qwen 1.5B 等級模型。
- 使用 GSM8K math reasoning task。
- 使用 reward functions 來鼓勵 correctness 與 answer format。
- 使用 W&B 觀察 `reward`, `kl`, `grad_norm`, `learning_rate`, `completion_length`。

### Differences

| Item | AMD Reference | Current Experiment |
|---|---|---|
| Hardware | AMD MI300X | UTD Juno H100 / Colab GPU |
| Backend | ROCm | CUDA |
| Inference acceleration | vLLM-assisted rollout generation | GB10B vLLM works for inference/evaluation, but not yet as TRL rollout server |
| Runtime stability | controlled benchmark environment | shared HPC queue + Colab instability |
| Training length | AMD reported smoother 200-step result | many short/partial runs due queue/time |
| Goal | official demo/blog result | resource-constrained reproduction |

因此目前成果適合說成：

```text
I reproduced the core GRPO fine-tuning workflow under my available H100 and GB10 resources.
Training was completed on Juno H100, and the merged GRPO model was served through GB10B vLLM for faster evaluation.
```

不建議說成：

```text
I exactly reproduced AMD's MI300X experiment.
```

目前與 AMD reference 最大的差距是：GB10B 的一般 vLLM OpenAI server 可以做 inference/evaluation，但 TRL GRPO training 的 `use_vllm=True` 需要 TRL-specific vLLM endpoints，例如 `/get_tensor_parallel_size/` 與 rollout/weight-sync endpoints。目前 GB10B server 回傳 `/get_tensor_parallel_size/ = 404`，所以它尚不能直接作為 TRL GRPO 的 rollout server。

---

## 4. Major Problems Encountered and Fixes

### 4.1 Colab Dependency Instability

遇到的問題：

- `torchao` version incompatible。
- `transformers` / `trl` / `accelerate` 版本不匹配。
- `GRPOTrainer._get_train_sampler()` error。
- Colab runtime 會斷線。
- `pip --force-reinstall` 會破壞 Colab 內建 torch / CUDA / pandas stack。

處理方式：

- pin dependencies:

```text
transformers==4.51.3
datasets==3.5.0
accelerate==1.6.0
trl==0.16.1
peft==0.15.2
bitsandbytes==0.45.5
wandb==0.19.11
protobuf<6
```

- Colab 不再主力訓練，只做 quick smoke test。

### 4.2 GRPO Batch / Generation Constraint

遇到的問題：

```text
global train batch size must be evenly divisible by num_generations
```

原因：

```text
GRPO 的 batch size 必須能被 num_generations 整除。
```

解法：

```bash
--per_device_train_batch_size 4
--num_generations 4
```

或：

```bash
--per_device_train_batch_size 2
--num_generations 2
```

### 4.3 Juno CUDA / PyTorch Version

早期 eval 有 CUDA mismatch warning，表示可能跑成 CPU 或 GPU 沒正確吃到。

解法是新增 Juno setup script：

```bash
scripts/setup_juno_env.sh
```

目標環境：

```text
python/3.11.11
cuda/12.4
torch==2.6.0+cu124
```

正常 log 應該看到：

```text
torch cuda: 12.4
cuda available: True
```

### 4.4 W&B API Key Did Not Enter Slurm Job

錯誤：

```text
wandb: ERROR api_key not configured (no-tty)
```

原因：

```text
Slurm batch job 沒有吃到 WANDB_API_KEY。
```

解法：

```bash
export WANDB_API_KEY='your_key'
echo ${#WANDB_API_KEY}
sbatch --export=ALL scripts/xxx.slurm
```

如果使用新版 `wandb_v1_...` key，長度通常是：

```text
86
```

---

## 5. Experiment Results So Far

### 5.1 Known Eval Results

| Model / Run | Eval Samples | exact_match | Notes |
|---|---:|---:|---|
| Early Colab run | 100 | `46/100` | early baseline-like result |
| Early H100 8gen checkpoint | 100 | `49/100` | timed out around checkpoint-25 |
| 4gen safe / format variants | 100 | `50/100` | no clear gain |
| 4gen complete 100step | 100 | `51/100` | complete but modest |
| 4gen strong reward checkpoint | 100 | `54/100` | previous best |
| r32 strong 150step | 100 | **`61/100`** | best 100-sample result |
| r32 strong checkpoint-140 | 500 | `271/500 = 54.2%` | close to final, but slightly lower |
| r32 strong 150step | 500 | **`274/500 = 54.8%`** | best stable result so far |
| Base model + GB10 vLLM chat | 500 | `273/500 = 54.6%` | same vLLM chat protocol baseline |
| r32 strong 150step merged + GB10 vLLM chat | 500 | **`340/500 = 68.0%`** | best same-protocol result |

### 5.2 Current Best Model

```bash
outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-150step
```

100-sample eval:

```text
exact_match = 0.6100 (61/100)
```

500-sample eval:

```text
exact_match = 0.5480 (274/500)
```

GB10B vLLM chat eval after merging LoRA into a full model:

```text
exact_match = 0.6800 (340/500)
```

The merged model served on GB10B:

```bash
outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-150step-merged
```

### 5.3 vLLM Evaluation and Attribution

GB10B vLLM was used as an inference/evaluation backend. The evaluation script ran from Juno, sent GSM8K prompts to the GB10B vLLM server, received generated answers, extracted the final answer, and computed exact match.

The pipeline was:

```text
Juno eval script
-> GB10B vLLM server
-> model generates answer on GB10B GPU
-> Juno receives generated output
-> Juno extracts final answer
-> Juno compares prediction with GSM8K gold answer
```

This is not training. Training requires gradient updates, reward computation inside the GRPO loop, LoRA parameter updates, and repeated model weight synchronization. The current GB10B setup provided fast inference/evaluation but did not participate in the training loop.

The correct same-protocol comparison is:

```text
Base model + GB10B vLLM chat eval: 54.6% (273/500)
GRPO model + GB10B vLLM chat eval: 68.0% (340/500)
Difference: +13.4 points, +67 correct answers
```

The improvement should be attributed to GRPO fine-tuning, not to vLLM alone. vLLM changes the serving backend and evaluation protocol; it does not change model weights.

### 5.4 Paired Error Analysis

Using the same 500 GSM8K examples under the vLLM chat evaluation protocol:

```text
Base wrong -> GRPO correct: 99 examples
Base correct -> GRPO wrong: 32 examples
Net gain: +67 correct examples
```

This supports that the improvement is not only a single aggregate number. The GRPO model fixed substantially more base-model failures than it introduced regressions.

### 5.5 Pending / Recent Jobs

| Job ID | Purpose |
|---:|---|
| `211825` | 500-sample eval of best model, result `54.8%` |
| `211830` | base model 500-sample baseline eval, result `45.4%` |
| `211872` | r32 strong 200step 500-sample eval, result `53.2%` |
| `211896` | r16 strong completion-192 500-sample eval, result `53.8%` |
| `211955` | r32 strong checkpoint-140 500-sample eval, result `54.2%` |

---

## 6. Interpretation of W&B Curves

### 6.1 Reward

Reward 有上升或高峰是好事，但 GRPO reward 本身很 noisy。

重點不是單一 peak，而是：

```text
reward trend + eval exact_match
```

如果 reward 高但 eval 沒變好，代表可能只是 reward hacking 或 sample noise。

### 6.2 Grad Norm

`train/grad_norm` 沒有固定越低越好。

健康狀態：

- 大多在 `0.2 - 1.0`。
- 偶爾 spike 到 `1.5 - 2.0` 可以接受。
- spike 後能回落，通常還可以。

危險狀態：

- 長期接近 0：可能沒在學。
- 持續升高到 `3 - 5+`：可能不穩。
- 與 `kl` / `loss` 同時爆：風險高。

目前 r32 strong 的 grad norm 比較震盪，但因為 100-sample eval 到 `61/100`，代表震盪有換到一定效果。不過 500-sample 掉到 `54.8%`，表示它可能還不夠穩。

### 6.3 KL

KL 代表模型偏離 base model 的程度。

判斷：

- 太低：可能太保守。
- 緩慢上升：正常。
- 突然大爆：可能訓練太激進。

更穩版本可以提高 `beta`，讓模型不要偏離 base model 太快。

---

## 7. Main Conclusion

### Did the training succeed?

是。以目前結果來看，GRPO 訓練已經成功跑通並產生有效模型。

理由：

- 訓練流程可以完整跑完。
- W&B 能正確記錄。
- LoRA checkpoint 可以 eval。
- 100-sample 從早期 `46%-54%` 推到 `61%`。
- 500-sample local HF eval 仍有 `54.8%`，高於 base model 的 `45.4%`。
- GB10B vLLM chat eval 中，GRPO model 達到 `68.0%`，高於同 protocol base model 的 `54.6%`。

### Is the improvement stable enough?

已經有 500-sample evidence 支持有效提升，但 full GSM8K test set 仍然是下一個更強驗證。

原因：

- `100` 題 eval 波動太大，所以不能只用 `61/100` 當最終結論。
- Local HF 500-sample eval 顯示 `45.4% -> 54.8%`。
- GB10B vLLM chat 500-sample eval 顯示 `54.6% -> 68.0%`。
- 絕對分數會受到 prompt format、chat template、max tokens、answer extraction、endpoint 設定影響。

真正穩定的歸因方式是：

```text
compare base model vs GRPO model under the same evaluation protocol
```

在目前兩套 protocol 中，GRPO model 都高於 base model。

---

## 8. Why 8 Generations Was Hard

8gen 理論上更接近許多 GRPO 設定，也能提供更多 group comparison。

但在目前資源下：

- 單 H100 不使用 vLLM 時，8gen 很慢。
- 早期 8gen 200step job 在約 37/200 step 時 timeout。
- completion length 越長，step time 越高。
- Juno queue/time limit 讓長 job 風險更大。

所以目前比較實際的折衷是：

```text
num_generations = 4
LoRA r = 32
max_completion_length = 128
max_steps = 150-200
```

---

## 9. Recommended Improvement Direction

### 9.1 Evaluation First

之後每個候選模型至少跑：

```text
500 samples
```

不要再只用 `100` samples 判斷最佳模型。

必要時最後跑完整 GSM8K test：

```text
1319 samples
```

### 9.2 Stable Training Variant

目前 best 的主要問題是：100 題很高，但 500 題回落。  
下一個穩定版建議：

```text
num_generations = 4
lora_r = 32
lora_alpha = 64
max_steps = 200
learning_rate = 1e-6
beta = 0.06
max_completion_length = 128
reward_weights = 2.0 0.8 1.2 0.25
```

對比目前 best：

| Parameter | Current Best | Stable Proposal |
|---|---:|---:|
| `learning_rate` | `2e-6` | `1e-6` |
| `beta` | `0.04` | `0.06` |
| `strict_format_weight` | `2.0` | `1.2` |
| `numeric_weight` | `0.5` | `0.25` |
| `lora_r` | `32` | `32` |
| `num_generations` | `4` | `4` |

目的：

- 降低更新幅度。
- 讓 KL 更受控。
- 避免 reward 權重過度推格式而犧牲 correctness。
- 保留 r32 的容量優勢。

### 9.3 Compare Intermediate Checkpoints

不一定最後一步最好。建議 eval：

```bash
checkpoint-100
checkpoint-120
checkpoint-140
checkpoint-150
```

例如：

```bash
sbatch --export=ALL,MODEL_PATH=outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-150step/checkpoint-120,MAX_SAMPLES=500 scripts/utd_eval_checkpoint.slurm
```

目前已完成的 checkpoint comparison：

```text
checkpoint-140 = 54.2% (271/500)
checkpoint-150 / final = 54.8% (274/500)
```

判讀：

```text
checkpoint-140 很接近 final，但 final/checkpoint-150 仍多答對 3 題。
目前最佳區間大約在 140-150 steps，而不是繼續拉到 200 steps。
```

### 9.4 Try Lower LR r32

如果 `r32 200step` 震盪太大，可以試：

```text
r32
learning_rate = 1e-6
beta = 0.06
max_steps = 200
```

這會犧牲一點衝高速度，但可能讓 500-sample 結果更穩。

### 9.5 Keep Completion Length Controlled

`max_completion_length=192` 可能讓模型有更多 reasoning 空間，但也會：

- 變慢。
- 增加不穩定 generation。
- 增加 format 出錯機率。

目前建議主線仍用：

```text
max_completion_length = 128
```

除非 `211833` 的 result 明顯更好。

---

## 10. Recommended Next Actions

### Immediate

1. 等 `211830` base model 500-sample eval。
2. 等 `211832` r32 200step。
3. 等 `211833` r16 completion-192。
4. 用 `grep` 整理所有 exact_match。

```bash
grep -R "exact_match" logs/eval-grpo-*.out logs/grpo-*.out
```

### If `211832` Beats 54.8% on 500 Samples

把它列為新 best，接著跑完整 1319 題。

### If `211832` Does Not Beat 54.8%

保留 `r32 strong 150step` 為目前 best，然後試穩定版：

```text
r32 stable 200step
lr 1e-6
beta 0.06
reward_weights 2.0 0.8 1.2 0.25
```

### If Base Model Baseline Is Close to 54.8%

代表目前 GRPO 還沒有穩定超越 base model，需要：

- 更精細 reward。
- 更穩 LR/beta。
- 更長但更保守訓練。
- eval intermediate checkpoints。

### If Base Model Baseline Is Much Lower Than 54.8%

代表 GRPO 有明確提升，可以進入 report/presentation 階段。

---

## 11. Suggested Final Reporting Language

可以這樣寫：

```text
I implemented a GRPO fine-tuning pipeline inspired by the AMD ROCm GRPO reference, using Qwen2.5-1.5B-Instruct on GSM8K with Hugging Face TRL and LoRA. Training was performed on UTD Juno H100, while GB10B vLLM was used as a fast inference/evaluation backend after merging the LoRA adapter into a full model. Under the same GB10B vLLM chat evaluation protocol on 500 GSM8K examples, the base model achieved 54.6% exact match, while the GRPO-tuned model achieved 68.0%, a +13.4 point improvement. vLLM was not responsible for training the model; it provided the serving backend used to evaluate both models consistently.
```

中文版本：

```text
本實驗參考 AMD ROCm GRPO fine-tuning workflow，使用 Qwen2.5-1.5B-Instruct 在 GSM8K 上進行 TRL GRPO + LoRA 訓練。訓練在 UTD Juno H100 上完成，LoRA 合併後的模型再放到 GB10B 透過 vLLM 進行快速推論與評估。在相同 GB10B vLLM chat evaluation protocol 下，base model 在 500 題 GSM8K 上達到 54.6% exact match，而 GRPO-tuned model 達到 68.0%，提升 +13.4 percentage points。vLLM 本身沒有訓練模型，它提供的是一致且較快的 inference/evaluation backend；模型能力提升來自 GRPO fine-tuning。
```

---

## 12. Current Best Commands

查所有 eval：

```bash
grep -R "exact_match" logs/eval-grpo-*.out logs/grpo-*.out
```

查 job 狀態：

```bash
squeue -u $USER
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed
```

評估目前 best，500 題：

```bash
sbatch --time=03:00:00 --export=ALL,MODEL_PATH=outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-150step,MAX_SAMPLES=500 scripts/utd_eval_checkpoint.slurm
```

評估 base model baseline，500 題：

```bash
sbatch --time=03:00:00 --export=ALL,MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct,MAX_SAMPLES=500 scripts/utd_eval_checkpoint.slurm
```

GB10B vLLM API eval，500 題：

```bash
python eval_gsm8k_vllm_api.py \
  --base_url http://10.180.72.205:8000/v1 \
  --model qwen25-15b-grpo-4gen-r32-strong-150 \
  --max_samples 500 \
  --endpoint chat \
  --output_jsonl logs/vllm-grpo-4gen-r32-strong-150-chat-500.jsonl
```

---

## 13. Bottom Line

目前最重要的結論：

```text
The GRPO pipeline works.
The model was successfully trained.
The best 100-sample result is 61%.
The local HF 500-sample result improved from 45.4% to 54.8%.
The GB10B vLLM chat 500-sample result improved from 54.6% to 68.0%.
The best defensible claim is a same-protocol GRPO gain of +13.4 points under GB10B vLLM chat evaluation.
```

下一步最重要的方向：

```text
Stop chasing only 100-sample peaks.
Use 500-sample eval as the main metric.
Try lower LR + higher beta for more stable GRPO.
Evaluate intermediate checkpoints.
Compare against base model baseline.
```

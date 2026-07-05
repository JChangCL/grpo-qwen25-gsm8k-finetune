# GRPO Fine-tuning Experiment Timeline and Results

## 1. Objective

The goal of this project was to reproduce an AMD-inspired GRPO fine-tuning workflow using the resources available to me.

The AMD reference used:

- Qwen2.5-1.5B-Instruct
- GSM8K
- Hugging Face TRL GRPO
- W&B tracking
- vLLM-assisted rollout generation
- Multi-GPU AMD MI300X environment

My implementation keeps the same core training idea, model family, dataset, reward-based GRPO method, and W&B metric tracking, but adapts the system to Colab, UTD Juno H100, and GB10B vLLM for inference/evaluation.

This is therefore an AMD-inspired GRPO reproduction, not a hardware-identical MI300X reproduction.

---

## 2. Final Current Status

The GRPO pipeline successfully trained a model that outperforms the base model on controlled 500-sample GSM8K evaluations.

| Model | Eval Samples | Exact Match |
|---|---:|---:|
| Base Qwen2.5-1.5B-Instruct, local HF eval | 500 | 45.4% (227/500) |
| Best GRPO model, local HF eval | 500 | 54.8% (274/500) |
| Local HF improvement | 500 | +9.4 points (+47 correct) |
| Base Qwen2.5-1.5B-Instruct, GB10B vLLM chat eval | 500 | 54.6% (273/500) |
| Best GRPO merged model, GB10B vLLM chat eval | 500 | 68.0% (340/500) |
| GB10B vLLM chat improvement | 500 | +13.4 points (+67 correct) |

Current best model:

```bash
outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-150step
```

Main conclusion:

```text
GRPO fine-tuning produced a measurable improvement over the base model.
The strongest same-protocol result is the GB10B vLLM chat evaluation:
54.6% for the base model versus 68.0% for the GRPO-tuned merged model.
```

Important attribution:

```text
vLLM did not train the model and did not change model weights.
Training happened on Juno H100 with TRL GRPO + LoRA.
GB10B vLLM was used as a faster inference/evaluation backend.
```

---

## 3. Core Implementation

### Model

```text
Qwen/Qwen2.5-1.5B-Instruct
```

### Dataset

```text
openai/gsm8k main
```

### Framework

```text
Hugging Face TRL GRPOTrainer
PEFT LoRA
Transformers
Accelerate
W&B
vLLM API evaluation on GB10B
```

### Reward Functions

The training used four custom reward functions:

| Reward Function | Purpose |
|---|---|
| `correctness_reward` | Checks whether final numeric answer matches GSM8K label |
| `soft_format_reward` | Rewards XML-like reasoning/answer format |
| `strict_format_reward` | Rewards exact XML structure |
| `numeric_answer_reward` | Rewards producing a parseable numeric answer |

The best-performing model used:

```bash
--reward_weights 2.0 1.0 2.0 0.5
```

Meaning:

```text
correctness = 2.0
soft format = 1.0
strict format = 2.0
numeric answer = 0.5
```

---

## 4. Evaluation Method

Two evaluation protocols were used.

The earlier controlled local evaluation used:

```text
Dataset: GSM8K test split
Metric: exact_match
Decoding: temperature = 0.0
Sampling: do_sample = False
Prompt format: same system prompt as training
Evaluation script: eval_gsm8k.py
```

The later GB10B evaluation used:

```text
Dataset: GSM8K test split
Metric: exact_match
Endpoint: OpenAI-compatible vLLM chat API
Evaluation script: eval_gsm8k_vllm_api.py
Server: GB10B at http://10.180.72.205:8000/v1
```

The most important rule is that model comparisons must be made within the same protocol. Therefore:

```text
45.4% -> 54.8% is the local HF evaluation comparison.
54.6% -> 68.0% is the GB10B vLLM chat evaluation comparison.
```

The absolute numbers are sensitive to prompt format, chat template, max token length, endpoint type, and answer extraction. The defensible claim is the same-protocol improvement.

---

## 5. Experiment Timeline

### Stage 1: Initial Colab Setup

Initial training was attempted on Google Colab.

Main issues:

- Runtime disconnections
- Dependency conflicts between `torch`, `trl`, `peft`, `transformers`, and `torchao`
- Colab package upgrades breaking CUDA/PyTorch compatibility
- Limited runtime stability for longer GRPO runs

Early result:

| Run | Eval Samples | Exact Match |
|---|---:|---:|
| Early Colab GRPO | 100 | 46.0% (46/100) |

Interpretation:

```text
Colab was useful for smoke testing, but not stable enough for longer GRPO experiments.
```

### Stage 2: Move to UTD Juno HPC

I moved training to UTD Juno to avoid Colab disconnections.

Important environment fixes:

```text
python/3.11.11
cuda/12.4
torch==2.6.0+cu124
```

The setup script was added to reduce CUDA mismatch issues:

```bash
scripts/setup_juno_env.sh
```

Expected GPU check:

```text
torch cuda: 12.4
cuda available: True
```

### Stage 3: 8-Generation Attempt

The first HPC direction tried to stay closer to AMD-style GRPO by using more generations.

Configuration direction:

```text
num_generations = 8
max_steps = 200
single H100
no vLLM
```

Result:

| Run | Eval Samples | Exact Match |
|---|---:|---:|
| Early H100 8gen checkpoint | 100 | 49.0% (49/100) |

Problem:

```text
8 generations was too slow without vLLM.
The job timed out before completing the full 200 steps.
```

Reason for change:

```text
Reduce num_generations from 8 to 4 to make complete runs feasible on a single H100.
```

### Stage 4: 4-Generation Sweeps

Several 4-generation variants were tested to balance training signal and wall-clock time.

Common direction:

```text
num_generations = 4
single H100
LoRA fine-tuning
W&B tracking
```

Results:

| Variant | Eval Samples | Exact Match | Interpretation |
|---|---:|---:|---|
| safe / format variants | 100 | about 50% | Stable but weak |
| strong reward checkpoint | 100 | 54.0% | Better reward weighting |
| complete 100step | 100 | 51.0% | Completed reliably but modest |

Reason for next change:

```text
The model needed more trainable adapter capacity and stronger correctness-oriented updates.
```

### Stage 5: LoRA r32 Strong Reward, 150 Steps

This became the current best model.

Key parameters:

```bash
--max_samples 1500
--max_steps 150
--learning_rate 2e-6
--per_device_train_batch_size 4
--gradient_accumulation_steps 4
--num_generations 4
--max_completion_length 128
--reward_weights 2.0 1.0 2.0 0.5
--lora_r 32
--lora_alpha 64
```

Results:

| Eval Samples | Exact Match |
|---:|---:|
| 100 | 61.0% (61/100) |
| 500 | 54.8% (274/500) |

Checkpoint-level result:

| Checkpoint | Eval Samples | Exact Match |
|---|---:|---:|
| checkpoint-140 | 500 | 54.2% (271/500) |
| checkpoint-150 / final | 500 | 54.8% (274/500) |

Interpretation:

```text
The 100-sample result was optimistic, but the 500-sample result still showed a real improvement over the base model.
The checkpoint-140 result was close to the final model, but checkpoint-150 remained slightly better by 3/500 examples.
```

### Stage 6: Base Model 500-Sample Baseline

To verify whether the GRPO improvement was real, the original base model was evaluated on the same 500-sample setup.

| Model | Eval Samples | Exact Match |
|---|---:|---:|
| Base Qwen2.5-1.5B-Instruct | 500 | 45.4% (227/500) |
| GRPO r32 strong 150step | 500 | 54.8% (274/500) |

Technical interpretation:

```text
The GRPO model answered 47 more questions correctly than the base model on the same 500 examples.
This supports that the gain comes from GRPO fine-tuning rather than random evaluation variance.
```

Approximate two-proportion comparison:

```text
Difference = 0.548 - 0.454 = 0.094
Approximate z = 2.99
Approximate p = 0.003
Approximate 95% CI = [+3.2%, +15.6%]
```

### Stage 7: r32 Strong Reward, 200 Steps

The next test checked whether training longer would improve performance.

Main change:

```text
max_steps: 150 -> 200
```

Result:

| Model | Eval Samples | Exact Match |
|---|---:|---:|
| r32 strong 150step | 500 | 54.8% (274/500) |
| r32 strong 200step | 500 | 53.2% (266/500) |

Interpretation:

```text
More steps did not improve performance.
The 200-step run likely started to over-optimize the training reward or drift too far from the base model.
```

### Stage 8: r16 Strong Reward with Longer Completion Length

This run tested whether a smaller LoRA rank with longer reasoning output would generalize better.

Main changes:

```text
lora_r: 32 -> 16
max_completion_length: 128 -> 192
max_steps: 150
```

Result:

| Model | Eval Samples | Exact Match |
|---|---:|---:|
| r32 strong 150step | 500 | 54.8% (274/500) |
| r16 c192 150step | 500 | 53.8% (269/500) |

Interpretation:

```text
The W&B curves looked more stable, especially for format rewards, but the final exact_match did not beat the r32 150step model.
Longer completions improved reasoning space but did not translate into better final-answer accuracy.
```

### Stage 9: LoRA Merge and GB10B vLLM Evaluation

After identifying the r32 strong 150-step model as the best local candidate, the LoRA adapter was merged into a full model and served on GB10B with vLLM.

Pipeline:

```text
Juno H100 training
-> LoRA checkpoint
-> merge LoRA into full model
-> copy merged model to GB10B
-> serve model with vLLM
-> evaluate GSM8K through vLLM API
```

This stage did not perform additional training. It used GB10B as a fast inference/evaluation backend.

Results under the same GB10B vLLM chat protocol:

| Model | Eval Samples | Exact Match |
|---|---:|---:|
| Base Qwen2.5-1.5B-Instruct | 500 | 54.6% (273/500) |
| GRPO r32 strong 150step merged | 500 | 68.0% (340/500) |
| Improvement | 500 | +13.4 points (+67 correct) |

Paired comparison on the same 500 examples:

```text
Base wrong -> GRPO correct: 99 examples
Base correct -> GRPO wrong: 32 examples
Net gain: +67 correct examples
```

Interpretation:

```text
The 68.0% result should not be attributed to vLLM itself.
vLLM provided faster and consistent inference/evaluation.
The model improvement came from GRPO fine-tuning on Juno H100.
```

### Stage 10: Attempted GB10B-Assisted GRPO Training

After confirming that GB10B vLLM worked for inference/evaluation, I attempted to use it as a vLLM rollout server for TRL GRPO training.

The GB10B server passed the OpenAI-compatible model check:

```text
/v1/models: OK
/health: OK
```

But it failed the TRL-specific endpoint check:

```text
/get_tensor_parallel_size/: 404 Not Found
```

Interpretation:

```text
The current GB10B vLLM server is suitable for inference/evaluation,
but it is not yet the TRL-specific vLLM server required by GRPOTrainer's use_vllm=True training path.
```

This is an infrastructure/version-compatibility issue rather than evidence that GRPO failed.

---

## 6. Main Results Table

| Rank | Model / Run | Eval Samples | Exact Match | Notes |
|---:|---|---:|---:|---|
| 1 | r32 strong 150step merged, GB10B vLLM chat | 500 | 68.0% (340/500) | Best same-protocol vLLM result |
| 2 | Base Qwen2.5-1.5B, GB10B vLLM chat | 500 | 54.6% (273/500) | vLLM chat baseline |
| 3 | r32 strong 150step, local HF eval | 500 | 54.8% (274/500) | Best local HF result |
| 4 | r32 strong checkpoint-140, local HF eval | 500 | 54.2% (271/500) | Close to final, but slightly lower |
| 5 | r16 c192 150step, local HF eval | 500 | 53.8% (269/500) | Stable W&B curves, slightly lower eval |
| 6 | r32 strong 200step, local HF eval | 500 | 53.2% (266/500) | Longer training did not help |
| 7 | Base Qwen2.5-1.5B, local HF eval | 500 | 45.4% (227/500) | Controlled local baseline |
| 8 | r32 strong 150step, local HF eval | 100 | 61.0% (61/100) | High but sample-size sensitive |
| 9 | Early H100 8gen checkpoint | 100 | 49.0% (49/100) | Timed out before full training |
| 10 | Early Colab run | 100 | 46.0% (46/100) | Initial smoke test |

---

## 7. Trend Analysis

### Trend 1: GRPO improves over the base model

The most important local HF comparison is:

```text
Base model 500: 45.4%
Best GRPO 500: 54.8%
Gain: +9.4 points
```

The strongest same-protocol vLLM comparison is:

```text
Base model + GB10B vLLM chat 500: 54.6%
GRPO model + GB10B vLLM chat 500: 68.0%
Gain: +13.4 points
```

Both comparisons support that the GRPO pipeline produced a useful model update.

### Trend 2: More training steps are not always better

```text
r32 150step: 54.8%
r32 200step: 53.2%
```

The 200-step version did not improve. This suggests the best region is likely around 120-150 steps for the current reward design.

The checkpoint comparison gives a more specific trend:

```text
checkpoint-140: 54.2%
checkpoint-150: 54.8%
step-200: 53.2%
```

This suggests that training improved slightly from step 140 to 150, but continuing to 200 steps reduced generalization.

### Trend 3: W&B reward is not the same as exact_match

The r16 c192 run had better-looking W&B curves, especially for format-related rewards, but it did not beat r32 150step on exact_match.

Reason:

```text
Training reward includes format and numeric-answer rewards.
Evaluation exact_match only checks whether the final answer is correct.
```

Therefore, reward improvements can reflect better formatting rather than better math accuracy.

### Trend 4: Longer completion length did not clearly help

```text
c128 r32 150step: 54.8%
c192 r16 150step: 53.8%
```

Longer reasoning space did not improve final answer accuracy in this setup. It also increases token cost.

### Trend 5: r32 adapter capacity helped

The best model used:

```text
lora_r = 32
lora_alpha = 64
```

This suggests the higher LoRA capacity helped correctness more than the more conservative r16 setup.

### Trend 6: vLLM changes throughput and evaluation protocol, not model weights

The vLLM result was much higher than earlier local results, but the correct interpretation is:

```text
vLLM did not make the base model smarter.
vLLM served models faster and used a chat API evaluation protocol.
The fair comparison is base-vLLM versus GRPO-vLLM.
```

Under that fair comparison, the GRPO gain was:

```text
54.6% -> 68.0%
```

---

## 8. Why the Changes Were Made

| Change | Reason | Outcome |
|---|---|---|
| 3B/other models -> 1.5B | Match AMD reference more closely and fit available compute | Successful |
| Colab -> Juno H100 | Avoid Colab disconnections | More reliable training |
| 8 generations -> 4 generations | 8gen was too slow without vLLM | Complete runs became feasible |
| LoRA r16 -> r32 | Increase adapter capacity | Best model used r32 |
| 150 steps -> 200 steps | Test whether longer training helps | Did not help |
| completion 128 -> 192 | Test longer reasoning output | More stable curves but lower exact_match |
| 100-sample eval -> 500-sample eval | Reduce random evaluation noise | More reliable comparison |
| base model eval added | Establish controlled baseline | Confirmed +9.4 point GRPO gain |
| `save_total_limit=2` -> 10 | Enable checkpoint sweep | Future runs can preserve more checkpoints |
| LoRA merge + GB10B vLLM eval | Evaluate trained model with faster serving backend | Confirmed +13.4 point same-protocol vLLM gain |
| GB10B as TRL rollout server | Try to approach AMD-style vLLM-assisted training | Blocked by missing TRL-specific endpoint |

---

## 9. Why vLLM Matters

The current implementation uses TRL GRPO for training. GB10B vLLM has been successfully used for inference/evaluation, but not yet for training-time rollout generation.

Without vLLM:

```text
Generation is slower.
Large num_generations such as 8 or 16 is hard to run.
Longer GRPO jobs are more likely to timeout.
```

With vLLM:

```text
Rollout generation becomes faster.
Higher num_generations becomes more practical.
The training can collect more comparison samples per prompt.
```

Important distinction:

```text
vLLM does not change the GRPO objective.
vLLM does not update model weights.
It improves serving throughput and, if integrated into training correctly, can make larger GRPO settings feasible.
```

Current limitation:

```text
GB10B currently runs an OpenAI-compatible vLLM server for inference/evaluation.
TRL GRPO training with use_vllm=True requires a TRL-specific vLLM server.
The current GB10B server returns 404 for /get_tensor_parallel_size/.
```

---

## 10. Technical Explanation of Success

The strongest local HF evidence is the controlled 500-sample comparison:

```text
Base model: 45.4%
GRPO model: 54.8%
```

Since both models used the same:

```text
dataset split
evaluation script
prompt format
decoding settings
metric
sample size
```

the main model-side difference is:

```text
the GRPO-trained LoRA adapter
```

Therefore, the result supports that the improvement comes from GRPO fine-tuning.

The strongest GB10B vLLM evidence is:

```text
Base model: 54.6%
GRPO merged model: 68.0%
```

Since both models were evaluated using the same GB10B vLLM chat protocol, this comparison supports a GRPO-driven gain under that protocol. The vLLM server provided inference only; it did not train or modify either model.

The improvement is not just a 100-sample lucky run, because the 500-sample evaluation still shows:

```text
+47 correct answers under local HF evaluation
+67 correct answers under GB10B vLLM chat evaluation
```

---

## 11. Current Weaknesses

### Weakness 1: Reward-target mismatch

The training reward includes formatting rewards, but the final metric only checks exact answer correctness.

This means the model may learn:

```text
better XML format
more numeric outputs
```

without always improving:

```text
final answer accuracy
```

### Weakness 2: Sparse correctness signal

GSM8K correctness reward is sparse:

```text
correct answer = reward
wrong answer = no correctness reward
```

With only 4 generations per prompt, the group-level learning signal can be noisy.

### Weakness 3: No vLLM acceleration

Without training-time vLLM rollout acceleration, larger generation counts are expensive.

This limited the practical search space:

```text
num_generations = 4 was feasible
num_generations = 8 was too slow
```

GB10B vLLM helped evaluation, but it has not yet been integrated as a TRL-compatible rollout server.

### Weakness 4: Checkpoint retention was initially too low

Earlier runs only preserved the last two checkpoints because:

```text
save_total_limit = 2
```

This limited checkpoint-level analysis. It has now been changed to:

```text
save_total_limit = 10
```

---

## 12. Next Recommended Experiments

### Experiment A: Preserve and evaluate more intermediate checkpoints

The current best run only preserved:

```text
checkpoint-140
checkpoint-150
```

The checkpoint-140 evaluation has now been completed:

```text
checkpoint-140 = 54.2% (271/500)
checkpoint-150 / final = 54.8% (274/500)
```

This means the final 150-step model is still slightly better, but the difference is only 3 examples out of 500.

Future runs should preserve more checkpoints with:

```bash
--save_total_limit 10
```

so steps such as 100, 110, 120, 130, 140, and 150 can be evaluated.

Example command for checkpoint-level evaluation:

```bash
sbatch --time=03:00:00 \
  --export=ALL,MODEL_PATH=outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-150step/checkpoint-140,MAX_SAMPLES=500 \
  scripts/utd_eval_checkpoint.slurm
```

### Experiment B: Correctness-dominant stable GRPO

The next training run should reduce format reward dominance and focus more directly on exact answer correctness.

Suggested configuration:

```bash
--learning_rate 1e-6
--beta 0.06
--reward_weights 3.0 0.5 0.5 0.1
--lora_r 32
--lora_alpha 64
--num_generations 4
--max_completion_length 128
--max_steps 150
--save_total_limit 10
```

Reason:

```text
Lower LR reduces unstable updates.
Higher beta limits drift from the base model.
Higher correctness weight aligns training more closely with exact_match.
Lower format weights reduce reward hacking on XML format.
```

### Experiment C: Full GSM8K test evaluation

Run full test-set evaluation for the base model and current best model.

```text
GSM8K test size = 1319 examples
```

This would provide the strongest final number for reporting.

### Experiment D: vLLM-assisted GRPO if multi-GPU access is available

If a TRL-compatible vLLM server can be started on GB10B or another GPU node, use vLLM for rollout generation.

Expected benefit:

```text
Faster generation
More feasible 8gen or 16gen GRPO
Better group-relative learning signal
```

Immediate requirement:

```text
The vLLM server must expose TRL-specific endpoints such as /get_tensor_parallel_size/.
An ordinary OpenAI-compatible vLLM server is enough for eval, but not enough for TRL GRPO training.
```

---

## 13. Suggested One-Paragraph Summary

I implemented an AMD-inspired GRPO fine-tuning pipeline using Qwen2.5-1.5B-Instruct, GSM8K, Hugging Face TRL GRPOTrainer, LoRA, and W&B tracking. The initial Colab setup was unstable, so training was moved to UTD Juno H100 with a fixed CUDA 12.4 / PyTorch 2.6 environment. After testing several configurations, the best model used 4 generations, LoRA rank 32, 150 training steps, max completion length 128, and reward weights favoring correctness and format. On a controlled local 500-sample GSM8K evaluation, the base model achieved 45.4% exact match, while the best GRPO model achieved 54.8%, a +9.4 point improvement. After merging the LoRA adapter and serving the model on GB10B with vLLM, the same-protocol vLLM chat evaluation showed a stronger result: 54.6% for the base model versus 68.0% for the GRPO model on 500 examples. vLLM was used for inference/evaluation only; the model improvement comes from GRPO fine-tuning.

---

## 14. Current Best Claim

The safest claim is:

```text
The GRPO pipeline successfully improved Qwen2.5-1.5B-Instruct on GSM8K under two same-protocol comparisons:
45.4% to 54.8% on local HF evaluation, and 54.6% to 68.0% on GB10B vLLM chat evaluation.
```

The stronger claim still needs full-test confirmation:

```text
The GRPO model improves over the base model on the full GSM8K test set.
```

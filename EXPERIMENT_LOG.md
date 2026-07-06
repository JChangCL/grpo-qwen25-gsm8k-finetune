# GRPO Fine-Tuning — Experiment Log

Model family: **Qwen2.5-1.5B-Instruct** · Method: **TRL GRPO + LoRA** · Rollout: **vLLM**
Hardware: **UTD Juno H100 80GB** (training + eval), GB10B (rollout server for the AMD split, WIP)
Software stack (Juno colocate venv): `trl 0.18.0 · vllm 0.8.5 · transformers 4.51.3 · torch 2.6.0+cu124 · peft 0.15.2`

---

## 1. Headline result (confirmed)

GRPO **beats the strong instruct base on GSM8K** with the right recipe:

| Model | GSM8K exact-match (full 1319 test, vLLM chat greedy) |
|---|---|
| Base Qwen2.5-1.5B-Instruct | 69.52% (917/1319) |
| **GRPO "Run4"** | **71.27% (940/1319) — +1.75 pt, +23 problems** |

Consistent across sample sizes and protocols (500-sample +2.2, plain-prompt +1.6, full-test +1.75), so it is a real gain, not noise.

Reference (AMD ROCm GRPO blog, same model): base 38.67% → GRPO-150 **42.76% (+3 pt)**. Our gain is smaller because our base is far stronger (see §5 protocol analysis).

---

## 2. GSM8K training runs (all Juno H100, vLLM colocate, LoRA r32/α64, 8 generations)

| Run | Job | Prompt | completion | steps | lr | beta | reward_weights | Key training metrics (final) | Outcome |
|---|---|---|---:|---:|---|---|---|---|---|
| Run1 | 263240 | raw | 128 | 150 | 1e-6 | 0.06 | 2/1/2/0.5 | clipped 1.0, corr 0.19 | completions all truncated → weak signal |
| Run2 | 263311 | raw | 256 | 150 | 1e-6 | 0.06 | 2/1/2/0.5 | clipped 1.0, corr 0.31, kl 1.09, grad_norm 20–25 | longer helped correctness but still no EOS; unstable |
| Run3 | 263941 | **chat** | 256 | 150 | 1e-6 | 0.10 | 2/1/2/0.5 | clipped 0.31, corr 0.56, kl **0.0005**, grad_norm 0.32 | healthy + stops naturally, but β too high → policy barely moved (≈base) |
| **Run4** | **264046** | **chat** | **512** | **300** | **5e-6** | **0.03** | **3/0.5/0.5/0.25** | clipped **0.0**, corr **0.625**, kl **0.007**, grad_norm 0.33, reward 2.44 | **best: stable, moved the policy, beat base** |

W&B project `grpo-gsm8k-simulation`; Run4 = run `nymhqisx`.

**What each change bought:**
- **raw → chat prompt** (Run2→Run3): the base is an instruct model; a raw completion prompt never emits an end-of-turn token, so every completion ran to `max_len` (`clipped_ratio=1.0`, `mean_terminated_length=0`). The chat template makes it stop naturally (clipped 1.0→0.31) and aligns train with eval. **This was the single most important fix for training health.**
- **completion 256 → 512**: removed the last truncation (clipped 0.31→0.0); more room for math reasoning.
- **beta 0.10 → 0.03 + lr 1e-6 → 5e-6 + steps 150 → 300**: Run3 was "healthy but frozen" — KL ≈ 0.0005 means it never left the base, so eval ≈ base. Loosening the KL penalty and pushing lr/steps moved the policy (KL 0.0005→0.007) **enough to actually improve accuracy** while staying stable (grad_norm ~0.33).
- **reward weights → correctness-heavy (3/0.5/0.5/0.25)**: the base already formats well, so weighting format was wasted; shifting weight to correctness directed the gain toward getting answers right.

---

## 3. GSM8K eval matrix (500-sample, vLLM greedy)

| Model | XML-prompt protocol | plain (no-scaffold) protocol |
|---|---|---|
| Base | 72.0% | 66.6% |
| Run2 | 72.2% | 66.0% |
| Run3 | 71.2% | 67.6% (+1.0 vs base) |
| **Run4** | **74.2% (+2.2)** | **68.2% (+1.6)** |

- Under the XML reasoning prompt the base is already saturated (72%), so only the strongest run (Run4) clears it.
- Under the plain protocol (base drops to 66.6% without the scaffold) the GRPO models show a clearer, AMD-style edge — they internalized reasoning that survives without the prompt.

---

## 4. MATH-500 (harder task — in progress)

| Model | MATH-500 exact-match (math_verify) |
|---|---|
| Base Qwen2.5-1.5B-Instruct | 45.6% (228/500) |
| Run4 (GSM8K-trained, transfer) | 43.8% — GSM8K training does **not** transfer to MATH |

MATH base 45.6% « GSM8K base 69.5% → ~large headroom. A dedicated **GRPO-on-MATH** run (nlile/hendrycks-MATH-benchmark train, contamination-filtered vs MATH-500; `\boxed{}` + math-equivalence reward; c1024) is training now — expected to show a much larger gain than GSM8K's +1.75.

---

## 5. Why absolute scores differ so much — eval protocol dominates

Same base model, different protocols:

| Protocol (same Base Qwen2.5-1.5B-Instruct) | GSM8K score |
|---|---|
| AMD strict harness (zero-shot, strict extract) | 38.67% |
| Prior report (GB10 vLLM chat, max_tokens 256) | 54.6% |
| Ours (chat, max_tokens 256, last-number fallback) | 63.8% |
| Ours (chat, max_tokens 512) | 72.0% |

**Lesson: the absolute number is almost meaningless without fixing the protocol; only the base→GRPO gain under one fixed protocol is meaningful.** The earlier "+13.4 pt (54.6→68)" was largely an artifact of an under-measured base (256-token truncation), not a real GRPO gain.

---

## 6. Key learnings

1. **vLLM rollout is the throughput unlock.** GRPO is generation-bound (8 rollouts/prompt). vLLM colocate cut wall-clock from **10+ hours (HF generate) to ~15 min** for the same run. This is the most practical win of the "AMD recipe."
2. **Match train/eval formatting.** Training on raw prompts while evaluating with a chat template is both unhealthy (no EOS) and a distribution mismatch. Use the chat template in training.
3. **Watch KL, not just reward.** A "healthy-looking" run (Run3) can have KL≈0 = the model never moved = no eval gain. You must let the policy move (lower beta / higher lr / more steps) to get real improvement.
4. **Reward should target the weakest skill.** A strong instruct base already formats; put reward weight on correctness.
5. **Task choice > model size for showing RL value.** GSM8K is near-ceiling for a 1.5B instruct model (base 69.5%), so gains are small. Harder tasks (MATH, competition math) have headroom where GRPO shines.

## 7. Improvement directions (ranked)

1. **Harder tasks (MATH / competition math)** — highest value, low cost (same pipeline). MATH base is only ~45%; expect large, defensible gains. **(in progress)**
2. **Longer chain-of-thought / test-time compute (R1-style)** — reward correctness on hard problems and let reasoning grow (completion 1–2k). Where real RL gains come from.
3. **Push Run4 recipe further on GSM8K** — Run5: even lower beta (0.01–0.02) / more steps (500) — but expect diminishing returns near the 72% ceiling.
4. **Scale to 7B** — raises absolute (~85% GSM8K base) but GRPO's marginal gain stays small; only worth it combined with a harder task.
5. **True cross-cluster split (H100 train ↔ GB10B vLLM rollout)** — engineering milestone (matches AMD topology), not an ML gain. Blocked by Juno driver 550 vs trl-1.7's vLLM ≥0.14 (needs a CUDA-forward-compat container); WIP.

## 8. Infrastructure notes (reproducibility)

- **Juno H100 = driver 550 / CUDA 12.4.** trl 1.7.1 needs vLLM 0.14–0.23 (CUDA 12.8+/driver 560+) → won't install natively. Working colocate stack is **trl 0.18.0 + vllm 0.8.5** (driver-550 native, has `vllm_mode="colocate"`).
- **Bare-metal vLLM colocate needs two env fixes** (baked into the sbatch): `CPATH` → a Python 3.11 `Python.h` (triton JITs a CUDA ext; node lacks `python3.11-devel`), and single-process `RANK/WORLD_SIZE/MASTER_ADDR` (vLLM's executor reads them).
- **Merge LoRA with `peft==0.15.2` and `device_map=cpu`**, and **without** the distributed env vars (RANK set → `mixed DTensor` error).
- Two Juno login nodes round-robin (`juno-l-01/-02`) with node-local `/tmp`; keep shared state on `/work`.

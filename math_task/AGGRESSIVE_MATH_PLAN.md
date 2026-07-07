# Aggressive MATH GRPO — Plan & Results

**Goal:** Test whether the MATH failure (MATH v2 reached train `corr_reward≈0.625`
but **MATH-500 eval 54.8% ≈ base 55.6%**) is caused by **insufficient policy
movement (KL≈0.0014)** — the same failure mode as GSM8K Run3 — and NOT by a
fundamental limit of final-answer GRPO.

The fix that worked on GSM8K (Run3→Run4): lower `beta`, raise `lr`, more `steps`,
longer `completion` → KL 0.0005 → **0.007**, which is what beat base. We now
apply the analogous push to MATH, where there is the most headroom (base 55.6%).

Base fixed reference (fixed `is_correct`): **MATH-500 = 55.6% (278/500)**, job 264570.
MATH v2 (fixed reward, KL≈0.0014): **54.8% (274/500)**, job 264569, W&B `qcroxolu`.

---

## Reward / eval sanity (verified before designing runs)

- **Reward bug is fixed.** `math_task/math_utils.py::is_correct` wraps the gold
  answer in `\boxed{...}` before `math_verify.parse` (unless it already contains
  `\boxed`/`$`): `gold_expr = g if ("\\boxed" in g or "$" in g) else f"\\boxed{{{g}}}"`.
  This is what restored train `corr_reward` 6% → 62.5% in v2.
- **Train and eval use the SAME fixed parser.** Both `train_grpo_math.py`
  (`correctness_reward`) and `eval_math.py` import `is_correct` from `math_utils`.
  There is no buggy code path left. ✅
- **Train/eval overlap guard is retained.** `prepare_math()` filters every train
  problem whose `problem` string appears in `HuggingFaceH4/MATH-500` before
  training and prints how many it removed. ✅ (The log's "0 overlap by unique_id"
  is enforced here by exact-problem-string match — equivalent in practice.)

**One caveat to watch, not a blocker:** `math_verify.verify` is order-sensitive
and can occasionally reject a correct free-form answer if the model emits extra
prose after `\boxed{}`. The eval already passes the full completion; if a run's
train `corr_reward` looks high but eval looks suspiciously flat, spot-check a few
`is_correct(completion, gold)` calls by hand before concluding "no gain".

---

## The run matrix (small, targeted — NOT a sweep)

All runs: Qwen2.5-1.5B-Instruct · TRL GRPO + LoRA r32/α64 · 8 generations ·
vLLM **colocate** on Juno H100 · fixed v2 reward weights `[correctness=2.0,
boxed=0.5]` · `max_samples=4000` · `save_steps=20` · dataset
`hendrycks/competition_math` (train split, MATH-500 filtered out).

| Run | sbatch | beta | lr | steps | completion | batch×accum | vLLM util | purpose |
|-----|--------|-----:|----:|------:|-----------:|:-----------:|:---------:|---------|
| **A** | `sbatch_math_aggressive_A.sh` | 0.02 | 1e-5 | 400 | 1024 | 16×1 | 0.35 | mildest push; safe first submit |
| **B** | `sbatch_math_aggressive_B.sh` | 0.01 | 1e-5 | 500 | 1024 | 16×1 | 0.35 | **primary** movement test |
| **C** | `sbatch_math_aggressive_C.sh` | 0.01 | 2e-5 | 500 | 1024 | 16×1 | 0.35 | 2× lr — strongest push |
| **D** | `sbatch_math_aggressive_D.sh` | 0.01 | 1e-5 | 500 | 1536 | 8×2 | 0.45 | long completion — test truncation |

Reference points: MATH v2 was `beta0.03 / lr5e-6 / 300s / c1024` → KL 0.0014.
GSM8K Run4 (the recipe that worked) was `beta0.03 / lr5e-6 / 300s / c512` → KL 0.007.

**Submit order recommendation:** B first (primary hypothesis), then A and C
(brackets B on lr/beta), then D (needs the OOM story confirmed — see below).

---

## Core hypothesis & decision rule

> **H1 (movement).** If KL rises from 0.0014 into **~0.005–0.02** *and*
> MATH-500 accuracy exceeds base **55.6%**, then MATH v2 failed primarily
> because the **policy didn't move** — same as GSM8K Run3. This validates the
> whole "watch KL, not reward" thesis on a second task.
>
> **H2 (ceiling of final-answer GRPO).** If KL clearly rises (≥0.005) **but eval
> stays ≈ base**, then final-answer GRPO is *not enough* for MATH at 1.5B. Only
> then do we move to the next directions: **process / verifier reward**, longer
> R1-style CoT, or **scale to 7B**. Do NOT jump to those until H1 is falsified.

**Do not judge a run by train `corr_reward` alone.** MATH v2 already proved that
high train reward (0.625) with KL≈0 gives **zero eval gain**. Eval delta over
base 55.6% is the only success metric; KL is the mechanism metric.

### Success / failure criteria (per run)

- ✅ **Success:** MATH-500 eval ≥ **57.6%** (≥ +2.0 pt over base 55.6%, i.e.
  comfortably past ~1 pt eval noise) with KL in [0.005, 0.02] and stable
  `grad_norm` (no divergence, no NaN).
- 🟡 **Movement-but-no-gain:** KL ≥ 0.005 but eval within ±1 pt of base → evidence
  for **H2** (final-answer GRPO insufficient).
- 🔴 **No movement:** KL < 0.003 → recipe still too gentle; the aggressive matrix
  itself failed to move the policy (reconsider lr/beta, not the research direction).
- ⚠️ **Instability:** `grad_norm` spikes / reward collapse (esp. C at lr 2e-5) →
  lower lr or shorten steps; report the divergence step.

---

## Metrics to record for every run

Pulled from W&B (project `grpo-gsm8k-simulation`) — TRL logs these each step:

| What | W&B / log key |
|------|---------------|
| train correctness reward | `rewards/correctness_reward/mean` |
| boxed / format reward | `rewards/boxed_format_reward/mean` (→ boxed rate = /0.5) |
| total reward curve | `reward` (and `reward_std`) |
| **KL** (the key metric) | `kl` |
| completion length | `completions/mean_length` (+ `completions/max_length`) |
| clipped / truncation ratio | `completions/clipped_ratio` |
| grad norm | `grad_norm` |
| final MATH-500 eval | from `eval_math.py` (record below) |
| **delta over base 55.6%** | `eval − 0.556` |

Fill in the results table as jobs finish:

| Run | job | KL | corr_reward | reward | mean_len | clipped | grad_norm | MATH-500 | Δ vs 55.6% | verdict |
|-----|-----|---:|------------:|-------:|---------:|--------:|----------:|:--------:|:----------:|:-------:|
| base | 264570 | — | — | — | — | — | — | 55.6% | 0 | ref |
| v2   | 264569 | 0.0014 | 0.625 | — | — | — | — | 54.8% | −0.8 | frozen |
| A    | 265834 |  |  |  |  |  |  |  |  | submitted |
| B    | 265833 |  |  |  |  |  |  |  |  | submitted |
| C    | 265835 |  |  |  |  |  |  |  |  | submitted |
| D    | 265836 |  |  |  |  |  |  |  |  | submitted |

**Submitted** 2026-07-07 (Juno H100, jobs 265833–265836). W&B in **offline** mode
(no API key was available at submit time) — metrics log to local `wandb/` on juno;
run `wandb sync wandb/offline-run-*` once a key is set to push them online.

---

## How to submit

```bash
# From a machine that can ssh to juno, with WANDB_API_KEY exported locally:
for S in A B C D; do
  printf '%s\n' "$WANDB_API_KEY" | ssh juno "IFS= read -r K; \
    WANDB_API_KEY=\$K sbatch --export=ALL math_task/sbatch_math_aggressive_${S}.sh"
done
# or submit one at a time (recommended: B, then A, C, then D).
```

## How to eval a finished run (merge LoRA → offline MATH-500)

`eval_math.py` loads a *full* model into vLLM, so merge the LoRA adapter first.

```bash
RUN=math-aggr-B-b01-lr1e5-500s-c1024          # e.g. for Run B
ADAPTER=outputs/qwen2.5-1.5b-math-grpo-${RUN}   # or a checkpoint-XXX inside it
MERGED=outputs/merged-${RUN}

# Merge on CPU, peft 0.15.2, WITHOUT the distributed env vars (see EXPERIMENT_LOG §8)
env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT \
  .venv-colocate/bin/python merge_lora.py \
    --adapter_path "$ADAPTER" --output_dir "$MERGED" \
    --dtype bfloat16 --device_map cpu --overwrite

# Eval MATH-500 (fixed is_correct via PYTHONPATH=math_task)
PYTHONPATH=math_task .venv-colocate/bin/python math_task/eval_math.py \
  --model "$MERGED" --tag "$RUN" --max_samples 500 --max_tokens 1024
# For Run D (c1536) bump eval budget: --max_tokens 1536
```

Record the `RESULT tag=... MATH exact_match=...` line into the table above and
into the repo `results.csv` (protocol `math500_vllm_chat_greedy`).

---

## Guardrails (scope of this experiment)

- Single-cluster **colocate only** — no GB10B / cross-cluster split (that is an
  infra milestone, EXPERIMENT_LOG §5, out of scope here).
- Keep the working Juno stack: `.venv-colocate` (trl 0.18.0 + vllm 0.8.5) and the
  `CPATH` + single-process `RANK/WORLD_SIZE` env fixes.
- This is a **4-run targeted matrix**, not a hyperparameter sweep. Stop after
  H1 is confirmed or falsified.

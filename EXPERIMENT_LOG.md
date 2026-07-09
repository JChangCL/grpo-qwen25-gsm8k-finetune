# GRPO Fine-Tuning — Experiment Log

Base model: **Qwen2.5-1.5B-Instruct** · Method: **TRL GRPO + LoRA (r32/α64, 8 generations)**
Rollout: **vLLM colocate** · Hardware: **UTD Juno H100 80GB** (training + eval)
Stack (Juno `.venv-colocate`): `trl 0.18.0 · vllm 0.8.5 · transformers 4.51.3 · torch 2.6.0+cu124 · peft 0.15.2`

---

## 0. TL;DR

- **GSM8K:** best run beats a strong base — **69.52% → 71.27% (+1.75 pt, confirmed on full 1319 test).**
- **MATH:** v2 stalled at **~base parity (54.8% vs 55.6%)** because the policy barely moved
  (KL≈0.0014). The **aggressive re-run fixed that**: moving the policy *stably* (run C,
  β0.01/lr2e-5/500s, KL 0.010) beats base — **55.6% → 57.6% (+2.0 pt)**. Gain is modest
  (1.5B near ceiling), and stability matters — run B moved but blew up (KL spike) → below base.
- **Core lesson:** at 1.5B, GRPO **sharpens existing ability but adds little**; gains are marginal because the base is near its ceiling. The single biggest lever is **letting the policy actually move (KL)** via lower beta / higher lr / more steps, plus **matching train/eval formatting** and **measuring the reward correctly.**
- **vLLM colocate cut training from 10+ hours → ~15 min.**
- **Cross-cluster split (H100↔GB10B)** not achieved — blocked on infrastructure (driver/version + container pull limits).

---

## 1. GSM8K training runs (Juno H100, colocate)

| Run | Job | Prompt | completion | steps | lr | beta | reward_weights | corr_reward | KL | grad_norm |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|
| Run1 | 263240 | raw | 128 | 150 | 1e-6 | 0.06 | 2/1/2/0.5 | 0.19 | — | — |
| Run2 | 263311 | raw | 256 | 150 | 1e-6 | 0.06 | 2/1/2/0.5 | 0.31 | 1.09 | 20–25 |
| Run3 | 263941 | **chat** | 256 | 150 | 1e-6 | 0.10 | 2/1/2/0.5 | 0.56 | **0.0005** | 0.32 |
| **Run4** | **264046** | **chat** | **512** | **300** | **5e-6** | **0.03** | **3/0.5/0.5/0.25** | **0.625** | **0.007** | 0.33 |

W&B project `grpo-gsm8k-simulation` (Run4 = `nymhqisx`).

**What each change bought**
- **raw → chat prompt** (Run2→Run3): an instruct model never emits an end-of-turn token on a raw completion prompt → every completion ran to `max_len` (`clipped_ratio=1.0`, `mean_terminated_length=0`), starving the reward signal. Chat template → stops naturally (clipped 1.0→0.31) and aligns train with eval. **Biggest fix for training health.**
- **completion 256→512:** removed the last truncation (clipped→0.0).
- **beta 0.10→0.03 + lr 1e-6→5e-6 + steps 150→300:** Run3 was "healthy but frozen" (KL≈0.0005 → never left base → eval≈base). Loosening KL + pushing lr/steps moved the policy (KL→0.007) **enough to beat base**, still stable.
- **correctness-heavy reward (3/0.5/0.5/0.25):** the base already formats; put weight on getting the answer right.

## 2. GSM8K eval matrix (vLLM greedy)

| Model | XML-prompt, 500 | plain (no scaffold), 500 | full 1319 (XML) |
|---|---|---|---|
| Base | 72.0% | 66.6% | 69.52% |
| Run2 | 72.2% | 66.0% | — |
| Run3 | 71.2% | 67.6% | — |
| **Run4** | **74.2% (+2.2)** | **68.2% (+1.6)** | **71.27% (+1.75, CONFIRMED)** |

Under the XML reasoning prompt the base is saturated (72%); only Run4 clears it. Under the plain protocol (base 66.6%) the GRPO edge is clearer — the model internalized reasoning that survives without the scaffold.

## 3. Eval-protocol analysis (why absolute numbers vary wildly)

Same **Base** Qwen2.5-1.5B-Instruct on GSM8K:

| Protocol | Score |
|---|---|
| AMD strict harness (zero-shot) | 38.67% |
| Prior report (GB10 vLLM chat, max_tokens 256) | 54.6% |
| Ours (chat, max_tokens 256, last-number fallback) | 63.8% |
| Ours (chat, max_tokens 512) | 72.0% |

**The absolute number is meaningless without a fixed protocol; only the base→GRPO gain under one protocol counts.** The earlier "+13.4 pt (54.6→68)" was mostly an under-measured base (256-token truncation), not a real gain.

**AMD reference (same model):** base 38.67% → GRPO-150 **42.76% (+3 pt).** Their bigger gain comes from a much weaker base (strict harness, no reasoning prompt) with lots of headroom.

## 4. MATH sub-project (`math_task/`)

Harder task chosen because GSM8K is near-ceiling for 1.5B. Eval = MATH-500; train = `nlile/hendrycks-MATH-benchmark` (train split, **verified 0 overlap** with MATH-500 by unique_id); reward = `\boxed{}` + math-equivalence (`math_verify`).

**Reward bug found & fixed (important):** `is_correct` returned False on LaTeX answers (e.g. `\dfrac{7}{20}`) because `math_verify.parse` doesn't parse a *bare* gold string — must wrap it: `parse("\\boxed{"+gold+"}")`. This under-counted everything.

| Run | Job | reward | corr_reward (train) | KL | MATH-500 eval |
|---|---|---|---:|---:|---|
| Base | — | — | — | — | **45.6% (buggy) → 55.6% (fixed)** |
| MATH v1 | 264170 | **broken** | 0.0625 | 0.0015 | (not meaningful) |
| MATH v2 | 264569 | fixed | **0.625** | **0.0014** | **54.8% ≈ base** |

**Interpretation:** the fix restored the *training* signal (corr 6%→62.5%), but eval still ≈ base because **KL≈0.0014 → the policy didn't move** (same failure mode as GSM8K Run3). On MATH the same β0.03/lr5e-6 moves *less* than on GSM8K (longer completions, sparser reward) → needs a more aggressive recipe to show a gain.

### 4a. Aggressive MATH runs (A/B/C/D) — *did the policy move?* ✅ (eval pending)

Recipe matrix from Improvement direction #1 (lower β, higher lr, more steps, longer
completion), colocate on Juno H100. Scripts `math_task/sbatch_math_aggressive_{A,B,C,D}.sh`;
plan + full table in `math_task/AGGRESSIVE_MATH_PLAN.md`. Metrics = mean(last-20 steps).

| Run | Job | β | lr | steps | compl | **KL** | KLmax | corr (train) | clip | mean_len | grad_norm | MATH-500 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v2 (ref) | 264569 | 0.03 | 5e-6 | 300 | 1024 | 0.0014 | — | 0.625 | — | — | — | 54.8% |
| **A** | 265851 | 0.02 | 1e-5 | 400 | 1024 | **0.0036** | 1.7 | 0.42 | 0.084 | 551 | 0.12 | 55.0% (−0.6) |
| **B** | 265887 | 0.01 | 1e-5 | 500 | 1024 | **0.0068** | 5.9e6 ⚠ | 0.52 | 0.062 | 527 | 0.15 | 54.6% (−1.0) |
| **C** | 265888 | 0.01 | 2e-5 | 500 | 1024 | **0.0100** | 0.16 | 0.52 | 0.053 | 494 | 0.15 | **57.6% (+2.0) ✅** |
| **D** | 265889 | 0.01 | 1e-5 | 500 | 1536 | **0.0073** | 0.54 | 0.53 | 0.016 | 574 | 0.13 | 56.8% (+1.2) |

Eval = MATH-500, fixed `is_correct`, vLLM greedy (job 268534). Base = 55.6% (278/500).

**Verdict — H1 CONFIRMED (qualified): moving the policy *stably* gives a real, modest gain.**
Eval tracks KL almost perfectly:
- **C is the cleanest, highest-KL mover (0.010, stable) AND the best model: 57.6%, +2.0 pt** —
  clears the success bar. **D** corroborates (in-band KL 0.0073 → +1.2). So v2's flat eval
  *was* substantially a policy-movement failure — move it and MATH-500 rises.
- **Stability matters as much as movement.** **B** had the same loosened β as C but a
  transient KL blow-up (KLmax≈5.9e6) → lands *below* base (54.6%). Don't just crank the
  leash; keep the update stable.
- **A ≈ base** (KL 0.0036, below band) reproduces v2's frozen behaviour — the control held.
- The win is **modest (+2 pt)** — consistent with #5 (1.5B near ceiling). This is *not* H2:
  final-answer GRPO does help on MATH; the ceiling, not the method, caps the size of the win.

**Winning MATH recipe: C = β0.01 / lr2e-5 / 500 steps / c1024, rw 2.0/0.5.**
Next: confirm C isn't 500-sample noise (+2pt = 10 problems) via a seed-repeat or full-test
eval; fix B-style instability with a KL/grad clip; only then escalate to process/verifier
reward or 7B.

**Infra fixes surfaced batch-submitting these four** (now folded into the scripts):
(1) `hendrycks/competition_math` is gated on the Hub → use `nlile/hendrycks-MATH-benchmark`;
(2) a hard-coded `MASTER_PORT=29500` makes co-scheduled colocate jobs collide with
`EADDRINUSE` — derive a per-job port (`20000+SLURM_JOB_ID%40000`).

## 5. Cross-cluster split (H100 train ↔ GB10B vLLM rollout) — NOT achieved

The original AMD topology. Status: **blocked on infrastructure, not attempted end-to-end.**
- Generation path Juno→GB10B verified working; GB10B `trl vllm-serve` runs with the correct TRL 1.x endpoints (`/init_communicator/`, `/update_named_param/`, …).
- Weight-sync direction is client→server (Juno→GB10B) = the working direction — promising.
- **Blocker:** trl 1.7.1 (to match GB10B) requires vLLM 0.14–0.23 → CUDA 12.8/13 → driver ≥560; **Juno H100 is driver 550**. venv can't satisfy it. Only path is a CUDA-forward-compat container; the NGC image pulls (~20 GB) repeatedly died on login-node OOM / srun time limits before the compat test ran. Cross-cluster NCCL remains unverified even past that.
- **Verdict:** an engineering milestone (doesn't change the ML result); needs either an admin driver bump or a persisted forward-compat container.

## 6. Key learnings

1. **vLLM colocate is the throughput unlock** — 10+ hr (HF generate) → ~15 min. GRPO is generation-bound.
2. **Match train/eval formatting** — chat template in training (natural EOS, no truncation, train=eval).
3. **Watch KL, not just reward.** Healthy reward curves with KL≈0 (Run3, MATH v2) = the model never moved = no eval gain. You must let it move.
4. **Measure the reward correctly.** The MATH LaTeX bug silently halved the signal; always sanity-check `reward(correct_output) == 1`.
5. **Task ≠ the bottleneck at 1.5B.** GRPO sharpens existing ability; a small model near its ceiling (GSM8K 69.5%, MATH 55.6%) has little to sharpen → marginal gains on both. AMD's +3 was on a base handicapped to 38.67%.

## 7. Improvement directions (ranked)

1. ~~**Aggressive MATH run**~~ ✅ **DONE (§4a):** β0.01/lr2e-5/500s/c1024 (run C) moved the
   policy (KL 0.010) and beat base **55.6% → 57.6% (+2.0 pt)**. Follow-ups: seed-confirm C
   (+2pt = 10/500, borderline), fix B-style KL instability, try more steps of C.
2. **Longer chain-of-thought / test-time compute (R1-style)** on hard problems.
3. **Scale to 7B** — more capacity to sharpen; raises absolute (~85% GSM8K) but GRPO gain stays modest.
4. **Process / verifier rewards** instead of final-answer-only.
5. **Cross-cluster split** — engineering only; needs driver bump or forward-compat container.

## 8. Infrastructure notes (reproducibility)

- **Juno H100 = driver 550 / CUDA 12.4.** Working colocate stack: **trl 0.18.0 + vllm 0.8.5** (driver-550 native, has `vllm_mode="colocate"`).
- **Bare-metal colocate needs two env fixes** (in the sbatch): `CPATH` → a Python 3.11 `Python.h` (triton JITs a CUDA ext; node lacks `python3.11-devel`), and single-process `RANK/WORLD_SIZE/MASTER_ADDR` (vLLM executor reads them).
- **Merge LoRA with `peft==0.15.2`, `device_map=cpu`, and WITHOUT** the distributed env vars (RANK set → `mixed DTensor` error). `peft 0.19` breaks against `transformers 4.51` (`ALL_PARALLEL_STYLES`).
- Two Juno login nodes round-robin (`juno-l-01/-02`) with node-local `/tmp`; keep shared state on `/work`.
- Eval protocols live in `scripts/eval_gsm8k_vllm_offline.py` (`--strict`/`--plain`) and `math_task/eval_math.py`.

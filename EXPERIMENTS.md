# Experiment Registry

專案管理入口。每個實驗一列;成績記在 [`results.csv`](results.csv);詳細分析看 [`GRPO_EXPERIMENT_REPORT.md`](GRPO_EXPERIMENT_REPORT.md)。

- **目標**:用 UTD H100 + GB10a/GB10b,靠近 AMD ROCm GRPO 配置(8 generations、vLLM-assisted rollout、~200 step),訓練出在**完整 GSM8K 1319 題**上穩定優於 base 的模型。
- **記錄**:W&B project `grpo-gsm8k-simulation`(每個 run 用下方 run_name;best checkpoint 存成 W&B Artifact)。
- **備份**:code / configs / scripts / reports / `results.csv` → GitHub;模型權重 → W&B Artifacts。
- **PM**:每個實驗開一個 GitHub Issue,label 標 `status:planned|running|done`,把 run_name 與 W&B 連結貼在 issue 裡。

## Hardware Topology (AMD-style)

複刻 AMD「1 GPU 專做 generation,其餘做 training」:

```text
H100 (Juno)      -> GRPO training (TRL GRPOTrainer + LoRA, --use_vllm true)
GB10a            -> trl vllm-serve rollout server for the main run
GB10b            -> 第二個平行 run 的 rollout server / 或 500 題 eval server
```

關鍵解鎖:GB10 必須用 `trl vllm-serve`(暴露 `/get_tensor_parallel_size/`、`/generate/`、weight-sync),
不能用普通 `vllm serve`(那個只有 OpenAI endpoint,TRL 會 404)。用 `scripts/check_trl_vllm_server.py` 驗證。

## Current Best

| 項目 | 值 |
|---|---|
| Checkpoint | `outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-150step` |
| Best same-protocol (GB10 vLLM chat, 500) | **68.0%** (340/500),base 54.6% → **+13.4pt** |
| Best local HF (500) | 54.8% (274/500),base 45.4% → +9.4pt |
| 最佳步數區間 | 140–150 steps(拉到 200 反而略降) |
| 尚未驗證 | 完整 1319 題 test |

## Experiment Matrix

status: `planned` / `running` / `done`。成績欄填 exact_match@samples,細節進 `results.csv`。

| ID | Config | num_gen | steps | lora_r | rollout | machine | status | best result | Issue |
|---|---|---:|---:|---:|---|---|---|---|---|
| E01 | 4gen-r32-strong-150 (current best) | 4 | 150 | 32 | none | H100 | done | 68.0%@500 (vllm) / 54.8%@500 (hf) | — |
| E02 | amd-8gen-vllm-r32-200 | 8 | 200 | 32 | GB10a | H100 | planned | — | — |
| E03 | amd-8gen-vllm-r32-200-stable (lr1e-6,beta0.06) | 8 | 200 | 32 | GB10b | H100 | planned | — | — |
| E04 | full-test eval of current best (1319) | — | — | 32 | GB10 | eval | planned | — | — |

> E02/E03 是本階段主線:8 generations + vLLM rollout,最接近 AMD。E03 用報告 §9.2 的穩定版超參,
> 目的是解決「100 題高、500 題回落」的不穩問題。

## Roadmap / Milestones

- [ ] **M0 Bootstrap**:SSH key 裝到 gb10a/gb10b、`gh auth login`、確認 gb10b 資訊。
- [ ] **M1 vLLM rollout 通**:GB10 起 `trl vllm-serve`,`check_trl_vllm_server.py` preflight passed。
- [ ] **M2 AMD-aligned 訓練跑通**:E02 完整跑完 200 step 不 timeout,W&B 有完整 reward/kl 曲線。
- [ ] **M3 穩定超越 base**:候選模型在 **500 題同協議** 穩定 > base。
- [ ] **M4 完整驗證**:最佳模型跑完整 **1319 題**,坐實最終數字。
- [ ] **M5 收尾**:best 模型存 W&B Artifact,`results.csv` / report 更新,GitHub 備份。

## How to log a result

```bash
# 訓練/eval 完成後,把 log 裡的 exact_match 收進 results.csv
bash scripts/collect_results.sh          # grep logs/*.out -> 追加到 results.csv
git add results.csv EXPERIMENTS.md && git commit -m "results: <run_name> <score>" && git push
```

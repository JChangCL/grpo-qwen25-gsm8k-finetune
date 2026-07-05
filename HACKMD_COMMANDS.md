# GRPO on UTD Juno 常用指令筆記

> Project: Qwen2.5-1.5B-Instruct + GSM8K + TRL GRPO  
> Repo: `grpo-qwen25-gsm8k-finetune`  
> Juno path: `/work/dal717586/grpo-qwen25-gsm8k-finetune`  
> W&B project: `grpo-gsm8k-simulation`

[TOC]

---

## 0. 目前重要結果

| 類型 | 結果 |
|---|---:|
| 早期 Colab baseline | `46/100` |
| 早期 Juno 8gen checkpoint | `49/100` |
| 4gen complete 100step | `51/100` |
| 舊 strong reward 最佳 | `54/100` |
| Base model 500 題 | `45.4% (227/500)` |
| r32 strong checkpoint-140 500 題 | `54.2% (271/500)` |
| 目前最佳：r32 strong 150step 500 題 | **`54.8% (274/500)`** |
| r32 strong 150step 100 題 | `61/100` |

目前最佳模型路徑：

```bash
outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-150step
```

---

## 1. SSH 連到 Juno

```bash
ssh dal717586@juno.utdallas.edu
```

如果學校文件使用另一個 hostname，也可以：

```bash
ssh dal717586@juno.hpcre.utdallas.edu
```

進去後切到專案：

```bash
cd /work/dal717586/grpo-qwen25-gsm8k-finetune
```

---

## 2. 更新 GitHub 最新程式

```bash
cd /work/dal717586/grpo-qwen25-gsm8k-finetune
git pull
```

確認有哪些檔案：

```bash
ls -lah
ls -lah scripts
```

---

## 3. W&B 設定

每次重新登入 Juno，如果要送 Slurm job，先確認 `WANDB_API_KEY` 有設好。

```bash
export WANDB_API_KEY='你的_wandb_key'
echo ${#WANDB_API_KEY}
```

如果是 W&B 新版 `wandb_v1_...` key，長度通常會顯示：

```text
86
```

送 job 時一定用：

```bash
sbatch --export=ALL scripts/xxx.slurm
```

檢查某個 job 是否有連到 W&B：

```bash
grep -i "wandb:" logs/grpo-r32-200-211832.err | head -30
```

正常會看到類似：

```text
wandb: Currently logged in as ...
wandb: Tracking run ...
wandb: View run at ...
```

> 注意：不要把真的 W&B key 貼到公開筆記或 GitHub。之前貼過的 key 建議去 W&B rotate。

---

## 4. 查看 Queue / Job 狀態

看自己所有正在跑或排隊的 job：

```bash
squeue -u $USER
```

比較清楚的格式：

```bash
squeue -u $USER -o "%.18i %.10P %.24j %.8u %.2t %.12M %.6D %R"
```

看預估開始時間：

```bash
squeue --start -u $USER
```

看特定 job：

```bash
squeue -j 211832,211833
squeue --start -j 211832,211833
```

看某個 job 詳細狀態：

```bash
scontrol show job 211832 | egrep "JobName|JobState|Reason|Priority|StartTime|TimeLimit|ReqTRES|StdOut|StdErr"
```

看完成/失敗紀錄：

```bash
sacct -j 211832,211833 --format=JobID,JobName,State,ExitCode,Elapsed
```

看 H100 整體隊列：

```bash
squeue -p h100 -o "%.18i %.10P %.24j %.8u %.2t %.12M %.6D %R"
```

看 H100 pending job 數量：

```bash
squeue -p h100 -t PD -h | wc -l
```

---

## 5. 取消 Job

取消單一 job：

```bash
scancel 211832
```

取消多個 job：

```bash
scancel 211832 211833
```

取消自己所有 job 要非常小心：

```bash
scancel -u $USER
```

---

## 6. 常用訓練 Job

### 6.1 目前最佳主線：r32 strong 150step

這個版本目前 eval `61/100`。

```bash
sbatch --export=ALL scripts/utd_grpo_h100_4gen_r32_strong_150.slurm
```

輸出模型：

```bash
outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-150step
```

### 6.2 改進版：r32 strong 200step

目標：看 150 step 繼續跑到 200 是否更好。

```bash
sbatch --export=ALL scripts/utd_grpo_h100_4gen_r32_strong_200.slurm
```

常見 log：

```bash
logs/grpo-r32-200-JOBID.out
logs/grpo-r32-200-JOBID.err
```

### 6.3 改進版：r16 strong completion 192

目標：看比較長回答長度 `max_completion_length=192` 是否更適合推理。

```bash
sbatch --export=ALL scripts/utd_grpo_h100_4gen_r16_strong_192_150.slurm
```

常見 log：

```bash
logs/grpo-r16-c192-JOBID.out
logs/grpo-r16-c192-JOBID.err
```

### 6.4 穩定版：4gen complete 100step

目標：保守跑完一次，確認流程完整。

```bash
sbatch --export=ALL scripts/utd_grpo_h100_4gen_complete_100.slurm
```

---

## 7. 查看訓練 Log

即時看 stdout：

```bash
tail -f logs/grpo-r32-200-211832.out
```

即時看 stderr：

```bash
tail -f logs/grpo-r32-200-211832.err
```

看最後 80 行：

```bash
tail -80 logs/grpo-r32-200-211832.out
tail -80 logs/grpo-r32-200-211832.err
```

快速抓錯誤：

```bash
grep -iE "error|traceback|failed|cuda|torch|no such|oom|out of memory|valueerror|runtimeerror" \
  logs/grpo-r32-200-211832.err logs/grpo-r32-200-211832.out
```

查 W&B 連線：

```bash
grep -i "wandb:" logs/grpo-r32-200-211832.err | head -30
```

查是否使用 GPU：

```bash
grep -iE "torch:|torch cuda|cuda available|gpu" logs/grpo-r32-200-211832.out
```

---

## 8. 評估模型 Eval

### 8.1 評估目前最佳模型，100 題

```bash
sbatch --export=ALL,MODEL_PATH=outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-150step,MAX_SAMPLES=100 scripts/utd_eval_checkpoint.slurm
```

### 8.2 評估目前最佳模型，500 題

```bash
sbatch --time=03:00:00 --export=ALL,MODEL_PATH=outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-150step,MAX_SAMPLES=500 scripts/utd_eval_checkpoint.slurm
```

### 8.3 評估原始 base model baseline

```bash
sbatch --time=03:00:00 --export=ALL,MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct,MAX_SAMPLES=500 scripts/utd_eval_checkpoint.slurm
```

### 8.4 評估某個 checkpoint

```bash
sbatch --export=ALL,MODEL_PATH=outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-150step/checkpoint-120,MAX_SAMPLES=100 scripts/utd_eval_checkpoint.slurm
```

### 8.5 查所有 eval 結果

```bash
grep -R "exact_match" logs/eval-grpo-*.out
```

查特定 eval：

```bash
cat logs/eval-grpo-211812.out
tail -80 logs/eval-grpo-211812.err
```

---

## 9. 找輸出模型與 Checkpoint

列出 outputs：

```bash
ls -lah outputs
```

找 checkpoint：

```bash
find outputs -maxdepth 3 -type d | grep checkpoint
```

看資料夾大小：

```bash
du -sh outputs/*
```

看某個模型內容：

```bash
ls -lah outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-150step
```

---

## 10. 結果判讀

### exact_match

`exact_match` 是目前最重要的指標。

| exact_match | 解讀 |
|---:|---|
| `46/100` | 早期 baseline 等級 |
| `50/100` | 有跑，但不一定有效 |
| `54/100` | 有初步提升 |
| `61/100` | 目前明確成功 |
| `65/100+` | 下一個值得追的目標 |

### train/grad_norm

健康狀況：

- 大多在 `0.2 - 1.0` 震盪：通常可以接受
- 偶爾 spike 到 `1.5 - 2.0`：可以接受
- 持續往上爆，或超過 `3 - 5` 很頻繁：可能太激進
- 接近 `0` 很久：可能沒在學

判斷時不要只看 `grad_norm`，要一起看：

```text
reward 上升 + kl 不爆 + grad_norm 不炸 + exact_match 變好
```

### train/kl

`kl` 太高代表模型偏離 base model 太多。

一般判斷：

- 很小且穩定：保守
- 慢慢上升：正常
- 突然爆高：可能不穩或 reward 太強

---

## 11. 常見錯誤

### 11.1 W&B api_key not configured

錯誤：

```text
wandb: ERROR api_key not configured (no-tty)
```

原因：送 Slurm job 時沒有帶到 `WANDB_API_KEY`。

解法：

```bash
export WANDB_API_KEY='你的_wandb_key'
echo ${#WANDB_API_KEY}
sbatch --export=ALL scripts/utd_grpo_h100_4gen_r32_strong_200.slurm
```

### 11.2 Job 是 PD，所以沒有 log

如果 job 還在 pending，log 可能還沒產生。

```bash
squeue -j JOBID
squeue --start -j JOBID
```

### 11.3 FAILED 但不知道原因

```bash
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed

grep -iE "error|traceback|failed|cuda|torch|no such|oom|out of memory|valueerror|runtimeerror" \
  logs/*JOBID*.err logs/*JOBID*.out
```

### 11.4 CUDA / Torch 版本問題

確認 log 中有：

```text
torch cuda: 12.4
cuda available: True
```

如果 eval 在 login node 上跑，可能會變 CPU 或記憶體爆掉。建議 eval 也用 Slurm：

```bash
sbatch --export=ALL,MODEL_PATH=模型路徑,MAX_SAMPLES=100 scripts/utd_eval_checkpoint.slurm
```

---

## 12. 備份結果

訓練腳本通常會自動打包：

```bash
outputs/grpo-h100-...-results-JOBID.tar.gz
```

手動打包目前最佳模型：

```bash
tar -czf outputs/best-r32-strong-150step-backup.tar.gz \
  outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-150step \
  logs
```

從 Juno 下載到 Mac：

```bash
scp dal717586@juno.utdallas.edu:/work/dal717586/grpo-qwen25-gsm8k-finetune/outputs/best-r32-strong-150step-backup.tar.gz .
```

---

## 13. 建議下一步

- 目前正式比較以 `500` 題為主，不再只看 `100` 題。
- Base model `45.4%`，目前最佳 GRPO `54.8%`，提升 `+9.4 points`。
- `checkpoint-140` 已評估：`54.2% (271/500)`；final/checkpoint-150 仍是目前最佳：`54.8% (274/500)`。
- `r32 200step` 與 `r16 c192` 都沒有超過目前最佳，因此主線仍保留 `r32 strong 150step`。
- 之後再試 `learning_rate=1e-6`，或在新 run 中用 `--save_total_limit 10` 保留更多中間 checkpoint。

---

## 14. 快速指令總表

```bash
# 進專案
cd /work/dal717586/grpo-qwen25-gsm8k-finetune

# 更新 repo
git pull

# W&B
export WANDB_API_KEY='你的_wandb_key'
echo ${#WANDB_API_KEY}

# 看 queue
squeue -u $USER
squeue --start -u $USER

# 查 job
scontrol show job JOBID | egrep "JobName|JobState|Reason|StartTime|StdOut|StdErr"
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed

# 送最佳主線訓練
sbatch --export=ALL scripts/utd_grpo_h100_4gen_r32_strong_150.slurm

# 送改進版
sbatch --export=ALL scripts/utd_grpo_h100_4gen_r32_strong_200.slurm
sbatch --export=ALL scripts/utd_grpo_h100_4gen_r16_strong_192_150.slurm

# eval best
sbatch --export=ALL,MODEL_PATH=outputs/qwen2.5-1.5b-gsm8k-grpo-h100-4gen-r32-strong-150step,MAX_SAMPLES=100 scripts/utd_eval_checkpoint.slurm

# eval baseline
sbatch --export=ALL,MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct,MAX_SAMPLES=500 scripts/utd_eval_checkpoint.slurm

# 看 eval 結果
grep -R "exact_match" logs/eval-grpo-*.out

# 找 checkpoints
find outputs -maxdepth 3 -type d | grep checkpoint

# 查錯
grep -iE "error|traceback|failed|cuda|torch|no such|oom|out of memory|valueerror|runtimeerror" logs/*JOBID*.err logs/*JOBID*.out

# 取消 job
scancel JOBID
```

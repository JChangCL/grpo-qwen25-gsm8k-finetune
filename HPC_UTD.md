# Running on UTD HPC

This guide adapts the Colab GRPO run to UTD HPC with Slurm.

Official UTD references:

- UTD HPC home: https://hpc.utdallas.edu/
- Juno system: https://hpc.utdallas.edu/systems-resources/juno/
- CIRC Slurm submitting workloads guide: https://docs.circ.utdallas.edu/user-guide/intro-to-hpc/submitting-jobs.html
- CIRC connecting guide: https://docs.circ.utdallas.edu/user-guide/intro-to-hpc/connecting.html

## 1. Connect

You must be on the UTD VPN before SSH.

```bash
ssh YOUR_NETID@ganymede.utdallas.edu
```

For Juno, use the hostname/account instructions provided by UTD HPC after your account is approved.

## 2. Clone the repo

```bash
git clone https://github.com/JChangCL/grpo-qwen25-gsm8k-finetune.git
cd grpo-qwen25-gsm8k-finetune
```

## 3. Check GPU partitions

```bash
sinfo
sinfo -o "%P %D %G %m %l"
```

For the Juno output shown on June 3, 2026, useful GPU partitions are:

```text
h100       gpu:nvidia_h100_80gb_hbm3 or gpu:nvidia_h100_nvl
a30        gpu:nvidia_a30
a30-2.12gb MIG slice, 12GB
a30-4.6gb  MIG slice, 6GB
```

Use `h100` for this GRPO run when available. The included script already uses:

```bash
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
```

If the partition changes later, edit:

```bash
nano scripts/utd_grpo_1gpu.slurm
```

Replace:

```bash
#SBATCH --partition=REPLACE_WITH_GPU_PARTITION
```

with the real GPU partition shown by `sinfo`.

## 4. Set secrets

Do not put tokens in the Slurm script. Export them before `sbatch`:

```bash
export WANDB_API_KEY="YOUR_NEW_WANDB_KEY"
export HF_TOKEN="YOUR_HF_READ_TOKEN"
```

If your shell does not export these into Slurm jobs, use:

```bash
sbatch --export=ALL scripts/utd_grpo_1gpu.slurm
```

## 5. Submit

```bash
sbatch --export=ALL scripts/utd_grpo_1gpu.slurm
```

The script creates a local `.venv`, installs the Python dependencies, checks whether PyTorch is available, and installs `torch` if needed. If UTD provides a preferred PyTorch module, edit `scripts/utd_grpo_1gpu.slurm` and load that module before the install section.

The main H100 job uses:

```text
outputs/qwen2.5-1.5b-gsm8k-grpo-hpc-8gen-200step
utd-hpc-qwen25-15b-gsm8k-grpo-8gen-200step
num_generations: 8
max_steps: 200
time limit: 8 hours
```

To also queue a smaller A30 run while the H100 job is pending:

```bash
sbatch --export=ALL scripts/utd_grpo_a30_1gpu.slurm
```

The A30 job uses a separate output directory and W&B run name:

```text
outputs/qwen2.5-1.5b-gsm8k-grpo-a30-100step
utd-a30-qwen25-15b-gsm8k-grpo-100step
```

To queue an H100 experiment with a larger GRPO group:

```bash
sbatch --export=ALL scripts/utd_grpo_h100_8gen.slurm
```

The 8-generation job uses:

```text
outputs/qwen2.5-1.5b-gsm8k-grpo-h100-8gen-100step
utd-h100-qwen25-15b-gsm8k-grpo-8gen-100step
time limit: 3 hours
```

To queue a stronger H100 run intended to better match the AMD-style reward curves:

```bash
sbatch --export=ALL scripts/utd_grpo_h100_8gen_strong_reward.slurm
```

This run uses a larger completion length, slightly higher learning rate, and reward weights:

```text
learning_rate: 2e-6
num_generations: 8
max_completion_length: 256
reward_weights: correctness=2.0, soft_format=1.0, strict_format=2.0, numeric=0.5
```

## Recommended Multi-Run Sweep

For one-H100-at-a-time training, prefer complete 4-generation runs over oversized 8-generation runs that may hit the time limit.

```bash
sbatch --export=ALL scripts/utd_grpo_h100_4gen_safe.slurm
sbatch --export=ALL scripts/utd_grpo_h100_4gen_strong_reward.slurm
sbatch --export=ALL scripts/utd_grpo_h100_4gen_format_heavy.slurm
sbatch --export=ALL scripts/utd_grpo_h100_4gen_long_strong.slurm
```

Suggested interpretation:

```text
4gen_safe: clean baseline that should finish.
4gen_strong_reward: best first bet for improvement.
4gen_format_heavy: checks whether strict XML formatting can be learned.
4gen_long_strong: longer run after the strong-reward setting looks promising.
```

If you want one expensive long-running 8-generation job, submit only one of these:

```bash
sbatch --export=ALL scripts/utd_grpo_h100_8gen_2day_strong.slurm
```

This requests the full H100 partition wall-time limit:

```text
time limit: 2 days
num_generations: 8
max_steps: 200
max_completion_length: 256
reward_weights: 2.0 1.0 2.0 0.5
save_steps: 10
```

Monitor:

```bash
squeue -u $USER
tail -f logs/grpo-qwen15b-JOBID.out
tail -f logs/grpo-qwen15b-JOBID.err
```

Cancel if needed:

```bash
scancel JOBID
```

## 6. Retrieve results

The job writes checkpoints and a tarball under:

```text
outputs/
logs/
outputs/grpo-hpc-results-JOBID.tar.gz
```

Copy results back to your local machine with:

```bash
scp YOUR_NETID@ganymede.utdallas.edu:/path/to/grpo-qwen25-gsm8k-finetune/outputs/grpo-hpc-results-JOBID.tar.gz .
```

Or move them to the HPC storage location recommended by your PI/group.

#!/usr/bin/env python3
"""Summarize aggressive MATH GRPO runs from their trainer_state.json log_history.

Prints a table of mean(last-N) for the metrics the plan tracks. Run on juno:
    .venv-colocate/bin/python math_task/summarize_runs.py
"""
import glob
import json

RUNS = {
    "A": "math-aggr-A-b02-lr1e5-400s-c1024",
    "B": "math-aggr-B-b01-lr1e5-500s-c1024",
    "C": "math-aggr-C-b01-lr2e5-500s-c1024",
    "D": "math-aggr-D-b01-lr1e5-500s-c1536",
}
N = 20


def latest_state(run_name: str) -> str | None:
    base = f"outputs/qwen2.5-1.5b-math-grpo-{run_name}"
    paths = glob.glob(base + "/checkpoint-*/trainer_state.json") + glob.glob(base + "/trainer_state.json")
    if not paths:
        return None
    return max(paths, key=lambda p: int(p.split("checkpoint-")[1].split("/")[0]) if "checkpoint-" in p else 10**9)


def series(hist, key):
    return [e[key] for e in hist if key in e]


def meanN(hist, key):
    s = series(hist, key)
    if not s:
        return float("nan")
    tail = s[-N:]
    return sum(tail) / len(tail)


cols = ["run", "steps", "KL(l20)", "KLmax", "corr", "boxed", "clip", "len", "gnorm", "reward"]
print(" | ".join(f"{c:>9}" for c in cols))
print("-" * (12 * len(cols)))
for tag, rn in RUNS.items():
    ts = latest_state(rn)
    if not ts:
        print(f"{tag:>9}  (no trainer_state.json yet)")
        continue
    h = json.load(open(ts))["log_history"]
    steps = h[-1].get("step", "?")
    kl = series(h, "kl")
    vals = [
        tag, str(steps),
        f"{meanN(h, 'kl'):.4f}", f"{(max(kl) if kl else 0):.2f}",
        f"{meanN(h, 'rewards/correctness_reward/mean'):.3f}",
        f"{meanN(h, 'rewards/boxed_format_reward/mean'):.2f}",
        f"{meanN(h, 'completions/clipped_ratio'):.3f}",
        f"{meanN(h, 'completions/mean_length'):.1f}",
        f"{meanN(h, 'grad_norm'):.2f}",
        f"{meanN(h, 'reward'):.3f}",
    ]
    print(" | ".join(f"{v:>9}" for v in vals))
print(f"\n(mean over last {N} logged steps; base MATH-500 = 55.6%, MATH v2 KL=0.0014 corr=0.625)")

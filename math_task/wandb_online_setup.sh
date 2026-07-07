#!/bin/bash
# ============================================================================
# W&B online setup + offline-run sync (Juno).
#
# The API key is READ FROM THE ENVIRONMENT (WANDB_API_KEY) — it is never
# hardcoded here or committed to git. Run it on juno (which has .venv-colocate
# and the offline wandb/ dirs). Pipe the key via stdin so it stays out of argv
# and shell history:
#
#   printf '%s\n' "<YOUR_WANDB_KEY>" | ssh juno 'IFS= read -r K; \
#     cd /work/dal717586/grpo-qwen25-gsm8k-finetune && \
#     WANDB_API_KEY=$K bash math_task/wandb_online_setup.sh --sync'
#
#   --sync   after login, push every wandb/offline-run-* to the W&B cloud.
#            (Safe to re-run: partial runs upload what exists; re-run when the
#             jobs finish to upload the rest.)
# ============================================================================
set -euo pipefail
: "${WANDB_API_KEY:?export WANDB_API_KEY (do not hardcode it in this file).}"

WB="${WB:-.venv-colocate/bin/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-grpo-gsm8k-simulation}"

# 1) Persist credentials to ~/.netrc so future runs authenticate automatically.
"$WB" login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 \
  && echo "wandb: logged in — ~/.netrc updated (persists across jobs)"

# 2) Stash the key privately so ONLINE sbatch submits can source it (the sbatch
#    guard `: \${WANDB_API_KEY:?}` needs the env var present at submit time).
umask 077
printf 'export WANDB_API_KEY=%s\n' "$WANDB_API_KEY" > "$HOME/.wandb_key"
echo "wandb: key stashed at ~/.wandb_key (600). For a future ONLINE run:"
echo "       source ~/.wandb_key && sbatch --export=ALL math_task/sbatch_math_aggressive_X.sh"

# 3) Optionally push offline runs to the cloud.
if [ "${1:-}" = "--sync" ]; then
  shopt -s nullglob
  runs=(wandb/offline-run-*)
  if [ ${#runs[@]} -eq 0 ]; then
    echo "wandb: no wandb/offline-run-* dirs to sync yet"
  else
    echo "wandb: syncing ${#runs[@]} offline run(s) to project '$WANDB_PROJECT'"
    for r in "${runs[@]}"; do
      echo ">>> sync $r"
      "$WB" sync --project "$WANDB_PROJECT" "$r" || echo "    (sync failed for $r — retry after the job finishes)"
    done
  fi
fi
echo "wandb: done."

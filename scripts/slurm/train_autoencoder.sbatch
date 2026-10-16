#!/bin/bash
# 4-GPU DDP autoencoder training on one JURECA dc-gpu-devel node.
# (JURECA account is "jinm16"; devel QOS: max 2 running jobs/user, 2 h wall limit.)
#
#   sbatch scripts/slurm/train_autoencoder.sbatch                                   # smoke test, then new run
#   RESUME=runs/<stamp>_ae_union sbatch scripts/slurm/train_autoencoder.sbatch      # continue a run
#
# Environment knobs: RUN_NAME, TRAIN_ARGS (extra args for the training script),
# TIME_LIMIT_MIN (default 108: leave ~12 min for the final checkpoint), SKIP_SMOKE=1,
# WANDB_MODE=offline|online (default: auto-detect internet on the node).
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=ae_union
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

TIME_LIMIT_MIN=${TIME_LIMIT_MIN:-108}
PY=.venv/bin/python
TORCHRUN=".venv/bin/torchrun --standalone --nproc_per_node=4"
export OMP_NUM_THREADS=4

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
nvidia-smi -L
$PY - <<'P'
import socket, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
for h in ("api.wandb.ai", "pypi.org"):
    try:
        socket.create_connection((h, 443), timeout=5).close(); print(h, "reachable")
    except OSError as e:
        print(h, "NOT reachable:", e)
P
echo "WANDB_MODE=${WANDB_MODE:-<auto: online if api.wandb.ai reachable, else offline>}"

if [[ -n "${RESUME:-}" ]]; then
    ARGS="--resume ${RESUME}"
else
    if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
        echo "=== smoke test (60 steps, wandb offline) ==="
        $TORCHRUN scripts/07_train_autoencoder.py --name smoke --max-steps 60 --log-every 10 \
            --val-every 30 --sample-every 30 --ckpt-every 30 --val-max-samples 256 \
            --warmup-steps 10 --wandb offline
        echo "=== smoke test passed ==="
    fi
    ARGS="--name ${RUN_NAME:-ae_union}"
fi

echo "=== training ==="
$TORCHRUN scripts/07_train_autoencoder.py ${ARGS} --time-limit-min "${TIME_LIMIT_MIN}" ${TRAIN_ARGS:-}
echo "end=$(date -Is)"

#!/bin/bash
# 4-GPU DDP MAE training on one JURECA dc-gpu-devel node (account jinm16; devel QOS: 2 h wall, 2 running jobs/user).
#
#   DATASET=artbench INIT=imagenet sbatch scripts/slurm/train_mae.sbatch     # smoke test, then a new run
#   DATASET=wikiart  INIT=scratch  EPOCHS=200 sbatch scripts/slurm/train_mae.sbatch
#   RESUME=runs/<stamp>_mae_artbench_imagenet sbatch scripts/slurm/train_mae.sbatch    # continue after the 2 h limit
#
# Environment knobs: DATASET (artbench|wikiart), INIT (imagenet|scratch), EPOCHS, RUN_NAME, TRAIN_ARGS (extra args),
# TIME_LIMIT_MIN (default 108: leaves ~12 min for the final checkpoint), SKIP_SMOKE=1,
# WANDB_MODE=offline|online (default: auto-detect internet on the node; offline runs: `wandb sync runs/<run>/wandb/offline-run-*`).
# After training: sbatch scripts/slurm/mae_embeddings.sbatch (extracts embeddings and runs the explorer).
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=mae_train
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

DATASET=${DATASET:-artbench}
INIT=${INIT:-imagenet}
TIME_LIMIT_MIN=${TIME_LIMIT_MIN:-108}
TORCHRUN=".venv/bin/torchrun --standalone --nproc_per_node=4"
export OMP_NUM_THREADS=4

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} dataset=$DATASET init=$INIT start=$(date -Is)"
nvidia-smi -L
.venv/bin/python - <<'P'
import socket, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
for h in ("api.wandb.ai", "pypi.org"):
    try:
        socket.create_connection((h, 443), timeout=5).close(); print(h, "reachable")
    except OSError as e:
        print(h, "NOT reachable:", e)
P

if [[ -n "${RESUME:-}" ]]; then
    ARGS="--resume ${RESUME}"
else
    if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
        echo "=== smoke test (60 steps, wandb offline) ==="
        $TORCHRUN scripts/12_train_mae.py --dataset "$DATASET" --init "$INIT" --name "smoke_mae_${DATASET}" \
            --max-steps 60 --log-every 10 --val-every 30 --sample-every 30 --ckpt-every 30 \
            --val-max-samples 256 --warmup-steps 10 --wandb offline
        echo "=== smoke test passed ==="
    fi
    ARGS="--dataset $DATASET --init $INIT --epochs ${EPOCHS:-60}"
    [[ -n "${RUN_NAME:-}" ]] && ARGS="$ARGS --name $RUN_NAME"
fi

echo "=== training ==="
$TORCHRUN scripts/12_train_mae.py ${ARGS} --time-limit-min "${TIME_LIMIT_MIN}" ${TRAIN_ARGS:-}
echo "end=$(date -Is)"

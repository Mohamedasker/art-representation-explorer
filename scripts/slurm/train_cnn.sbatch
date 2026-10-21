#!/bin/bash
# 4-GPU DDP supervised ResNet training (style/genre/artist heads) on one JURECA dc-gpu-devel node.
#
#   sbatch scripts/slurm/train_cnn.sbatch                               # smoke tests, then new run
#   RESUME=runs/<stamp>_cnn_resnet50 sbatch scripts/slurm/train_cnn.sbatch
#
# Knobs: RUN_NAME, TRAIN_ARGS (extra args, e.g. "--epochs 40 --pretrained"), TIME_LIMIT_MIN (default 108),
#        SKIP_SMOKE=1, WANDB_MODE.
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=cnn_union
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
TIME_LIMIT_MIN=${TIME_LIMIT_MIN:-108}
TORCHRUN=".venv/bin/torchrun --standalone --nproc_per_node=4"
export OMP_NUM_THREADS=4
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"; nvidia-smi -L

if [[ -n "${RESUME:-}" ]]; then
    ARGS="--resume ${RESUME}"
else
    if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
        echo "=== smoke: CNN (60 steps) ==="
        $TORCHRUN scripts/10_train_cnn.py --name smoke_cnn --max-steps 60 --log-every 10 --val-every 30 \
            --ckpt-every 30 --val-max-samples 512 --warmup-steps 10 --wandb offline
        echo "=== smoke: autoencoder script after refactor (20 steps) ==="
        $TORCHRUN scripts/07_train_autoencoder.py --name smoke_ae --max-steps 20 --log-every 10 --val-every 20 \
            --sample-every 20 --ckpt-every 20 --val-max-samples 256 --warmup-steps 5 --wandb disabled
        echo "=== smoke tests passed ==="
    fi
    ARGS="--name ${RUN_NAME:-cnn_resnet50}"
fi
echo "=== training ==="
$TORCHRUN scripts/10_train_cnn.py ${ARGS} --time-limit-min "${TIME_LIMIT_MIN}" ${TRAIN_ARGS:-}
echo "end=$(date -Is)"

#!/bin/bash
# 4-GPU DDP training of the inversion encoder (scripts/25_train_gan_encoder.py) for a trained StyleGAN2 (generator frozen).
#
#   GAN_RUN=runs/<stamp>_gan_union sbatch scripts/slurm/train_gan_encoder.sbatch      # smoke test, then the run
#   RESUME=runs/<stamp>_gan_enc sbatch scripts/slurm/train_gan_encoder.sbatch         # continue after the time limit
#
# Knobs: GAN_RUN (a run dir with gan_final.pt, else its checkpoints/last.pt is used), EPOCHS (default 5), TRAIN_ARGS,
#        TIME_LIMIT_MIN (default 108), SKIP_SMOKE=1, WANDB_MODE=offline|online.
# After training: GAN_RUN=... ENC_RUN=runs/<stamp>_gan_enc sbatch scripts/slurm/embed_gan.sbatch
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=gan_enc
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
TIME_LIMIT_MIN=${TIME_LIMIT_MIN:-108}
TORCHRUN=".venv/bin/torchrun --standalone --nproc_per_node=4"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"; nvidia-smi -L

if [[ -n "${RESUME:-}" ]]; then
    ARGS="--resume ${RESUME}"
else
    GAN_RUN=${GAN_RUN:?set GAN_RUN=runs/<stamp>_gan_union}
    GAN_CK=$([[ -f "$GAN_RUN/gan_final.pt" ]] && echo "$GAN_RUN/gan_final.pt" || echo "$GAN_RUN/checkpoints/last.pt")
    if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
        echo "=== smoke test (30 steps) ==="
        $TORCHRUN scripts/25_train_gan_encoder.py --gan "$GAN_CK" --name smoke_gan_enc --max-steps 30 --log-every 10 \
            --val-every 15 --val-n 256 --ckpt-every 15 --warmup-steps 5 --wandb offline
        echo "=== smoke test passed ==="
    fi
    ARGS="--gan $GAN_CK --epochs ${EPOCHS:-5}"
fi
echo "=== training ==="
$TORCHRUN scripts/25_train_gan_encoder.py ${ARGS} --time-limit-min "${TIME_LIMIT_MIN}" ${TRAIN_ARGS:-}
echo "end=$(date -Is)"

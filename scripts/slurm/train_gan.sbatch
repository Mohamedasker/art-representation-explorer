#!/bin/bash
# 4-GPU StyleGAN2 training (scripts/20_train_gan.py) on ArtBench-10 + WikiArt train splits jointly, on one JURECA GPU
# node (account jinm16), resumable and self-chaining.
#
#   sbatch scripts/slurm/train_gan.sbatch                           # devel (2 h): smoke test, then a new run (--time-limit)
#   SMOKE_ONLY=1 sbatch scripts/slurm/train_gan.sbatch              # just the 60-step smoke test
#   RESUME=runs/<stamp>_gan_union sbatch scripts/slurm/train_gan.sbatch
#   # long run on dc-gpu (24 h limit), self-resubmits (same run dir, same wandb run) until the target kimg is reached:
#   SKIP_SMOKE=1 TIME_LIMIT_MIN=1410 CHAIN=1 NEXT_TIME=24:00:00 KIMG=20000 \
#       sbatch --partition=dc-gpu --time=24:00:00 scripts/slurm/train_gan.sbatch
#
# Knobs: RUN_NAME (default gan_union), KIMG (target thousand images), TRAIN_ARGS (extra args), TIME_LIMIT_MIN (default 108 =
#        devel limit minus 12 min), SKIP_SMOKE=1, SMOKE_ONLY=1, CHAIN=1 (resubmit with RESUME= when the run stopped at the
#        time limit; NEXT_TIME / NEXT_PARTITION = --time / --partition of that job), WANDB_MODE=offline|online.
#        Offline wandb runs: `wandb sync runs/<run>/wandb/offline-run-*` from the login node.
# After training: sbatch scripts/slurm/embed_gan.sbatch
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=gan_union
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
TIME_LIMIT_MIN=${TIME_LIMIT_MIN:-108}
PY=.venv/bin/python
TORCHRUN=".venv/bin/torchrun --standalone --nproc_per_node=4"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}      # Inception-v3 weights for the FID metric (cached from the login node)
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"; nvidia-smi -L

if [[ -n "${RESUME:-}" ]]; then
    RUN_DIR="${RESUME}"
    ARGS="--resume ${RESUME}"
    [[ -n "${KIMG:-}" ]] && ARGS="$ARGS --kimg $KIMG"
else
    RUN_NAME=${RUN_NAME:-gan_union}
    if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
        echo "=== smoke test (60 iterations, wandb offline) ==="
        $TORCHRUN scripts/20_train_gan.py --name smoke_gan --max-steps 60 --log-every 10 --fid-n 200 --sample-kimg 4 \
            --ckpt-kimg 4 --wandb offline
        echo "=== smoke test passed ==="
        [[ "${SMOKE_ONLY:-0}" == "1" ]] && exit 0
    fi
    ARGS="--name $RUN_NAME --kimg ${KIMG:-20000}"
    RUN_DIR=""
fi

echo "=== training ==="
$TORCHRUN scripts/20_train_gan.py ${ARGS} --time-limit-min "${TIME_LIMIT_MIN}" ${TRAIN_ARGS:-}
echo "end=$(date -Is)"

# find the run dir of a fresh run (newest <stamp>_<RUN_NAME>) to chain on
if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR=$(ls -d runs/*_"${RUN_NAME}" | tail -1)
fi
STATUS=$($PY -c "import json; print(json.load(open('$RUN_DIR/status.json'))['status'])" 2>/dev/null || echo unknown)
echo "run=$RUN_DIR status=$STATUS"
if [[ "${CHAIN:-0}" == "1" && "$STATUS" == "time_limit" ]]; then
    NEXT=$(sbatch --parsable --partition="${NEXT_PARTITION:-${SLURM_JOB_PARTITION:-dc-gpu}}" \
           --time="${NEXT_TIME:-24:00:00}" --job-name="${SLURM_JOB_NAME:-gan_union}" \
           --export=ALL,RESUME="$RUN_DIR",CHAIN=1,TIME_LIMIT_MIN="${NEXT_TIME_LIMIT_MIN:-1410}",NEXT_TIME="${NEXT_TIME:-24:00:00}",NEXT_PARTITION="${NEXT_PARTITION:-}",NEXT_TIME_LIMIT_MIN="${NEXT_TIME_LIMIT_MIN:-1410}",KIMG="${KIMG:-}" \
           scripts/slurm/train_gan.sbatch)
    echo "chained next job $NEXT"
fi

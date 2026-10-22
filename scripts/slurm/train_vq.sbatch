#!/bin/bash
# 4-GPU DDP training of the VQ image tokenizer (scripts/21_train_vq.py) on one JURECA dc-gpu node, resumable and self-chaining.
# Needs the pixel cache of scripts/20_prepare_ar_data.py (data/ar_tokens/pixels128_*.npy, built once on the login node).
#
#   sbatch scripts/slurm/train_vq.sbatch                                  # devel (2 h): smoke test, then a new run
#   RESUME=runs/<stamp>_vq_artbench_wikiart sbatch scripts/slurm/train_vq.sbatch
#   # the real run: normal partition, no smoke test, resubmits itself after the time limit, and when it has finished it
#   # tokenises the datasets (scripts/22_tokenize_ar.py) and submits the GPT training (train_ar.sbatch):
#   SKIP_SMOKE=1 CHAIN=1 NEXT_AR=1 TIME_LIMIT_MIN=1410 NEXT_TIME=24:00:00 sbatch --partition=dc-gpu --time=24:00:00 scripts/slurm/train_vq.sbatch
#
# Environment knobs: EPOCHS, RUN_NAME, TRAIN_ARGS (extra args for scripts/21_train_vq.py), TIME_LIMIT_MIN (default 108 = devel limit
# minus 12 min), SKIP_SMOKE=1, CHAIN=1 (resubmit with RESUME= at the time limit; NEXT_TIME = its --time),
# NEXT_AR=1 (after a finished run: tokenise + submit train_ar.sbatch, which takes AR_EPOCHS / AR_RUN_NAME / AR_TRAIN_ARGS from the environment),
# WANDB_MODE=offline|online.   JURECA: account jinm16; devel QOS: 2 running jobs/user, 2 h wall.
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=ar_vq
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
TIME_LIMIT_MIN=${TIME_LIMIT_MIN:-108}
PY=.venv/bin/python
TORCHRUN=".venv/bin/torchrun --standalone --nproc_per_node=4"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
nvidia-smi -L
echo "WANDB_MODE=${WANDB_MODE:-<auto: online if api.wandb.ai reachable, else offline>}"

if [[ -n "${RESUME:-}" ]]; then
    RUN_DIR="${RESUME}"
    ARGS="--resume ${RESUME}"
else
    RUN_NAME=${RUN_NAME:-vq_artbench_wikiart}
    if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
        echo "=== smoke test (40 steps, wandb offline) ==="
        $TORCHRUN scripts/21_train_vq.py --name smoke_vq --max-steps 40 --log-every 5 --val-every 20 --sample-every 40 \
            --ckpt-every 20 --val-n 256 --warmup-steps 10 --revive-every 10 --wandb offline
        echo "=== smoke test passed ==="
    fi
    ARGS="--name $RUN_NAME --epochs ${EPOCHS:-40}"
    RUN_DIR=""
fi

echo "=== training ==="
$TORCHRUN scripts/21_train_vq.py ${ARGS} --time-limit-min "${TIME_LIMIT_MIN}" ${TRAIN_ARGS:-}
echo "end=$(date -Is)"

if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR=$(ls -d runs/*_"${RUN_NAME}" | tail -1)
fi
STATUS=$($PY -c "import json; print(json.load(open('$RUN_DIR/status.json'))['status'])" 2>/dev/null || echo unknown)
echo "run=$RUN_DIR status=$STATUS"
if [[ "${CHAIN:-0}" == "1" && "$STATUS" == "time_limit" ]]; then
    NEXT=$(sbatch --parsable --partition="${SLURM_JOB_PARTITION:-dc-gpu}" --time="${NEXT_TIME:-24:00:00}" \
           --job-name="${SLURM_JOB_NAME:-ar_vq}" \
           --export=ALL,RESUME="$RUN_DIR",CHAIN=1,TIME_LIMIT_MIN="$TIME_LIMIT_MIN" scripts/slurm/train_vq.sbatch)
    echo "chained next job $NEXT"
elif [[ "$STATUS" == "finished" && "${NEXT_AR:-0}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES=0 $PY scripts/22_tokenize_ar.py --tokenizer "$RUN_DIR/vq_final.pt"
    NEXT=$(sbatch --parsable --partition="${SLURM_JOB_PARTITION:-dc-gpu}" --time="${NEXT_TIME:-24:00:00}" \
           --job-name=ar_gpt --export=ALL,TOKENIZER="$RUN_DIR/vq_final.pt",CHAIN=1,NEXT_EMBED=1,SKIP_SMOKE="${SKIP_SMOKE:-0}",TIME_LIMIT_MIN="$TIME_LIMIT_MIN",NEXT_TIME="${NEXT_TIME:-24:00:00}" \
           scripts/slurm/train_ar.sbatch)
    echo "submitted GPT training $NEXT"
fi

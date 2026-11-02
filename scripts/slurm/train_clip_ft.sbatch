#!/bin/bash
# 4-GPU DDP gentle fine-tuning of CLIP ViT-B/32 on ONE dataset (scripts/28_train_clip_ft.py), resumable and self-chaining.
#
#   DATASET=artbench sbatch scripts/slurm/train_clip_ft.sbatch     # smoke test, then the run (devel, 2 h)
#   DATASET=wikiart  sbatch scripts/slurm/train_clip_ft.sbatch
#   RESUME=runs/<stamp>_clip_ft_artbench sbatch scripts/slurm/train_clip_ft.sbatch
#   # on a normal partition, no smoke test, self-resubmits until the run has finished:
#   DATASET=artbench SKIP_SMOKE=1 TIME_LIMIT_MIN=1410 CHAIN=1 NEXT_TIME=24:00:00 \
#       sbatch --partition=dc-gpu --time=24:00:00 scripts/slurm/train_clip_ft.sbatch
#
# Knobs: DATASET (artbench|wikiart), RUN_NAME (default clip_ft_<dataset>), TRAIN_ARGS (extra args, e.g. "--epochs 3"),
#        TIME_LIMIT_MIN (default 108 = devel limit minus 12 min), SKIP_SMOKE=1, SMOKE_ONLY=1, CHAIN=1 (resubmit with
#        RESUME= when the run stopped at the time limit; NEXT_TIME = --time of that job), WANDB_MODE=offline|online.
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=clip_ft
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
TIME_LIMIT_MIN=${TIME_LIMIT_MIN:-108}
PY=.venv/bin/python
TORCHRUN=".venv/bin/torchrun --standalone --nproc_per_node=4"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"; nvidia-smi -L
export HF_HUB_OFFLINE=1      # compute nodes have no internet: the OpenAI CLIP weights come from the HF cache

if [[ -n "${RESUME:-}" ]]; then
    RUN_DIR="${RESUME}"
    RUN_NAME=$(basename "$RUN_DIR" | cut -d_ -f2-)
    ARGS="--resume ${RESUME}"
else
    DATASET=${DATASET:?set DATASET=artbench or DATASET=wikiart}
    RUN_NAME=${RUN_NAME:-clip_ft_$DATASET}
    if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
        echo "=== smoke: CLIP fine-tune, dataset=$DATASET (40 steps) ==="
        $TORCHRUN scripts/28_train_clip_ft.py --dataset "$DATASET" --name smoke_clip_ft_$DATASET --max-steps 40 \
            --log-every 10 --val-every 20 --ckpt-every 20 --val-max-samples 512 --warmup-steps 10 \
            --generic-max-images 1000 --wandb offline
        echo "=== smoke test passed ==="
        [[ "${SMOKE_ONLY:-0}" == "1" ]] && exit 0
    fi
    ARGS="--dataset $DATASET --name $RUN_NAME"
    RUN_DIR=""
fi

echo "=== training ==="
$TORCHRUN scripts/28_train_clip_ft.py ${ARGS} --time-limit-min "${TIME_LIMIT_MIN}" ${TRAIN_ARGS:-}
echo "end=$(date -Is)"

# find the run dir of a fresh run (newest <stamp>_<RUN_NAME>) to chain on
if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR=$(ls -d runs/*_"${RUN_NAME}" | tail -1)
fi
STATUS=$($PY -c "import json; print(json.load(open('$RUN_DIR/status.json'))['status'])" 2>/dev/null || echo unknown)
echo "run=$RUN_DIR status=$STATUS"
if [[ "${CHAIN:-0}" == "1" && "$STATUS" == "time_limit" ]]; then
    NEXT=$(RESUME="$RUN_DIR" sbatch --parsable --partition="${SLURM_JOB_PARTITION:-dc-gpu}" \
           --time="${NEXT_TIME:-24:00:00}" --job-name="${SLURM_JOB_NAME:-clip_ft}" \
           --export=ALL,RESUME="$RUN_DIR",CHAIN=1,TIME_LIMIT_MIN="$TIME_LIMIT_MIN" scripts/slurm/train_clip_ft.sbatch)
    echo "chained next job $NEXT"
fi

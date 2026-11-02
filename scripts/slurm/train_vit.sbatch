#!/bin/bash
# 4-GPU DDP supervised ViT-B/16 training (style/genre/artist heads, scripts/19_train_vit.py) on one JURECA GPU node,
# resumable and self-chaining.
#
#   INIT=scratch  sbatch scripts/slurm/train_vit.sbatch      # smoke test, then a new run from scratch (devel, 2 h)
#   INIT=imagenet sbatch scripts/slurm/train_vit.sbatch      # ... fine-tuned from the ImageNet-1k weights
#   RESUME=runs/<stamp>_vit_scratch sbatch scripts/slurm/train_vit.sbatch
#   # long run on a normal partition (24 h limit), no smoke test, self-resubmits until the run has finished:
#   INIT=scratch SKIP_SMOKE=1 TIME_LIMIT_MIN=1410 CHAIN=1 NEXT_TIME=24:00:00 \
#       sbatch --partition=dc-gpu --time=24:00:00 scripts/slurm/train_vit.sbatch
#
# Knobs: INIT (scratch|imagenet), RUN_NAME (default vit_scratch / vit_imagenet_ft), TRAIN_ARGS (extra args, e.g.
#        "--epochs 100"), TIME_LIMIT_MIN (default 108 = devel limit minus 12 min), SKIP_SMOKE=1, SMOKE_ONLY=1,
#        CHAIN=1 (resubmit with RESUME= when the run stopped at the time limit; NEXT_TIME = --time of that job,
#        default 24:00:00, use 02:00:00 on devel), WANDB_MODE=offline|online.
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=vit_union
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
TIME_LIMIT_MIN=${TIME_LIMIT_MIN:-108}
PY=.venv/bin/python
TORCHRUN=".venv/bin/torchrun --standalone --nproc_per_node=4"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"; nvidia-smi -L
export HF_HUB_OFFLINE=1      # compute nodes have no internet: the ImageNet weights come from the HF cache

if [[ -n "${RESUME:-}" ]]; then
    RUN_DIR="${RESUME}"
    ARGS="--resume ${RESUME}"
else
    INIT=${INIT:-scratch}
    RUN_NAME=${RUN_NAME:-$([[ $INIT == imagenet ]] && echo vit_imagenet_ft || echo vit_scratch)}
    if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
        echo "=== smoke: ViT, init=$INIT (60 steps) ==="
        $TORCHRUN scripts/19_train_vit.py --init "$INIT" --name smoke_vit_$INIT --max-steps 60 --log-every 10 \
            --val-every 30 --ckpt-every 30 --val-max-samples 512 --warmup-steps 10 --wandb offline
        echo "=== smoke test passed ==="
        [[ "${SMOKE_ONLY:-0}" == "1" ]] && exit 0
    fi
    ARGS="--init $INIT --name $RUN_NAME"
    RUN_DIR=""
fi

echo "=== training ==="
$TORCHRUN scripts/19_train_vit.py ${ARGS} --time-limit-min "${TIME_LIMIT_MIN}" ${TRAIN_ARGS:-}
echo "end=$(date -Is)"

# find the run dir of a fresh run (newest <stamp>_<RUN_NAME>) to chain on
if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR=$(ls -d runs/*_"${RUN_NAME}" | tail -1)
fi
STATUS=$($PY -c "import json; print(json.load(open('$RUN_DIR/status.json'))['status'])" 2>/dev/null || echo unknown)
echo "run=$RUN_DIR status=$STATUS"
if [[ "${CHAIN:-0}" == "1" && "$STATUS" == "time_limit" ]]; then
    NEXT=$(RESUME="$RUN_DIR" sbatch --parsable --partition="${SLURM_JOB_PARTITION:-dc-gpu}" \
           --time="${NEXT_TIME:-24:00:00}" --job-name="${SLURM_JOB_NAME:-vit_union}" \
           --export=ALL,RESUME="$RUN_DIR",CHAIN=1,TIME_LIMIT_MIN="$TIME_LIMIT_MIN" scripts/slurm/train_vit.sbatch)
    echo "chained next job $NEXT"
fi

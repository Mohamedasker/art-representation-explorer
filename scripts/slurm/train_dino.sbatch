#!/bin/bash
# 4-GPU DDP DINO v1 (ViT-B/16, scripts/29_train_dino.py) on ArtBench-10 or WikiArt on one JURECA GPU node, resumable and
# self-chaining.  Needs the 256 px caches (python scripts/29_cache_images.py, run once on the login node) and, for
# INIT=imagenet, pretrained/dino_vitb16_full/dino_vitbase16_pretrain_full_checkpoint.pth (see src/models/dino.py).
#
#   DATASET=artbench INIT=scratch  sbatch scripts/slurm/train_dino.sbatch   # smoke test, then a run (devel, 2 h)
#   DATASET=wikiart  INIT=imagenet sbatch scripts/slurm/train_dino.sbatch
#   RESUME=runs/<stamp>_dino_scratch_artbench sbatch scripts/slurm/train_dino.sbatch
#   # long run on a normal partition, self-resubmits until the run has finished:
#   DATASET=artbench INIT=scratch TRAIN_ARGS="--epochs 200" TIME_LIMIT_MIN=700 CHAIN=1 NEXT_TIME=12:00:00 \
#       sbatch --partition=dc-gpu --time=12:00:00 scripts/slurm/train_dino.sbatch
#
# Knobs: DATASET (artbench|wikiart), INIT (scratch|imagenet), RUN_NAME (default dino_scratch_<ds> / dino_adapt_<ds>),
#        TRAIN_ARGS (extra args, e.g. "--epochs 200 --probe-every 5"), TIME_LIMIT_MIN (default 108 = devel limit minus 12 min),
#        SKIP_SMOKE=1, SMOKE_ONLY=1, CHAIN=1 (resubmit with RESUME= when the run stopped at the time limit; NEXT_TIME = --time
#        of that job), WANDB_MODE=offline|online.
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=dino
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
TIME_LIMIT_MIN=${TIME_LIMIT_MIN:-108}
PY=.venv/bin/python
TORCHRUN=".venv/bin/torchrun --standalone --nproc_per_node=4"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"; nvidia-smi -L

if [[ -n "${RESUME:-}" ]]; then
    RUN_DIR="${RESUME}"
    ARGS="--resume ${RESUME}"
else
    DATASET=${DATASET:-artbench}
    INIT=${INIT:-scratch}
    RUN_NAME=${RUN_NAME:-dino_$([[ $INIT == imagenet ]] && echo adapt || echo scratch)_${DATASET}}
    if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
        echo "=== smoke: DINO ${INIT} on ${DATASET} (40 steps) ==="
        $TORCHRUN scripts/29_train_dino.py --dataset "$DATASET" --init "$INIT" --name smoke_${RUN_NAME} --max-steps 40 \
            --epochs 1 --log-every 5 --ckpt-every 20 --warmup-epochs 0.05 --probe-every 0 --probe-train-n 1024 \
            --probe-test-n 512 --wandb offline
        echo "=== smoke test passed ==="
        [[ "${SMOKE_ONLY:-0}" == "1" ]] && exit 0
    fi
    ARGS="--dataset $DATASET --init $INIT --name $RUN_NAME"
    RUN_DIR=""
fi

echo "=== training ==="
$TORCHRUN scripts/29_train_dino.py ${ARGS} --time-limit-min "${TIME_LIMIT_MIN}" ${TRAIN_ARGS:-}
echo "end=$(date -Is)"

# find the run dir of a fresh run (newest <stamp>_<RUN_NAME>) to chain on
if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR=$(ls -d runs/*_"${RUN_NAME}" | tail -1)
fi
STATUS=$($PY -c "import json; print(json.load(open('$RUN_DIR/status.json'))['status'])" 2>/dev/null || echo unknown)
echo "run=$RUN_DIR status=$STATUS"
if [[ "${CHAIN:-0}" == "1" && "$STATUS" == "time_limit" ]]; then
    NEXT=$(RESUME="$RUN_DIR" sbatch --parsable --partition="${SLURM_JOB_PARTITION:-dc-gpu}" \
           --time="${NEXT_TIME:-12:00:00}" --job-name="${SLURM_JOB_NAME:-dino}" \
           --export=ALL,RESUME="$RUN_DIR",CHAIN=1,TIME_LIMIT_MIN="$TIME_LIMIT_MIN",NEXT_TIME="${NEXT_TIME:-12:00:00}" \
           scripts/slurm/train_dino.sbatch)
    echo "chained next job $NEXT"
fi

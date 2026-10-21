#!/bin/bash
# 4-GPU DDP SimCLR training on one JURECA GPU node, resumable and self-chaining.
#
#   DATASET=artbench AUG_SET=baseline sbatch scripts/slurm/train_simclr.sbatch                 # smoke test, then a new run (devel, 2 h)
#   RESUME=runs/<stamp>_simclr_artbench_baseline sbatch scripts/slurm/train_simclr.sbatch      # continue a run
#   # long runs on a normal partition (24 h limit), no smoke test, self-resubmits until the run has finished:
#   DATASET=wikiart AUG_SET=color_heavy SKIP_SMOKE=1 TIME_LIMIT_MIN=1410 CHAIN=1 NEXT_TIME=24:00:00 \
#       sbatch --partition=dc-gpu --time=24:00:00 scripts/slurm/train_simclr.sbatch
#
# Environment knobs: DATASET (artbench|wikiart), AUG_SET, RUN_NAME, TRAIN_ARGS (extra args for scripts/13_train_simclr.py),
# TIME_LIMIT_MIN (default 108 = devel limit minus 12 min; use limit-12 min for other partitions), SKIP_SMOKE=1,
# CHAIN=1 (resubmit this script with RESUME= when the run stopped at the time limit; NEXT_TIME = --time of that next job, default 24:00:00, must be 02:00:00 on devel), WANDB_MODE=offline|online.
# JURECA account is "jinm16"; devel QOS: 2 running jobs/user, 2 h wall limit.
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=simclr
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

TIME_LIMIT_MIN=${TIME_LIMIT_MIN:-108}
PY=.venv/bin/python
TORCHRUN=".venv/bin/torchrun --standalone --nproc_per_node=4"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
nvidia-smi -L
echo "WANDB_MODE=${WANDB_MODE:-<auto: online if api.wandb.ai reachable, else offline>}"

if [[ -n "${RESUME:-}" ]]; then
    RUN_DIR="${RESUME}"
    ARGS="--resume ${RESUME}"
else
    DATASET=${DATASET:?set DATASET=artbench|wikiart}
    AUG_SET=${AUG_SET:-baseline}
    RUN_NAME=${RUN_NAME:-simclr_${DATASET}_${AUG_SET}}
    if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
        echo "=== smoke test (30 steps, wandb offline) ==="
        $TORCHRUN scripts/13_train_simclr.py --dataset "$DATASET" --aug-set "$AUG_SET" --name smoke_simclr \
            --max-steps 30 --log-every 5 --warmup-epochs 0.2 --probe-train-n 1024 --probe-test-n 512 \
            --ckpt-every 15 --wandb offline --num-workers 16
        echo "=== smoke test passed ==="
    fi
    ARGS="--dataset $DATASET --aug-set $AUG_SET --name $RUN_NAME"
    RUN_DIR=""
fi

echo "=== training ==="
$TORCHRUN scripts/13_train_simclr.py ${ARGS} --time-limit-min "${TIME_LIMIT_MIN}" ${TRAIN_ARGS:-}
echo "end=$(date -Is)"

# find the run dir of a fresh run (newest <stamp>_<RUN_NAME>) to chain on
if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR=$(ls -d runs/*_"${RUN_NAME}" | tail -1)
fi
STATUS=$($PY -c "import json,sys; print(json.load(open('$RUN_DIR/status.json'))['status'])" 2>/dev/null || echo unknown)
echo "run=$RUN_DIR status=$STATUS"
if [[ "${CHAIN:-0}" == "1" && "$STATUS" == "time_limit" ]]; then
    NEXT=$(RESUME="$RUN_DIR" sbatch --parsable --partition="${SLURM_JOB_PARTITION:-dc-gpu}" \
           --time="${NEXT_TIME:-24:00:00}" --job-name="${SLURM_JOB_NAME:-simclr}" \
           --export=ALL,RESUME="$RUN_DIR",CHAIN=1,TIME_LIMIT_MIN="$TIME_LIMIT_MIN" scripts/slurm/train_simclr.sbatch)
    echo "chained next job $NEXT"
fi

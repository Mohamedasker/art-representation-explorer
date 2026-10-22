#!/bin/bash
# 4-GPU DDP training of the autoregressive image transformer (scripts/23_train_ar.py) on one JURECA dc-gpu node, resumable
# and self-chaining.  Needs the VQ tokens of scripts/22_tokenize_ar.py (done automatically when train_vq.sbatch ran with NEXT_AR=1).
#
#   TOKENIZER=runs/<stamp>_vq_artbench_wikiart/vq_final.pt sbatch scripts/slurm/train_ar.sbatch        # devel: smoke test + run
#   RESUME=runs/<stamp>_ar_gpt_artbench_wikiart sbatch scripts/slurm/train_ar.sbatch
#   TOKENIZER=... SKIP_SMOKE=1 CHAIN=1 TIME_LIMIT_MIN=1410 NEXT_TIME=24:00:00 sbatch --partition=dc-gpu --time=24:00:00 scripts/slurm/train_ar.sbatch
#
# Environment knobs: TOKENIZER (new run), RESUME, AR_EPOCHS (default 60), AR_RUN_NAME, AR_TRAIN_ARGS (extra args for scripts/23_train_ar.py), TIME_LIMIT_MIN (default 108), SKIP_SMOKE=1,
# CHAIN=1 (resubmit at the time limit; NEXT_TIME = its --time; a finished run is followed by the layer probe, scripts/24_probe_ar_layers.py),
# WANDB_MODE=offline|online.  Then: AR_RUN=<run> sbatch scripts/slurm/embed_ar.sbatch   Devel QOS: 2 running jobs/user, 2 h wall.
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=ar_gpt
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
TIME_LIMIT_MIN=${TIME_LIMIT_MIN:-108}
PY=.venv/bin/python
TORCHRUN=".venv/bin/torchrun --standalone --nproc_per_node=4"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
nvidia-smi -L

if [[ -n "${RESUME:-}" ]]; then
    RUN_DIR="${RESUME}"
    ARGS="--resume ${RESUME}"
else
    TOKENIZER=${TOKENIZER:?set TOKENIZER=runs/<vq run>/vq_final.pt}
    RUN_NAME=${AR_RUN_NAME:-ar_gpt_artbench_wikiart}
    if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
        echo "=== smoke test (30 steps, small model, wandb offline) ==="
        $TORCHRUN scripts/23_train_ar.py --tokenizer "$TOKENIZER" --name smoke_ar --max-steps 30 --log-every 5 --val-every 15 \
            --ckpt-every 15 --warmup-steps 5 --n-layer 2 --d-model 256 --n-head 4 --n-per-style 2 --wandb offline
        echo "=== smoke test passed ==="
    fi
    ARGS="--tokenizer $TOKENIZER --name $RUN_NAME --epochs ${AR_EPOCHS:-60}"
    RUN_DIR=""
fi

echo "=== training ==="
$TORCHRUN scripts/23_train_ar.py ${ARGS} --time-limit-min "${TIME_LIMIT_MIN}" ${AR_TRAIN_ARGS:-}
echo "end=$(date -Is)"

if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR=$(ls -d runs/*_"${RUN_NAME}" | tail -1)
fi
STATUS=$($PY -c "import json; print(json.load(open('$RUN_DIR/status.json'))['status'])" 2>/dev/null || echo unknown)
echo "run=$RUN_DIR status=$STATUS"
if [[ "${CHAIN:-0}" == "1" && "$STATUS" == "time_limit" ]]; then
    NEXT=$(sbatch --parsable --partition="${SLURM_JOB_PARTITION:-dc-gpu}" --time="${NEXT_TIME:-24:00:00}" \
           --job-name="${SLURM_JOB_NAME:-ar_gpt}" \
           --export=ALL,RESUME="$RUN_DIR",CHAIN=1,TIME_LIMIT_MIN="$TIME_LIMIT_MIN",NEXT_TIME="${NEXT_TIME:-24:00:00}" \
           scripts/slurm/train_ar.sbatch)
    echo "chained next job $NEXT"
elif [[ "$STATUS" == "finished" ]]; then
    CUDA_VISIBLE_DEVICES=0 $PY scripts/24_probe_ar_layers.py --checkpoint "$RUN_DIR/checkpoints/best.pt"
    echo "layer probe written; next: AR_RUN=$RUN_DIR sbatch scripts/slurm/embed_ar.sbatch"
fi

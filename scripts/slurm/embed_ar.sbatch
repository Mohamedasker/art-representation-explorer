#!/bin/bash
# Autoregressive-model embeddings for both datasets: train + test split files + merged _all file, then the explorer pages.
#
#   * GPT hidden states (probed layer)     -> representations/<dataset>_ar_gpt_{train,test,all}.npz   (1024-D)
#   * VQ tokenizer latents (4x4-pooled)    -> representations/<dataset>_ar_vq_{train,test,all}.npz    (512-D)
#
#   AR_RUN=runs/<stamp>_ar_gpt_artbench_wikiart VQ_RUN=runs/<stamp>_vq_artbench_wikiart sbatch scripts/slurm/embed_ar.sbatch
#   (the GPT checkpoint is $AR_CKPT, default $AR_RUN/checkpoints/best.pt; the layer comes from $AR_RUN/layer_probe.json
#    (scripts/24_probe_ar_layers.py) or AR_LAYER=N; VQ_RUN defaults to the tokenizer recorded in the GPT run; EXPLORER=0
#    skips the page building)
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=embed_ar
#SBATCH --output=logs/slurm/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
AR_RUN=${AR_RUN:?set AR_RUN=runs/<GPT run>}
AR_CKPT=${AR_CKPT:-$AR_RUN/checkpoints/best.pt}
if [[ -z "${VQ_RUN:-}" ]]; then
    VQ_CKPT=$(.venv/bin/python -c "import json; print(json.load(open('$AR_RUN/config.json'))['info']['tokenizer'])")
else
    VQ_CKPT=$VQ_RUN/vq_final.pt
fi
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is) gpt=$AR_CKPT vq=$VQ_CKPT"
LAYER_ARG=""
[[ -n "${AR_LAYER:-}" ]] && LAYER_ARG="--ar-layer $AR_LAYER"
EXTRACT=".venv/bin/python scripts/02_extract_features.py --split all --batch-size 128 --num-workers 30"

JOBS=()
for ds in artbench wikiart; do
    JOBS+=("--model ar_gpt --checkpoint $AR_CKPT $LAYER_ARG --dataset $ds")
    JOBS+=("--model ar_vq --checkpoint $VQ_CKPT --dataset $ds")
done
PIDS=()
for g in 0 1 2 3; do
    (for ((i = g; i < ${#JOBS[@]}; i += 4)); do
         CUDA_VISIBLE_DEVICES=$g $EXTRACT ${JOBS[$i]} || exit 1
     done) &
    PIDS+=($!)
done
fail=0
for p in "${PIDS[@]}"; do wait "$p" || fail=1; done
echo "extraction done fail=$fail $(date -Is)"

if [[ "${EXPLORER:-1}" == "1" && $fail == 0 ]]; then
    export OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64 NUMBA_NUM_THREADS=64
    for ds in artbench wikiart; do
        .venv/bin/python -u scripts/08_build_explorer.py --skip-existing \
            --repr $(ls representations/${ds}_ar_*_all.npz | tr '\n' ' ') || fail=1
    done
    .venv/bin/python scripts/relink_explorer.py
fi
echo "end=$(date -Is) fail=$fail"
exit $fail

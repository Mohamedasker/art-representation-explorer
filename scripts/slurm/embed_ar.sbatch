#!/bin/bash
# Autoregressive-model embeddings for both datasets: train + test split files + merged _all file, then the explorer pages.
#
#   * iGPT hidden states (probed layer)   -> representations/<dataset>_ar_igpt_{train,test,all}.npz  (512-D)   [IGPT_RUN]
#   * VQ tokens + GPT hidden states        -> representations/<dataset>_ar_gpt_{train,test,all}.npz   (1024-D)  [AR_RUN]
#   * VQ tokenizer latents (4x4-pooled)    -> representations/<dataset>_ar_vq_{train,test,all}.npz    (512-D)   [AR_RUN or VQ_RUN]
#
#   IGPT_RUN=runs/<stamp>_ar_igpt_artbench_wikiart AR_RUN=runs/<stamp>_ar_gpt_artbench_wikiart sbatch scripts/slurm/embed_ar.sbatch
#   (every variable is optional, what is not given is skipped; the checkpoint is $IGPT_CKPT / $AR_CKPT, default
#    <run>/checkpoints/best.pt (lowest validation loss); the layer comes from <run>/layer_probe.json
#    (scripts/24_probe_ar_layers.py) or IGPT_LAYER=N / AR_LAYER=N; the tokenizer is the one recorded in the AR_RUN
#    (or VQ_RUN=<vq run>); EXPLORER=0 skips the page building)
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
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
EXTRACT=".venv/bin/python scripts/02_extract_features.py --split all --batch-size 128 --num-workers 30"
VQ_CKPT=""
if [[ -n "${VQ_RUN:-}" ]]; then
    VQ_CKPT=$VQ_RUN/vq_final.pt
elif [[ -n "${AR_RUN:-}" ]]; then
    VQ_CKPT=$(.venv/bin/python -c "import json; print(json.load(open('$AR_RUN/config.json'))['info']['tokenizer'])")
fi
JOBS=()
for ds in artbench wikiart; do
    if [[ -n "${IGPT_RUN:-}" ]]; then
        JOBS+=("--model ar_igpt --checkpoint ${IGPT_CKPT:-$IGPT_RUN/checkpoints/best.pt} ${IGPT_LAYER:+--ar-layer $IGPT_LAYER} --dataset $ds")
    fi
    if [[ -n "${AR_RUN:-}" ]]; then
        JOBS+=("--model ar_gpt --checkpoint ${AR_CKPT:-$AR_RUN/checkpoints/best.pt} ${AR_LAYER:+--ar-layer $AR_LAYER} --dataset $ds")
    fi
    [[ -n "$VQ_CKPT" ]] && JOBS+=("--model ar_vq --checkpoint $VQ_CKPT --dataset $ds")
done
echo "jobs: ${JOBS[*]}"
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

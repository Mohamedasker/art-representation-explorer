#!/bin/bash
# ViT-B/16 embeddings (final-layer CLS token, 768-D) for both datasets: train + test split files + merged _all file
# (the explorer builds its pages from the _all files), then the explorer pages of these representations.
#
#   * frozen ImageNet ViT                         -> representations/<dataset>_vit_imagenet_{train,test,all}.npz
#   * ViT from scratch  (VIT_SCRATCH=<run dir>)   -> representations/<dataset>_vit_supervised_{train,test,all}.npz
#   * ViT ImageNet-init (VIT_FT=<run dir>)        -> representations/<dataset>_vit_supervised_ft_{train,test,all}.npz
#
#   VIT_SCRATCH=runs/<stamp>_vit_scratch VIT_FT=runs/<stamp>_vit_imagenet_ft sbatch scripts/slurm/embed_vit.sbatch
#   (a run's vit_final.pt is used if present, else its checkpoints/last.pt; weights that are not given are skipped;
#    EXPLORER=0 skips the page building)
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=embed_vit
#SBATCH --output=logs/slurm/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
EXTRACT=".venv/bin/python scripts/02_extract_features.py --split all --batch-size 128 --num-workers 30"
ck() { [[ -f "$1/vit_final.pt" ]] && echo "$1/vit_final.pt" || echo "$1/checkpoints/last.pt"; }

# (model args, dataset) jobs, spread over the 4 GPUs round-robin
JOBS=()
for ds in artbench wikiart; do
    JOBS+=("--model vit_imagenet --dataset $ds")
    [[ -n "${VIT_SCRATCH:-}" ]] && JOBS+=("--model vit_supervised --checkpoint $(ck "$VIT_SCRATCH") --dataset $ds")
    [[ -n "${VIT_FT:-}" ]] && JOBS+=("--model vit_supervised --checkpoint $(ck "$VIT_FT") --dataset $ds")
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
            --repr $(ls representations/${ds}_vit_*_all.npz | tr '\n' ' ') || fail=1
    done
    .venv/bin/python scripts/relink_explorer.py
fi
echo "end=$(date -Is) fail=$fail"
exit $fail

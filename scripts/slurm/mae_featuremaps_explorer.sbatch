#!/bin/bash
# MAE feature maps for the feature-map viewer + explorer pages of the MAE embeddings (same pipeline as the other models).
#   sbatch scripts/slurm/mae_featuremaps_explorer.sbatch
# Env: DATASETS (default "artbench wikiart"), FM_MODELS (default: all MAE models that have a finished run),
#      FM=0 skips the feature maps, EXPLORER=0 skips the explorer pages.
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=mae_explorer
#SBATCH --output=logs/slurm/%x-%j.out

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
export OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64 NUMBA_NUM_THREADS=64
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"

if [[ "${FM:-1}" == "1" ]]; then
    FM_MODELS=${FM_MODELS:-"mae_imagenet mae_adapt_artbench mae_adapt_wikiart mae_scratch_artbench mae_scratch_wikiart"}
    .venv/bin/python -u scripts/11_build_featuremaps.py --models $FM_MODELS || echo "feature maps FAILED"
fi

if [[ "${EXPLORER:-1}" == "1" ]]; then
    .venv/bin/python scripts/merge_splits.py
    for ds in ${DATASETS:-artbench wikiart}; do
        .venv/bin/python -u scripts/08_build_explorer.py --skip-existing \
            --repr $(ls representations/${ds}_mae_*_all.npz | tr '\n' ' ')
    done
    .venv/bin/python scripts/rerender_explorer.py
fi
echo "end=$(date -Is)"

#!/bin/bash
# Extract the supervised CNN's embeddings for ArtBench + WikiArt and rebuild the explorer.
#
#   RUN=runs/<stamp>_cnn_resnet50 sbatch scripts/slurm/cnn_embeddings_explorer.sbatch
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --job-name=cnn_embed
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
RUN=${RUN:?set RUN=runs/<stamp>_<name>}
export OMP_NUM_THREADS=4
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} run=$RUN start=$(date -Is)"
[[ -f "$RUN/cnn_final.pt" ]] || { echo "missing $RUN/cnn_final.pt (training unfinished? resume it first)"; exit 1; }

for ds in artbench wikiart; do
    gpu=$([[ $ds == artbench ]] && echo 0 || echo 1)
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/02_extract_features.py \
        --model cnn_supervised --checkpoint "$RUN/cnn_final.pt" --dataset "$ds" --split train \
        --batch-size 128 --num-workers 24 --out "representations/${ds}_cnn_supervised_train.npz" &
done
wait

export OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64
.venv/bin/python scripts/08_build_explorer.py
echo "end=$(date -Is)"

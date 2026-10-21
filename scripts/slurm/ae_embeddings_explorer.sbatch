#!/bin/bash
# Finish the autoencoder run (if needed), extract its embeddings for ArtBench + WikiArt and
# rebuild the explorer (PCA / UMAP / t-SNE pages, same pipeline as the other representations).
#
#   RUN=runs/<stamp>_ae_union sbatch scripts/slurm/ae_embeddings_explorer.sbatch
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --job-name=ae_embed
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
RUN=${RUN:?set RUN=runs/<stamp>_<name>}
POOL=${POOL:-4}
export OMP_NUM_THREADS=4
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} run=$RUN start=$(date -Is)"

# 1) finish training if it stopped at the time limit (last.pt -> autoencoder_final.pt)
if [[ ! -f "$RUN/autoencoder_final.pt" ]]; then
    .venv/bin/torchrun --standalone --nproc_per_node=4 scripts/07_train_autoencoder.py --resume "$RUN"
fi
CK="$RUN/autoencoder_final.pt"

# 2) embeddings: one GPU per dataset, in parallel
for ds in artbench wikiart; do
    gpu=$([[ $ds == artbench ]] && echo 0 || echo 1)
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/02_extract_features.py \
        --model autoencoder --checkpoint "$CK" --ae-pool "$POOL" --dataset "$ds" --split train \
        --batch-size 128 --num-workers 24 --out "representations/${ds}_autoencoder_train.npz" &
done
wait

# 3) explorer over all representations (cached projections of the old ones are reused)
export OMP_NUM_THREADS=64
.venv/bin/python scripts/08_build_explorer.py
echo "end=$(date -Is)"

#!/bin/bash
# Embeddings of the whole datasets for every model, train and test kept in SEPARATE files:
#   representations/<dataset>_<model>_train.npz and ..._test.npz  (+ manifest.json), then the attribute tables.
# Existing per-split files are reused, so only missing splits are extracted.
# MERGE=1 additionally writes one merged <dataset>_<model>_all.npz per model (train + test; the bundle keeps a
# per-image `split` array, so train and test stay identifiable; scripts/08_build_explorer.py --prefer-all uses it).
#
#   sbatch scripts/slurm/embed_splits.sbatch
#   MERGE=1 AE_CKPT=... CNN_CKPT=... sbatch scripts/slurm/embed_splits.sbatch
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --job-name=embed_splits
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
AE_CKPT=${AE_CKPT:-runs/20261015-124529_ae_union/autoencoder_final.pt}
CNN_CKPT=${CNN_CKPT:-runs/20261015-175907_cnn_resnet50/cnn_final.pt}
export OMP_NUM_THREADS=4
SPLIT_MODE=$([[ "${MERGE:-0}" == "1" ]] && echo all || echo both)
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"

run_dataset() {          # $1 = dataset, $2 = gpu
    local ds=$1 gpu=$2
    export CUDA_VISIBLE_DEVICES=$gpu
    for spec in "clip_vitb32" "dinov2_vitb14" "resnet50" \
                "autoencoder --checkpoint $AE_CKPT" "cnn_supervised --checkpoint $CNN_CKPT"; do
        .venv/bin/python scripts/02_extract_features.py --model $spec --dataset "$ds" --split ${SPLIT_MODE} \
            --batch-size 128 --num-workers 24
    done
}
run_dataset artbench 0 &
run_dataset wikiart 1 &
wait

# attributes per split (CLIP concept scores, metadata, pixel statistics)
export OMP_NUM_THREADS=32
.venv/bin/python -u scripts/09_build_attributes.py --split train test --workers 96
echo "end=$(date -Is)"

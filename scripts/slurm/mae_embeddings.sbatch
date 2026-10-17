#!/bin/bash
# MAE embeddings (mean-pooled encoder patch tokens, unmasked input, 768-D) for both datasets, train and test files:
#   * the frozen ImageNet MAE            -> representations/<dataset>_mae_imagenet_{train,test}.npz
#   * every MAE trained here, on BOTH datasets (own dataset + cross-dataset)
#                                        -> representations/<dataset>_mae_<adapt|scratch>_<train-dataset>_{train,test}.npz
# then the merged *_all.npz files (scripts/merge_splits.py).  Existing per-split files are reused.
#
#   MAE_RUNS="runs/<stamp>_mae_artbench_imagenet runs/<stamp>_mae_wikiart_imagenet" sbatch scripts/slurm/mae_embeddings.sbatch
#   (a run's mae_final.pt is used if present, else its checkpoints/last.pt)
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=mae_embed
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
export OMP_NUM_THREADS=4
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
EXTRACT=".venv/bin/python scripts/02_extract_features.py --split both --batch-size 128 --num-workers 30"

ck() { [[ -f "$1/mae_final.pt" ]] && echo "$1/mae_final.pt" || echo "$1/checkpoints/last.pt"; }

# stream 0/1: frozen ImageNet MAE on one dataset each; streams 2/3 start with the trained MAEs (cross-dataset first)
declare -a PIDS
slot=0
for ds in artbench wikiart; do
    (CUDA_VISIBLE_DEVICES=$slot $EXTRACT --model mae_imagenet --dataset $ds
     for run in ${MAE_RUNS:-}; do        # own-dataset embeddings of every trained MAE
         if [[ $(.venv/bin/python -c "import json,sys;print(json.load(open('$run/config.json'))['info']['train_dataset'])") == $ds ]]; then
             CUDA_VISIBLE_DEVICES=$slot $EXTRACT --model mae --checkpoint "$(ck $run)" --dataset $ds
         fi
     done) &
    PIDS+=($!); slot=$((slot + 1))
done
for ds in artbench wikiart; do
    (for run in ${MAE_RUNS:-}; do        # cross-dataset embeddings
         if [[ $(.venv/bin/python -c "import json,sys;print(json.load(open('$run/config.json'))['info']['train_dataset'])") != $ds ]]; then
             CUDA_VISIBLE_DEVICES=$slot $EXTRACT --model mae --checkpoint "$(ck $run)" --dataset $ds
         fi
     done) &
    PIDS+=($!); slot=$((slot + 1))
done
for p in "${PIDS[@]}"; do wait "$p"; done

.venv/bin/python scripts/merge_splits.py
echo "end=$(date -Is)"

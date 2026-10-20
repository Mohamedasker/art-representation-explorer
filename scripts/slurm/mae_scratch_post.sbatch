#!/bin/bash
# After the from-scratch MAE runs (scripts/slurm/train_mae.sbatch, INIT=scratch): finish them if they stopped at the
# time limit, extract their embeddings on both datasets, build the feature maps and the explorer pages.
# Submit chained to the two training jobs (partition dc-gpu: no 2 h cap, no devel QOS job limit):
#   sbatch --dependency=afterany:<train_job_1>:<train_job_2> scripts/slurm/mae_scratch_post.sbatch
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=05:00:00
#SBATCH --job-name=mae_scratch_post
#SBATCH --output=logs/slurm/%x-%j.out

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
MAE_RUNS=auto bash scripts/slurm/mae_embeddings.sbatch || exit 1
FM_MODELS="mae_scratch_artbench mae_scratch_wikiart" bash scripts/slurm/mae_featuremaps_explorer.sbatch
echo "end=$(date -Is)"

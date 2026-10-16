#!/bin/bash
# Build explorer pages.   ARGS="..." sbatch scripts/slurm/explorer_build.sbatch
#   e.g. ARGS="--extra-sets 1 --full --only default extra"   (10k sample + another 10k, linked to the full pages)
#        ARGS="--extra-sets 1 --full --only full"             (all images; large pages, no t-SNE)
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=explorer
#SBATCH --output=logs/slurm/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
exec scripts/slurm/explorer_sets.sh ${ARGS:-}

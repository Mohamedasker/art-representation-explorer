#!/bin/bash
# Invariance part of scripts/17_compare_simclr.py (GPU) for both datasets; the static part is run on the CPU.
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=compare_simclr
#SBATCH --output=logs/slurm/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
export OMP_NUM_THREADS=4
for ds in artbench wikiart; do
    .venv/bin/python -u scripts/17_compare_simclr.py --dataset $ds --skip-static --out docs/simclr_invariance_$ds
done

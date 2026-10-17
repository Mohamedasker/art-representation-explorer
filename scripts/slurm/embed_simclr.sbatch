#!/bin/bash
# SimCLR embeddings (2048-D pooled encoder output), train + test split files + merged _all file (the explorer builds its pages from the _all files).
#   Frozen ImageNet SimCLR v1 on both datasets (CPU is fine, ~15 min):
#       ENTRIES="imagenet@artbench,wikiart" sbatch --partition=dc-cpu --gres=none scripts/slurm/embed_simclr.sbatch
#   Trained models (GPU): ENTRIES="runs/<run>/simclr_final.pt@artbench runs/<run2>/simclr_final.pt@wikiart"
#       sbatch scripts/slurm/embed_simclr.sbatch
#   (entry = <checkpoint | imagenet>@<dataset>[,<dataset>...]; each entry is extracted for every listed dataset)
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=embed_simclr
#SBATCH --output=logs/slurm/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
ENTRIES=${ENTRIES:?set ENTRIES="imagenet@artbench,wikiart ..."}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
fail=0
for e in $ENTRIES; do
    ck=${e%@*}; dss=${e#*@}
    if [[ "$ck" == "imagenet" ]]; then MARGS="--model simclr_imagenet"; else MARGS="--model simclr --checkpoint $ck"; fi
    for ds in ${dss//,/ }; do
        .venv/bin/python scripts/02_extract_features.py $MARGS --dataset "$ds" --split all \
            --batch-size 128 --num-workers ${NUM_WORKERS:-24} || fail=1
    done
done
echo "end=$(date -Is) fail=$fail"
exit $fail

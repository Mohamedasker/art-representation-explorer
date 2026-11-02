#!/bin/bash
# Embeddings (teacher CLS token, 768-D, L2-normalised) of the DINO v1 models trained with scripts/29_train_dino.py, for both
# datasets: train + test split files + merged _all file, then their explorer pages.
#
#   DINO_RUNS="runs/<stamp>_dino_adapt_artbench runs/<stamp>_dino_adapt_wikiart runs/<stamp>_dino_scratch_artbench \
#              runs/<stamp>_dino_scratch_wikiart" sbatch scripts/slurm/dino_embeddings.sbatch
#   (a run's dino_final.pt is used if present, else its checkpoints/last.pt; EXPLORER=0 skips the page building)
#   -> representations/<dataset>_dino_{adapt,scratch}_<train dataset>_{train,test,all}.npz
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=embed_dino
#SBATCH --output=logs/slurm/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
export OMP_NUM_THREADS=4
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
EXTRACT=".venv/bin/python scripts/02_extract_features.py --split all --batch-size 128 --num-workers 30"
ck() { [[ -f "$1/dino_final.pt" ]] && echo "$1/dino_final.pt" || echo "$1/checkpoints/last.pt"; }

# (model args, dataset) jobs, spread over the 4 GPUs round-robin
JOBS=()
for ds in artbench wikiart; do
    for r in ${DINO_RUNS:?set DINO_RUNS to the run directories}; do
        JOBS+=("--model dino_train --checkpoint $(ck "$r") --dataset $ds")
    done
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
            --repr $(ls representations/${ds}_dino_adapt_*_all.npz representations/${ds}_dino_scratch_*_all.npz | tr '\n' ' ') || fail=1
    done
    .venv/bin/python scripts/relink_explorer.py
fi
echo "end=$(date -Is) fail=$fail"
exit $fail

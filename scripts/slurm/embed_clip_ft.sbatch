#!/bin/bash
# Embeddings of the fine-tuned CLIPs (scripts/28_train_clip_ft.py) for BOTH datasets (train + test split files + merged
# _all file), each as the fine-tuned model (alpha = 1) and as WiSE-FT interpolations with the original image tower, then
# the explorer pages of these representations.  The frozen original (exact model) is extracted too: representations/<dataset>_clip_vitb32_qgelu_*.npz.
#
#   CLIP_FT_ARTBENCH=runs/<stamp>_clip_ft_artbench CLIP_FT_WIKIART=runs/<stamp>_clip_ft_wikiart \
#       sbatch scripts/slurm/embed_clip_ft.sbatch
#   (a run's clip_ft_final.pt is used if present; weights that are not given are skipped; ALPHAS="1.0 0.5" are the
#    interpolation weights of the fine-tuned model (-> clip_ft_<ds>, clip_ft_<ds>_wise50); EXPLORER=0 skips the pages)
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=embed_clip_ft
#SBATCH --output=logs/slurm/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
EXTRACT=".venv/bin/python scripts/02_extract_features.py --split all --batch-size 256 --num-workers 30"
ALPHAS=${ALPHAS:-"1.0 0.5"}

# (checkpoint, alpha, evaluated dataset) jobs, spread over the 4 GPUs round-robin
JOBS=()
for ds in artbench wikiart; do
    JOBS+=("--model clip_vitb32_qgelu --dataset $ds")
done
for run in "${CLIP_FT_ARTBENCH:-}" "${CLIP_FT_WIKIART:-}"; do
    [[ -z "$run" ]] && continue
    for a in $ALPHAS; do
        for ds in artbench wikiart; do
            JOBS+=("--model clip_ft --checkpoint $run/clip_ft_final.pt --clip-alpha $a --dataset $ds")
        done
    done
done
[[ ${#JOBS[@]} == 0 ]] && { echo "nothing to do: set CLIP_FT_ARTBENCH and/or CLIP_FT_WIKIART"; exit 1; }
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

if [[ $fail == 0 ]]; then
    # fit vs. forgetting table (zero-shot style / CIFAR / kNN, near-duplicates of the other dataset excluded)
    EV=()
    [[ -n "${CLIP_FT_ARTBENCH:-}" ]] && EV+=(--run "artbench=$CLIP_FT_ARTBENCH")
    [[ -n "${CLIP_FT_WIKIART:-}" ]] && EV+=(--run "wikiart=$CLIP_FT_WIKIART")
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/29_eval_clip_ft.py "${EV[@]}" --alphas $ALPHAS || fail=1
fi

if [[ "${EXPLORER:-1}" == "1" && $fail == 0 ]]; then
    export OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64 NUMBA_NUM_THREADS=64
    for ds in artbench wikiart; do
        .venv/bin/python -u scripts/08_build_explorer.py --skip-existing \
            --repr representations/${ds}_clip_vitb32_qgelu_all.npz $(ls representations/${ds}_clip_ft_*_all.npz | tr '\n' ' ') || fail=1
    done
    .venv/bin/python scripts/relink_explorer.py
fi
echo "end=$(date -Is) fail=$fail"
exit $fail

#!/bin/bash
# Embeddings of the ResNet-50s trained on ONE dataset (scripts/10_train_cnn.py --sources artbench|wikiart) for both
# datasets (train + test + merged _all file), then their explorer pages.
#   representations/<dataset>_cnn_supervised_<artbench|wikiart>_{train,test,all}.npz
#
#   AB_RUN=runs/<stamp>_cnn_resnet50_artbench WA_RUN=runs/<stamp>_cnn_resnet50_wikiart sbatch scripts/slurm/embed_cnn_sources.sbatch
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=cnn_src_embed
#SBATCH --output=logs/slurm/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
AB_RUN=${AB_RUN:?set AB_RUN=runs/<stamp>_cnn_resnet50_artbench}
WA_RUN=${WA_RUN:?set WA_RUN=runs/<stamp>_cnn_resnet50_wikiart}
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
for r in "$AB_RUN" "$WA_RUN"; do [[ -f "$r/cnn_final.pt" ]] || { echo "missing $r/cnn_final.pt"; exit 1; }; done
EXTRACT=".venv/bin/python scripts/02_extract_features.py --model cnn_supervised --split all --batch-size 128 --num-workers 30"

JOBS=()
for ds in artbench wikiart; do
    JOBS+=("--checkpoint $AB_RUN/cnn_final.pt --dataset $ds" "--checkpoint $WA_RUN/cnn_final.pt --dataset $ds")
done
PIDS=()
for g in 0 1 2 3; do
    (CUDA_VISIBLE_DEVICES=$g $EXTRACT ${JOBS[$g]} || exit 1) &
    PIDS+=($!)
done
fail=0
for p in "${PIDS[@]}"; do wait "$p" || fail=1; done
echo "extraction done fail=$fail $(date -Is)"

if [[ $fail == 0 ]]; then
    export OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64 NUMBA_NUM_THREADS=64
    for ds in artbench wikiart; do
        .venv/bin/python -u scripts/08_build_explorer.py --skip-existing \
            --repr representations/${ds}_cnn_supervised_artbench_all.npz representations/${ds}_cnn_supervised_wikiart_all.npz || fail=1
    done
    .venv/bin/python scripts/relink_explorer.py
fi
echo "end=$(date -Is) fail=$fail"
exit $fail

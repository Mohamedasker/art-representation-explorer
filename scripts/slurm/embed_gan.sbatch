#!/bin/bash
# StyleGAN2 embeddings of the WHOLE datasets (train + test split files + merged _all file), then the explorer pages:
#   * discriminator features (512-D)       GAN_RUN=<run dir>   -> representations/<dataset>_gan_disc_{train,test,all}.npz
#   * inversion-encoder latent w (512-D)   ENC_RUN=<run dir>   -> representations/<dataset>_gan_enc_{train,test,all}.npz
#
#   GAN_RUN=runs/<stamp>_gan_union ENC_RUN=runs/<stamp>_gan_enc sbatch scripts/slurm/embed_gan.sbatch
#   (a run's gan_final.pt / gan_enc_final.pt is used if present, else its checkpoints/last.pt; weights that are not given
#    are skipped; EXPLORER=0 skips the page building)
#SBATCH --account=jinm16
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=embed_gan
#SBATCH --output=logs/slurm/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
EXTRACT=".venv/bin/python scripts/02_extract_features.py --split all --batch-size 128 --num-workers 30"
ck() { [[ -f "$1/$2" ]] && echo "$1/$2" || echo "$1/checkpoints/last.pt"; }

JOBS=()
for ds in artbench wikiart; do
    [[ -n "${GAN_RUN:-}" ]] && JOBS+=("--model gan_disc --checkpoint $(ck "$GAN_RUN" gan_final.pt) --dataset $ds")
    [[ -n "${ENC_RUN:-}" ]] && JOBS+=("--model gan_enc --checkpoint $(ck "$ENC_RUN" gan_enc_final.pt) --dataset $ds")
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
            --repr $(ls representations/${ds}_gan_*_all.npz | tr '\n' ' ') || fail=1
    done
    .venv/bin/python scripts/relink_explorer.py
fi
echo "end=$(date -Is) fail=$fail"
exit $fail

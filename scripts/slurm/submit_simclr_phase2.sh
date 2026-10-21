#!/bin/bash
# Queue the SimCLR augmentation-variant runs (7 sets x ArtBench, WikiArt; baseline already done), 200 epochs each, on dc-gpu.
# Refuses to run while other jobs of this user (e.g. the MAE session's mae_*) are queued/running, unless FORCE=1.
#   scripts/slurm/submit_simclr_phase2.sh            # all
#   SETS="color_heavy rotation" DATASETS=artbench scripts/slurm/submit_simclr_phase2.sh
cd "$(dirname "$0")/../.."
SETS=${SETS:-"color_heavy edge_texture rotation geometry_heavy no_color blur_heavy crop_only"}
DATASETS=${DATASETS:-"artbench wikiart"}
if [[ "${FORCE:-0}" != "1" ]] && squeue -u "$USER" -h -o "%j" | grep -qv '^simclr'; then
    echo "other jobs of $USER are queued/running:"; squeue -u "$USER" -o "%i %j %T"; echo "wait for them or set FORCE=1"; exit 1
fi
for ds in $DATASETS; do
    if [[ $ds == artbench ]]; then T=02:30:00; M=138; else T=04:00:00; M=225; fi
    for aug in $SETS; do
        if ls -d runs/*_simclr_${ds}_${aug} >/dev/null 2>&1; then echo "skip $ds/$aug (run dir exists)"; continue; fi
        j=$(DATASET=$ds AUG_SET=$aug TRAIN_ARGS="--epochs 200" CHAIN=1 NEXT_TIME=$T TIME_LIMIT_MIN=$M \
            sbatch --parsable --partition=dc-gpu --time=$T --job-name=simclr_${ds:0:2}_${aug} scripts/slurm/train_simclr.sbatch)
        echo "submitted $ds/$aug job $j"
    done
done

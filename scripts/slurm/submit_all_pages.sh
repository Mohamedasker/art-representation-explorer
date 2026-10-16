#!/bin/bash
# Submit the build of the merged train + test explorer pages (retries until the per-user job limit allows it).
cd /p/project1/cjinm16/asker/art-representation-explorer
ARGS="--with-all --repr $(ls representations/*_all.npz | tr '\n' ' ') --extra-sets 1 --only default extra --skip-existing"
until J=$(ARGS="$ARGS" sbatch --parsable scripts/slurm/explorer_build.sbatch 2>/dev/null); do sleep 60; done
echo "submitted $J"

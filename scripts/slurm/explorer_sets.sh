#!/bin/bash
# Build the extra sample sets / full-dataset explorer pages.  Usage: explorer_sets.sh [08_build_explorer.py args...]
cd /p/project1/cjinm16/asker/art-representation-explorer
export OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64 NUMBA_NUM_THREADS=64
.venv/bin/python -u scripts/08_build_explorer.py "$@"

#!/bin/bash
cd /p/project1/cjinm16/asker/art-representation-explorer
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
.venv/bin/python -u scripts/09_build_attributes.py --workers 48 || exit 1
.venv/bin/python -u scripts/08_build_explorer.py

#!/bin/bash
# Keeps the explorer builds going until every page exists: submits whatever is missing whenever a job slot is
# free, then re-renders the landing page.  Prints one line per step; the last line is "ALL DONE" (or "GAVE UP").
cd /p/project1/cjinm16/asker/art-representation-explorer
PY=.venv/bin/python
ALL_REPRS=$(ls representations/*_all.npz 2>/dev/null | tr '\n' ' ')
rounds=0
while true; do
    miss=$($PY scripts/check_explorer_pages.py)
    nq=$(squeue -u "$USER" -h -n explorer -o %i | wc -l)
    if [[ -z "$miss" && "$nq" -eq 0 ]]; then
        $PY scripts/rerender_explorer.py > /dev/null 2>&1
        echo "ALL DONE: every explorer page is built, landing page refreshed"
        exit 0
    fi
    # submit missing kinds, one job per kind, only if no job of the explorer family is queued for it
    if [[ -n "$miss" && "$nq" -lt 2 && $rounds -lt 12 ]]; then
        for kind in $miss; do
            case $kind in
                per_split) A="--extra-sets 1 --full --only default extra --skip-existing" ;;
                full)      A="--extra-sets 1 --full --only full --skip-existing" ;;
                all)       A="--with-all --repr $ALL_REPRS --extra-sets 1 --only default extra --skip-existing" ;;
            esac
            if J=$(ARGS="$A" sbatch --parsable scripts/slurm/explorer_build.sbatch 2>/dev/null); then
                echo "submitted $kind build as job $J (missing: $miss)"; rounds=$((rounds+1)); nq=$((nq+1))
                [[ "$nq" -ge 2 ]] && break
            fi
        done
    fi
    if [[ $rounds -ge 12 && "$nq" -eq 0 ]]; then echo "GAVE UP: pages still missing ($miss) after $rounds submissions"; exit 1; fi
    sleep 120
done

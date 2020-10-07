#!/usr/bin/env bash

# Reformats imports and source code so that you don't have to

set -xe

SRC_FILES=(src/ tests/ setup.py)

# sometimes we need a couple of runs to get to a setting that all the tools are
# happy with
n_runs=2
for run in $(seq 1 $n_runs); do
    echo "Reformatting source code (run $run/$n_runs)"
    yapf -ir ${SRC_FILES[@]}
    echo "Sorting imports (repeat $run/$n_runs)"
    isort ${SRC_FILES[@]}
    echo "Removing unused imports (run $run/$n_runs)"
    autoflake --in-place --expand-star-imports --remove-all-unused-imports -r ${SRC_FILES[@]}
done

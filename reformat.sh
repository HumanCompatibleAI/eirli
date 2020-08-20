#!/usr/bin/env bash

# Reformats imports and source code so that you don't have to

set -xe

SRC_FILES=(src/ tests/ setup.py)

echo "Sorting imports"
isort -r ${SRC_FILES[@]}
echo "Removing unused imports"
autoflake --in-place --expand-star-imports --remove-all-unused-imports -r ${SRC_FILES[@]}
echo "Reformatting source code"
yapf -ir ${SRC_FILES[@]}

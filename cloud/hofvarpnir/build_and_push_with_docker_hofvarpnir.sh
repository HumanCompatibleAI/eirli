#!/bin/bash

set -euo pipefail

# get dir containing this bash script (fall back to cwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd || pwd)"

# script is $SCRIPT_DIR/../build_and_push_docker.sh (we get canonical path here)
path_to_build_and_push_docker="$(realpath "$SCRIPT_DIR/../build_and_push_docker.sh")"

# now exec that script under bash with -u 10002 -n sam -d humancompatibleai -p
exec bash "$path_to_build_and_push_docker" -u 10002 -n sam -d humancompatibleai -p
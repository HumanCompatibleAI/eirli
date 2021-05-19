#!/bin/bash

# Simple test of new joint training script on MAGICAL.

set -e

ray_address="localhost:42000"
ray_ncpus=1
ray_ngpus=0.25
base_cfgs=("n_batches=30000" "env_use_magical")
repl_configs=("repl_noid" "repl_vae" "repl_fd" "repl_id")
env_names=("MatchRegions" "MoveToCorner" "MoveToRegion")
nseeds=3

submit_expt() {
    # submit joint training experiment to local Ray server
    python -m il_representations.scripts.joint_training_cluster \
        --ray-address="$ray_address" --ray-ncpus "$ray_ncpus" \
        --ray-ngpus "$ray_ngpus" "${base_cfgs[@]}" "$@" &
}

lower() {
    # convert string to lowercase
    echo "$1" | tr "[:upper:]" "[:lower:]"
}

for repl_config in "${repl_configs[@]}"; do
    for env_name in "${env_names[@]}"; do
        lower_env="$(lower "$env_name")"
        # launch $nseeds duplicates
        for _ in $(seq "$nseeds"); do
            # using test variant demos
            # (we don't explicitly seed; leave that to Sacred)
            submit_expt exp_ident="${repl_config}_test_variant_repl" \
                "${repl_config[@]}" "env_cfg.task_name=${env_name}-Demo-v0" \
                "repl_data_${lower_env}_demos_test"
        done
    done
done
wait

#!/bin/bash

# Another test variant baseline that trains for 100k epochs.

set -e

ray_address="localhost:42000"
ray_ncpus=1
ray_ngpus=0.2
base_cfgs=("n_batches=100000" "env_use_magical"
           "bc.short_eval_interval=2000")
env_names=("FixColour" "FindDupe" "ClusterColour"
           "MakeLine" "ClusterShape" "MatchRegions"
           "MoveToCorner" "MoveToRegion")

submit_expt() {
    python -m il_representations.scripts.joint_training_cluster \
        --ray-address="$ray_address" --ray-ncpus "$ray_ncpus" \
        --ray-ngpus "$ray_ngpus" "${base_cfgs[@]}" "$@" &
}

lower() {
    # convert string to lowercase
    echo "$1" | tr "[:upper:]" "[:lower:]"
}

launch_seed() {
    for env_name in "${env_names[@]}"; do
        # "oracle" BC baselines that get access to test data
        submit_expt exp_ident="repl_noid_test_variant_cheating_100k" \
            "${repl_config[@]}" "env_cfg.task_name=${env_name}-Demo-v0" \
            "bc_data_${lower_env}_demos_test" \
            "repl_data_${lower_env}_demos_test"
    done
}

# Ray was dying when I tried launching all jobs at once :(
launch_seed
launch_seed
wait
launch_seed
launch_seed
wait
launch_seed
wait

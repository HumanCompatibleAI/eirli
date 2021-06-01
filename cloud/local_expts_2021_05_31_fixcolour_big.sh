#!/bin/bash


# Testing on FixColour with the huge new TestAll dataset that I generated. All
# runs do 100k batches of optimisation. The specific algo configs are:
#
# - Standard noid config as a control.
# - A noid config that gets access to test data for BC (!).
# - An id config that gets access to test data for repL, but not BC.
#
# Aim is just to get signs of life with the bigger dataset.

set -e

ray_address="localhost:42000"
ray_ncpus=1
ray_ngpus=0.24
# 100k batches this time (large # of batches seems necessary)
base_cfgs=("n_batches=100000" "env_use_magical" "bc.short_eval_interval=5000")
repl_configs=("repl_noid" "repl_id")
env_names=("FixColour")

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

launch_seed() {
    for repl_config in "${repl_configs[@]}"; do
        for env_name in "${env_names[@]}"; do
            lower_env="$(lower "$env_name")"
            # using test variant demos
            # (we don't explicitly seed; leave that to Sacred)
            submit_expt exp_ident="${repl_config}_test_variant_repl" \
                "${repl_config[@]}" "env_cfg.task_name=${env_name}-Demo-v0" \
                "repl_data_${lower_env}_demos_test"
        done

        # "oracle" BC baselines that get access to test data
        submit_expt exp_ident="repl_noid_test_variant_cheating" \
            "${repl_config[@]}" "env_cfg.task_name=${env_name}-Demo-v0" \
            "bc_data_${lower_env}_demos_test" \
            "repl_data_${lower_env}_demos_test"
    done
}

launch_seed
launch_seed
launch_seed
launch_seed
launch_seed
wait

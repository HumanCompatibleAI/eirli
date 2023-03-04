#!/bin/bash

# Additional joint training experiments:
# - Baselines that get access to test variant data
# - CPC runs
# Both on all MAGICAL tasks.

set -e

ray_address="localhost:42000"
ray_ncpus=1
base_cfgs=("n_batches=30000" "env_use_magical" "bc.short_eval_interval=2000")
repl_configs=("repl_noid" "repl_tcpc8")
env_names=("FixColour" "FindDupe" "ClusterColour" "MakeLine" "ClusterShape" "MatchRegions" "MoveToCorner" "MoveToRegion")

submit_expt_pt3() {
    # submit joint training experiment to local Ray server (0.3 GPUs)
    python -m il_representations.scripts.joint_training_cluster \
        --ray-address="$ray_address" --ray-ncpus "$ray_ncpus" \
        --ray-ngpus 0.3 "${base_cfgs[@]}" "$@" &
}

submit_expt_pt24() {
    # submit joint training experiment to local Ray server (0.24 GPUs)
    python -m il_representations.scripts.joint_training_cluster \
        --ray-address="$ray_address" --ray-ncpus "$ray_ncpus" \
        --ray-ngpus 0.24 "${base_cfgs[@]}" "$@" &
}

lower() {
    # convert string to lowercase
    echo "$1" | tr "[:upper:]" "[:lower:]"
}

launch_seed() {
    for env_name in "${env_names[@]}"; do
        for repl_config in "${repl_configs[@]}"; do
            lower_env="$(lower "$env_name")"
            # using test variant demos
            # (we don't explicitly seed; leave that to Sacred)
            submit_expt_pt3 exp_ident="${repl_config}_test_variant_repl" \
                "${repl_config[@]}" "env_cfg.task_name=${env_name}-Demo-v0" \
                "repl_data_${lower_env}_demos_test"
        done

        # "oracle" BC baselines that get access to test data
        submit_expt_pt24 exp_ident="repl_noid_test_variant_cheating" \
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

#!/bin/bash

# TODO(sam):
# - **DONE** get right set of repL algos
# - Make sure repL algos have the right batch size.
# - Add both -augs and -noaugs BC baselines
# - Make submit_expt use the right GPU fraction

set -e

ray_address="localhost:42000"
ray_ncpus=1
nseeds=5
base_cfgs=("n_batches=30000" "env_use_magical" "bc.short_eval_interval=2000")
repl_configs=("repl_noid" "repl_id" "repl_vae" "repl_fd" "repl_simclr" "repl_tcpc8")
env_names=("MatchRegions" "MoveToCorner" "MoveToRegion")
gpu_default=0.15
declare -A gpu_overrides=(
    ["repl_tcpc8"]="0.3"
    ["repl_simclr"]="0.3"
)
gpu_config() {
    # figures out GPU configuration string based on repL config, if any
    repl_config="$1"
    override="${gpu_overrides[$repl_config]}"
    if [ -z "$override" ]; then
        override="$gpu_default"
    fi
    echo "tune_run_kwargs.resources_per_trial.gpu=$override"
}
submit_expt() {
    # submit joint training experiment to local Ray server (0.3 GPUs)
    # FIXME(sam): --ray-ngpus arg is wrong; should come from gpu_config()
    python -m il_representations.scripts.joint_training_cluster \
        --ray-address="$ray_address" --ray-ncpus "$ray_ncpus" \
        --ray-ngpus 0.3 --nseeds "$nseeds" "${base_cfgs[@]}" "$@" &
}

lower() {
    # convert string to lowercase
    echo "$1" | tr "[:upper:]" "[:lower:]"
}

launch() {
    for env_name in "${env_names[@]}"; do
        for repl_config in "${repl_configs[@]}"; do
            lower_env="$(lower "$env_name")"
            # using test variant demos
            # (we don't explicitly seed; leave that to Sacred)
            submit_expt_pt3 exp_ident="${repl_config}_test_variant_repl" \
                "${repl_config[@]}" "env_cfg.task_name=${env_name}-Demo-v0" \
                "repl_data_${lower_env}_demos_test"
        done
    done
}

launch

#!/bin/bash

# Joint train runs for NeurIPS benchmarks track:
# - Everything uses the same augmentations (augs_neurips_repl_bc), except the
#   no-augs control.
# - Algos: control (noid), no-augs control (noid_noaugs), ID, FD, VAE, SimCLR,
#   TCPC-8.
# - IL data: 5 demos
# - repL data: 5 demos + all random rollouts
# - Batch size of 64 for all repL algos except the contrastive ones, which are
#   384 due to defaults.

# TODO(sam):
# - **DONE** get right set of repL algos
# - **DONE** Make sure repL algos have the right batch size.
# - **DONE** Make sure all methods use the same augs.
# - Make sure everything is using just 5 demos.
# - **DONE** Add both -augs and -noaugs BC baselines
# - **DONE** Make submit_expt use the right GPU fraction

set -e

ray_address="localhost:42000"
ray_ncpus=1
nseeds=5
base_cfgs=("n_batches=30000" "env_use_magical" "bc.short_eval_interval=2000"
           "augs_neurips_repl_bc" "bc_data_5demos" "repl_data_5demos_random")
repl_configs=("repl_noid" "repl_noid_noaugs" "repl_id"
              "repl_vae" "repl_fd" "repl_simclr" "repl_tcpc8")
env_names=("MatchRegions" "MoveToCorner" "MoveToRegion")
gpu_default=0.15
declare -A gpu_overrides=(
    ["repl_tcpc8"]="0.3"
    ["repl_simclr"]="0.3"
)
gpu_config() {
    # figures out GPU configuration string based on repL config, if any
    override="$gpu_default"
    for cfg_string in "$@"; do
        this_override="${gpu_overrides[$cfg_string]}"
        if [ ! -z "$this_override" ]; then
            override="$this_override"
        fi
    done
    echo "tune_run_kwargs.resources_per_trial.gpu=$override"
}
submit_expt() {
    # submit joint training experiment to local Ray server
    n_gpus="$(gpu_config "$@")"
    python -m il_representations.scripts.joint_training_cluster \
        --ray-address="$ray_address" --ray-ncpus "$ray_ncpus" \
        --ray-ngpus "$n_gpus" --nseeds "$nseeds" "${base_cfgs[@]}" "$@" &
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
            submit_expt exp_ident="${repl_config}_test_variant_repl" \
                "${repl_config[@]}" "env_cfg.task_name=${env_name}-Demo-v0" \
                "repl_data_${lower_env}_demos_test"
        done
    done
}

launch

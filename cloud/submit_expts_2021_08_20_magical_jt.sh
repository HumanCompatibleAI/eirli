#!/bin/bash

# Joint train runs for NeurIPS benchmarks track:
# - Everything uses the same augmentations (augs_neurips_repl_bc), except the
#   no-augs control.
# - Algos: control (noid), no-augs control (noid_noaugs), ID, FD, VAE, SimCLR,
#   TCPC-8.
# - IL data: 5 demos
# - repL data: 5 demos + all random rollouts
# - Batch size of 64 for all repL algos except the contrastive ones, which are
#   384.
# - RepL weight is always 1.0, except in the controls where it is 0.0 (I can't
#   disable repL entirely, so I just zero the weight).

set -e

ray_ncpus=1.0
cluster_cfg_path="./gcp_cluster_sam_new_vis.yaml"
nseeds=5
base_cfgs=("n_batches=30000" "env_use_magical" "bc.short_eval_interval=2000"
           "augs_neurips_repl_bc" "bc_data_5demos" "repl_data_5demos_random")
repl_configs=("repl_simclr" "repl_id" "repl_tcpc8" "repl_vae" "repl_fd"
              "repl_noid" "repl_noid_noaugs")
env_names=("MatchRegions" "MoveToCorner" "MoveToRegion")
gpu_default=0.11
declare -A gpu_overrides=(
    ["repl_tcpc8"]="0.21"
    ["repl_simclr"]="0.21"
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
    echo "$override"
}
submit_expt() {
    # submit joint training experiment to local Ray server
    n_gpus="$(gpu_config "$@")"
    ray submit --tmux "$cluster_cfg_path" ./submit_joint_training_cluster.py \
        --ray-ncpus "$ray_ncpus" --ray-ngpus "$n_gpus" --nseeds "$nseeds" \
        "${base_cfgs[@]}" "$@"
}

launch() {
    for env_name in "${env_names[@]}"; do
        for repl_config in "${repl_configs[@]}"; do
            # 5 demos + random rollouts for repL; just 5 demos for BC
            # (actual data config is in base_cfgs)
            submit_expt exp_ident="${repl_config}_5demos_random" \
                "${repl_config[@]}" "env_cfg.task_name=${env_name}-Demo-v0"
        done
    done
}

launch

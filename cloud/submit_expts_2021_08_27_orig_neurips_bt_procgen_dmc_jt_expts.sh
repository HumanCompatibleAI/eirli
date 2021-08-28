#!/bin/bash

# Joint training Procgen/DMC runs from the NeurIPS benchmarks track submission

set -e

ray_ncpus=1.0
cluster_cfg_path="./gcp_cluster_sam_new_vis.yaml"
nseeds=5
base_cfgs=("model_save_interval=25000" "repl.batch_save_interval=25000" "n_batches=1000000")
dmc_tasks=("cheetah-run" "finger-spin" "reacher-easy")  # omitting walker-walk
procgen_tasks=("jumper" "coinrun" "fruitbot" "miner")
repl_configs=("repl_fd" "repl_id" "repl_simclr" "repl_tcpc8" "repl_vae")
gpu_default=0.11
declare -A gpu_overrides=(
    ["repl_tcpc8"]="0.16"
    ["repl_simclr"]="0.16"
)
gpu_config() {
    # figures out GPU configuration string based on repL config, if any
    override="$gpu_default"
    for cfg_string in "$@"; do  # iterate through configs
        this_override="${gpu_overrides[$cfg_string]}"
        if [ ! -z "$this_override" ]; then
            # if this config has a GPU override, we update GPU config
            # accordingly
            override="$this_override"
        fi
    done
    echo "$override"
}
submit_expt() {
    # submit joint training experiment to Ray autoscaler cluster
    n_gpus="$(gpu_config "$@")"
    ray submit --tmux "$cluster_cfg_path" ./submit_joint_training_cluster.py \
        --ray-ncpus "$ray_ncpus" --ray-ngpus "$n_gpus" --nseeds "$nseeds" \
        "${base_cfgs[@]}" "$@"
}

launch() {
    for repl_config in "${repl_configs[@]}"; do
        # always demos + random rollouts for repL, hence the exp_ident
        # (see base_cfgs for actual dataset config)
        submit_expt exp_ident="${repl_config}_demos_random" \
            "${repl_config[@]}" "$@"
    done
}

for procgen_task in "${procgen_tasks[@]}"; do
    launch "env_cfg.benchmark_name=procgen" "env_cfg.task_name=$procgen_task"
done
for dmc_task in "${dmc_tasks[@]}"; do
    launch "env_cfg.benchmark_name=dm_control" "env_cfg.task_name=$dmc_task"
done

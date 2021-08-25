#!/bin/bash

# Procgen/DMC joint train runs for NeurIPS benchmarks track:
# - Everything uses the same augmentations (augs_neurips_repl_bc), except the
#   no-augs control.
# - Algos: control (noid), no-augs control (noid_noaugs), ID, FD, VAE, SimCLR,
#   TCPC-8.
# - IL data: however many demos are on disk.
# - repL data: same demos + all available random rollouts too.
# - Batch size of 64 for all repL algos except the contrastive ones, which are
#   192.
# - RepL weight is always 1.0, except in the controls where it is 0.0 (I can't
#   disable repL entirely, so I just zero the weight).

set -e

ray_ncpus=1.0
cluster_cfg_path="./gcp_cluster_sam_new_vis.yaml"
nseeds=5
n_batches=500000  # hopefully finishes fast enough
base_cfgs=("n_batches=$n_batches" "bc.short_eval_interval=20000"
           "augs_neurips_repl_bc" "repl_data_demos_random")
repl_configs=("repl_simclr_192" "repl_tcpc8_192" "repl_id" "repl_vae" "repl_fd"
              "repl_noid" "repl_noid_noaugs")
gpu_default=0.11
declare -A gpu_overrides=(
    ["repl_tcpc8"]="0.16"
    ["repl_simclr"]="0.16"
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
    for repl_config in "${repl_configs[@]}"; do
        # always demos + random rollouts for repL, hence the exp_ident
        # (see base_cfgs for actual dataset config)
        submit_expt exp_ident="${repl_config}_demos_random" \
            "${repl_config[@]}" "$@"
    done
}

for procgen_task in jumper coinrun fruitbot miner; do
    launch "env_cfg.benchmark_name=procgen" "env_cfg.task_name=$procgen_task"
done
for dmc_task in finger-spin cheetah-run reacher-easy; do
    launch "env_cfg.benchmark_name=dm_control" "env_cfg.task_name=$dmc_task"
done

#!/bin/bash

# MAGICAL joint train runs for NeurIPS benchmarks track:
# - Everything uses the same augmentations (augs_neurips_repl_bc), except the
#   no-augs control.
# - Algos: control (noid), no-augs control (noid_noaugs), ID, FD, VAE, SimCLR,
#   TCPC-8.
# - IL data: 5 demos for MAGICAL, however many demos are on disk for the rest.
# - repL data: same demos + all available random rollouts too.
# - Batch size of 64 for all repL algos except the contrastive ones, which are
#   192.
# - RepL weight is always 1.0, except in the controls where it is 0.0 (I can't
#   disable repL entirely, so I just zero the weight).

set -e

ray_ncpus=1.0
ray_address="localhost:42000"
nseeds=5
magical_base_cfgs=(
    "n_batches=30000" "env_use_magical" "bc.short_eval_interval=2000"
    "augs_neurips_repl_bc" "bc_data_5demos" "repl_data_5demos_random"
    "disable_extra_saves_and_eval")
magical_env_names=("MatchRegions" "MoveToCorner" "MoveToRegion")
repl_configs=(
    "repl_simclr_192" "repl_tcpc8_192" "repl_id" "repl_vae" "repl_fd"
    "repl_noid" "repl_noid_noaugs")
gpu_default=0.2
declare -A gpu_overrides=(
    ["repl_tcpc8_192"]="0.3"
    ["repl_simclr_192"]="0.3"
)
gpu_config() {
    # figures out GPU configuration string based on repL config, if any
    override="$gpu_default"
    for cfg_string in "$@"; do
        this_override="${gpu_overrides[$cfg_string]}"
        if [ -n "$this_override" ]; then
            override="$this_override"
        fi
    done
    echo "$override"
}

submit_expt() {
    # submit joint training experiment to local Ray server
    n_gpus="$(gpu_config "$@")"
    python -m il_representations.scripts.joint_training_cluster \
        --ray-address="$ray_address" --ray-ncpus "$ray_ncpus" \
        --ray-ngpus "$n_gpus" --nseeds "$nseeds" "$@" &
}

magical_launch_all() {
    for env_name in "${magical_env_names[@]}"; do
        for repl_config in "${repl_configs[@]}"; do
            # 5 demos + random rollouts for repL; just 5 demos for BC
            # (actual data config is in base_cfgs)
            submit_expt "${magical_base_cfgs[@]}" \
                exp_ident="${repl_config}_5demos_random" "${repl_config[@]}" \
                "env_cfg.task_name=${env_name}-Demo-v0"
        done
    done
}

magical_launch_all
wait

#!/bin/bash

# SimCLR ablations for rebuttal.

set -e

# WARNING: 5 demos here!
base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5"
           "cfg_data_il_5demos")
cluster_cfg_path="./gcp_cluster_sam_new_vis.yaml"
declare -a repl_configs=("cfg_repl_simclr" "cfg_repl_simclr_asymm_proj"
                         "cfg_repl_simclr_no_proj" "cfg_repl_simclr_ceb_loss"
                         "cfg_repl_simclr_momentum")
gpu_default=0.2
declare -A gpu_overrides=()
gpu_config() {
    # figures out GPU configuration string based on repL config, if any
    repl_config="$1"
    override="${gpu_overrides[$repl_config]}"
    if [ -z "$override" ]; then
        override="$gpu_default"
    fi
    echo "tune_run_kwargs.resources_per_trial.gpu=$override"
}
declare -a magical_envs=("MatchRegions-Demo-v0" "MoveToRegion-Demo-v0" "MoveToCorner-Demo-v0")
declare -a mg_dataset_configs=("cfg_data_repl_5demos_random")

submit_expt() {
    # submit experiment to cluster using given args
    ray submit --tmux "$cluster_cfg_path" \
        ./submit_pretrain_n_adapt.py -- "${base_cfgs[@]}" "$@"
}

for magical_env in "${magical_envs[@]}"; do
    for repl_config in "${repl_configs[@]}"; do
        for mg_dataset_config in "${mg_dataset_configs[@]}"; do
            echo -e "\n *** TRAINING $repl_config ON $magical_env WITH DATASET $mg_dataset_config *** \n "
            submit_expt cfg_repl_5k_il cfg_repl_augs cfg_use_magical \
                "$repl_config" "$(gpu_config "$repl_config")" \
                "$mg_dataset_config" cfg_il_bc_20k_nofreeze \
                exp_ident="neurips_repl_bc_${repl_config}_${mg_dataset_config}" \
                "env_cfg.task_name=$magical_env"
        done
    done
    for use_augs in augs noaugs; do
        submit_expt cfg_repl_none cfg_use_magical cfg_il_bc_20k_nofreeze \
            "$(gpu_config "no_repl")" "cfg_il_bc_${use_augs}" \
            exp_ident="neurips_control_bc_${use_augs}" \
            "env_cfg.task_name=$magical_env"
    done
done

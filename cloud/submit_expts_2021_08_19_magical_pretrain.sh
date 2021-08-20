#!/bin/bash

# What this file does:
# - repL/augs: Dynamics, InvDyn, SimCLR, TemporalCPC, VAE, BC+augs, BC-augs
# - Tasks: MTC, MTR, MR
# - #seeds: 5
# Does only BC, and only demos+random repL data

set -e

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=1")

cluster_cfg_path="./gcp_cluster_sam_new_vis.yaml"

declare -a repl_configs=("icml_inv_dyn" "icml_dynamics" "cfg_repl_tcpc8"
                         "cfg_repl_simclr" "icml_vae")
declare -a magical_envs=("MatchRegions-Demo-v0" "MoveToRegion-Demo-v0" "MoveToCorner-Demo-v0")
declare -a mg_dataset_configs=("cfg_data_repl_demos_random")

submit_expt() {
    # submit experiment to cluster using given args
    ray submit --tmux "$cluster_cfg_path" \
        ./submit_pretrain_n_adapt.py -- "${base_cfgs[@]}" "$@"
}


for magical_env in "${magical_envs[@]}"; do
    for repl_config in "${repl_configs[@]}"; do
        for mg_dataset_config in "${mg_dataset_configs[@]}"; do
            echo -e "\n *** TRAINING $repl_config ON $magical_env WITH DATASET $mg_dataset_config *** \n "
            submit_expt icml_il_on_repl_sweep cfg_repl_augs cfg_use_magical \
                "$repl_config" "$mg_dataset_config" cfg_il_bc_20k_nofreeze  \
                exp_ident="neurips_repl_bc_${repl_config}_${mg_dataset_config}" \
                "env_cfg.task_name=$magical_env"
        done
    done
    for use_augs in augs noaugs; do
        submit_expt icml_control cfg_use_magical  cfg_il_bc_20k_nofreeze \
            "cfg_il_bc_${use_augs}" exp_ident="neurips_control_bc_${use_augs}" \
            "env_cfg.task_name=$magical_env"
    done
done

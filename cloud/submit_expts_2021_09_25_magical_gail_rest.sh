#!/bin/bash

# What this file does:
# - repL/augs: Dynamics, InvDyn, SimCLR, TemporalCPC, VAE, GAIL+augs, GAIL-augs
# - Tasks: MTC, MTR, MR
# - #seeds: 5
# - repL batch size of 64 for all the non-contrastive methods, 384 for
#   contrastive ones.
# - RepL datasets: 5 demos + some large number of random rollouts. (IL uses 5
#   demos too.)
# Does only GAIL, and only demos+random repL data

set -e

# WARNING: 5 demos here!
base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5"
           "cfg_data_il_5demos")
cluster_cfg_path="./gcp_cluster_sam_new_vis.yaml"
# 6 runs per GPU by default
gpu_default=0.3
declare -A gpu_overrides=(
    # 4 runs per GPU for TCPC8/SimCLR
    ["cfg_repl_tcpc8"]="0.3"
    ["cfg_repl_simclr"]="0.3"
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
declare -a mg_dataset_configs=("cfg_data_repl_5demos_random")

# declare -a repl_configs=("icml_inv_dyn" "icml_dynamics" "cfg_repl_tcpc8"
#                          "cfg_repl_simclr" "icml_vae")
# declare -a magical_envs=("MatchRegions-Demo-v0" "MoveToRegion-Demo-v0" "MoveToCorner-Demo-v0")

# we are evaluating on only a subset of cases for the rebuttal; we can do the
# rest later (which will require 4x as much compute)
declare -a repl_configs=("icml_inv_dyn" "icml_dynamics" "cfg_repl_tcpc8" "cfg_repl_simclr" "icml_vae")
declare -a magical_envs=("MatchRegions-Demo-v0" "MoveToRegion-Demo-v0" "MoveToCorner-Demo-v0")

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
                "$mg_dataset_config" "gail_mr_config_2021_03_29" \
                "env_cfg.task_name=$magical_env" \
                exp_ident="neurips_repl_gail_${repl_config}_${mg_dataset_config}"
        done
    done
    for use_augs in augs noaugs; do
        submit_expt cfg_repl_none cfg_use_magical \
            "$(gpu_config "no_repl")" "gail_mr_config_2021_03_29" \
            "gail_disc_${use_augs}" "env_cfg.task_name=$magical_env" \
            exp_ident="neurips_control_gail_${use_augs}"
    done
done

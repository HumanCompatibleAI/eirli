#!/bin/bash

set -e

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu")

cluster_cfg_path="./gcp_cluster_sam.yaml"

declare -a control_configs=("control_no_ortho_init")
declare -a dmc_envs=("finger-spin" "cheetah-run")
declare -a magical_envs=("MatchRegions" "MoveToRegion")

submit_expt() {
    # submit experiment to cluster using given args
    ray submit --tmux "$cluster_cfg_path" \
        ./submit_pretrain_n_adapt.py -- "${base_cfgs[@]}" "$@"
}


for control_config in "${control_configs[@]}"; do
    for magical_env in "${magical_envs[@]}"; do
        echo -e "\n *** TRAINING $control_config ON $magical_env *** \n "
        submit_expt icml_il_on_repl_sweep cfg_use_magical \
                    "$control_config" cfg_il_bc_20k_nofreeze  \
                    stages_to_run=IL_ONLY \
                    exp_ident="actual_${control_config}_${mg_dataset_config}" \
                    "env_cfg.task_name=$magical_env"
    done
    for dmc_env in "${dmc_envs[@]}"; do
        echo -e "\n *** TRAINING $control_config ON $dmc_env *** \n "
        submit_expt icml_il_on_repl_sweep cfg_use_dm_control \
                    "$control_config" cfg_il_bc_200k_nofreeze \
                    stages_to_run=IL_ONLY \
                    exp_ident="actual_${control_config}_${dmc_dataset_config}" \
                    "env_cfg.task_name=$dmc_env"
    done
done

#!/bin/bash

set -e

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu")

cluster_cfg_path="./gcp_cluster_sam.yaml"

declare -a testing_configs=("icml_inv_dyn" "icml_dynamics" "icml_ac_tcpc"
                            "icml_identity_cpc" "icml_vae"
                            "control_ortho_init" "control_no_ortho_init")
# Sam: skipping control_lsi_… configs because I don't expect them to make a difference.
# (this hyperparameter is also only used in DMC)
#                             "control_lsi_one" "control_lsi_zero")

declare -a dmc_envs=("finger-spin" "cheetah-run")

declare -a magical_envs=("MatchRegions" "MoveToRegion")

declare -a dmc_dataset_configs=("cfg_data_repl_demos_random" "cfg_data_repl_random")

declare -a mg_dataset_configs=("cfg_data_repl_demos_random" "cfg_data_repl_random"
                               "cfg_data_repl_demos_magical_mt" "cfg_data_repl_rand_demos_magical_mt")

submit_expt() {
    # submit experiment to cluster using given args
    ray submit --tmux "$cluster_cfg_path" \
        ./submit_pretrain_n_adapt.py -- "${base_cfgs[@]}" "$@"
}


for test_config in "${testing_configs[@]}"; do
    for magical_env in "${magical_envs[@]}"; do
        for mg_dataset_config in "${mg_dataset_configs[@]}"; do
            echo -e "\n *** TRAINING $test_config ON $magical_env WITH DATASET $mg_dataset_config *** \n "
            submit_expt icml_il_on_repl_sweep cfg_use_magical \
                        "$test_config" "$mg_dataset_config" cfg_il_bc_20k_nofreeze  \
                        exp_ident="${test_config}_${mg_dataset_config}" \
                        "env_cfg.task_name=$magical_env"
        done
    done
    for dmc_env in "${dmc_envs[@]}"; do
        for dmc_dataset_config in "${dmc_dataset_configs[@]}"; do
            echo -e "\n *** TRAINING $test_config ON $dmc_env WITH DATASET $dmc_dataset_config *** \n "
            submit_expt icml_il_on_repl_sweep cfg_use_dm_control \
                        "$test_config" "$dmc_dataset_config" cfg_il_bc_200k_nofreeze \
                        exp_ident="${test_config}_${dmc_dataset_config}" \
                        "env_cfg.task_name=$dmc_env"
        done
    done
done

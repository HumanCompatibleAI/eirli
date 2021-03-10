#!/bin/bash

set -e

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=1")
cluster_cfg_path="./gcp_cluster_sam.yaml"
declare -a testing_configs=("icml_identity_cpc" "icml_temporal_cpc_asym_proj"
                            "icml_tcpc_no_augs" "icml_tceb"
                            "icml_tcpc_momentum" "icml_four_tcpc")
declare -a dmc_envs=("finger-spin" "cheetah-run")
declare -a magical_envs=("MatchRegions" "MoveToCorner" "MoveToRegion")
declare -a dataset_configs=("cfg_data_repl_random")

submit_expt() {
    # submit experiment to cluster using given args
    ray submit --tmux "$cluster_cfg_path" \
        ./submit_pretrain_n_adapt.py -- "${base_cfgs[@]}" "$@"
}

for magical_env in "${magical_envs[@]}"; do
    for dataset_config in "${dataset_configs[@]}"; do
        for test_config in "${testing_configs[@]}"; do
            echo -e "\n ***TRAINING $test_config ON $magical_env *** \n "
            submit_expt icml_il_on_repl_sweep cfg_use_magical \
                "$test_config" "$dataset_config" cfg_il_bc_20k_nofreeze  \
                exp_ident="ablation_${test_config}_${dataset_config}" "env_cfg.task_name=$magical_env"
        done
    done
    # control run
    submit_expt cfg_use_magical icml_control \
        "$dataset_config" cfg_il_bc_20k_nofreeze  \
        exp_ident="ablation_control" "env_cfg.task_name=$magical_env"
done

for dmc_env in "${dmc_envs[@]}"; do
    for dataset_config in "${dataset_configs[@]}"; do
        for test_config in "${testing_configs[@]}"; do
            echo -e "\n ***TRAINING $test_config ON  $dmc_env *** \n "
            submit_expt icml_il_on_repl_sweep cfg_use_dm_control \
                "$test_config" "$dataset_config" cfg_il_bc_200k_nofreeze \
                exp_ident="ablation_${test_config}_${dataset_config}" "env_cfg.task_name=$dmc_env"
        done
    done
    # control run
    submit_expt cfg_use_dm_control icml_control \
        "$dataset_config" cfg_il_bc_200k_nofreeze \
        exp_ident="ablation_control" "env_cfg.task_name=$dmc_env"
done

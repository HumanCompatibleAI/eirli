#!/bin/bash

# Try to provoke hang that occurs when running contrastive algorithms on random
# data

set -e

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=1")
cluster_cfg_path="./gcp_cluster_sam.yaml"
declare -a testing_configs=("icml_identity_cpc")
declare -a dmc_envs=("cheetah-run")
declare -a dataset_configs=("cfg_data_repl_random")

submit_expt() {
    # submit experiment to cluster using given args
    ray submit --tmux "$cluster_cfg_path" \
        ./submit_pretrain_n_adapt.py -- "${base_cfgs[@]}" "$@"
}

for dmc_env in "${dmc_envs[@]}"; do
    for dataset_config in "${dataset_configs[@]}"; do
        for test_config in "${testing_configs[@]}"; do
            echo -e "\n ***TRAINING $test_config ON  $dmc_env *** \n "
            submit_expt icml_il_on_repl_sweep cfg_use_dm_control \
                "$test_config" "$dataset_config" cfg_il_bc_200k_nofreeze \
                exp_ident="ablation_${test_config}_${dataset_config}" "env_cfg.task_name=$dmc_env"
        done
    done
done

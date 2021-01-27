#!/bin/bash

set -e

base_cfgs=("cfg_base_skopt_1cpu_pt25gpu_no_retry" "icml_tuning")
cluster_cfg_path="./gcp_cluster_sam.yaml"
tuning_configs=("main_contrastive_tuning" "tune_momentum" "tune_projection_heads" "tune_vae" "tune_ceb" )
dmc_envs=("finger-spin" "cheetah-run")
magical_envs=("MatchRegions" "MoveToRegion")

submit_expt() {
    # submit experiment to cluster using given args
    ray submit --tmux "$cluster_cfg_path" \
        ./submit_pretrain_n_adapt.py -- "${base_cfgs[@]}" "$@"
}

for algo_config in "${tuning_configs[@]}"; do
    for magical_env in "${magical_envs[@]}"; do
        echo -e "\n ***TRAINING $algo_config ON $magical_env *** \n "
        submit_expt cfg_use_magical cfg_il_bc_20k_nofreeze "$algo_config" exp_ident="$algo_config" env_cfg.task_name="$magical_env"
    done
    for dmc_env in "${dmc_envs[@]}"; do
        echo -e "\n ***TRAINING $algo_config ON  $dmc_env *** \n "
        submit_expt cfg_use_dm_control cfg_il_bc_200k_nofreeze "$algo_config" exp_ident="$algo_config" env_cfg.task_name="$dmc_env"
    done
done

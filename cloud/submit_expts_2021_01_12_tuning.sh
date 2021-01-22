#!/bin/bash

set -e

base_cfgs=("cfg_base_skopt_4cpu_pt3gpu_no_retry" "icml_tuning")
cluster_cfg_path="./gcp_cluster_sam.yaml"
tuning_configs=("tune_ceb" "tune_momentum"
                "main_contrastive_tuning" "tune_vae",
                "tune_projection_heads")
dmc_envs=("finger-spin" "cheetah-run")
magical_envs=("MatchRegions" "MoveToRegion")

submit_expt() {
    ray submit --tmux "$cluster_cfg_path" \
        ./submit_pretrain_n_adapt.py -- ${base_cfgs[@]} $@
}

for algo_config in ${tuning_configs[@]}; do
    for dmc_env in ${dmc_envs[@]}; do
        echo -e "\n ***TRAINING $algo_config ON  $dmc_env *** \n "
        submit_expt cfg_use_dm_control $algo_config exp_ident=$algo_config env_cfg.task_name=$dmc_env
    done
    for magical_env in ${magical_envs[@]}; do
        echo -e "\n ***TRAINING $algo_config ON $magical_env *** \n "
        submit_expt cfg_use_magical $algo_config exp_ident=$algo_config env_cfg.task_name=$magical_env
    done
done

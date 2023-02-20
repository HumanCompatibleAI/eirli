#!/bin/bash

set -e

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "il_train.freeze_encoder=True"
           "tune_run_kwargs.num_samples=5")
cluster_cfg_path="./gcp_cluster_sam.yaml"
declare -a non_control_configs=("icml_inv_dyn" "icml_dynamics" "icml_ac_tcpc"
                                "icml_identity_cpc" "icml_vae")
declare -a control_configs=("control_ortho_init" "control_no_ortho_init")
declare -a dmc_envs=("finger-spin" "cheetah-run")
declare -a magical_envs=("MatchRegions-Demo-v0" "MoveToRegion-Demo-v0")
declare -a dmc_dataset_configs=("cfg_data_repl_demos_random" "cfg_data_repl_random")
declare -a mg_dataset_configs=("cfg_data_repl_demos_random" "cfg_data_repl_random"
                               "cfg_data_repl_demos_magical_mt"
                               "cfg_data_repl_rand_demos_magical_mt")

submit_expt() {
    # submit experiment to cluster using given args
    ray submit --tmux "$cluster_cfg_path" \
        ./submit_pretrain_n_adapt.py -- "${base_cfgs[@]}" "$@"
}


# control runs
for control_config in "${control_configs[@]}"; do
    for magical_env in "${magical_envs[@]}"; do
        echo -e "\n *** TRAINING CONTROL $control_config ON $magical_env *** \n "
        submit_expt icml_il_on_repl_sweep cfg_use_magical \
                    "$control_config" cfg_il_bc_20k_nofreeze  \
                    exp_ident="froco_${control_config}" \
                    "env_cfg.task_name=$magical_env"
    done
    for dmc_env in "${dmc_envs[@]}"; do
        echo -e "\n *** TRAINING CONTROL $control_config ON $dmc_env *** \n "
        submit_expt icml_il_on_repl_sweep cfg_use_dm_control \
                    "$control_config" cfg_il_bc_200k_nofreeze \
                    exp_ident="froco_${control_config}" \
                    "env_cfg.task_name=$dmc_env"
    done
done


# non-control configs
for test_config in "${non_control_configs[@]}"; do
    for magical_env in "${magical_envs[@]}"; do
        for mg_dataset_config in "${mg_dataset_configs[@]}"; do
            echo -e "\n *** TRAINING $test_config ON $magical_env WITH DATASET $mg_dataset_config *** \n "
            submit_expt icml_il_on_repl_sweep cfg_use_magical \
                        "$test_config" "$mg_dataset_config" cfg_il_bc_20k_nofreeze  \
                        exp_ident="froco_${test_config}_${mg_dataset_config}" \
                        "env_cfg.task_name=$magical_env"
        done
    done
    for dmc_env in "${dmc_envs[@]}"; do
        for dmc_dataset_config in "${dmc_dataset_configs[@]}"; do
            echo -e "\n *** TRAINING $test_config ON $dmc_env WITH DATASET $dmc_dataset_config *** \n "
            submit_expt icml_il_on_repl_sweep cfg_use_dm_control \
                        "$test_config" "$dmc_dataset_config" cfg_il_bc_200k_nofreeze \
                        exp_ident="froco_${test_config}_${dmc_dataset_config}" \
                        "env_cfg.task_name=$dmc_env"
        done
    done
done

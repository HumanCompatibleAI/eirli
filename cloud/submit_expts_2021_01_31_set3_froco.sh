#!/bin/bash

set -e

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu")

cluster_cfg_path="./gcp_cluster_sam.yaml"

declare -a testing_configs=("icml_inv_dyn" "icml_dynamics" "icml_ac_tcpc"
                            "icml_vae" "control_no_ortho_init")
declare -a mg_dataset_configs=("cfg_data_repl_demos_random")

submit_expt() {
    # submit experiment to cluster using given args
    ray submit --tmux "$cluster_cfg_path" \
        ./submit_pretrain_n_adapt.py -- "${base_cfgs[@]}" "$@"
}


for test_config in "${testing_configs[@]}"; do
    for mg_dataset_config in "${mg_dataset_configs[@]}"; do
        echo -e "\n *** TRAINING $test_config ON $magical_env WITH DATASET $mg_dataset_config *** \n "
        submit_expt icml_il_on_repl_sweep cfg_use_magical \
                    "$test_config" "$mg_dataset_config" cfg_il_bc_20k_nofreeze  \
                    il_train.freeze_encoder=True \
                    exp_ident="froco_${test_config}_${mg_dataset_config}" \
                    "env_cfg.task_name=MoveToRegion-Demo-v0"
    done
done

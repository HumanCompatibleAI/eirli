#!/bin/bash

# Like runs on 2021-03-30, but:
#
# - Training repL for 20k batches instead of 5k
# - Using single-task -TestAll data for repL training, not multitask -Demo data
# - Adding some runs with a frozen GAIL discriminator

set -euo pipefail

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu"
           "gail_mr_config_2021_03_29"
           "tune_run_kwargs.num_samples=5"
           "tune_run_kwargs.resources_per_trial.gpu=0.25"
           "ray_init_kwargs.address=localhost:42000")
# declare -a repl_configs=("icml_inv_dyn" "icml_dynamics" "icml_identity_cpc" "icml_vae")
# (dropping dynamics on the basis that it's probably testing something
# ~identical to the VAE)
declare -a repl_configs=("icml_inv_dyn" "icml_identity_cpc" "icml_vae" "icml_dynamics")
declare -a magical_envs=("MatchRegions-Demo-v0" "MoveToCorner-Demo-v0" "MoveToRegion-Demo-v0")
# declare -a magical_envs=("MatchRegions-Demo-v0" "MoveToRegion-Demo-v0" "MoveToCorner-Demo-v0")
# declare -a mg_dataset_configs=("cfg_data_repl_random" "cfg_data_repl_rand_demos_magical_mt")
declare -a mg_dataset_configs=("cfg_data_repl_rand_demos_magical_mt")

do_magical_repl_run() {
    python -m il_representations.scripts.pretrain_n_adapt \
        with "${base_cfgs[@]}" stages_to_run=REPL_AND_IL cfg_use_magical "$@" &
}

for magical_env in "${magical_envs[@]}"; do
    for repl_config in "${repl_configs[@]}"; do
        for mg_dataset_config in "${mg_dataset_configs[@]}"; do
            echo -e "\n *** TRAINING $repl_config ON $magical_env WITH DATASET $mg_dataset_config *** \n "
            do_magical_repl_run "env_cfg.task_name=$magical_env" "$repl_config" \
                "$mg_dataset_config" "env_cfg.task_name=$magical_env" \
                exp_ident="magigail_${repl_config}_${mg_dataset_config}"
        done
    done
done
wait


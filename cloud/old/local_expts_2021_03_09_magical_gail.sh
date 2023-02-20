#!/bin/bash

set -e

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "cfg_il_gail_magical_250k_nofreeze"
           "tune_run_kwargs.num_samples=3"
           "ray_init_kwargs.address=localhost:42000")
# declare -a repl_configs=("icml_inv_dyn" "icml_dynamics" "icml_identity_cpc" "icml_vae")
# (dropping dynamics on the basis that it's probably testing something
# ~identical to the VAE)
declare -a repl_configs=("icml_inv_dyn" "icml_identity_cpc" "icml_vae")
declare -a control_configs=("control_no_ortho_init")
declare -a magical_envs=("MatchRegions-Demo-v0" "MoveToCorner-Demo-v0")
# declare -a magical_envs=("MatchRegions-Demo-v0" "MoveToRegion-Demo-v0" "MoveToCorner-Demo-v0")
# declare -a mg_dataset_configs=("cfg_data_repl_random" "cfg_data_repl_rand_demos_magical_mt")
declare -a mg_dataset_configs=("cfg_data_repl_rand_demos_magical_mt")

submit_expt() {
    # submit experiments to local Ray server
    python -m il_representations.scripts.pretrain_n_adapt \
        with "${base_cfgs[@]}" "$@" &
}

for magical_env in "${magical_envs[@]}"; do
    for repl_config in "${repl_configs[@]}"; do
        for mg_dataset_config in "${mg_dataset_configs[@]}"; do
            echo -e "\n *** TRAINING $repl_config ON $magical_env WITH DATASET $mg_dataset_config *** \n "
            submit_expt stages_to_run=REPL_AND_IL cfg_use_magical "$repl_config" \
                "$mg_dataset_config" "env_cfg.task_name=$magical_env" \
                exp_ident="magigail_${repl_config}_${mg_dataset_config}"
        done
    done
    for control_config in "${control_configs[@]}"; do
        submit_expt stages_to_run=IL_ONLY cfg_use_magical \
            "$control_config" exp_ident="magigail_${control_config}" \
            "env_cfg.task_name=$magical_env"
    done
done
wait

#!/bin/bash

# What this file does:
# - repL/augs: Dynamics, InvDyn, SimCLR, TemporalCPC, VAE, GAIL+augs, GAIL-augs
# - Tasks: MAGICAL 3x, DMC 3x, Procgen 4x
# - #seeds: 5
# - repL batch size of 64 for all the non-contrastive methods, 192 for
#   contrastive ones.
# - RepL datasets: 5 demos + some large number of random rollouts. (IL uses 5
#   demos too.)
# Does only GAIL, and only demos+random repL data

set -e

# WARNING: 5 demos here!
magical_base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5"
                   "cfg_data_il_5demos" "cfg_use_magical"
                   "cfg_data_repl_5demos_random" "gail_mr_config_2021_03_29")

magical_envs=("MatchRegions-Demo-v0" "MoveToRegion-Demo-v0" "MoveToCorner-Demo-v0")

dmc_base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5"
               "env_cfg.benchmark_name=dm_control"
               "cfg_il_gail_dmc_500k_nofreeze")
dmc_envs=("finger-spin" "cheetah-run" "reacher-easy")

procgen_base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5"
                   "env_cfg.benchmark_name=procgen"
                   "gail_procgen_500k_config_2021_11_05")
procgen_envs=("coinrun" "ninja" "fruitbot" "jumper")

repl_configs=("icml_inv_dyn" "icml_dynamics" "cfg_repl_tcpc8_192"
              "cfg_repl_simclr_192" "icml_vae")

cluster_cfg_path="./gcp_cluster_sam_new_vis.yaml"
local=local

# 7 runs per GPU by default
gpu_default=0.14
declare -A gpu_overrides=(
    ["cfg_repl_tcpc8"]="0.16"
    ["cfg_repl_simclr"]="0.16"
    ["cfg_repl_tcpc8_192"]="0.16"
    ["cfg_repl_simclr_192"]="0.16"
)

gpu_config() {
    # figures out GPU configuration string based on repL config, if any
    override="$gpu_default"
    for cfg_string in "$@"; do
        this_override="${gpu_overrides[$cfg_string]}"
        if [ -n "$this_override" ]; then
            override="$this_override"
        fi
    done
    echo "tune_run_kwargs.resources_per_trial.gpu=$override"
}

submit_expt() {
    if [ "$local" == local ]; then
        # submit experiment to local ray server using given args
        python -m il_representations.scripts.pretrain_n_adapt run with \
            "ray_init_kwargs.address=localhost:42000" "$(gpu_config "$@")" "$@" &
    else
        # submit experiment to cluster using given args
        ray submit --tmux "$cluster_cfg_path" ./submit_pretrain_n_adapt.py -- \
            "$(gpu_config "$@")" "$@"
    fi
}

for magical_env in "${magical_envs[@]}"; do
    for repl_config in "${repl_configs[@]}"; do
        echo -e "\n *** TRAINING $repl_config ON $magical_env *** \n "
        submit_expt "${magical_base_cfgs[@]}" cfg_repl_5k_il cfg_repl_augs \
            "$repl_config" "env_cfg.task_name=$magical_env" \
            exp_ident="neurips_repl_gail_${repl_config}"
    done
    for use_augs in augs noaugs; do
        submit_expt "${magical_base_cfgs[@]}" cfg_repl_none \
            "gail_disc_${use_augs}" "env_cfg.task_name=$magical_env" \
            exp_ident="neurips_control_gail_${use_augs}"
    done
done

# --------------------------------
# GAIL + Procgen
# --------------------------------

for procgen_env in "${procgen_envs[@]}"; do
    for repl_config in "${repl_configs[@]}"; do
        echo -e "\n *** TRAINING $repl_config ON $procgen_env *** \n "
        submit_expt "${procgen_base_cfgs[@]}" cfg_repl_5k_il cfg_repl_augs \
            "$repl_config" "env_cfg.task_name=$procgen_env" \
            exp_ident="neurips_repl_gail_${repl_config}"
    done
    for use_augs in augs noaugs; do
        submit_expt "${procgen_base_cfgs[@]}" cfg_repl_none \
            "gail_disc_${use_augs}" "env_cfg.task_name=$procgen_env" \
            exp_ident="neurips_control_gail_${use_augs}"
    done
done

# --------------------------------
# GAIL + DMC
# --------------------------------

for dmc_env in "${dmc_envs[@]}"; do
    for repl_config in "${repl_configs[@]}"; do
        echo -e "\n *** TRAINING $repl_config ON $dmc_env *** \n "
        submit_expt "${dmc_base_cfgs[@]}" cfg_repl_5k_il cfg_repl_augs \
                    "$repl_config" "env_cfg.task_name=$dmc_env" \
                    exp_ident="neurips_repl_gail_${repl_config}"
    done
    for use_augs in augs noaugs; do
        submit_expt "${dmc_base_cfgs[@]}" cfg_repl_none \
                    "gail_disc_${use_augs}" "env_cfg.task_name=$dmc_env" \
                    exp_ident="neurips_control_gail_${use_augs}"
    done
done

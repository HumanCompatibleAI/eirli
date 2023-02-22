#!/bin/bash

# What this file does:
# - repL/augs: Dynamics, InvDyn, SimCLR, TemporalCPC, VAE, BC+augs, BC-augs
# - Tasks: 3 MAGICAL tasks, 4 Proxgen tasks, 3 DMC tasks
# - #seeds: 5
# - repL batch size of 64 for all the non-contrastive methods, 192 for
#   contrastive ones (so that we finish on time).
# - RepL datasets: demos + random rollouts. MAGICAL is limited to 5 demos; all
#   other datasets (MAGICAL random rollouts, demos/rollouts for rest) are
#   unlimited.
# - # of steps: 20k for MAGICAL, 500,000 for the rest (should be doing more, but
#   # we have limited time)
# Note that this file does only BC, and only demos+random repL data

set -e

# WARNING: unlimited demos here! (for DMC/Procgen)
local_ray_address="localhost:42000"
cluster_cfg_path="./gcp_cluster_sam_new_vis.yaml"
local=local
dmc_procgen_base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5")
dmc_procgen_n_batches=500000  # in the interests of finishing on time
repl_configs=("icml_inv_dyn" "icml_dynamics" "cfg_repl_tcpc8_192"
              "cfg_repl_simclr_192" "icml_vae")
dataset_configs=("cfg_data_repl_demos_random")
dmc_procgen_env_configs=(
    # dmc envs
    "env_cfg.benchmark_name=dm_control env_cfg.task_name=finger-spin"
    "env_cfg.benchmark_name=dm_control env_cfg.task_name=cheetah-run"
    "env_cfg.benchmark_name=dm_control env_cfg.task_name=reacher-easy"
)

gpu_default=0.11
declare -A gpu_overrides=(
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
            "ray_init_kwargs.address=$local_ray_address" "$(gpu_config "$@")" "$@" &
    else
        # submit experiment to cluster using given args
        ray submit --tmux "$cluster_cfg_path" ./submit_pretrain_n_adapt.py -- \
            "$(gpu_config "$@")" "$@"
    fi
}

for env_config_str in "${dmc_procgen_env_configs[@]}"; do
    # deliberately split on words
    # shellcheck disable=SC2206
    env_cfg_arr=($env_config_str)
    for repl_config in "${repl_configs[@]}"; do
        for dataset_config in "${dataset_configs[@]}"; do
            echo -e "\n *** TRAINING $repl_config ON '${env_cfg_arr[*]}' WITH DATASET $dataset_config *** \n "
            submit_expt "${dmc_procgen_base_cfgs[@]}" cfg_repl_5k_il \
                cfg_repl_augs "$repl_config" "$(gpu_config "$repl_config")" \
                "$dataset_config" cfg_il_bc_500k_nofreeze \
                exp_ident="neurips_repl_bc_${repl_config}_${dataset_config}" \
                "${env_cfg_arr[@]}" "il_train.bc.n_batches=$dmc_procgen_n_batches"
        done
    done
    for use_augs in augs noaugs; do
        submit_expt "${dmc_procgen_base_cfgs[@]}" cfg_repl_none \
            cfg_il_bc_500k_nofreeze "$(gpu_config "no_repl")" \
            "cfg_il_bc_${use_augs}" \
            exp_ident="neurips_control_bc_${use_augs}" "${env_cfg_arr[@]}" \
            "il_train.bc.n_batches=$dmc_procgen_n_batches"
    done
done
wait
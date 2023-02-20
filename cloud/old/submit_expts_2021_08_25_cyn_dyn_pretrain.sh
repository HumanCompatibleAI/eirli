#!/bin/bash

# What this file does:
# - repL/augs: Dynamics
# - Tasks: DMC FS/CR/RE, Procgen CR/M/FB/J
# - #seeds: 5
# - repL batch size of ??? (whatever the default is)
# - RepL datasets: demos ??? (whatever the default is)
# Aims to use ~identical configs to what Cynthia used.

set -e

base_cfgs=(
    "cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5"
    "tune_run_kwargs.max_failures=0" "cfg_force_use_repl" "cfg_il_bc_nofreeze"
)
cluster_cfg_path="./gcp_cluster_sam_new_vis.yaml"
declare -a repl_configs=("cfg_repl_dyn")
gpu_default=0.11
declare -A gpu_overrides=()
gpu_config() {
    # figures out GPU configuration string based on repL config, if any
    repl_config="$1"
    override="${gpu_overrides[$repl_config]}"
    if [ -z "$override" ]; then
        override="$gpu_default"
    fi
    echo "tune_run_kwargs.resources_per_trial.gpu=$override"
}

submit_expt() {
    # submit experiment to cluster using given args
    ray submit --tmux "$cluster_cfg_path" \
        ./submit_pretrain_n_adapt.py -- "${base_cfgs[@]}" "$@"
}


for repl_config in "${repl_configs[@]}"; do
    # DMC expts
    submit_expt "$(gpu_config "$repl_config")" \
        "${repl_config}" \
        cfg_bench_micro_sweep_dm_control \
        exp_ident="cyn-${repl_config}" \
        il_train.bc.n_batches=1000000
    # procgen expts
    submit_expt "$(gpu_config "$repl_config")" \
        "${repl_config}" \
        cfg_bench_procgen_cmfn \
        exp_ident="cyn-${repl_config}" \
        il_train.bc.n_batches=1000000
done

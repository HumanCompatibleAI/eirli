#!/bin/bash

# Pretraining Procgen/DMC runs from the NeurIPS benchmarks track submission

set -e

# WARNING: unlimited demos here!
base_cfgs=("tune_run_kwargs.num_samples=5" "cfg_il_bc_nofreeze")
dmc_cfgs=("il_train.bc.n_batches=4000000")
procgen_cfgs=("il_train.bc.n_batches=2000000")
repl_configs=("cfg_repl_inv_dyn" "cfg_repl_dyn" "cfg_repl_simclr"
                         "cfg_repl_temporal_cpc" "cfg_repl_vae")
cluster_cfg_path="./gcp_cluster_sam_new_vis.yaml"
gpu_default=0.11
declare -A gpu_overrides=(
    ["cfg_repl_tcpc8"]="0.16"
    ["cfg_repl_simclr"]="0.16"
)
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

launch_expts() {
    # usage: launch_expts (dmc|procgen) [args…]
    # (the first arg is used to set the exp_ident, and the remaining args are
    # passed straight through to submit_expt)
    prefix="$1"  # will be "dmc" or "procgen"
    shift
    # BC w/ augs
    submit_expt "$(gpu_config "no_repl")" cfg_repl_none \
        "exp_ident=${prefix}-augs" "$@"
    # BC w/o augs
    submit_expt "$(gpu_config "no_repl")" cfg_repl_none \
        "exp_ident=${prefix}-noaugs" il_train.bc.augs=None "$@"
    for repl_config in "${repl_configs[@]}"; do
        # repL (w/ augs, since that's the default)
        submit_expt "$(gpu_config "$repl_config")" cfg_force_use_repl \
            "${repl_config}" "exp_ident=${prefix}-${repl_config}" "$@"
    done
}

launch_expts dmc     "${dmc_cfgs[@]}"     "cfg_bench_micro_sweep_dm_control"
launch_expts procgen "${procgen_cfgs[@]}" "cfg_bench_procgen_cmfn"

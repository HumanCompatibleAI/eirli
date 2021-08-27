#!/bin/bash

# Local submission file that re-does Cynthia's dynamics experiments.
# Specifically:
# - repL/augs: Dynamics
# - Tasks: DMC FS/CR/RE, Procgen CR/M/FB/J
# - #seeds: 5
# - repL batch size of ??? (whatever the default is)
# - RepL datasets: demos ??? (whatever the default is)
# Aims to use ~identical configs to what Cynthia used.

set -e

base_cfgs=(
    "ray_init_kwargs.address=localhost:42000"
    "cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5"
    "tune_run_kwargs.max_failures=0" "cfg_force_use_repl"
    "cfg_il_bc_nofreeze"
)
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
    # submit experiment to local Ray server using given args
    python -m il_representations.scripts.pretrain_n_adapt with \
           "${base_cfgs[@]}" "$@"
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

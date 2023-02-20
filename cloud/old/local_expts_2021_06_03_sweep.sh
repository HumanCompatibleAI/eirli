#!/bin/bash

# Big sweep over many repL algorithms and many environments, from both DMC and
# MAGICAL. Sam is using this for his quals slides, but it's generally useful to
# get an overview of how well different things work.

set -euo pipefail

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu"
           "tune_run_kwargs.num_samples=5"
           "tune_run_kwargs.resources_per_trial.gpu=0.3"
           "ray_init_kwargs.address=localhost:42000")
repl_configs=("icml_inv_dyn" "icml_identity_cpc" "icml_tcpc_8step" "icml_vae" "icml_dynamics")
declare -A magical_env_names=(
    ["matchregions"]="MatchRegions-Demo-v0"
    ["movetocorner"]="MoveToCorner-Demo-v0"
    ["movetoregion"]="MoveToRegion-Demo-v0"
)
dmc_env_names=("cheetah-run" "finger-spin")
declare -A magical_il_flags=(
    ["gail"]="gail_mr_config_2021_03_29"
    ["bc"]="cfg_il_bc_20k_nofreeze"
)
declare -A dmc_il_flags=(
    ["gail"]="cfg_il_gail_dmc_250k_nofreeze"
    ["bc"]="cfg_il_bc_200k_nofreeze"
)
common_control_flags=("stages_to_run=IL_ONLY" "control_no_ortho_init")
common_repl_flags=("stages_to_run=REPL_AND_IL")

submit_expt() {
    # submit experiments to local Ray server
    python -m il_representations.scripts.pretrain_n_adapt \
        with "${base_cfgs[@]}" "$@" &
}

submit_control() {
    submit_expt "${common_control_flags[@]}" "$@"
}

# MAGICAL experiments

for magical_env_nick in "${!magical_env_names[@]}"; do
    full_env_name="${magical_env_names[$magical_env_nick]}"
    for il_alg in bc gail; do
        for repl_config in "${repl_configs[@]}"; do
            # alias for running repL experiments
            submit_repl() {
                submit_expt cfg_use_magical "${common_repl_flags[@]}" "${magical_il_flags[$il_alg]}" \
                    "$repl_config" "$@"
            }

            # demos + random rollouts
            submit_repl "cfg_data_repl_demos_random" "env_cfg.task_name=$full_env_name" \
                exp_ident="${il_alg}_${repl_config}_rand_demos"
            # multitask demos + random rollouts
            submit_repl "cfg_data_repl_rand_demos_magical_mt" "env_cfg.task_name=$full_env_name" \
                exp_ident="${il_alg}_${repl_config}_rand_demos_magical_mt"
            # test variant demos + random rollouts
            submit_repl "env_cfg.task_name=${full_env_name}" "repl.is_multitask=True" \
                "cfg_data_repl_${magical_env_nick}_rand_demos_magical_test" \
                "cfg_data_repl_rand_demos_magical_mt" "env_cfg.task_name=$full_env_name" \
                exp_ident="${il_alg}_${repl_config}_rand_demos_test"
        done

        # controls
        submit_control cfg_use_magical \
            "${magical_il_flags[$il_alg]}" "env_cfg.task_name=$full_env_name" \
            exp_ident="${il_alg}_control"
    done
    # one env at a time (avoids "too many open files" error)
    wait
done
wait

# DMC experiments

for dmc_env in "${dmc_env_names[@]}"; do
    for il_alg in bc gail; do
        for repl_config in "${repl_configs[@]}"; do
            # demos + random rollouts
            submit_expt "env_cfg.benchmark_name=dm_control" \
                "${common_repl_flags[@]}" "${dmc_il_flags[$il_alg]}" \
                "$repl_config" "cfg_data_repl_demos_random" \
                "env_cfg.task_name=$dmc_env" \
                exp_ident="${il_alg}_${repl_config}_rand_demos"
        done

        # controls
        submit_control cfg_use_magical \
            "${dmc_il_flags[$il_alg]}" "env_cfg.task_name=$dmc_env" \
            exp_ident="${il_alg}_control"
    done
    # no 'wait' (we do all envs at once, don't worry about 'too many open files'
    # since we have fewer expts)
done
wait

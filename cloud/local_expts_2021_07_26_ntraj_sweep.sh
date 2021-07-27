#!/bin/bash

# Sweeps over repL algorithms on three MAGICAL environments. This time we vary
# the number of trajectories provided to BC to see whether giving very few
# trajectories actually helps.

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
declare -A magical_il_flags=(
    ["gail"]="gail_mr_config_2021_03_29"
    ["bc"]="cfg_il_bc_20k_nofreeze"
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

for il_alg in bc gail; do
    for bc_ntrajs in 1 25; do
        ntraj_opt="il_train.bc.n_trajs=${bc_ntrajs}"
        for magical_env_nick in "${!magical_env_names[@]}"; do
            full_env_name="${magical_env_names[$magical_env_nick]}"
            for repl_config in "${repl_configs[@]}"; do
                # alias for running repL experiments
                submit_repl() {
                    submit_expt cfg_use_magical "${common_repl_flags[@]}" "${magical_il_flags[$il_alg]}" \
                        "$repl_config" "$ntraj_opt" "$@"
                }

                # demos + random rollouts
                submit_repl "cfg_data_repl_demos_random" "env_cfg.task_name=$full_env_name" \
                    exp_ident="${il_alg}_${repl_config}_rand_demos_${bc_ntrajs}t"
                # Leaving these out for now, might add them back in later on.
                # # multitask demos + random rollouts
                # submit_repl "cfg_data_repl_rand_demos_magical_mt" "env_cfg.task_name=$full_env_name" \
                #     exp_ident="${il_alg}_${repl_config}_rand_demos_magical_mt_${bc_ntrajs}t"
                # # test variant demos + random rollouts
                # submit_repl "env_cfg.task_name=${full_env_name}" "repl.is_multitask=True" \
                #     "cfg_data_repl_${magical_env_nick}_rand_demos_magical_test" \
                #     "cfg_data_repl_rand_demos_magical_mt" "env_cfg.task_name=$full_env_name" \
                #     exp_ident="${il_alg}_${repl_config}_rand_demos_test_${bc_ntrajs}t"
            done

            # controls
            submit_control cfg_use_magical \
                "${magical_il_flags[$il_alg]}" "env_cfg.task_name=$full_env_name" \
                "$ntraj_opt" exp_ident="${il_alg}_control_${bc_ntrajs}t"
        done
        # one env at a time (avoids "too many open files" error)
        wait
    done
done
wait

#!/bin/bash

# Run one seed of each repL algorithm on each MAGICAL environment.

set -e

# WARNING: only one sample here!
base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=3"
           "ray_init_kwargs.address=localhost:42000"
           "tune_run_kwargs.resources_per_trial.gpu=0.5")
env_names=("MatchRegions" "MoveToCorner" "MoveToRegion" "ClusterColour"
           "ClusterShape" "FindDupe" "MakeLine" "FixColour")

# let's cast a wiiiide net
declare -a repl_configs=("icml_tcpc_8step" "icml_inv_dyn" "icml_identity_cpc"
                         "icml_vae" "icml_dynamics")

submit_expt() {
    # submit experiments to local Ray server
    python -m il_representations.scripts.pretrain_n_adapt \
           with "${base_cfgs[@]}" "$@" &
}

launch_repl_for_env() {
    echo -e "\n *** REPL+IL RUN WITH $* *** \n "
    submit_expt stages_to_run=REPL_AND_IL cfg_use_magical \
        cfg_il_bc_20k_nofreeze repl.batches_per_epoch=2000 repl.n_epochs=10 il_train.bc.n_batches=40000 "$@"
}

lower() {
    echo "$1" | tr "[:upper:]" "[:lower:]"
}

# repL runs
for repl_config in "${repl_configs[@]}"; do
    for env_name in "${env_names[@]}"; do
        lower_env="$(lower "$env_name")"

        # using test variant demos
        launch_repl_for_env exp_ident="${repl_config}_test_variant_demos" \
                            "${repl_config[@]}" "env_cfg.task_name=${env_name}-Demo-v0" \
                            "cfg_data_repl_${lower_env}_demos_magical_test" \
                            repl.is_multitask=True

        # using (demo variant demos) + (test/demo variant random rollouts)
        # (note that we don't use anything from other tasks)
        launch_repl_for_env exp_ident="${repl_config}_demos_and_test_rollouts" \
                            "${repl_config[@]}" "env_cfg.task_name=${env_name}-Demo-v0" \
                            "cfg_data_repl_${lower_env}_test_demos_and_all_random" \
                            repl.is_multitask=True
    done
done
wait

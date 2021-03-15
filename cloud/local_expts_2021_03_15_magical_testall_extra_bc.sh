#!/bin/bash

# Additional experiments after failure of experiments on the 11th to find
# anything interesting.

set -e

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5" "ray_init_kwargs.address=localhost:42000"
           "tune_run_kwargs.resources_per_trial.gpu=0.5")

# let's cast a wiiiide net
declare -a repl_configs=("icml_inv_dyn" "icml_dynamics" "icml_identity_cpc" "icml_vae")

submit_expt() {
    # submit experiments to local Ray server
    python -m il_representations.scripts.pretrain_n_adapt \
           with "${base_cfgs[@]}" "$@" &
}

launch_for_env() {
    for repl_config in "${repl_configs[@]}"; do
        echo -e "\n *** TRAINING $repl_config WITH $@ *** \n "
        submit_expt stages_to_run=REPL_AND_IL cfg_use_magical \
                    "$repl_config" "$mg_dataset_config" cfg_il_bc_20k_nofreeze  \
                    "$@"
    done
}

launch_for_env exp_ident="mtest_st_demos_matchregions" env_cfg.task_name=MatchRegions-Demo-v0 \
    cfg_data_repl_demos
launch_for_env exp_ident="mtest_st_rand_demos_matchregions" env_cfg.task_name=MatchRegions-Demo-v0 \
    cfg_data_repl_demos_random
launch_for_env exp_ident="mtest_st_test_demos_matchregions" env_cfg.task_name=MatchRegions-Demo-v0 \
    cfg_data_repl_matchregions_demos_magical_test
launch_for_env exp_ident="mtest_st_test_rand_demos_matchregions" env_cfg.task_name=MatchRegions-Demo-v0 \
    cfg_data_repl_matchregions_rand_demos_magical_test

launch_for_env exp_ident="mtest_st_demos_movetoregion" env_cfg.task_name=MoveToRegion-Demo-v0 \
               cfg_data_repl_demos
launch_for_env exp_ident="mtest_st_rand_demos_movetoregion" env_cfg.task_name=MoveToRegion-Demo-v0 \
               cfg_data_repl_demos_random
launch_for_env exp_ident="mtest_st_test_demos_movetoregion" env_cfg.task_name=MoveToRegion-Demo-v0 \
               cfg_data_repl_movetoregion_demos_magical_test
launch_for_env exp_ident="mtest_st_test_rand_demos_movetoregion" env_cfg.task_name=MoveToRegion-Demo-v0 \
               cfg_data_repl_movetoregion_rand_demos_magical_test

launch_for_env exp_ident="mtest_st_demos_movetocorner" env_cfg.task_name=MoveToCorner-Demo-v0 \
               cfg_data_repl_demos
launch_for_env exp_ident="mtest_st_rand_demos_movetocorner" env_cfg.task_name=MoveToCorner-Demo-v0 \
               cfg_data_repl_demos_random
launch_for_env exp_ident="mtest_st_test_demos_movetocorner" env_cfg.task_name=MoveToCorner-Demo-v0 \
               cfg_data_repl_movetocorner_demos_magical_test
launch_for_env exp_ident="mtest_st_test_rand_demos_movetocorner" env_cfg.task_name=MoveToCorner-Demo-v0 \
               cfg_data_repl_movetocorner_rand_demos_magical_test

wait

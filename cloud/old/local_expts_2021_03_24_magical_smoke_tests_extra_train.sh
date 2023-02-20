#!/bin/bash

# Followup to experiments on 22nd in which I move repL training from 5k to 20k,
# and IL training from 20k to 40k. The hope is that this will improve variants
# that train on test data, at the cost of taking 2x to 3x as long. I did this
# because I noticed the VAE runs weren't producing reasonable outputs.

set -e

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5" "ray_init_kwargs.address=localhost:42000"
           "tune_run_kwargs.resources_per_trial.gpu=0.25")

# let's cast a wiiiide net
# (update: disabling identity CPC because it uses lots of memory, and svm is
# currently running low)
# declare -a repl_configs=("icml_inv_dyn" "icml_dynamics" "icml_identity_cpc" "icml_vae")
declare -a repl_configs=("icml_inv_dyn" "icml_dynamics" "icml_vae")

submit_expt() {
    # submit experiments to local Ray server
    python -m il_representations.scripts.pretrain_n_adapt \
           with "${base_cfgs[@]}" "$@" &
}

launch_repl_for_env() {
    echo -e "\n *** REPL+IL RUN WITH $@ *** \n "
    submit_expt stages_to_run=REPL_AND_IL cfg_use_magical \
        cfg_il_bc_20k_nofreeze repl.batches_per_epoch=2000 repl.n_epochs=10 il_train.bc.n_batches=40000 "$@"
}

launch_control_for_env() {
    echo -e "\n *** CONTROL RUN WITH $@ *** \n "
    submit_expt stages_to_run=IL_ONLY cfg_use_magical control_no_ortho_init \
        cfg_il_bc_20k_nofreeze il_train.bc.n_batches=40000 "$@"
}

# controls
launch_control_for_env exp_ident="mtest_control_moartrain" env_cfg.task_name=MatchRegions-Demo-v0
launch_control_for_env exp_ident="mtest_control_test_moartrain" env_cfg.task_name=MatchRegions-Demo-v0 \
    cfg_data_il_matchregions_demos_magical_test

launch_control_for_env exp_ident="mtest_control_moartrain" env_cfg.task_name=MoveToCorner-Demo-v0
launch_control_for_env exp_ident="mtest_control_test_moartrain" env_cfg.task_name=MoveToCorner-Demo-v0 \
    cfg_data_il_movetocorner_demos_magical_test

launch_control_for_env exp_ident="mtest_control_moartrain" env_cfg.task_name=MoveToRegion-Demo-v0
launch_control_for_env exp_ident="mtest_control_test_moartrain" env_cfg.task_name=MoveToRegion-Demo-v0 \
    cfg_data_il_movetoregion_demos_magical_test

# repL runs
for repl_config in "${repl_configs[@]}"; do
    # for MatchRegions
    launch_repl_for_env exp_ident="mtest_st_${repl_config}_demos_moartrain" "${repl_config[@]}" env_cfg.task_name=MatchRegions-Demo-v0 \
        cfg_data_repl_demos

    launch_repl_for_env exp_ident="mtest_st_${repl_config}_rand_demos_moartrain" "${repl_config[@]}" env_cfg.task_name=MatchRegions-Demo-v0 \
        cfg_data_repl_demos_random

    launch_repl_for_env exp_ident="mtest_st_${repl_config}_test_demos_moartrain" "${repl_config[@]}" env_cfg.task_name=MatchRegions-Demo-v0 \
        cfg_data_repl_matchregions_demos_magical_test repl.is_multitask=True

    launch_repl_for_env exp_ident="mtest_st_${repl_config}_test_rand_demos_moartrain" "${repl_config[@]}" env_cfg.task_name=MatchRegions-Demo-v0 \
        cfg_data_repl_matchregions_rand_demos_magical_test repl.is_multitask=True



    # for MoveToRegion
    launch_repl_for_env exp_ident="mtest_st_${repl_config}_demos_moartrain" "${repl_config[@]}" env_cfg.task_name=MoveToRegion-Demo-v0 \
        cfg_data_repl_demos

    launch_repl_for_env exp_ident="mtest_st_${repl_config}_rand_demos_moartrain" "${repl_config[@]}" env_cfg.task_name=MoveToRegion-Demo-v0 \
        cfg_data_repl_demos_random

    launch_repl_for_env exp_ident="mtest_st_${repl_config}_test_demos_moartrain" "${repl_config[@]}" env_cfg.task_name=MoveToRegion-Demo-v0 \
        cfg_data_repl_movetoregion_demos_magical_test repl.is_multitask=True

    launch_repl_for_env exp_ident="mtest_st_${repl_config}_test_rand_demos_moartrain" "${repl_config[@]}" env_cfg.task_name=MoveToRegion-Demo-v0 \
        cfg_data_repl_movetoregion_rand_demos_magical_test repl.is_multitask=True



    # for MoveToCorner
    launch_repl_for_env exp_ident="mtest_st_${repl_config}_demos_moartrain" "${repl_config[@]}" env_cfg.task_name=MoveToCorner-Demo-v0 \
        cfg_data_repl_demos

    launch_repl_for_env exp_ident="mtest_st_${repl_config}_rand_demos_moartrain" "${repl_config[@]}" env_cfg.task_name=MoveToCorner-Demo-v0 \
        cfg_data_repl_demos_random

    launch_repl_for_env exp_ident="mtest_st_${repl_config}_test_demos_moartrain" "${repl_config[@]}" env_cfg.task_name=MoveToCorner-Demo-v0 \
        cfg_data_repl_movetocorner_demos_magical_test repl.is_multitask=True

    launch_repl_for_env exp_ident="mtest_st_${repl_config}_test_rand_demos_moartrain" "${repl_config[@]}" env_cfg.task_name=MoveToCorner-Demo-v0 \
        cfg_data_repl_movetocorner_rand_demos_magical_test repl.is_multitask=True
done
wait

#!/bin/bash

# Additional experiments after failure of experiments on the 11th to find
# anything interesting.

set -e

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5" "ray_init_kwargs.address=localhost:42000"
           "tune_run_kwargs.resources_per_trial.gpu=0.25")

submit_expt() {
    # submit experiments to local Ray server
    python -m il_representations.scripts.pretrain_n_adapt \
           with "${base_cfgs[@]}" "$@" &
}

launch_for_env() {
    submit_expt stages_to_run=IL_ONLY cfg_use_magical control_no_ortho_init \
                cfg_il_bc_20k_nofreeze "$@"
}

launch_for_env exp_ident="mtest_control_test_matchregions" env_cfg.task_name=MatchRegions-Demo-v0 \
    cfg_data_il_matchregions_demos_magical_test
launch_for_env exp_ident="mtest_control_test_movetoregion" env_cfg.task_name=MoveToRegion-Demo-v0 \
    cfg_data_il_movetoregion_demos_magical_test
launch_for_env exp_ident="mtest_control_test_movetocorner" env_cfg.task_name=MoveToCorner-Demo-v0 \
    cfg_data_il_movetocorner_demos_magical_test

wait

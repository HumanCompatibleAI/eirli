#!/bin/bash

# Like runs on 2021-04-02, but:
#
# - Default config is fixed so that runs do not use frozen encoder by default
# - Frozen encoder runs are taken out, since they were fine last time

set -euo pipefail

# note extra repL training with 'repl.batches_per_epoch=2000' and
# 'repl.n_epochs=10' (vs. defaults of 1000 and 5, for 5k batches total)
base_cfgs=("cfg_base_5seed_1cpu_pt25gpu"
           "gail_mr_config_2021_03_29"
           "tune_run_kwargs.num_samples=5"
           "tune_run_kwargs.resources_per_trial.gpu=0.25"
           "ray_init_kwargs.address=localhost:42000"
           "repl.batches_per_epoch=2000"
           "repl.n_epochs=10")
declare -a repl_configs=("icml_inv_dyn" "icml_identity_cpc" "icml_vae" "icml_dynamics")
declare -a env_dataset_configs=(
    "env_cfg.task_name=MatchRegions-Demo-v0 cfg_data_repl_matchregions_rand_demos_magical_test repl.is_multitask=True"
    "env_cfg.task_name=MoveToCorner-Demo-v0 cfg_data_repl_movetocorner_rand_demos_magical_test repl.is_multitask=True"
    "env_cfg.task_name=MoveToRegion-Demo-v0 cfg_data_repl_movetoregion_rand_demos_magical_test repl.is_multitask=True"
)

do_magical_repl_run() {
    python -m il_representations.scripts.pretrain_n_adapt \
        with "${base_cfgs[@]}" stages_to_run=REPL_AND_IL cfg_use_magical "$@" &
}

for repl_config in "${repl_configs[@]}"; do
    for env_dataset_config in "${env_dataset_configs[@]}"; do
        echo -e "\n *** RUNS WITH $repl_config AND ENV/DATASET $env_dataset_config *** \n "
        # nothing frozen
        # (note that we re-split env_dataset_config)
        do_magical_repl_run "$repl_config" $env_dataset_config \
            exp_ident="magigail_${repl_config}_rand_demos_test_fixed_2021_04_13"
    done
done
wait

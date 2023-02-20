#!/bin/bash

# Single run of identity CPC on MAGICAL MatchRegions using TestAll data. Doing
# this for the purpose of my visualisation experiments.

set -e

# WARNING: only one sample here!
base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=1" "ray_init_kwargs.address=localhost:42000"
           "tune_run_kwargs.resources_per_trial.gpu=0.25")

# let's cast a wiiiide net
# (update: disabling identity CPC because it uses lots of memory, and svm is
# currently running low)
# declare -a repl_configs=("icml_inv_dyn" "icml_dynamics" "icml_identity_cpc" "icml_vae")
declare -a repl_configs=("icml_identity_cpc")

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

# repL runs
for repl_config in "${repl_configs[@]}"; do
    # for MatchRegions
    launch_repl_for_env exp_ident="mtest_st_${repl_config}_test_demos_moartrain" "${repl_config[@]}" env_cfg.task_name=MatchRegions-Demo-v0 \
        cfg_data_repl_matchregions_demos_magical_test repl.is_multitask=True
done
wait

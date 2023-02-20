#!/bin/bash

# FIXME(sam): things to do with this file:
#
# - [DONE] Run it on multiple envs.
# - Have it run without repL.
# - Make it submit to GCP.

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=1"
           "tune_run_kwargs.num_samples=5" "tune_run_kwargs.max_failures=0"
           "tune_run_kwargs.resources_per_trial.gpu=0.2"
           "cfg_il_gail_dmc_500k_nofreeze" "venv_opts.n_envs=8")
cluster_cfg_path="./gcp_cluster_cyn_expt.yaml"
exp_id="gail_smoke_test_no_repl"
benchmark_name="procgen"
task_names=("coinrun" "ninja" "fruitbot" "jumper")

submit_expt() {
    # submit experiment to cluster using given args
    ray submit --tmux "$cluster_cfg_path" \
        ./submit_pretrain_n_adapt.py -- "${base_cfgs[@]}" "$@"
}

for task_name in "${task_names[@]}"; do
    submit_expt exp_ident="${exp_id}" cfg_repl_none \
        env_cfg.benchmark_name="${benchmark_name}" \
        env_cfg.task_name="${task_name}"
done

# cfg_bench_full_sweep_procgen \
# cfg_force_use_repl \

# env_cfg.benchmark_name=${benchmark_name} \
# env_cfg.task_name=${task_name} \
# il_train.gail.disc_augs=None \

# For debugging
# repl.batches_per_epoch=10 \
# il_train.gail.total_timesteps=$total_timesteps \
# il_test.n_rollouts=1 \

#!/bin/bash

# Smoke test of GAIL + DMC

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5"
           "tune_run_kwargs.max_failures=0"
           "tune_run_kwargs.resources_per_trial.gpu=0.15"
           "cfg_il_gail_dmc_500k_nofreeze" "venv_opts.n_envs=8")
cluster_cfg_path="./gcp_cluster_sam_new_vis.yaml"
exp_id="gail_smoke_test_dmc_no_repl"
benchmark_name="dm_control"
task_names=("finger-spin" "cheetah-run" "reacher-easy")

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

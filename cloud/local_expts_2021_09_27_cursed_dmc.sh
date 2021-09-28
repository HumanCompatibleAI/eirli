#!/bin/bash

# Run DMC environments with near-identical config to Cynthia, except that I'm
# using parallel_workers=0.

base_cfgs=("cfg_base_5seed_1cpu_pt25gpu" "tune_run_kwargs.num_samples=5"
           "tune_run_kwargs.max_failures=0"
           "tune_run_kwargs.resources_per_trial.gpu=0.3"
           "cfg_il_gail_dmc_500k_nofreeze"
           "venv_opts.n_envs=32"
           "venv_opts.parallel_workers=None"
           "venv_opts.venv_parallel=True")
exp_id="gail_smoke_test_dmc_no_repl_32env_nopw"
benchmark_name="dm_control"

run_expt() {
    python -m il_representations.scripts.pretrain_adapt run with \
        -- "${base_cfgs[@]}" "$@"
}

launch_on_task() {
    task_name="$*"
    run_expt exp_ident="${exp_id}" cfg_repl_none \
        env_cfg.benchmark_name="${benchmark_name}" \
        env_cfg.task_name="${task_name}"
}

CUDA_VISIBLE_DEVICES=1 launch_on_task finger-spin &
CUDA_VISIBLE_DEVICES=2 launch_on_task cheetah-run &
CUDA_VISIBLE_DEVICES=3 launch_on_task reacher-easy &
wait

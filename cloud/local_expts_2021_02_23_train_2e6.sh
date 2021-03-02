#!/bin/bash

# Investigating what happens when we have a big network (resnet18) and lots of
# training (2e6 batches).

set -e

# note that this requires Ray to be running on port 42000
base_cfgs=("cfg_base_3seed_4cpu_pt3gpu" "tune_run_kwargs.resources_per_trial.cpu=1"
           "ray_init_kwargs.address=localhost:42000" "venv_opts.n_envs=5")

for bench in \
    "env_cfg.benchmark_name=magical env_cfg.task_name=MoveToCorner" \
    "env_cfg.benchmark_name=magical env_cfg.task_name=MatchRegions" \
    "env_cfg.benchmark_name=dm_control env_cfg.task_name=cheetah-run" \
    "env_cfg.benchmark_name=dm_control env_cfg.task_name=finger-spin" \
    ; do
    python -m il_representations.scripts.pretrain_n_adapt with \
        "${base_cfgs[@]}" $bench cfg_repl_none cfg_il_bc_20k_nofreeze \
        il_train.bc.n_batches=2000000 il_train.bc.lr=0.0001 \
        il_train.bc.save_every_n_batches=100000 \
        il_train.encoder_kwargs.obs_encoder_cls=Resnet18 \
        il_train.encoder_kwargs.representation_dim=512 \
        exp_ident=control_il_resnet18_2m &
    sleep 1
done

wait

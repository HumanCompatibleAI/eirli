#!/usr/bin/env bash
# pretrain_n_adapt with MAGICALCNN + Adam + 200 trajs
CUDA_VISIBLE_DEVICES=1 xvfb-run -a python src/il_representations/scripts/pretrain_n_adapt.py with \
 cfg_base_3seed_1cpu_pt2gpu_2envs \
 cfg_repl_none \
 cfg_il_bc_nofreeze \
 tune_run_kwargs.num_samples=2 \
 tune_run_kwargs.resources_per_trial.gpu=0.5 \
 exp_ident=magical-small \
 il_train.bc.n_batches=400000 \
 il_train.bc.batch_size=512 \
 il_train.encoder_kwargs.obs_encoder_cls=MAGICALCNN \
 il_train.encoder_kwargs.arch_str=MAGICALCNN-small \
 env_cfg.benchmark_name=dm_control \
 env_cfg.task_name=finger-spin \
 env_cfg.n_traj=200

#CUDA_VISIBLE_DEVICES=2 python src/il_representations/scripts/pretrain_n_adapt.py with cfg_base_3seed_1cpu_pt2gpu_2envs cfg_repl_none cfg_il_bc_nofreeze tune_run_kwargs.num_samples=2 exp_ident=64-128-256 il_train.bc.n_batches=400000 il_train.encoder_kwargs.obs_encoder_cls=MAGICALCNN benchmark.benchmark_name=dm_control benchmark.dm_control_env=finger-spin benchmark.n_traj=200

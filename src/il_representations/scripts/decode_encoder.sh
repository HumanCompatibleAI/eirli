#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 xvfb-run -a python src/il_representations/scripts/pretrain_n_adapt.py with \
  repl.algo=VariationalAutoencoder
# cfg_base_3seed_1cpu_pt2gpu_2envs \
# cfg_repl_autoencoder \
# tune_run_kwargs.num_samples=2 \
# tune_run_kwargs.resources_per_trial.gpu=0.5 \
# exp_ident=decode_encoder \
# il_train.bc.n_batches=200000 \
# il_train.bc.batch_size=384 \
# il_train.encoder_kwargs.load_path=./runs/chain_runs/38/il_train/1/policy_final.pt \
# env_cfg.benchmark_name=dm_control \
# env_cfg.task_name=finger-spin

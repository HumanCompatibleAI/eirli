repl_epochs=1
bc_trajs=10
bc_batches=1

CUDA_VISIBLE_DEVICES=0 python src/il_representations/scripts/pretrain_n_adapt.py with \
  cfg_repl_simclr \
  cfg_il_bc_nofreeze \
  tune_run_kwargs.num_samples=1 \
  tune_run_kwargs.resources_per_trial.gpu=1 \
  env_cfg.benchmark_name=dm_control \
  env_cfg.task_name=finger-spin \
  repl.n_epochs=$repl_epochs \
  repl.algo_params.batch_size=32 \
  repl.algo_params.representation_dim=2048 \
  il_train.bc.n_trajs=$bc_trajs \
  il_train.bc.n_batches=$bc_batches \
  exp_ident=repl_epoch_${repl_epochs}_bc_${bc_trajs}_trajs_${bc_batches}_batches


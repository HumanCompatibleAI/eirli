for seed_var in 1 2 3 4 5
do
  #success condition
#  CUDA_VISIBLE_DEVICES="1" xvfb-run -a python -m il_representations.scripts.run_rep_learner with algo='MoCo' algo_params.representation_dim=10\
#  benchmark.benchmark_name='dm_control' benchmark.n_envs=1 use_random_rollouts=False \
#  pretrain_epochs=50 algo_params.encoder_kwargs.momentum_weight=0.995\
#   algo_params.loss_calculator_kwargs.use_batch_neg=True seed=$seed_var
#
#
#  # failure condition I expect to collapse down
#  CUDA_VISIBLE_DEVICES="1" xvfb-run -a python -m il_representations.scripts.run_rep_learner with algo='MoCo' algo_params.representation_dim=10\
#  benchmark.benchmark_name='dm_control' benchmark.n_envs=1 use_random_rollouts=False \
#  pretrain_epochs=50 algo_params.encoder_kwargs.momentum_weight=0.995\
#   algo_params.loss_calculator_kwargs.use_batch_neg=False seed=$seed_var
#

#  # failure condition I expect to collapse up
#  CUDA_VISIBLE_DEVICES="1" xvfb-run -a python -m il_representations.scripts.run_rep_learner with algo='MoCo' algo_params.representation_dim=10\
#  benchmark.benchmark_name='dm_control' benchmark.n_envs=1 use_random_rollouts=False \
#  pretrain_epochs=50 algo_params.encoder_kwargs.momentum_weight=0.50\
#   algo_params.loss_calculator_kwargs.use_batch_neg=False seed=$seed_var
#
#  CUDA_VISIBLE_DEVICES="1" xvfb-run -a python -m il_representations.scripts.run_rep_learner with algo='MoCo' algo_params.representation_dim=10\
#  benchmark.benchmark_name='dm_control' benchmark.n_envs=1 use_random_rollouts=False \
#  pretrain_epochs=50 algo_params.encoder_kwargs.momentum_weight=0.95\
#   algo_params.loss_calculator_kwargs.use_batch_neg=False seed=$seed_var

    CUDA_VISIBLE_DEVICES="1" xvfb-run -a python -m il_representations.scripts.run_rep_learner with algo='MoCo' algo_params.representation_dim=10\
  benchmark.benchmark_name='dm_control' benchmark.n_envs=1 use_random_rollouts=False \
  pretrain_epochs=50 algo_params.encoder_kwargs.momentum_weight=0.95\
   algo_params.loss_calculator_kwargs.use_batch_neg=True seed=$seed_var

done
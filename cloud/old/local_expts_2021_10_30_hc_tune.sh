#!/bin/bash

set -euo pipefail

# use il_train.gail.total_timesteps to control how long this trains for
# (on perceptron, it seems like 16 processes give a load average of like 400 or
# so; I imagine the ideal is closer to 10 or fewer, so I'm going for
# 1/0.33*4 ~= 64/5 ~= 12 processes below)
DISPLAY=:420 exec python -m il_representations.scripts.pretrain_n_adapt with \
    ray_init_kwargs.address="localhost:42000" tune_run_kwargs.num_samples=1000 \
    tune_run_kwargs.resources_per_trial.{gpu=0.33,cpu=5} \
    gail_tune_hc_2021_10_30 $@

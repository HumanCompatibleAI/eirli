#!/bin/bash

set -euo pipefail

# use il_train.gail.total_timesteps to control how long this trains for
DISPLAY=:420 python -m il_representations.scripts.pretrain_n_adapt with \
    ray_init_kwargs.address="localhost:42000" tune_run_kwargs.num_samples=1000 \
    tune_run_kwargs.resources_per_trial.{gpu=0.15,cpu=4} gail_tune_hc_2021_10_30

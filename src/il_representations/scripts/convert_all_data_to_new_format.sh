#!/bin/bash

# Convert all demonstrations etc. to new data format

set -e

gen_demos() {
    xvfb-run -a python -m il_representations.scripts.mkdataset_demos with n_traj_total=1 $@
}

gen_random() {
    xvfb-run -a python -m il_representations.scripts.mkdataset_random with n_timesteps_min=64 $@
}

for mag_pfx in ClusterColour ClusterShape FixColour FindDupe MatchRegions MakeLine MoveToCorner MoveToRegion; do
    echo "Working on MAGICAL/${mag_pfx}"
    mag_env_opts=(env_cfg.benchmark_name=magical env_cfg.magical_env_prefix="${mag_pfx}")
    echo "Demonstrations:"
    gen_demos ${mag_env_opts[@]}
    echo "Random rollouts:"
    gen_random ${mag_env_opts[@]}
done

for dmc_name in finger-spin cheetah-run walker-walk cartpole-swingup reacher-easy ball-in-cup-catch; do
    echo "Working on DMC/${dmc_name}"
    dmc_env_opts=(env_cfg.benchmark_name=dm_control env_cfg.dm_control_env_name="${dmc_name}")
    echo "Demonstrations:"
    gen_demos ${dmc_env_opts[@]}
    echo "Random rollouts:"
    gen_random ${dmc_env_opts[@]}
done

for atari_name in PongNoFrameskip-v4; do
    echo "Working on Atari/${atari_name}"
    atari_env_opts=(env_cfg.benchmark_name=atari env_cfg.atari_env_id="${atari_name}")
    echo "Demonstrations:"
    gen_demos ${atari_env_opts[@]}
    echo "Random rollouts:"
    gen_random ${atari_env_opts[@]}
done

echo "Done!"
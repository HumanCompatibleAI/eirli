#!/bin/bash

# Convert all demonstrations etc. to new data format

set -e

gen_demos() {
    xvfb-run -a python -m il_representations.scripts.mkdataset_demos $@
}

for mag_pfx in ClusterColour ClusterShape FixColour FindDupe MatchRegions MakeLine MoveToCorner MoveToRegion; do
    echo "Converting MAGICAL/${mag_pfx}"
    gen_demos with env_cfg.benchmark_name=magical env_cfg.magical_env_prefix="${mag_pfx}"
done

for dmc_name in finger-spin cheetah-run walker-walk cartpole-swingup reacher-easy ball-in-cup-catch; do
    echo "Converting DMC/${dmc_name}"
    gen_demos with env_cfg.benchmark_name=dm_control env_cfg.dm_control_env_name="${dmc_name}"
done

for atari_name in PongNoFrameskip-v4; do
    echo "Converting Atari/${atari_name}"
    gen_demos with env_cfg.benchmark_name=atari env_cfg.atari_env_id="${atari_name}"
done

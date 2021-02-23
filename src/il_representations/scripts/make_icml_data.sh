#!/bin/bash

# Convert all demonstrations etc. to new data format

set -e

NEW_DATA_ROOT="data-icml"
# randomness used to generate random rollouts, and to select which subset of
# trajectories is used from the larger demonstration set
SEED=797821
MAGICAL_NTRAJ=5
DMC_NTRAJ=250
RAND_TIMESTEPS=250000

truncate_demos() {
    xvfb-run -a python -m il_representations.scripts.truncate_datasets_icml run \
        with seed="$SEED" new_data_root="$NEW_DATA_ROOT" $@
}

gen_demos() {
    # use "with n_traj_total=N" to write fewer demos
    xvfb-run -a python -m il_representations.scripts.mkdataset_demos run \
        with seed="$SEED" env_data.data_root="$NEW_DATA_ROOT" $@
}

gen_random() {
    xvfb-run -a python -m il_representations.scripts.mkdataset_random run \
        with seed="$SEED" env_data.data_root="$NEW_DATA_ROOT" \
        venv_opts.venv_parallel=True venv_opts.n_envs=8 \
        n_timesteps_min="$RAND_TIMESTEPS" $@
}

do_magical() {
    mag_pfx="$1"
    echo "Working on MAGICAL/${mag_pfx}"
    mag_env_opts=(env_cfg.benchmark_name=magical env_cfg.task_name="${mag_pfx}")

    echo "Truncating demonstrations to $MAGICAL_NTRAJ trajectories"
    truncate_demos ${mag_env_opts[@]} n_traj="$MAGICAL_NTRAJ"

    echo "repL-ifying demonstrations:"
    gen_demos ${mag_env_opts[@]}

    echo "Generating repL-ificated random rollouts:"
    gen_random ${mag_env_opts[@]}
}

do_dmc() {
    dmc_name="$1"
    echo "Working on DMC/${dmc_name}"
    dmc_env_opts=(env_cfg.benchmark_name=dm_control env_cfg.task_name="${dmc_name}")

    echo "Truncating demonstrations to $DMC_NTRAJ trajectories"
    truncate_demos ${dmc_env_opts[@]} n_traj="$DMC_NTRAJ"

    echo "repL-ifying demonstrations:"
    gen_demos ${dmc_env_opts[@]}

    echo "Generating repL-ificated random rollouts:"
    gen_random ${dmc_env_opts[@]}
}

do_magical ClusterColour &
do_magical ClusterShape &
do_magical FixColour &
do_magical FindDupe &
wait
do_magical MatchRegions &
do_magical MakeLine &
do_magical MoveToCorner &
do_magical MoveToRegion &
wait

# These use more memory because the trajectories are long. Thus we run them
# sequentially.
do_dmc finger-spin
do_dmc cheetah-run
do_dmc walker-walk
do_dmc cartpole-swingup
do_dmc reacher-easy
do_dmc ball-in-cup-catch

echo "Done!"

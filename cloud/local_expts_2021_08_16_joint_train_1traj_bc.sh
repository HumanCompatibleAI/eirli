#!/bin/bash

# Sanity test: when doing joint training, does a BC auxiliary objective help a
# BC primary objective when the BC auxiliary objective gets access to held-out
# data?

set -e

ray_address="localhost:42000"
ray_ncpus=1
base_cfgs=("n_batches=30000" "env_use_magical" "bc.short_eval_interval=2000")
repl_configs=("repl_noid" "repl_ap")
env_names=("MatchRegions" "MoveToCorner" "MoveToRegion")

submit_expt_pt2() {
    # submit joint training experiment to local Ray server (0.3 GPUs)
    python -m il_representations.scripts.joint_training_cluster \
        --ray-address="$ray_address" --ray-ncpus "$ray_ncpus" \
        --ray-ngpus 0.2 "${base_cfgs[@]}" "$@" &
}

launch_seed() {
    n_trajs=1
    # t_suffix is placed at end of name to indicate how many trajectories there are
    if [ "${n_trajs}" = None ]; then
        t_suffix="_all_traj"
    else
        t_suffix="_${n_trajs}_traj"
    fi

    for env_name in "${env_names[@]}"; do
        for repl_config in "${repl_configs[@]}"; do
            # using train variant demos
            submit_expt_pt2 \
                exp_ident="${repl_config}_demos_repl${t_suffix}" \
                "${repl_config[@]}" "env_cfg.task_name=${env_name}-Demo-v0" \
                "repl_data_demos" "bc.n_trajs=${n_trajs}"
        done
    done
}

# launch a bunch of seeds (should be able to do all at once, there's only
# 2*3*5=30 runs)
launch_seed
launch_seed
launch_seed
launch_seed
launch_seed
wait

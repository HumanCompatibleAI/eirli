#!/bin/bash

# Joint training repL + BC experiments

# FIXME(sam): here are the main things I want in here:
# - [DONE] Have both unrestricted runs and one-traj runs.
# - [DONE] Envs: MR, MTC, MTR
# - repL data configs: demos+random on train env, also that plus demos+random on
#   test env.
# - [DONE] RepL methods: TCPC-8, ID, FD, VAE

set -e

ray_address="localhost:42000"
ray_ncpus=1
base_cfgs=("n_batches=30000" "env_use_magical" "bc.short_eval_interval=2000")
repl_configs=("repl_noid" "repl_tcpc8" "repl_id" "repl_vae" "repl_fd")
env_names=("MatchRegions" "MoveToCorner" "MoveToRegion")

submit_expt_pt3() {
    # submit joint training experiment to local Ray server (0.3 GPUs)
    python -m il_representations.scripts.joint_training_cluster \
        --ray-address="$ray_address" --ray-ncpus "$ray_ncpus" \
        --ray-ngpus 0.3 "${base_cfgs[@]}" "$@" &
}

submit_expt_pt24() {
    # submit joint training experiment to local Ray server (0.24 GPUs)
    python -m il_representations.scripts.joint_training_cluster \
        --ray-address="$ray_address" --ray-ncpus "$ray_ncpus" \
        --ray-ngpus 0.24 "${base_cfgs[@]}" "$@" &
}

lower() {
    # convert string to lowercase
    echo "$1" | tr "[:upper:]" "[:lower:]"
}

launch_seed() {
    for n_trajs in 1 None; do
        # t_suffix is placed at end of name to indicate how many trajectories there are
        if [ "${n_trajs}" = None ]; then
            t_suffix="_all_traj"
        else
            t_suffix="_${n_trajs}_traj"
        fi

        for env_name in "${env_names[@]}"; do
            for repl_config in "${repl_configs[@]}"; do
                lower_env="$(lower "$env_name")"
                # using train variant demos
                submit_expt_pt3 \
                    exp_ident="${repl_config}_demos_repl${t_suffix}" \
                    "${repl_config[@]}" "env_cfg.task_name=${env_name}-Demo-v0" \
                    "repl_data_demos" "bc.n_trajs=${n_trajs}"

                # using train variant demos + random rollouts
                submit_expt_pt3 \
                    exp_ident="${repl_config}_demos_random_repl${t_suffix}" \
                    "${repl_config[@]}" "env_cfg.task_name=${env_name}-Demo-v0" \
                    "repl_data_demos_random" "bc.n_trajs=${n_trajs}"

                # using test variant demos
                submit_expt_pt3 \
                    exp_ident="${repl_config}_demos_test_variant_repl${t_suffix}" \
                    "${repl_config[@]}" "env_cfg.task_name=${env_name}-Demo-v0" \
                    "repl_data_${lower_env}_demos_test" "bc.n_trajs=${n_trajs}"
            done

            # "oracle" BC baselines that get access to test data
            submit_expt exp_ident="repl_noid_demos_test_variant_cheating${t_suffix}" \
                "env_cfg.task_name=${env_name}-Demo-v0" \
                "bc_data_${lower_env}_demos_test" \
                "repl_data_${lower_env}_demos_test" "bc.n_trajs=${n_trajs}"
        done
    done
}

# Ray was dying when I tried launching all jobs at once :(
launch_seed
wait
launch_seed
wait
launch_seed
wait
launch_seed
wait
launch_seed
wait

#!/bin/bash

# Converts demonstrations to webdataset format. Really just a combination of
# convert_all_data_to_new_format and convert_testall_data.sh.

set -e

declare -A magical_dirs_by_suffix=(
    ["demo"]="/scratch/sam/il-demos-quals/magical-demo"
    ["testall"]="/scratch/sam/il-demos-quals/magical-testall"
)

gen_demos() {
    # use "with n_traj_total=N" to write fewer demos
    xvfb-run -a python -m il_representations.scripts.mkdataset_demos run with "$@"
}

gen_random() {
    xvfb-run -a python -m il_representations.scripts.mkdataset_random run \
        with venv_opts.venv_parallel=True venv_opts.n_envs=8 n_timesteps_min=150000 "$@"
}

find_demo_dir() {
    env_name="$1";
    shift
    # convert CamelCaseNames to dashed-names-like-this, omitting anything in
    # input after the first dash (e.g. MoveToCorner-Demo-v0 -> move-to-corner)
    nocamel="$(echo "$env_name" \
        | cut -d - -f 1 \
        | sed 's/\([a-z]\)\([A-Z]\)/\1-\2/g' \
        | tr '[:upper:]' '[:lower:]')"
    # get the 'demo' or 'testall' part
    suffix="$(echo "$env_name" \
        | cut -d - -f 2 \
        | tr '[:upper:]' '[:lower:]')"
    # look up 'demo' or 'testall' part in hash
    search_dir="$(readlink -f "${magical_dirs_by_suffix[$suffix]}")"
    # now use wildcard match to look up corresponding directory
    real_path="$(find "${search_dir}" -name "${nocamel}*" | head -n 1)"
    if [[ -z "$real_path" ]]; then
        >&2 echo "Could not find directory for '${env_name}' in '${search_dir}'" \
             "(looked for something starting with '${nocamel}')"
        # exit 1 rather than return 1 seems a bit strong, but whatever
        exit 1
    fi
    echo "${real_path}"
}

for magical_env_name in ClusterColour-Demo-v0 \
        MoveToCorner-TestAll-v0 ClusterShape-Demo-v0 \
        FixColour-Demo-v0 FindDupe-Demo-v0 MatchRegions-Demo-v0 \
        MakeLine-Demo-v0 MoveToCorner-Demo-v0 MoveToRegion-Demo-v0 \
        MatchRegions-TestAll-v0 MoveToRegion-TestAll-v0; do
    echo "Generating demos for MAGICAL/${magical_env_name}"
    mag_env_opts=("env_cfg.benchmark_name=magical" "env_cfg.task_name=${magical_env_name}")
    gen_demos "${mag_env_opts[@]}" \
        "env_data.magical_demo_dirs.${magical_env_name}=$(find_demo_dir "$magical_env_name")" &
done
wait

for magical_env_pfx in ClusterColour \
        ClusterShape \
        FixColour FindDupe MatchRegions \
        MakeLine MoveToCorner MoveToRegion; do
    for magical_env_sfx in "-Demo-v0" "-TestAll-v0"; do
        magical_env_name="${magical_env_pfx}${magical_env_sfx}"
        echo "Generating rand. rollouts on MAGICAL/${magical_env_name}"
        mag_env_opts=("env_cfg.benchmark_name=magical" "env_cfg.task_name=${magical_env_name}")
        gen_random "${mag_env_opts[@]}" &
    done
done
wait

for dmc_name in finger-spin cheetah-run reacher-easy; do
    echo "Working on DMC/${dmc_name}"
    dmc_env_opts=("env_cfg.benchmark_name=dm_control" "env_cfg.task_name=${dmc_name}")
    echo "Demonstrations:"
    gen_demos "${dmc_env_opts[@]}" &
    echo "Random rollouts:"
    gen_random "${dmc_env_opts[@]}" &
done
wait

echo "Done!"

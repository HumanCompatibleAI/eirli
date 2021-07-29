#!/bin/bash

# Convert all my TestAll demonstrations to new format (& generate new rollouts).

set -e

data_dir="${HOME}/repos/magical/demos-ea-testall"

gen_demos() {
    # use "with n_traj_total=N" to write fewer demos
    xvfb-run -a python -m il_representations.scripts.mkdataset_demos run -D with "$@"
}

gen_random() {
    xvfb-run -a python -m il_representations.scripts.mkdataset_random run \
        with venv_opts.venv_parallel=True venv_opts.n_envs=8 n_timesteps_min=50000 "$@"
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
    # now use wildcard match to look up corresponding directory
    real_path="$(find "${data_dir}" -name "${nocamel}*" | head -n 1)"
    if [[ -z "$real_path" ]]; then
        >&2 echo "Could not find directory for '${env_name}' in '${data_dir}'" \
             "(looked for something starting with '${nocamel}')"
        # exit 1 rather than return 1 seems a bit strong, but whatever
        exit 1
    fi
    echo "${real_path}"
}

for mag_pfx in ClusterColour-TestAll-v0 ClusterShape-TestAll-v0 FixColour-TestAll-v0 FindDupe-TestAll-v0 MatchRegions-TestAll-v0 MakeLine-TestAll-v0 MoveToCorner-TestAll-v0 MoveToRegion-TestAll-v0; do
    echo "Working on MAGICAL/${mag_pfx}"
    mag_env_opts=("env_cfg.benchmark_name=magical" "env_cfg.task_name=${mag_pfx}")
    echo "Demonstrations:"
    gen_demos "${mag_env_opts[@]}" "env_data.magical_demo_dirs.${mag_pfx}=$(find_demo_dir "$mag_pfx")"
    echo "Random rollouts:"
    gen_random "${mag_env_opts[@]}""
done

echo "Done!"

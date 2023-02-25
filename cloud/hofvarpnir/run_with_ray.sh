#!/bin/bash

# Entrypoint for Docker container that runs Ray + Xvfb.
# The root process will always be the Ray server, so if Ray goes down then the
# entire outer container dies.
# However, the arguments you pass to the container will be executed in a
# subshell after the Ray server starts.
# This is a bit weird, but the point is that we can run multiple Python
# experiment scripts that all need a Ray server while having only one Ray
# instance.
#
# We really don't _need_ Ray on Hofvarpnir's infrastructure, but I don't want to
# remove it because all of our configs depend on it, and removing Ray would
# probably break something.

set -euo pipefail
ulimit -n 65536

# Start an X server
echo 'Starting Xvfb'
for job in $(pgrep '^Xvfb\b'); do kill "$job"; done \
    && rm -f ~/.Xauthority \
    && Xvfb -screen 0 640x480x16 -nolisten tcp -nolisten unix \
        -auth ~/.Xauthority -maxclients 2048 :0 &
disown
export DISPLAY=:0

# Start a loop to kill Ray once all the tasks finish
ray_kill_loop() {
    # wait 2 minutes before first check, since it might take a while to queue up
    # jobs (this means the container will take a while to actually die if the
    # desired command fails)
    sleep 120
    while true; do
        echo "Iterating through Ray kill loop"
        # test whether Ray is running any tasks
        resource_regex=' [0-9]+\.[0-9]+/[0-9]+\.[0-9]+ (CPU|GPU|accelerator_type).*'
        resources_used="$(ray status | grep -E "$resource_regex" | sed -E 's? ([0-9]+\.[0-9]+)/.*?\1?' || echo "command failed")"
        resources_used_line="$(echo "$resources_used" | paste -s -d ' ')"
        # resources_used should have 1 or more lines (probably 3 lines, for
        # CPU/GPU/accelerator_type:G) with something _other than_ 0.0 on at
        # least one line; if we don't get this then we panic and kill everything
        if [[ "$resources_used_line" =~ ^(0.0)?(\ 0.0)*$ ]]; then
            echo "Resources line was '$resources_used_line' (this means no resources used if it's all 0.0s, otherwise an error)"
            echo "Gracefully stopping Ray"
            sleep 5
            ray stop || echo "Error $? shutting down Ray"
            # now kill any process starting with Ray, Xvfb or python
            # (I don't think this is necessary if Ray stops, since Ray should be at the root of the container)
            for job in pgrep '^(ray|Xvfb|python)'; do kill -9 "$job" || echo "Error $? killing other process #$job"; done
            sleep 5
            exit 0
        fi
        # shorter sleep here, since jobs should be constantly running
        sleep 30
    done
    # if this is doing to die then it should kill EVERYTHING on its way out
}
ray_kill_loop &
disown

# this function executes "$@" in a subshell after waiting 10s
# (meant to be executed in a disowned background job)
launch_actual_command() {
    sleep 10
    exec "$@"
}
launch_actual_command "$@" &
disown
# get ID of last launched job
actual_command_pid="$!"

check_command_finished() {
    # loop to check if actual command has finished (& warn when it does)
    while true; do
        if ! ps -p "$actual_command_pid" > /dev/null; then
            echo "Actual command finished"
            break
        fi
        sleep 2
    done
}
check_command_finished &

# Start a blocking Ray head server; exits once ray stops
echo "Starting Ray"
# use 10GB object store memory (arg is in bytes); if we don't do this then Ray
# tries to use some insane amount of /dev/shm
exec ray start --head --port=42000 --block \
    --object-store-memory=10000000000
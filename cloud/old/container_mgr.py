#!/usr/bin/env python3
"""Script to manage a monolothic Docker container/mini-VM with Ray, Xvfb, tmux,
and a bunch of experiment processes."""
import argparse
from dataclasses import dataclass
from enum import Enum
import os
import shlex
import subprocess
import re
import pathlib
import io
from typing import Optional


def latest_image_name(image_name: str) -> str:
    """Iterate over all tags an image and return the latest one.

    Shells out to Docker to figure out which tags `image_name` has."""
    cmd = ["docker", "image", "ls", "--format", "{{.Tag}}", image_name]
    output = subprocess.check_output(cmd, shell=False).decode("utf-8")
    tags = output.strip().splitlines()
    tag_re = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+-r[0-9]+$")
    lines_by_match = []
    for tag in tags:
        # extract numbers from tag using tag_re, converting each group of the
        # match to an integer
        match = tag_re.match(tag)
        if match is None:
            continue
        groups = [int(g) for g in match.groups()]
        lines_by_match.append((groups, tag))

    # error out if lines_by_match is empty
    if not lines_by_match:
        raise Exception(
            f"No valid 'NNNN.NN.NN-rN'-style tags found for image "
            f"{image_name} (all tags: {', '.join(tags)}"
        )

    # now find latest tag
    _, latest_tag = max(lines_by_match)

    return f"{image_name}:{latest_tag}"


# make enum of container states (CONTAINER_RUNNING, CONTAINER_DOES_NOT_EXIST, or OTHER_STATUS)
class ContainerStatus(Enum):
    RUNNING = "running"
    DOES_NOT_EXIST = "does not exist"
    OTHER_STATUS = "other"


@dataclass
class ContainerStatusMessage:
    status: ContainerStatus
    message: Optional[str]


def get_container_status(container_name: str) -> ContainerStatusMessage:
    """Return the state of the container with name `container_name`."""
    cmd = ["docker", "inspect", "--format", "{{.State.Status}}", container_name]
    try:
        output = subprocess.check_output(cmd, shell=False).decode("utf-8")
    except subprocess.CalledProcessError:
        return ContainerStatusMessage(
            status=ContainerStatus.DOES_NOT_EXIST, message=None
        )
    output = output.strip()
    if output == "running":
        status = ContainerStatus.RUNNING
    else:
        status = ContainerStatus.OTHER_STATUS
    return ContainerStatusMessage(status=status, message=output)


def get_ilr_code_dir() -> pathlib.Path:
    """Get the location of the IL representations repository."""
    # return parent of the directory that contains this file
    return pathlib.Path(__file__).absolute().parent.parent


def start(args):
    # check if container is already running
    container_status = get_container_status(args.name)
    if container_status.status != ContainerStatus.DOES_NOT_EXIST:
        # container exists already
        if container_status.status == ContainerStatus.RUNNING:
            # bail out if it's running
            print(f"Container {args.name} is already running. Exiting early.")
            return
        else:
            # raise an error if it has any other status
            raise Exception(
                f"Container '{args.name}' is in state "
                f"{container_status.message}; please fix this manually "
                "(e.g. by removing the container if is stopped)."
            )

    # if args.image does not have a tag, add the latest one
    if ":" not in args.image:
        image_name = latest_image_name(args.image)
        print(f"Using inferred latest image name '{image_name}'")
    else:
        image_name = args.image

    # Start docker container if it's not running already. Main process will be
    # Ray.
    ilr_code_dir = get_ilr_code_dir()
    ray_start_cmd = [
        "docker",
        "run",
        "-dti",
        "--mount",
        f"type=bind,src={ilr_code_dir},dst=/root/il-rep",
        "--mount",
        f"type=bind,src={ilr_data_dir},dst=/root/il-rep/data",
        "--mount",
        f"type=bind,src={ilr_runs_dir},dst=/root/il-rep/runs",
        "--workdir",
        "/root/il-rep" "--name",
        args.name,
        image_name,
        "ray",
        "start",
        "--head",
        "--port",
        str(args.port),
        "--block",
    ]
    # run ray_start_cmd without capturing output
    subprocess.check_call(ray_start_cmd)

    try:
        # misc. setup scripts (start X server, start tmux, etc.)
        docker_setup_script = io.BytesIO(
            "set -euo pipefail; "
            "touch ~/.Xauthority && chmod 0600 ~/.Xauthority; "
            "./cloud/ray-init-scripts/start_x_server.sh; "
            f"tmux new-session -ds '{args.name}'"
            "exit;\n".encode("utf-8")
        )
        docker_setup_cmds = [
            "docker",
            "exec",
            "-i",
            args.name,
            "bash",
        ]
        subprocess.check_call(ray_start_cmd, stdin=docker_setup_script)
    except Exception as ex:
        # terminate the container if we run into an error
        print(f"Error running docker setup script: {ex}")
        print("Stopping container...")
        subprocess.check_call(["docker", "stop", args.name])
        print("Re-raising exception...")
        raise


def assert_container_running(container_name: str) -> None:
    """Check that container with name `container_name` is running. Raise
    `Exception` if it is not."""
    status = get_container_status(container_name)
    if status.status != ContainerStatus.RUNNING:
        raise Exception(
            f"Container is not running. Container status: {status.status} "
            f"(message: '{status.message}')"
        )


def tmux_attach(args):
    """Attach to the tmux session with name `args.name` in container with name
    `args.name`."""
    assert_container_running(args.name)
    docker_setup_cmds = [
        "docker",
        "exec",
        "-it",
        args.name,
        "tmux",
        "a",
        args.name,
    ]
    os.execvp("docker", docker_setup_cmds)


def tmux_exec(args):
    """Execute a command in the tmux session with name `args.name` in container
    with name `args.name`. Will execute in a new tmux window."""
    assert_container_running(args.name)
    # these commands are read by bash initially
    command_file = io.BytesIO(
        (
            "source ~/.bashrc; " + " ".join(shlex.quote(arg) for arg in args.command)
        ).encode("utf-8")
    )
    # create a new bash session and run the supplied commmand with send-keys
    tmux_exec_cmds = [
        "docker",
        "exec",
        "-it",
        args.name,
        "tmux",
        "new-window",
        "-n",
        args.name,
        "bash",
        ### SKETCHY COPILOT STUFF:
        ### # make bash execute commands on stdin and then drop to an interactive shell
        ### "-c",
        ### "cat <&0 | bash",
        ### # send the command file to the bash process
        ### "--",
        ### "-",
    ]


# create new parser with three subcommands/subparsers:
# - start: start the container.
# - exec: run a command in the container.
# - tmux-exec: run a command in the container in a new tmux window.
parser = argparse.ArgumentParser()
parser.add_argument("--name", default="mono", help="name of the container")
parser.add_argument(
    "--image",
    default="humancompatibleai/il-representations",
    help="image to use (omit the tag to use the most recent, ordered by date)",
)
subparsers = parser.add_subparsers()

start_parser = subparsers.add_parser("start")
start_parser.set_defaults(func=start)
start_parser.add_argument(
    "--port", type=int, default=42000, help="port for Ray to bind on"
)

tmux_attach_parser = subparsers.add_parser("tmux-attach")
tmux_attach_parser.set_defaults(func=tmux_attach)

tmux_exec_parser = subparsers.add_parser("tmux-exec")
tmux_exec_parser.set_defaults(func=tmux_exec)
# accept arbitrary number of positional arguments, which will be used as a
# command to execute in tmux
tmux_exec_parser.add_argument(
    "command", nargs="+", help="command to execute in new tmux window"
)


def main():
    # XXX(stoyer): this file seems unnecessary
    raise NotImplementedError(
        "this is untested and probably broken; I gave up on it after the lab "
        "moved to Kubernetes-based execution and scheduling for our compute "
        "resources"
    )
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

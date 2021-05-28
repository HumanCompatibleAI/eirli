#!/usr/bin/env python3
"""Thin wrapper to run joint_training.py on a Ray server."""

import argparse
import faulthandler
import logging
import signal

from docopt import docopt
import ray
from sacred.arg_parser import get_config_updates
from sacred.utils import ensure_wellformed_argv


class ArgumentParseFailed(Exception):
    """Raised when Sacred fails to parse arguments."""


def hacky_run_commandline(exp, argv):
    """Version of `Experiment.run_commandline` that DOESN'T DO SYS.EXIT(1) WHEN
    IT FAILS TO PARSE ARGS grrrrr."""
    argv = ensure_wellformed_argv(argv)
    short_usage, usage, internal_usage = exp.get_usage()
    args = docopt(internal_usage, [str(a) for a in argv[1:]], help=False)

    cmd_name = args.get("COMMAND") or exp.default_command
    config_updates, named_configs = get_config_updates(args["UPDATE"])

    err = exp._check_command(cmd_name)
    if not args["help"] and err:
        raise ArgumentParseFailed(short_usage + "\n" + err)

    if exp._handle_help(args, usage):
        # I don't know what this branch is even for in the original code
        raise ArgumentParseFailed(
            "given help command somehow (this shouldn't happen)")

    return exp.run(
        cmd_name,
        config_updates,
        named_configs,
        info={},
        meta_info={},
        options=args,
    )


def run_joint_training_remote(extra_args):
    """This function executes the experiment. It is intended to be wrapped with
    ray.remote and run as a Ray task."""
    from il_representations.scripts.joint_training import add_fso, train_ex
    add_fso()
    argv = ['placeholder arg because only argv[1:] is used', 'train']
    if extra_args:
        argv.append('with')
        argv.extend(extra_args)
    hacky_run_commandline(train_ex, argv)


def main(args, sacred_args):
    """Main entry point on the Ray client. Essentially just spins up a Ray task
    & then exits once it terminates."""
    faulthandler.register(signal.SIGUSR1)
    logging.basicConfig(level=logging.INFO)

    ray_opts = {}
    if args.ray_ncpus is not None:
        ray_opts['num_cpus'] = args.ray_ncpus
    if args.ray_ngpus is not None:
        ray_opts['num_gpus'] = args.ray_ngpus

    logging.info(
        f"Running remotely on Ray instance at {args.ray_address} with "
        f"options {ray_opts}")

    ray.init(address=args.ray_address)

    if ray_opts:
        remote_decorator = ray.remote(**ray_opts)
    else:
        # ray.remote doesn't support passing in no kwargs; must pass in
        # function directly instead (weird but whatever)
        remote_decorator = ray.remote
    remote_handle = remote_decorator(run_joint_training_remote)
    remote_run = remote_handle.remote(sacred_args)
    return ray.get(remote_run)


# allow_abbrev stops argparse from swallowing genuine Sacred arguments that
# happen to be prefixes of the --ray-* arguments below. e.g. you can do
# "./script.py --ray-address localhost:6060 some_namedconfig variable.foo=42".
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--ray-address',
                    default='localhost:42000',
                    help='address of Ray server')
parser.add_argument('--ray-ncpus',
                    default=None,
                    type=float,
                    help='number of CPUs for task')
parser.add_argument('--ray-ngpus',
                    default=None,
                    type=float,
                    help='number of GPUs for task')

if __name__ == '__main__':
    main(*parser.parse_known_args())

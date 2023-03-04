#!/usr/bin/env python3
"""This is a wrapper around `joint_training_cluster` for use with `ray submit`.
As with `submit_pretrain_n_adapt.py`, you can omit the "with" part of the
command when you use this (it will automatically prepend 'with' and some
additional default settings to whatever you pass it)."""

import os
import sys

from il_representations.scripts.joint_training_cluster import main, parser

if __name__ == '__main__':
    os.chdir(os.path.expanduser('~/il-rep/'))
    main(*parser.parse_known_args(args=[
        # the autoscaler always launches a Ray server on this address (which I
        # think is the default Redis port), so we connect to that
        '--ray-address=localhost:6379',
        *sys.argv[1:],
    ]))

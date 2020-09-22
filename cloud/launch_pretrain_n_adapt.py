#!/usr/bin/env python3
"""This is a wrapper around `pretrain_n_adapt` for use with `ray submit`. You
should omit the "with" part of the command when you use this (it will
automatically prepend 'with' and some additional default settings to whatever
you pass it)."""

import os
import sys

from il_representations.scripts.pretrain_n_adapt import main

if __name__ == '__main__':
    os.chdir(os.path.expanduser('~/il-rep/'))
    main([
        # the autoscaler always launches a Ray server on this address;
        # we want to connect to it instead of making a new one
        sys.argv[0], 'run', 'with', 'ray_init_kwargs.address=localhost:6379',
        *sys.argv[1:]])

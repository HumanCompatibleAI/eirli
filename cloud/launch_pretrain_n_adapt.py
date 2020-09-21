#!/usr/bin/env python3
"""This is a wrapper around `pretrain_n_adapt` for use with `ray submit`."""

import os
from il_representations.scripts.pretrain_n_adapt

def main():
    os.chdir(os.path.expanduser('~/il-rep/'))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Recursively collate repL and IL training configs from some directory.
Deduplicate the configs (using equivalence modulo seed), group the configs by
exp_ident and benchmark/task name, and print out to console."""
import argparse
import copy
import io
import operator
import os
import pprint
import sys
from typing import Iterable, Iterator, List, TextIO, Tuple

import jsonpickle
import tqdm

from il_representations.utils import hash_configs as _hash_configs

parser = argparse.ArgumentParser()
parser.add_argument('search_dir',
                    help='directory to search for config files in')


def config_generator(search_dir: str,
                     *,
                     config_name: str = 'config.json') -> Iterator[str]:
    """Recursively walk a directory looking for config.json files."""
    for root_dir, subdirs, files in os.walk(search_dir,
                                            followlinks=True,
                                            topdown=True):
        if config_name in files:
            yield os.path.join(root_dir, config_name)


def filter_configs(
    config_generator: Iterator[str],
    *,
    allowed_subdirs: Iterable[str] = ('repl', 'il_train')
) -> Iterator[Tuple[str, str]]:
    """Filter config.json paths returned by some config generator to include
    only those paths whose grandparent directory is `allowed_subdirs`."""
    allowed_set = frozenset(allowed_subdirs)
    for yielded_dir in config_generator:
        dirname = os.path.dirname(os.path.dirname(yielded_dir))
        base_dir = os.path.basename(dirname)
        if base_dir in allowed_set:
            yield base_dir, yielded_dir


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as fp:
        # jsonpickle doesn't seem to support loading straight from a buffer
        # (maybe because it can't control framing?)
        config_string = fp.read()
        return jsonpickle.decode(config_string)


def hash_config(config_dict: dict, *,
                forbidden=('seed', 'encoder_path')) -> str:
    """Hash config dict, modulo seed, encoder path, etc."""
    filtered = copy.copy(config_dict)
    for forbidden_key in forbidden:
        if forbidden_key in filtered:
            del filtered[forbidden_key]
    return _hash_configs(filtered)


def format_exp_ident(config_dict: dict, *, key: str = 'exp_ident') -> str:
    if key in config_dict:
        return str(config_dict[key])
    return 'missing'


def format_task_and_benchmark(config_dict: dict) -> str:
    try:
        env_cfg = config_dict['env_cfg']
        bench_name = env_cfg['benchmark_name']
        task_name = env_cfg['task_name']
        return f'{bench_name}/{task_name}'
    except KeyError:
        return 'missing/incomplete'


def collect_configs(subdir_path_iter: Iterator[Tuple[str, str]]) -> List[dict]:
    by_hash = {}
    wrapped_iter = tqdm.tqdm(subdir_path_iter,
                             desc='loading config.json files',
                             unit=' files')
    for base_dir, config_path in wrapped_iter:
        config_dict = load_config(config_path)
        config_hash = hash_config(config_dict)
        if config_hash in by_hash:
            # We don't save the other dicts since we won't have space to print
            # them all out. We WILL save paths though, just so we can dig
            # through the relevant paths at our leisure :)
            by_hash[config_hash]['paths'].append(config_path)
        else:
            by_hash[config_hash] = {
                'paths': [config_path],
                'hash': config_hash,
                'config_dict': config_dict,
                'subdir': base_dir,
            }
    return sorted(by_hash.values(), key=operator.itemgetter('hash'))


class IndentedStream(io.TextIOBase, TextIO):
    """Text mode pseudo-file that indents text written to it."""
    def __init__(self, indent: int, *, inner_stream: TextIO = sys.stdout):
        self.inner_stream = inner_stream
        self.indent_str = ' ' * indent
        self.indent_next = True

    def write(self, data: str) -> int:
        written = 0
        for char in data:
            if self.indent_next:
                written += self.inner_stream.write(self.indent_str)
                self.indent_next = False
            written += self.inner_stream.write(char)
            if char == '\n':
                self.indent_next = True
        return written

    def flush(self):
        self.inner_stream.flush()

    def close(self):
        raise NotImplementedError("You should close inner_stream directly!")


def common_prefix_len(strs: List[str]) -> int:
    """Find length of common prefix in a list of strings."""
    first_string = strs[0]
    longest_init = len(first_string)
    for this_string in strs[1:]:
        enum_zip_iter = enumerate(zip(first_string[:longest_init],
                                      this_string),
                                  start=1)
        for prefix_len, (c1, c2) in enum_zip_iter:
            if c1 == c2:
                longest_init = prefix_len
            else:
                break
    return longest_init


def strip_common_prefix(strs: List[str]) -> List[str]:
    pl = common_prefix_len(strs)
    prefix = strs[0][:pl]
    with_prefix_removed = [st[pl:] for st in strs]
    return prefix, with_prefix_removed


def do_final_printing(hash_dict: dict):
    """Print out one of the subdicts returned as values by
    collect_configs()."""
    print(f"Config for '{hash_dict['hash']}'")
    print('  Experiment type:', hash_dict['subdir'])
    print('  exp_ident:', format_exp_ident(hash_dict['config_dict']))
    print('  Task info:', format_task_and_benchmark(hash_dict['config_dict']))

    # this stream is for pprinting, so we get top-level indent offset on the
    # printed data structures
    indent_stream_4 = IndentedStream(4)
    print('  Example config dict:')
    pprint.pprint(hash_dict['config_dict'],
                  width=120,
                  sort_dicts=True,
                  stream=indent_stream_4)

    n_paths = len(hash_dict['paths'])
    path_prefix, with_path_prefix_removed = strip_common_prefix(
        hash_dict['paths'])
    if n_paths <= 1:
        print(f"  Hash corresponds to path {hash_dict['paths']}")
    else:
        print(f'  Hash corresponds to {n_paths} paths in {path_prefix}:')
        pprint.pprint(with_path_prefix_removed,
                      width=120,
                      stream=indent_stream_4)


def main(args):
    search_dir = args.search_dir  # str
    subdir_path_iter = filter_configs(config_generator(search_dir))
    collected = collect_configs(subdir_path_iter)
    for hash_dict in collected:
        do_final_printing(hash_dict)
        # print extra newline
        print()


if __name__ == '__main__':
    main(parser.parse_args())

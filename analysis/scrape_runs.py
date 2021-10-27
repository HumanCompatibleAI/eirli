"""Tuning runs are not saving their search-alg-*.pkl files (maybe because
`CheckpointFIFOScheduler` is broken), so I'm writing this script to collate
some existing runs. It's a day before the NeurIPS BT rebuttal period closes, so
this is going to be kind of hacky. I should fix the tuning script if I want to
do more HP tuning in future."""

import collections
import glob
import os
import pathlib

import jsonpickle
import numpy as np
import sacred
from sacred import Experiment

sacred.SETTINGS['CAPTURE_MODE'] = 'no'  # workaround for sacred issue#740
collate_ex = Experiment('collate', ingredients=[])


@collate_ex.config
def base_config():
    # where to search for eval.json files
    search_dir = os.getcwd()

    # number of 'best' runs to show
    n_best = 10

    _ = locals()
    del _


def flatten_dict(d):
    """Flatten a nested dict into a single-level dict with
    'keys/separated/like/this'.

    Copy-pasted from plot.ipynb."""
    out_dict = {}
    if isinstance(d, dict):
        key_iter = d.items()
    else:
        assert isinstance(d, list), type(d)
        # we flatten lists into dicts of the form {0: <first elem>, 1: <second
        # elem>, …}
        key_iter = ((str(idx), v) for idx, v in enumerate(d))
    for key, value in key_iter:
        if isinstance(value, (dict, list)):
            value = flatten_dict(value)
            for subkey, subvalue in value.items():
                out_dict[f'{key}/{subkey}'] = subvalue
        else:
            out_dict[key] = value
    return out_dict


def order_dict(d):
    """Convert a dict to an OrderedDict with sorted keys (non-recursive)."""
    return collections.OrderedDict(sorted(d.items()))


def find_common_items(dicts):
    """Given an iterable of dicts, this function creates a dict containing only
    the items that are common between all of them."""
    common_items = None
    for d in dicts:
        if common_items is None:
            # set common_items to be a copy of the first dict
            common_items = dict(d)
            continue

        # delete any keys that are not present in both dicts
        for key in common_items.keys() - d.keys():
            del common_items[key]

        for key, value in list(common_items.items()):
            if value != d[key]:
                del common_items[key]

    if common_items is None:
        common_items = {}

    return common_items


def print_dict(d, indent=2):
    prefix = ' ' * indent
    for k, v in d.items():
        print(prefix, k, ': ', v, sep='')


@collate_ex.main
def run(search_dir, n_best):
    pattern = pathlib.Path(search_dir) / '**' / 'il_test' / '*' / 'eval.json'
    eval_iter = glob.iglob(str(pattern), recursive=True)
    records = []
    for eval_path in eval_iter:
        with open(eval_path, 'r') as fp:
            eval_json_str = fp.read()
        eval_contents = jsonpickle.loads(eval_json_str)
        train_return = eval_contents['train_level']['return_mean']
        test_return = eval_contents['test_level']['return_mean']
        # usually the policy is in the same directory as the config.json file
        pol_path = pathlib.Path(eval_contents['policy_path'])
        config_path = pol_path.parent / 'config.json'
        with open(config_path, 'r') as fp:
            config_json_str = fp.read()
        config_dict = jsonpickle.loads(config_json_str)
        gail_config = config_dict['gail']
        flat_gail_config = order_dict(flatten_dict(gail_config))
        records.append({
            'train_return': train_return,
            'test_return': test_return,
            'config': flat_gail_config,
        })

    # extract the config keys that are always the same for ALL runs
    common_config = order_dict(find_common_items(r['config'] for r in records))
    # add an extra key to each record for the deduplicated config dict, which
    # does not contain any of the common config keys
    for record in records:
        record['simple_config'] = collections.OrderedDict(
            (k, v) for k, v in record['config'].items()
            if k not in common_config)

    # plot percentiles for return
    returns_by_variant = {
        'train': [r['train_return'] for r in records],
        'test': [r['test_return'] for r in records],
    }
    for variant in ['train', 'test']:
        returns = returns_by_variant[variant]
        n_obs = len(returns)
        print(f'Distribution of returns for {variant} environment '
              f'(computed on {n_obs} observations):')
        percentiles = [10, 25, 50, 75, 90]
        percentile_values = np.percentile(returns, percentiles)
        for percentile, value in zip(percentiles, percentile_values):
            print(f'  {percentile}%ile: {value}')

    # select the K best runs
    if len(records) < n_best:
        print(f'You asked for the n_best={n_best} best runs, but this script '
              f'found only {len(records)} on disk.')
    sorted_records = sorted(records, key=lambda r: r['train_return'],
                            reverse=True)
    best_records = sorted_records[:n_best]
    print(f'Showing configs for the {len(best_records)} best runs, sorted by '
          'train environment score')

    # print returns
    print('Train returns for the best:',
          ', '.join('%.3g' % r['train_return'] for r in best_records))
    print('Test returns for the best: ',
          ', '.join('%.3g' % r['test_return'] for r in best_records))

    # print shared config
    print('\nCommon config for all runs:')
    print_dict(common_config)

    # print range of config values for each key
    print('\nAdditional config keys for best runs:')
    all_keys = set()
    for record in best_records:
        all_keys.update(record['simple_config'].keys())
    big_best_config_dict = collections.OrderedDict([
        (k, [r['simple_config'].get(k) for r in best_records])
        for k in all_keys
    ])

    def stringify(x):
        if isinstance(x, float):
            return '%.4g' % x
        return str(x)

    # TODO(sam): smarter stringification based on type (mostly: it should print
    # shorter floats somehow)
    stringified_dict = collections.OrderedDict([
        (k, ', '.join(map(stringify, v)))
        for k, v in big_best_config_dict.items()
    ])
    print_dict(stringified_dict)


def unify_dicts(dicts):
    all_keys = set()
    for d in dicts:
        all_keys.update(d.keys())


def main(argv=None):
    """We have a separate main() function so that we can execute this code from
    other scripts."""
    collate_ex.run_commandline(argv)


if __name__ == '__main__':
    main()

"""
This file should support:
1. Running representation pretrain for X seeds, then adapt on an IL algo for Y seeds.
2. Grid search over different configurations

Some design questions:
1. Is it possible to collect the above runs in the same folder?
"""
import itertools
from sacred import Experiment
from il_representations.scripts.run_rep_laerner import represent_ex
from il_representations.scripts.il_train import il_train_ex

pretrain_adapt_ex = Experiment('pretrain_n_adapt')


@pretrain_adapt_ex.named_config
def grid_search_config():
    pretrain_params = {'seed': [1, 2, 3], 'lr': [0.1, 0.2]}
    pretrain_configs = get_grid_combinations(pretrain_params)

    adapt_params = {'seed': [4, 5, 6], 'lr': [0.01, 0.02]}
    adapt_configs = get_grid_combinations(adapt_params)


@pretrain_adapt_ex.named_config
def pretrain_n_adapt():
    pretrain_configs = [{}]
    adapt_config = [{}]


def get_grid_combinations(configs):
    """
    This method will return a list of key-value combinations over a dictionary. It's useful for conducting grid search
    over specified hyperparameters.
    Example:
        input: {1: [a, b], 2: [c, d]}
        output: [{1: a, 2: c}, {1: a, 2: d}, {1: b, 2: c}, {1: b, 2: d}]
    """
    keys, values = zip(*configs.items())
    config_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return config_grid


@represent_ex.main
def run(pretrain_configs, adapt_configs):
    for pretrain_config in pretrain_configs:
        # Run representation pretrain
        pretrain_run = represent_ex.run(config_updates=pretrain_config)

    for adapt_config in adapt_configs:
        # Run imitation learning adaptation
        adapt_run = il_train_ex(config_updates=adapt_config)



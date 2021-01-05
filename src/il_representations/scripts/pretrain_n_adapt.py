import collections
import copy
import enum
import logging
import os
import os.path as osp
import weakref

import numpy as np
import ray
from ray import tune
from ray.tune.integration.docker import DockerSyncer
from ray.tune.schedulers import FIFOScheduler
from ray.tune.suggest.skopt import SkOptSearch
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
import skopt

from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)
from il_representations.scripts import experimental_conditions  # noqa: F401
from il_representations.scripts.hp_tuning import make_hp_tuning_configs
from il_representations.scripts.il_test import il_test_ex
from il_representations.scripts.il_train import il_train_ex
from il_representations.scripts.run_rep_learner import represent_ex
from il_representations.scripts.utils import detect_ec2, sacred_copy, update

sacred.SETTINGS['CAPTURE_MODE'] = 'sys'  # workaround for sacred issue#740
chain_ex = Experiment(
    'chain',
    ingredients=[
        # explicitly list every ingredient we want to configure
        represent_ex,
        il_train_ex,
        il_test_ex,
        env_cfg_ingredient,
        env_data_ingredient,
        venv_opts_ingredient,
    ])
cwd = os.getcwd()


class StagesToRun(str, enum.Enum):
    """These enum flags are used to control whether the script tunes RepL, or
    IL, or both."""
    REPL_AND_IL = "REPL_AND_IL"
    REPL_ONLY = "REPL_ONLY"
    IL_ONLY = "IL_ONLY"


# Add configs to experiment for hyperparameter tuning
# This is to allow us to separate out tuning configs into their own file
make_hp_tuning_configs(chain_ex)


def get_stages_to_run(stages_to_run):
    """Convert a string (or enum) to StagesToRun object."""
    upper_str = stages_to_run.upper()
    try:
        stage = StagesToRun(upper_str)
    except ValueError as ex:
        options = [f"'{s.name}'" for s in StagesToRun]
        raise ValueError(
            f"Could not convert '{stages_to_run}' to StagesToRun ({ex}). "
            f"Available options are {', '.join(options)}")
    return stage


class CheckpointFIFOScheduler(FIFOScheduler):
    """Variant of FIFOScheduler that periodically saves the given search
    algorithm. Useful for, e.g., SkOptSearch, where it is helpful to be able to
    re-instantiate the search object later on."""

    # FIXME: this is a stupid hack, inherited from another project. There
    # should be a better way of saving skopt internals as part of Ray Tune.
    # Perhaps defining a custom trainable would do the trick?
    def __init__(self, search_alg):
        self.search_alg = weakref.proxy(search_alg)

    def on_trial_complete(self, trial_runner, trial, result):
        rv = super().on_trial_complete(trial_runner, trial, result)
        # references to _local_checkpoint_dir and _session_dir are a bit hacky
        checkpoint_path = os.path.join(
            trial_runner._local_checkpoint_dir,
            f'search-alg-{trial_runner._session_str}.pkl')
        self.search_alg.save(checkpoint_path + '.tmp')
        os.rename(checkpoint_path + '.tmp', checkpoint_path)
        return rv


def expand_dict_keys(config_dict):
    """Some Ray Tune hyperparameter search options do not supported nested
    dictionaries for configuration. To emulate nested dictionaries, we use a
    plain dictionary with keys of the form "level1:level2:…". . The colons are
    then separated out by this function into a nested dict (e.g. {'level1':
    {'level2': …}}). Example:

    >>> expand_dict_keys({'x:y': 42, 'z': 4, 'x:u:v': 5, 'w': {'s:t': 99}})
    {'x': {'y': 42, 'u': {'v': 5}}, 'z': 4, 'w': {'s': {'t': 99}}}
    """
    dict_type = type(config_dict)
    new_dict = dict_type()

    for key, value in config_dict.items():
        dest_dict = new_dict

        parts = key.split(':')
        for part in parts[:-1]:
            if part not in dest_dict:
                # create a new sub-dict if necessary
                dest_dict[part] = dict_type()
            else:
                assert isinstance(dest_dict[part], dict)
            dest_dict = dest_dict[part]
        if isinstance(value, dict):
            # recursively expand nested dicts
            value = expand_dict_keys(value)
        dest_dict[parts[-1]] = value

    return new_dict


def run_single_exp(inner_ex_config, shared_configs, tune_config_updates,
                   log_dir, exp_name):
    """
    Run a specified experiment. We could not pass each Sacred experiment in
    because they are not pickle serializable, which is not supported by Ray
    (when running this as a remote function).

    params:
        inner_ex_config: The current experiment's default config.
        shared_configs: a dict containing config values that can be shared
            between experiments (keys should be limited to 'env_cfg',
            'env_data', and 'venv_opts').
        tune_config_updates: The config generated by Ray tune for
            hyperparameter tuning
        log_dir: The log directory of current chain experiment.
        exp_name: Specify the experiment type in ['repl', 'il_train',
            'il_test']
    """
    # we need to run the workaround in each raylet, so we do it at the start of
    # run_single_exp
    sacred.SETTINGS['CAPTURE_MODE'] = 'sys'  # workaround for sacred issue#740
    from il_representations.scripts.il_test import il_test_ex
    from il_representations.scripts.il_train import il_train_ex
    from il_representations.scripts.run_rep_learner import represent_ex

    if exp_name == 'repl':
        inner_ex = represent_ex
        allowed_shared_config_keys = {'env_cfg', 'env_data'}
    elif exp_name == 'il_train':
        inner_ex = il_train_ex
        allowed_shared_config_keys = {'env_cfg', 'env_data', 'venv_opts'}
    elif exp_name == 'il_test':
        inner_ex = il_test_ex
        allowed_shared_config_keys = {'env_cfg', 'venv_opts'}
    else:
        raise ValueError(f"cannot process exp type '{exp_name}'")

    all_allowed_keys = {
        # Depending on the value of exp_name, some of these keys will not be
        # used in this function, so we ignore them later. (this is a complete
        # list of all keys that could be used over all values of exp_name,
        # though)
        'repl', 'il_train', 'il_test', 'env_cfg', 'env_data', 'venv_opts',
    }
    assert tune_config_updates.keys() <= all_allowed_keys, \
        tune_config_updates.keys()
    assert shared_configs.keys() <= all_allowed_keys, shared_configs.keys()

    inner_ex_dict = dict(inner_ex_config)
    # combine with environment config (`env_cfg`, `env_data`, etc.)
    shared_configs_subset = {
        k: v for k, v in shared_configs.items()
        if k in allowed_shared_config_keys
    }
    merged_config = update(inner_ex_dict, shared_configs_subset)
    # now combine with rest of config values, form Ray
    merged_config = update(
        merged_config, tune_config_updates.get(exp_name, {}))
    for root_key in allowed_shared_config_keys:
        merged_config = update(
            merged_config,
            {root_key: tune_config_updates.get(root_key, {})})
    observer = FileStorageObserver(osp.join(log_dir, exp_name))
    inner_ex.observers.append(observer)
    ret_val = inner_ex.run(config_updates=merged_config)
    return ret_val.result


def setup_run(config):
    """To be run before an experiment"""

    # generate a new random seed
    # TODO(sam): use the same seed for different configs, but different seeds
    # within each repeat of a single config (to reduce variance)
    rng = np.random.RandomState()

    # copy config so that we don't mutate in-place
    config = copy.deepcopy(config)

    return rng, config


def report_experiment_result(sacred_result):
    """To be run after an experiment."""

    filtered_result = {
        k: v
        for k, v in sacred_result.items() if isinstance(v, (int, float))
    }
    logging.info(
        f"Got sacred result with keys {', '.join(filtered_result.keys())}")
    tune.report(**filtered_result)


def run_end2end_exp(rep_ex_config, il_train_ex_config, il_test_ex_config,
                    shared_configs, config, log_dir):
    """
    Run representation learning, imitation learning's training and testing
    sequentially.

    Params:
        rep_ex_config: Config of represent_ex. It's the default config plus any
            modifications we might have made in an macro_experiment config
            update.
        il_train_ex_config: Config of il_train_ex. It's the default config plus
            any modifications we might have made in an macro_experiment config
            update.
        il_test_ex_config: Config of il_test_ex. It's the default config plus
            any modifications we might have made in an macro_experiment config
            update.
        shared_configs: Config keys shared between two or more experiments.
        config: The config generated by Ray tune for hyperparameter tuning
        log_dir: The log directory of current chain experiment.
    """
    rng, tune_config_updates = setup_run(config)
    del config  # I want a new name for it

    # Run representation learning
    tune_config_updates['repl'].update({
        'seed': rng.randint(1 << 31),
    })
    pretrain_result = run_single_exp(rep_ex_config, shared_configs,
                                     tune_config_updates, log_dir, 'repl')

    # Run il train
    tune_config_updates['il_train'].update({
        'encoder_path':
        pretrain_result['encoder_path'],
        'seed':
        rng.randint(1 << 31),
    })
    il_train_result = run_single_exp(il_train_ex_config, shared_configs,
                                     tune_config_updates, log_dir, 'il_train')

    # Run il test
    tune_config_updates['il_test'].update({
        'policy_path':
        il_train_result['model_path'],
        'seed':
        rng.randint(1 << 31),
    })
    il_test_result = run_single_exp(il_test_ex_config, shared_configs,
                                    tune_config_updates, log_dir, 'il_test')

    report_experiment_result(il_test_result)


def run_repl_only_exp(rep_ex_config, shared_configs, config, log_dir):
    """Experiment that runs only representation learning."""
    rng, tune_config_updates = setup_run(config)
    del config

    tune_config_updates['repl'].update({
        'seed': rng.randint(1 << 31),
    })

    pretrain_result = run_single_exp(rep_ex_config, shared_configs,
                                     tune_config_updates, log_dir, 'repl')
    report_experiment_result(pretrain_result)
    logging.info("RepL experiment completed")


def run_il_only_exp(il_train_ex_config, il_test_ex_config, shared_configs,
                    config, log_dir):
    """Experiment that runs only imitation learning."""
    rng, tune_config_updates = setup_run(config)
    del config

    tune_config_updates['il_train'].update({'seed': rng.randint(1 << 31)})
    il_train_result = run_single_exp(il_train_ex_config, shared_configs,
                                     tune_config_updates, log_dir, 'il_train')
    tune_config_updates['il_test'].update({
        'policy_path':
        il_train_result['model_path'],
        'seed':
        rng.randint(1 << 31),
    })
    il_test_result = run_single_exp(il_test_ex_config, shared_configs,
                                    tune_config_updates, log_dir, 'il_test')
    report_experiment_result(il_test_result)


@chain_ex.config
def base_config():
    exp_name = "grid_search"
    # the repl, il_train and il_test experiments will have this value as their
    # exp_ident settings
    exp_ident = None
    # Name of the metric to optimise. By default, this will be automatically
    # selected based on the value of stages_to_run.
    metric = None
    stages_to_run = StagesToRun.REPL_ONLY
    spec = {
        # DO NOT ADD ANYTHING TO THESE BY DEFAULT.
        # They will affect unit tests and also every other use of the script.
        # If you really want to make a permanent change to a default, then
        # change the `repl`, `il_train`, `il_test`, etc. dictionaries at the
        # *top level of this config*, rather than within `spec` (which is
        # *intended for Tune grid search).
        'repl': {},
        'il_train': {},
        'il_test': {},
        'env_cfg': {},
        'env_data': {},
        'venv_opts': {},
    }
    # "use_skopt" will use scikit-optimize. This will ignore the 'spec' dict
    # above; instead, you need to declare an appropriate skopt_space. Use this
    # mode for hyperparameter tuning.
    use_skopt = False
    skopt_search_mode = None
    skopt_space = collections.OrderedDict()
    skopt_ref_configs = []
    # use 'true' if you want to do cluster-specific things like using
    # DockerSyncer
    on_cluster = False

    tune_run_kwargs = dict(num_samples=1,
                           max_failures=2,
                           fail_fast=False,
                           resources_per_trial=dict(
                               cpu=1,
                               gpu=0,  # TODO change back to 0.32?
                           ))
    ray_init_kwargs = dict(
        object_store_memory=None,
        include_dashboard=False,
    )

    _ = locals()
    del _


@chain_ex.named_config
def cfg_use_magical():
    # see il_representations/envs/config for examples of what should go here
    env_cfg = {
        'benchmark_name': 'magical',
        # MatchRegions is of intermediate difficulty
        # (TODO(sam): allow MAGICAL to load data from _all_ tasks at once, so
        # we can try multi-task repL)
        'task_name': 'MatchRegions',
        # we really need magical_remove_null_actions=True for BC; for RepL it
        # shouldn't matter so much (for action-based RepL methods)
        'magical_remove_null_actions': False,
    }

    _ = locals()
    del _


@chain_ex.named_config
def cfg_use_dm_control():
    env_cfg = {
        'benchmark_name': 'dm_control',
        # walker-walk is difficult relative to other dm-control tasks that we
        # use, but RL solves it quickly. Plateaus around 850-900 reward (see
        # https://docs.google.com/document/d/1YrXFCmCjdK2HK-WFrKNUjx03pwNUfNA6wwkO1QexfwY/edit#).
        'task_name': 'reacher-easy',
    }

    _ = locals()
    del _


@chain_ex.named_config
def cfg_base_3seed_4cpu_pt3gpu():
    """Basic config that does three samples per config, using 5 CPU cores and
    0.3 of a GPU. Reasonable idea for, e.g., GAIL on svm/perceptron."""
    use_skopt = False
    tune_run_kwargs = dict(num_samples=3,
                           # retry on (node) failure
                           max_failures=2,
                           fail_fast=False,
                           resources_per_trial=dict(
                               cpu=5,
                               gpu=0.32,
                           ))

    _ = locals()
    del _


@chain_ex.named_config
def cfg_base_3seed_1cpu_pt2gpu_2envs():
    """Another config that uses only one CPU per run, and .2 of a GPU. Good for
    running GPU-intensive algorithms (repL, BC) on GCP."""
    use_skopt = False
    tune_run_kwargs = dict(num_samples=3,
                           # retry on node failure
                           max_failures=3,
                           fail_fast=False,
                           resources_per_trial=dict(
                               cpu=1,
                               gpu=0.2,
                           ))
    venv_opts = {
        'n_envs': 2,
    }

    _ = locals()
    del _


@chain_ex.named_config
def cfg_base_3seed_1cpu_pt5gpu_2envs():
    """As above, but one GPU per run."""
    use_skopt = False
    tune_run_kwargs = dict(num_samples=3,
                           # retry on node failure
                           max_failures=3,
                           fail_fast=False,
                           resources_per_trial=dict(
                               cpu=1,
                               gpu=0.5,
                           ))
    venv_opts = {
        'n_envs': 2,
    }

    _ = locals()
    del _


@chain_ex.named_config
def cfg_base_3seed_1cpu_1gpu_2envs():
    """As above, but one GPU per run."""
    use_skopt = False
    tune_run_kwargs = dict(num_samples=3,
                           # retry on node failure
                           max_failures=3,
                           fail_fast=False,
                           resources_per_trial=dict(
                               cpu=1,
                               gpu=1,
                           ))
    venv_opts = {
        'n_envs': 2,
    }

    _ = locals()
    del _


@chain_ex.named_config
def cfg_no_log_to_driver():
    # disables sending stdout of Ray workers back to head node
    # (only useful for huge clusters)
    ray_init_kwargs = {
        'log_to_driver': False,
    }

    _ = locals()
    del _


@chain_ex.named_config
def cfg_bench_short_sweep_magical():
    """Sweeps over four easiest MAGICAL instances."""
    spec = dict(env_cfg=tune.grid_search(
        # MAGICAL configs
        [
            {
                'benchmark_name': 'magical',
                'task_name': magical_env_name,
                'magical_remove_null_actions': True,
            } for magical_env_name in [
                'MoveToCorner',
                'MoveToRegion',
                'FixColour',
                'MatchRegions',
                # 'FindDupe',
                # 'MakeLine',
                # 'ClusterColour',
                # 'ClusterShape',
            ]
        ]))

    _ = locals()
    del _


@chain_ex.named_config
def cfg_bench_short_sweep_dm_control():
    """Sweeps over four easiest dm_control instances."""
    spec = dict(env_cfg=tune.grid_search(
        # dm_control configs
        [
            {
                'benchmark_name': 'dm_control',
                'task_name': dm_control_env_name
            } for dm_control_env_name in [
                # to gauge how hard these are, see
                # https://docs.google.com/document/d/1YrXFCmCjdK2HK-WFrKNUjx03pwNUfNA6wwkO1QexfwY/edit#heading=h.akt76l1pl1l5
                'reacher-easy',
                'finger-spin',
                'ball-in-cup-catch',
                'cartpole-swingup',
                # 'cheetah-run',
                # 'walker-walk',
                # 'reacher-easy',
            ]
        ]))

    _ = locals()
    del _


@chain_ex.named_config
def cfg_bench_micro_sweep_magical():
    """Tiny sweep over MAGICAL configs, both of which are "not too hard",
    but still provide interesting generalisation challenges."""
    spec = dict(env_cfg=tune.grid_search(
        [
            {
                'benchmark_name': 'magical',
                'task_name': magical_env_name,
                'magical_remove_null_actions': True,
            } for magical_env_name in [
                'MoveToRegion', 'MatchRegions', 'MoveToCorner'
            ]
        ]))

    _ = locals()
    del _


@chain_ex.named_config
def cfg_bench_micro_sweep_dm_control():
    """Tiny sweep over two dm_control configs (finger-spin is really easy for
    RL, and cheetah-run is really hard for RL)."""
    spec = dict(env_cfg=tune.grid_search(
        [
            {
                'benchmark_name': 'dm_control',
                'task_name': dm_control_env_name
            } for dm_control_env_name in [
                'finger-spin', 'cheetah-run', 'reacher-easy'
            ]
        ]))

    _ = locals()
    del _


@chain_ex.named_config
def cfg_bench_one_task_magical():
    """Just one simple MAGICAL config."""
    env_cfg = {
        'benchmark_name': 'magical',
        'task_name': 'MatchRegions',
        'magical_remove_null_actions': True,
    }

    _ = locals()
    del _


@chain_ex.named_config
def cfg_bench_magical_mr():
    """Bench on MAGICAL MatchRegions."""
    env_cfg = {
        'benchmark_name': 'magical',
        'task_name': 'MatchRegions',
        'magical_remove_null_actions': True,
    }

    _ = locals()
    del _


@chain_ex.named_config
def cfg_bench_magical_mtc():
    """Bench on MAGICAL MoveToCorner."""
    env_cfg = {
        'benchmark_name': 'magical',
        'task_name': 'MoveToCorner',
        'magical_remove_null_actions': True,
    }

    _ = locals()
    del _


@chain_ex.named_config
def cfg_bench_one_task_dm_control():
    """Just one simple dm_control config."""
    env_cfg = {
        'benchmark_name': 'dm_control',
        'task_name': 'cheetah-run',
    }

    _ = locals()
    del _


@chain_ex.named_config
def cfg_base_repl_5000_batches():
    repl = {
        'batches_per_epoch': 500,
        'n_epochs': 10,
    }

    _ = locals()
    del _


@chain_ex.named_config
def cfg_base_repl_10000_batches():
    repl = {
        'batches_per_epoch': 1000,
        'n_epochs': 10,
    }

    _ = locals()
    del _


@chain_ex.named_config
def cfg_force_use_repl():
    stages_to_run = StagesToRun.REPL_AND_IL

    _ = locals()
    del _


@chain_ex.named_config
def cfg_repl_none():
    stages_to_run = StagesToRun.IL_ONLY

    _ = locals()
    del _


@chain_ex.named_config
def cfg_repl_moco():
    stages_to_run = StagesToRun.REPL_AND_IL
    repl = {
        'algo': 'MoCoWithProjection',
    }

    _ = locals()
    del _


@chain_ex.named_config
def cfg_repl_simclr():
    stages_to_run = StagesToRun.REPL_AND_IL
    repl = {
        'algo': 'SimCLR',
    }

    _ = locals()
    del _


@chain_ex.named_config
def cfg_repl_temporal_cpc():
    stages_to_run = StagesToRun.REPL_AND_IL
    repl = {
        'algo': 'TemporalCPC',
    }

    _ = locals()
    del _


@chain_ex.named_config
def cfg_data_repl_demos_random():
    """Training on both demos and random rollouts for the current
    environment."""
    repl = {
        'dataset_configs': [{'type': 'demos'}, {'type': 'random'}],
    }
    _ = locals()
    del _


@chain_ex.named_config
def cfg_data_repl_random():
    """Training on both demos and random rollouts for the current
    environment."""
    repl = {
        'dataset_configs': [{'type': 'random'}],
    }
    _ = locals()
    del _


@chain_ex.named_config
def cfg_data_repl_demos_magical_mt():
    """Multi-task training on all MAGICAL tasks."""
    repl = {
        'dataset_configs': [
            {
                'type': 'demos',
                'env_cfg': {
                    'benchmark_name': 'magical',
                    'task_name': magical_task_name,
                }
            } for magical_task_name in [
                'MoveToCorner',
                'MoveToRegion',
                'MatchRegions',
                'MakeLine',
                'FixColour',
                'FindDupe',
                'ClusterColour',
                'ClusterShape',
            ]
        ],
    }
    _ = locals()
    del _


@chain_ex.named_config
def cfg_data_il_5traj():
    """Use only 5 trajectories for IL training."""
    il_train = {
        'n_traj': 5,
    }
    _ = locals()
    del _


@chain_ex.named_config
def cfg_data_il_hc_extended():
    """Use extended HalfCheetah dataset for IL training."""
    env_data = {
        'dm_control_demo_patterns': {
            'cheetah-run':
            'data/dm_control/extended-cheetah-run-*500traj.pkl.gz',
        }
    }
    _ = locals()
    del _


@chain_ex.named_config
def cfg_repl_ceb():
    stages_to_run = StagesToRun.REPL_AND_IL
    repl = {
        'algo': 'CEB',
    }

    _ = locals()
    del _


@chain_ex.named_config
def cfg_repl_vae():
    repl = {
        'algo': 'VariationalAutoencoder',
        'algo_params': {'batch_size': 32},
    }
    _ = locals()
    del _


@chain_ex.named_config
def cfg_repl_inv_dyn():
    repl = {
        'algo': 'InverseDynamicsPrediction',
        'algo_params': {'batch_size': 32},
    }
    _ = locals()
    del _


@chain_ex.named_config
def cfg_il_bc_nofreeze():
    il_train = {
        'algo': 'bc',
        'bc': {
            'n_batches': 15000,
        },
        'freeze_encoder': False,
    }

    _ = locals()
    del _


@chain_ex.named_config
def cfg_il_bc_freeze():
    il_train = {
        'algo': 'bc',
        'bc': {
            'n_batches': 15000,
        },
        'freeze_encoder': True,
    }

    _ = locals()
    del _


def cfg_il_gail_nofreeze():
    il_train = {
        'algo': 'gail',
        'gail': {
            # TODO(sam): increase this to a default that works for most
            # environments.
            'total_timesteps': 200000,
        },
        'freeze_encoder': False,
    }
    venv_opts = {
        'n_envs': 32,
        'parallel': True,
        'parallel_workers': 8,
    }

    _ = locals()
    del _


class WrappedConfig():
    def __init__(self, config_dict):
        self.config_dict = config_dict


def trainable_function(config):
    # "config" argument is passed in by Ray Tune
    shared_config_keys = ['env_cfg', 'env_data', 'venv_opts']
    log_dir = config['log_dir']
    stages_to_run = config['stages_to_run']
    wrapped_config_keys = config['wrapped_config_keys']

    del config['log_dir']
    del config['stages_to_run']
    del config['wrapped_config_keys']

    if stages_to_run == StagesToRun.REPL_AND_IL:
        keys_to_add = [
            'env_cfg', 'env_data', 'venv_opts', 'il_train', 'il_test',
            'repl',
        ]
    elif stages_to_run == StagesToRun.IL_ONLY:
        keys_to_add = [
            'env_cfg', 'env_data', 'venv_opts', 'il_train', 'il_test',
        ]
    elif stages_to_run == StagesToRun.REPL_ONLY:
        keys_to_add = ['env_cfg', 'env_data', 'repl']
    else:
        raise ValueError(f"stages_to_run has invalid value {config['stages_to_run']}")

    inflated_configs = {}
    for key in wrapped_config_keys:
        # Unwrap all wrapped ingredient baseline configs
        # that were passed around as Ray parameters
        assert f"{key}_frozen" in config, f"No version of {key} config " \
                                          f"(under {key}_frozen) found in config"

        inflated_configs[key] = config[f"{key}_frozen"].config_dict

        # Delete the keys for cleanliness' sake
        del config[f"{key}_frozen"]
    shared_configs = {k:inflated_configs[k] for k in shared_config_keys}
    logging.warning(f"Config keys: {config.keys()}")
    config = expand_dict_keys(config)

    # add empty update dicts if necessary to avoid crashing
    # FIXME(sam): decide whether this is the appropriate defensive thing to
    # do. It would be nice if we caught errors where the user tries to,
    # e.g., tune repL, but does not specify anything to tune _over_.

    for key in keys_to_add:
        if key not in config:
            config[key] = {}

    if stages_to_run == StagesToRun.REPL_AND_IL:
        run_end2end_exp(rep_ex_config=inflated_configs['repl'],
                        il_train_ex_config=inflated_configs['il_train'],
                        il_test_ex_config=inflated_configs['il_test'],
                        shared_configs=shared_configs, config=config,
                        log_dir=log_dir)
    if stages_to_run == StagesToRun.IL_ONLY:
        run_il_only_exp(il_train_ex_config=inflated_configs['il_train'],
                        il_test_ex_config=inflated_configs['il_test'],
                        shared_configs=shared_configs, config=config,
                        log_dir=log_dir)
    if stages_to_run == StagesToRun.REPL_ONLY:
        run_repl_only_exp(rep_ex_config=inflated_configs['repl'],
                          shared_configs=shared_configs, config=config,
                          log_dir=log_dir)


@chain_ex.main
def run(exp_name, metric, spec, repl, il_train, il_test, env_cfg, env_data,
        venv_opts, tune_run_kwargs, ray_init_kwargs, stages_to_run, use_skopt,
        skopt_search_mode, skopt_ref_configs, skopt_space, exp_ident,
        on_cluster):
    print(f"Ray init kwargs: {ray_init_kwargs}")
    rep_ex_config = sacred_copy(repl)
    il_train_ex_config = sacred_copy(il_train)
    il_test_ex_config = sacred_copy(il_test)
    env_cfg_config = sacred_copy(env_cfg)
    env_data_config = sacred_copy(env_data)
    venv_opts_config = sacred_copy(venv_opts)
    spec = sacred_copy(spec)
    stages_to_run = get_stages_to_run(stages_to_run)
    log_dir = os.path.abspath(chain_ex.observers[0].dir)

    # set default exp_ident
    if rep_ex_config['exp_ident'] is None:
        rep_ex_config['exp_ident'] = exp_ident
    if il_train_ex_config['exp_ident'] is None:
        il_train_ex_config['exp_ident'] = exp_ident
    if il_test_ex_config['exp_ident'] is None:
        il_test_ex_config['exp_ident'] = exp_ident

    ingredient_configs_dict = {
        'repl': rep_ex_config,
        'il_train': il_train_ex_config,
        'il_test': il_test_ex_config,
        'env_cfg': env_cfg_config,
        'env_data': env_data_config,
        'venv_opts': venv_opts_config,
    }
    # List comprehension to make the keys() object an actual list
    wrapped_config_keys = [k for k in ingredient_configs_dict.keys()]

    if metric is None:
        # choose a default metric depending on whether we're running
        # representation learning, IL, or both
        metric = {
            # return_mean is returned by il_test.run()
            StagesToRun.REPL_AND_IL:
            'return_mean',
            StagesToRun.IL_ONLY:
            'return_mean',
            # repl_loss is returned by run_rep_learner.run()
            StagesToRun.REPL_ONLY:
            'repl_loss',
        }[stages_to_run]

    # We remove unnecessary keys from the "spec" that we pass to Ray Tune. This
    # ensures that Ray Tune doesn't try to tune over things that can't affect
    # the outcome.

    if stages_to_run == StagesToRun.IL_ONLY \
       and 'repl' in spec:
        logging.warning(
            "You only asked to tune IL, so I'm removing the representation "
            "learning config from the Tune spec.")
        del spec['repl']

    if stages_to_run == StagesToRun.REPL_ONLY \
       and 'il_train' in spec:
        logging.warning(
            "You only asked to tune RepL, so I'm removing the imitation "
            "learning config from the Tune spec.")
        del spec['il_train']

    # make Ray run from this directory
    ray_dir = os.path.join(log_dir)
    os.makedirs(ray_dir, exist_ok=True)
    # Ray Tune will change the directory when tuning; this next step ensures
    # that pwd-relative data_roots remain valid.
    env_data_config['data_root'] = os.path.abspath(
        os.path.join(cwd, env_data_config['data_root']))

    if detect_ec2():
        ray.init(address="auto", **ray_init_kwargs)
    else:
        ray.init(**ray_init_kwargs)

    if use_skopt:
        assert skopt_search_mode in {'min', 'max'}, \
            'skopt_search_mode must be "min" or "max", as appropriate for ' \
            'the metric being optimised'
        assert len(skopt_space) > 0, "was passed an empty skopt_space"

        # do some sacred_copy() calls to ensure that we don't accidentally put
        # a ReadOnlyDict or ReadOnlyList into our optimizer
        skopt_space = sacred_copy(skopt_space)
        skopt_search_mode = sacred_copy(skopt_search_mode)
        skopt_ref_configs = sacred_copy(skopt_ref_configs)
        metric = sacred_copy(metric)

        # In addition to the actual spaces we're searching over, we also need to
        # store the baseline config values in Ray to avoid Ray issue #12048
        skopt_space['log_dir'] = skopt.space.Categorical(categories=(log_dir,))
        skopt_space['stages_to_run'] = skopt.space.Categorical(categories=(stages_to_run,))
        skopt_space['wrapped_config_keys'] = skopt.space.Categorical(categories=(wrapped_config_keys,))
        for ref_config in skopt_ref_configs:
            ref_config['log_dir'] = log_dir
            ref_config['stages_to_run'] = stages_to_run
            ref_config['wrapped_config_keys'] = wrapped_config_keys

        for ing_name, ing_config in ingredient_configs_dict.items():
            frozen_config = WrappedConfig(ing_config)

            # Create a Categorical skopt search space with a single element:
            # the frozen config. This means that Ray's config dictionary
            # will contain the same `frozen_config` object on every trial
            skopt_space[f"{ing_name}_frozen"] = skopt.space.Categorical(categories=(frozen_config,))
            for ref_config in skopt_ref_configs:
                ref_config[f"{ing_name}_frozen"] = frozen_config

        sorted_space = collections.OrderedDict([
            (key, value) for key, value in sorted(skopt_space.items())
        ])
        for k, v in list(sorted_space.items()):
            # cast each value in sorted_space to a skopt Dimension object, then
            # make the name of the Dimension object match the corresponding key
            new_v = skopt.space.check_dimension(v)
            new_v.name = k
            sorted_space[k] = new_v
        skopt_optimiser = skopt.optimizer.Optimizer([*sorted_space.values()],
                                                    base_estimator='RF')
        algo = SkOptSearch(skopt_optimiser,
                           list(sorted_space.keys()),
                           metric=metric,
                           mode=skopt_search_mode,
                           points_to_evaluate=[[
                               ref_config_dict[k] for k in sorted_space.keys()
                           ] for ref_config_dict in skopt_ref_configs])
        tune_run_kwargs = {
            'search_alg': algo,
            'scheduler': CheckpointFIFOScheduler(algo),
            **tune_run_kwargs,
        }
        # completely remove 'spec'
        if spec:
            logging.warning("Will ignore everything in 'spec' argument")
        spec = {}
    else:
        # In addition to the actual spaces we're searching over, we also need
        # to store the baseline config values in Ray to avoid Ray issue #12048.
        # We create a grid search with a single value of the WrappedConfig
        # object.
        for ing_name, ing_config in ingredient_configs_dict.items():
            frozen_config = WrappedConfig(ing_config)
            spec[f"{ing_name}_frozen"] = tune.grid_search([frozen_config])
        spec['log_dir'] = log_dir
        spec['stages_to_run'] = stages_to_run
        spec['wrapped_config_keys'] = wrapped_config_keys

    if on_cluster:
        # use special syncer which is able to attach to autoscaler's Docker
        # container once it connects to worker machines (necessary for GCP)
        assert 'sync_config' not in tune_run_kwargs, \
            "set on_cluster=True, which overrides sync_config for tune.run, " \
            "but sync_config was already supplied (and set to " \
            f"{tune_run_kwargs['sync_config']})"
        tune_run_kwargs = {
            **tune_run_kwargs,
            'sync_config': tune.SyncConfig(sync_to_driver=DockerSyncer),
        }

    rep_run = tune.run(
        trainable_function,
        name=exp_name,
        config=spec,
        local_dir=ray_dir,
        **tune_run_kwargs,
    )
    if use_skopt:
        # get_best_config() requires a mode, but skopt_search_mode is only
        # guaranteed to be non-None when tuning with skopt
        best_config = rep_run.get_best_config(
            metric=metric, mode=skopt_search_mode)
        logging.info(f"Best config is: {best_config}")
    logging.info("Results available at: ")
    logging.info(rep_run._get_trial_paths())


def main(argv=None):
    # This function is here because it gets called from other scripts. Please
    # don't delete!
    chain_ex.observers.append(FileStorageObserver('runs/chain_runs'))
    chain_ex.run_commandline(argv)


if __name__ == '__main__':
    main()

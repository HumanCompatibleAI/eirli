import collections
import copy
import faulthandler
from glob import glob
import logging
import os
import os.path as osp
import signal
from time import time
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
from il_representations.configs.chain_configs import make_chain_configs
from il_representations.configs.hp_tuning import make_hp_tuning_configs
from il_representations.configs.icml_hp_tuning import make_icml_tuning_configs
from il_representations.configs.icml_experiment_configs import make_dataset_experiment_configs
from il_representations.scripts.il_test import il_test_ex
from il_representations.scripts.il_train import il_train_ex
from il_representations.scripts.run_rep_learner import represent_ex
from il_representations.script_utils import detect_ec2, sacred_copy, update, StagesToRun, ReuseRepl
from il_representations.utils import hash_configs, up, WrappedConfig


sacred.SETTINGS['CAPTURE_MODE'] = 'no'  # workaround for sacred issue#740
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

make_dataset_experiment_configs(chain_ex)
make_icml_tuning_configs(chain_ex)

# Add configs to experiment for hyperparameter tuning
# This is to allow us to separate out tuning configs into their own file
make_hp_tuning_configs(chain_ex)

# Add all other configs
# DO NOT ADD MORE CONFIGS TO pretrain_n_adapt.py! Add them to a separate file
make_chain_configs(chain_ex)


def get_stages_to_run(stages_to_run):
    """Convert a string (or enum) to StagesToRun object."""
    upper_str = stages_to_run.upper()
    try:
        stage = StagesToRun(upper_str)
    except ValueError as ex:
        # pytype: disable=missing-parameter
        options = [f"'{s.name}'" for s in StagesToRun]
        # pytype: enable=missing-parameter
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
    {'level2': …}}).

    Example:

    ```
    >>> expand_dict_keys({'x:y': 42, 'z': 4, 'x:u:v': 5, 'w': {'s:t': 99}})
    {'x': {'y': 42, 'u': {'v': 5}}, 'z': 4, 'w': {'s': {'t': 99}}}
    ```
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


def run_single_exp(merged_config, log_dir, exp_name):
    """Run a specified experiment. We could not pass each Sacred experiment in
    because they are not pickle serializable, which is not supported by Ray
    (when running this as a remote function).

    Args:
        merged_config: The config that should be used in the sub-experiment;
                       formed by calling merge_configs()
        log_dir: The log directory of current chain experiment.
        exp_name: Specify the experiment type in `['repl', 'il_train',
            'il_test']`.
    """
    # we need to run the workaround in each raylet, so we do it at the start of
    # run_single_exp
    sacred.SETTINGS['CAPTURE_MODE'] = 'no'  # workaround for sacred issue#740
    from il_representations.scripts.il_test import il_test_ex
    from il_representations.scripts.il_train import il_train_ex
    from il_representations.scripts.run_rep_learner import represent_ex

    if exp_name == 'repl':
        inner_ex = represent_ex
    elif exp_name == 'il_train':
        inner_ex = il_train_ex
    elif exp_name == 'il_test':
        inner_ex = il_test_ex
    else:
        raise NotImplementedError(
            f"exp_name must be one of repl, il_train, or il_test. Value passed in: {exp_name}")

    observer = FileStorageObserver(osp.join(log_dir, exp_name))
    inner_ex.observers.append(observer)
    ret_val = inner_ex.run(config_updates=merged_config)
    return {
        "type": exp_name,
        "result": ret_val.result,
        # FIXME(sam): this is dependent on us having exactly one
        # observer, and it being a FileStorageObserver
        "dir": ret_val.observers[0].dir,
    }


def report_final_experiment_results(results_dict):
    """To be run after an experiment."""
    logging.info(
        f"Got experiment result with keys {', '.join(results_dict.keys())}")
    tune.report(**results_dict)


def relative_symlink(src, dst):
    link_dir_abs, link_fn = os.path.split(os.path.abspath(dst))
    if not link_fn:
        raise ValueError(f"path dst='{dst}' has empty basename")
    # absolute path to src, and path relative to link_dir
    src_abspath = os.path.abspath(src)
    src_relpath = os.path.relpath(src_abspath, start=link_dir_abs)

    os.makedirs(link_dir_abs, exist_ok=True)
    link_dir_fd = os.open(link_dir_abs, os.O_RDONLY)
    try:
        # both src_relpath and link_fn are relative to link_dir, which is
        # represented by the file descriptor link_dir_fd
        os.symlink(src_relpath, link_fn, dir_fd=link_dir_fd)
    finally:
        os.close(link_dir_fd)


def cache_repl_encoder(repl_encoder_path, repl_directory_dir,
                       config_hash, seed, config_path=None):
    """A utility function for taking a trained repl encoder and symlinking it,
    with appropriate searchable directory name, to the repl run directory

    Args:
        repl_encoder_path: A path to an encoder checkpoint file. Assumed to be
            within a /checkpoints dir within a run of the `repl` experiment
        repl_directory_dir: The directory where the symlinked listing of repl
            encoder should be stored
        config_hash: The hash identifying the config attached to this repl run
        seed: The seed for this repl run
        config_path: The path to the config file for this repl run. If None,
            will try to search relative to repl_encoder_path
    """
    if config_path is None:
        config_path = os.path.join(up(up(up(repl_encoder_path))), 'config.json')
    dir_name = f"{config_hash}_{seed}_{round(time())}"
    logging.info(f"Symlinking encoder path under the directory {dir_name}")
    encoder_link = os.path.join(repl_directory_dir, dir_name, 'repl_encoder')
    config_link = os.path.join(repl_directory_dir, dir_name, 'config.json')
    relative_symlink(repl_encoder_path, encoder_link)
    relative_symlink(config_path, config_link)


def get_repl_dir(log_dir):
    """A utility function for returning a repl directory location relative to
    logdir, and creating one if it does not exist."""
    repl_dir = os.path.join(up(up(log_dir)), 'all_repl')
    if not os.path.exists(repl_dir):
        os.makedirs(repl_dir)
    return repl_dir


def resolve_env_cfg(merged_repl_config):
    merged_config_copy = copy.deepcopy(merged_repl_config)

    # If the task we are training on is multitask, we do not want to
    # include env_cfg.task_name in the canonical repl hash, since it will not be used
    if merged_config_copy.get('is_multitask', False):
        del merged_config_copy['env_cfg']['task_name']

    # Do not hash based on env_data, which just contains path information,
    # and thus will vary based on path setup by machine
    del merged_config_copy['env_data']
    return merged_config_copy


def run_end2end_exp(*, rep_ex_config, il_train_ex_config, il_test_ex_config,
                    env_cfg_config, env_data_config, venv_opts_config,
                    reuse_repl, repl_encoder_path, full_run_start_time,
                    log_dir):
    """Run representation learning, imitation learning's training and testing
    sequentially.

    Args:
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
        reuse_repl: An enum value determining whether, in our end2end experiment,
            we should kick off a new RepL run, or else attempt to load one in
            that previously exists
        repl_encoder_path: A string parameter, set by default to None, allows
            us to hardcode a path from which to load a pretrained RepL encoder.
            In this situation, hardcoding overrides inference of which prior
            RepL run we should read in
        log_dir: The log directory of current chain experiment.
    """
    rng = np.random.RandomState()

    # Get the directory to store repl runs, and the hash for this config
    # Used to facilitate reuse of repl runs
    repl_dir = get_repl_dir(log_dir)
    merged_repl_config = update(
        rep_ex_config,
        {
            'env_cfg': env_cfg_config,
            'env_data': env_data_config
        },
    )
    config_to_hash = resolve_env_cfg(merged_repl_config)
    repl_hash = hash_configs(config_to_hash)
    pretrained_encoder_path = None

    # If we are open to reading in a pretrained repl encoder
    if repl_encoder_path is not None:
        assert reuse_repl != ReuseRepl.NO, "You've set a specific encoder path, but also turned off RepL encoder reuse"
    if reuse_repl in (ReuseRepl.YES, ReuseRepl.IF_AVAILABLE):
        # If we have hardcoded an encoder path, check if it exists, and, if so, use it
        if repl_encoder_path is not None:
            assert os.path.exists(repl_encoder_path), f"Hardcoded encoder path {repl_encoder_path} does not exist"
            pretrained_encoder_path = repl_encoder_path
        else:
            # If we are searching for a prior repl run based on hashed config
            # This is the default case
            existing_repl_runs = glob(os.path.join(repl_dir, f"{repl_hash}_*"))

            # If no matching repl run is found, we will fall through to training repl as normal
            if len(existing_repl_runs) > 0:
                timestamps = [el.split('_')[-1] for el in existing_repl_runs]

                # Don't read in any Repl runs completed after the start of the full run
                valid_timestamps = [ts for ts in timestamps if int(ts) < full_run_start_time]
                if len(valid_timestamps) > 0:
                    most_recent_run = existing_repl_runs[np.argmax(valid_timestamps)]
                    pretrained_encoder_path = os.path.join(most_recent_run, 'repl_encoder')
                    logging.info(f"Loading encoder from {pretrained_encoder_path}")

            if pretrained_encoder_path is None:
                assert reuse_repl != ReuseRepl.YES, "Set repl_reuse to YES, but no run was found; erroring out"
                logging.info(f"No encoder found that existed prior to full run start time")

    # If none of the branches above have found a pretrained path,
    # proceed with repl training as normal
    if pretrained_encoder_path is None:
        if merged_repl_config.get('seed') is None:
            merged_repl_config['seed'] = rng.randint(1 << 31)

        pretrain_rv = run_single_exp(merged_repl_config, log_dir, 'repl')

        pretrained_encoder_path = pretrain_rv['result']['encoder_path']
        # Once repl training finishes, symlink the result to the repl directory
        cache_repl_encoder(pretrained_encoder_path,
                           repl_dir,
                           repl_hash,
                           merged_repl_config['seed'])
    else:
        pretrain_rv = {
            "type": "cached_repl",
            "result": {
                "encoder_path": pretrained_encoder_path,
            },
        }

    # Run il train
    merged_il_train_config = update(
        {'seed': rng.randint(1 << 31)},
        il_train_ex_config,
        {
            'encoder_path': pretrained_encoder_path,
            'env_cfg': env_cfg_config,
            'env_data': env_data_config,
            'venv_opts': venv_opts_config,
        },
    )
    il_train_rv = run_single_exp(merged_il_train_config, log_dir, 'il_train')
    il_train_model_path = il_train_rv['result']['model_path']

    # Run il test
    merged_il_test_config = update(
        {'seed': rng.randint(1 << 31)},
        il_test_ex_config,
        {
            'policy_path': il_train_model_path,
            'env_cfg': env_cfg_config,
            'venv_opts': venv_opts_config,
        },
    )
    il_test_rv = run_single_exp(merged_il_test_config, log_dir, 'il_test')

    report_final_experiment_results({
        "all_experiment_rvs": [pretrain_rv, il_train_rv, il_test_rv],
        **il_test_rv["result"],
    })


def run_repl_only_exp(*, rep_ex_config, env_cfg_config, env_data_config,
                      log_dir):
    """Experiment that runs only representation learning."""
    rng = np.random.RandomState()

    # set up config
    merged_repl_config = update(
        rep_ex_config,
        {
            'env_cfg': env_cfg_config,
            'env_data': env_data_config
        },
    )
    repl_hash = hash_configs(merged_repl_config)
    merged_repl_config.setdefault('seed', rng.randint(1 << 31))

    pretrain_rv = run_single_exp(merged_repl_config, log_dir, 'repl')

    # caching
    repl_dir = get_repl_dir(log_dir)
    cache_repl_encoder(pretrain_rv['result']['encoder_path'],
                       repl_dir,
                       repl_hash,
                       merged_repl_config['seed'])

    # report result
    report_final_experiment_results({
        "all_experiment_rvs": [pretrain_rv],
        **pretrain_rv["result"],
    })
    logging.info("RepL experiment completed")


def run_il_only_exp(*, il_train_ex_config, il_test_ex_config, env_cfg_config,
                    env_data_config, venv_opts_config, log_dir):
    """Experiment that runs only imitation learning."""
    rng = np.random.RandomState()

    merged_il_train_config = update(
        {'seed': rng.randint(1 << 31)},
        il_train_ex_config,
        {
            'env_cfg': env_cfg_config,
            'env_data': env_data_config,
            'venv_opts': venv_opts_config,
        },
    )
    il_train_rv = run_single_exp(merged_il_train_config, log_dir,
                                 'il_train')
    il_policy_path = il_train_rv['result']['model_path']
    merged_il_test_config = update(
        {'seed': rng.randint(1 << 31)},
        il_test_ex_config,
        {
            'policy_path': il_policy_path,
            'env_cfg': env_cfg_config,
            'venv_opts': venv_opts_config,
        },
    )
    il_test_rv = run_single_exp(merged_il_test_config, log_dir, 'il_test')

    report_final_experiment_results({
        "all_experiment_rvs": [il_train_rv, il_test_rv],
        **il_test_rv["result"],
    })


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

    # An enum for whether to reuse Repl or train it again from scratch
    # Available options are YES, NO, and IF_AVAILABLE Setting to YES will error
    # if no prior runs exist with a matching config IF_AVAILABLE will use a run
    # if it exists, and rerun otherwise This code is designed to ensure that
    # repl reuse only happens if it can be done in a way that seeds can be
    # consistent across tasks (i.e. only loads encoders completed before main
    # Ray script was started
    reuse_repl = ReuseRepl.NO

    # Set to a non-None string to force reading in that saved encoder in
    # end2end. BE CAREFUL, since this will override the check that ensures you
    # only read in an encoder that matches your own repl config, and this could
    # lead to inconsistencies
    repl_encoder_path = None

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


def trainable_function(config):
    # The 'config' dict that gets passed in looks something like this:
    #
    # {
    #     'some:nested:hyperparam': value1,
    #     'another:nested:hyperparam': value2,
    #     '+base_config': base_config_dict,
    # }
    #
    # The hyperparameters are specified with the 'foo:bar:baz' naming
    # convention because it is not possible to pass nested dicts straight into
    # skopt (Tune does not have this limitation). The '+base_config' dict
    # contains config keys that do not vary across training runs; it is a
    # 'pseudo-hyperparameter' that we use to get around limitations with Ray
    # Tune (see Ray bug #12048). To inflate this dictionary into a useful dict
    # of config values, we do the following:
    #
    # 1. Separate '+base_config' from the hyperparameter keys (foo:bar:baz
    #    etc.).
    # 2. 'Inflate' the nested hyperparameter names into a real nested dict.
    # 3. Do a nested dict merge which updates +base_config with all the
    #    inflated hyperparameters from step (2).
    # 4. Treat the dict produced by steps (1)-(3) as keyword arguments to an
    #    inner function that actually runs the experiment. This step ensures
    #    that we do not omit any keys from 'config', or provide extraneous keys
    #    that are not used at all in 'trainable_function'.
    logging.basicConfig(level=logging.INFO)

    # first separate hyperparameters from base config
    hyperparameters = dict(config)  # so we don't modify Tune data structures
    del config  # defensive
    base_config = dict(hyperparameters.pop('+base_config').config_dict)

    # Inflate nested hyparparemeters.
    hyperparameters = expand_dict_keys(hyperparameters)

    # Do a recursive dict merge.
    inner_kwargs = update(base_config, hyperparameters)
    del base_config, hyperparameters  # from now on, just use 'inner_kwargs'
    logging.warning(f'Config keys: {inner_kwargs.keys()}')

    # Treat the result as kwargs to an inner function which actually runs the
    # experiment.

    def trainable_function_inner(
            *,
            # 'meta' config variables that help us decide which nested Sacred
            # experiments to run
            stages_to_run, log_dir, reuse_repl, repl_encoder_path,
            run_start_time,
            # Optional ingredient configs
            il_train=None, il_test=None, repl=None, env_cfg=None,
            env_data=None, venv_opts=None):

        if stages_to_run == StagesToRun.REPL_AND_IL:
            run_end2end_exp(rep_ex_config=repl or {},
                            il_train_ex_config=il_train or {},
                            il_test_ex_config=il_test or {},
                            env_cfg_config=env_cfg or {},
                            env_data_config=env_data or {},
                            venv_opts_config=venv_opts or {},
                            reuse_repl=reuse_repl,
                            repl_encoder_path=repl_encoder_path,
                            log_dir=log_dir,
                            full_run_start_time=run_start_time)
        if stages_to_run == StagesToRun.IL_ONLY:
            run_il_only_exp(il_train_ex_config=il_train or {},
                            il_test_ex_config=il_test or {},
                            env_cfg_config=env_cfg or {},
                            env_data_config=env_data or {},
                            venv_opts_config=venv_opts or {},
                            log_dir=log_dir)
        if stages_to_run == StagesToRun.REPL_ONLY:
            run_repl_only_exp(rep_ex_config=repl or {},
                              env_cfg_config=env_cfg or {},
                              env_data_config=env_data or {},
                              log_dir=log_dir)

    return trainable_function_inner(**inner_kwargs)


@chain_ex.main
def run(exp_name, metric, spec, repl, il_train, il_test, env_cfg, env_data,
        venv_opts, tune_run_kwargs, ray_init_kwargs, stages_to_run, use_skopt,
        skopt_search_mode, skopt_ref_configs, skopt_space, exp_ident,
        reuse_repl, repl_encoder_path, on_cluster):
    faulthandler.register(signal.SIGUSR1)

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
    for inner_ex_config in [rep_ex_config, il_train_ex_config, il_test_ex_config]:
        if inner_ex_config['exp_ident'] is None:
            inner_ex_config['exp_ident'] = exp_ident

    ingredient_configs_dict = {
        'repl': rep_ex_config,
        'il_train': il_train_ex_config,
        'il_test': il_test_ex_config,
        'env_cfg': env_cfg_config,
        'env_data': env_data_config,
        'venv_opts': venv_opts_config,
    }

    # These are values from the config that we want to be set for all experiments,
    # and that we need to pass in through either the spec or the skopt config so
    # they will be reset properly in case of experiment failure
    needed_config_params = {'log_dir': log_dir,
                            'stages_to_run': stages_to_run,
                            'reuse_repl': reuse_repl,
                            'repl_encoder_path': repl_encoder_path,
                            'run_start_time': round(time())}

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

    # The config values in this wrapped dict are constant in all trials, and
    # will be passed in to `trainable_function()` as a pseudo-hyperparameter in
    # order to get around Ray bug #12048. See `trainable_function()` docs for
    # explanation of precisely how this works.
    base_config = WrappedConfig(
        update(needed_config_params, ingredient_configs_dict))

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

        # In addition to the actual spaces we're searching over, we also need
        # to store the baseline config values in Ray to avoid Ray issue #12048.
        base_space = skopt.space.Categorical([base_config])
        skopt_space['+base_config'] = base_space
        for ref_config in skopt_ref_configs:
            ref_config['+base_config'] = base_config

        sorted_space = collections.OrderedDict([
            (key, value) for key, value in sorted(skopt_space.items())
        ])
        for k, v in list(sorted_space.items()):
            # Cast each value in sorted_space to a skopt Dimension object, then
            # make the name of the Dimension object match the corresponding
            # key. This is the step that converts tuple ranges---like `(1e-3,
            # 2.0, 'log-uniform')`---to actual skopt `Space` objects.
            try:
                new_v = skopt.space.check_dimension(v)
            except ValueError:
                # Raise actually-informative value error instead
                raise ValueError(f"Dimension issue: k:{k} v: {v}")
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
        # As with skopt, we store baseline config values to avoid Ray issue
        # #12048. We create a grid search with a single value of the
        # WrappedConfig object.
        spec['+base_config'] = base_config

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
    return rep_run.results


def main(argv=None):
    # This function is here because it gets called from other scripts. Please
    # don't delete!
    chain_ex.observers.append(FileStorageObserver('runs/chain_runs'))
    chain_ex.run_commandline(argv)


if __name__ == '__main__':
    main()

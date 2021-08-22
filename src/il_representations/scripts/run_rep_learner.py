import faulthandler
import logging
import os
import signal

import imitation.util.logger as imitation_logger
import numpy as np
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch
from torch.optim.adam import Adam

from il_representations import algos
from il_representations.algos.representation_learner import \
    RepresentationLearner
from il_representations.algos.utils import set_global_seeds
from il_representations.configs.run_rep_learner_configs import \
    make_run_rep_learner_configs
from il_representations.envs import auto
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient)
from il_representations.utils import RepLSaveExampleBatchesCallback

sacred.SETTINGS['CAPTURE_MODE'] = 'no'  # workaround for sacred issue#740
represent_ex = Experiment(
    # We take env_cfg_ingredient to determine which task we need to load data
    # for, and env_data_ingredient gives us the paths to that data.
    'repl', ingredients=[env_cfg_ingredient, env_data_ingredient])


@represent_ex.config
def default_config():
    # exp_ident is an arbitrary string. Set it to a meaningful value to help
    # you identify runs in viskit.
    exp_ident = None

    # `dataset_configs` is a list of data sources to use for representation
    # learning. During representation learning, batches will be constructed by
    # drawing from each of these data sources with equal probability. For
    # instance, using `dataset_configs = [{'type': 'demos'}, {'type':
    # 'random'}]` interleaves demonstrations and (saved) random rollouts in
    # equal proportion. See docs for `load_new_style_ilr_dataset()` for more
    # information on the syntax of `dataset_configs`.
    dataset_configs = [{'type': 'demos'}]
    algo = "ActionConditionedTemporalCPC"
    torch_num_threads = 1
    algo_params = {
        'representation_dim': 128,
        'augmenter_kwargs': {
            # augmenter_spec is a comma-separated list of enabled
            # augmentations. Consult docstring for
            # imitation.augment.StandardAugmentations to see available
            # augmentations.
            "augmenter_spec":
            "translate,rotate,gaussian_blur,color_jitter_ex",
        },
    }
    device = "auto"
    optimizer_cls = Adam
    optimizer_kwargs = {'lr': 1e-4}
    scheduler_cls = None
    scheduler_kwargs = {}

    # For repL, an 'epoch' is just a fixed number of batches, configured with
    # batches_per_epoch.
    #
    # This makes it possible to balance data sources when we do multitask
    # training. For instance, if the datasets for two different tasks are
    # different lengths, then we truncate or repeat them so they are both of
    # length `batches_per_epoch / 2`. If we did not truncate/repeat, then the
    # shorter dataset would run out before the longer one, and the network
    # would end up training on more samples from the longer dataset.
    batches_per_epoch = 1000
    n_epochs = 5

    # how often should we save repL batch data?
    # (set to None to disable completely)
    repl_batch_save_interval = 1000
    # how often to dump logs (in #batches)
    log_interval = 100
    # how often to save checkpoints (in #batches)
    save_interval = 1000

    # set this to True if you want repL encoder hashing to ignore env_cfg
    is_multitask = False

    # If True, return the representation model as `run.results['model']`.
    debug_return_model = False

    _ = locals()
    del _


def initialize_non_features_extractor(sb3_model):
    # This is a hack to get around the fact that you can't initialize only some
    # of the components of a SB3 policy upon creation, and we in fact want to
    # keep the loaded representation frozen, but orthogonally initialize other
    # components.
    sb3_model.policy.init_weights(sb3_model.policy.mlp_extractor, np.sqrt(2))
    sb3_model.policy.init_weights(sb3_model.policy.action_net, 0.01)
    sb3_model.policy.init_weights(sb3_model.policy.value_net, 1)
    return sb3_model


def config_specifies_task_name(dataset_config_dict):
    if 'env_cfg' not in dataset_config_dict:
        return False
    return 'task_name' in dataset_config_dict['env_cfg']


make_run_rep_learner_configs(represent_ex)


@represent_ex.main
def run(dataset_configs, algo, algo_params, seed, batches_per_epoch, n_epochs,
        torch_num_threads, repl_batch_save_interval, is_multitask,
        debug_return_model, optimizer_cls, optimizer_kwargs, scheduler_cls,
        scheduler_kwargs, log_interval, save_interval, _config):
    faulthandler.register(signal.SIGUSR1)
    set_global_seeds(seed)

    # TODO fix to not assume FileStorageObserver always present
    log_dir = represent_ex.observers[0].dir
    if torch_num_threads is not None:
        torch.set_num_threads(torch_num_threads)

    logging.basicConfig(level=logging.INFO)
    imitation_logger.configure(log_dir, ["stdout", "csv", "tensorboard"])

    # setup environment & dataset
    webdatasets, combined_meta = auto.load_wds_datasets(
        configs=dataset_configs)
    color_space = combined_meta['color_space']
    observation_space = combined_meta['observation_space']
    action_space = combined_meta['action_space']

    # callbacks for saving example batches
    def make_batch_saver(interval):
        save_video = algo in ['Autoencoder', 'VariationalAutoencoder']
        print(f'In run_rep_learner, save_video={save_video}')
        return RepLSaveExampleBatchesCallback(
            save_interval_batches=interval,
            dest_dir=os.path.join(log_dir, 'batch_saves'),
            color_space=color_space,
            save_video=save_video)
    repl_callbacks = []
    repl_end_callbacks = []
    if repl_batch_save_interval is not None:
        # this gets called at every batch, so we need a nonzero interval
        reg_save_callback = make_batch_saver(repl_batch_save_interval)
    else:
        # if there's no specified interval, we set the interval so high that it
        # will only run once (at the start)
        reg_save_callback = make_batch_saver(n_epochs * batches_per_epoch + 1)
    repl_callbacks.append(reg_save_callback)
    # this callback gets called once at the end to guarantee that we always
    # save the last batch
    repl_end_callbacks.append(make_batch_saver(0))

    if isinstance(algo, str):
        algo = getattr(algos, algo)

    # instantiate algo
    dataset_configs_multitask = np.all([config_specifies_task_name(config_dict)
                                        for config_dict in dataset_configs])
    if is_multitask:
        assert dataset_configs_multitask, "Parameter `is_multitask` is set, but dataset_configs contain configs " \
                                          "referencing the current task_name"
    else:
        assert not dataset_configs_multitask, "dataset_configs implies a multitask training setup, but " \
                                              "is_multitask is set to False; please fix to make consistent"
    assert issubclass(algo, RepresentationLearner)
    algo_params = dict(algo_params)
    algo_params['augmenter_kwargs'] = {
        'color_space': color_space,
        **algo_params['augmenter_kwargs'],
    }
    logging.info(f"Running {algo} with parameters: {algo_params}")
    model = algo(
        observation_space=observation_space,
        action_space=action_space,
        **algo_params)

    # setup model
    loss_record, most_recent_encoder_path = model.learn(
        datasets=webdatasets,
        batches_per_epoch=batches_per_epoch,
        n_epochs=n_epochs,
        callbacks=repl_callbacks,
        log_dir=log_dir,
        end_callbacks=repl_end_callbacks,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        scheduler_cls=scheduler_cls,
        scheduler_kwargs=scheduler_kwargs,
        log_interval=log_interval,
        save_interval=save_interval,
    )

    result = {
        'encoder_path': most_recent_encoder_path,
        # return average loss from final epoch for HP tuning
        'repl_loss': loss_record[-1],
    }

    if debug_return_model:
        # Used for serialization validation testing in test_base_algos.py.
        result['model'] = model

    return result


if __name__ == '__main__':
    represent_ex.observers.append(FileStorageObserver('runs/rep_learning_runs'))
    represent_ex.run_commandline()

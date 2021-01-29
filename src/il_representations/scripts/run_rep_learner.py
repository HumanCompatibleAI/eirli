import logging
import os

import numpy as np
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch

from il_representations import algos
from il_representations.algos.representation_learner import \
    RepresentationLearner
from il_representations.algos.utils import LinearWarmupCosine
from il_representations.envs import auto
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient)
from il_representations.utils import RepLSaveCallback

sacred.SETTINGS['CAPTURE_MODE'] = 'sys'  # workaround for sacred issue#740
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
    n_envs = 1
    algo_params = {
        'representation_dim': 128,
        'optimizer': torch.optim.Adam,
        'optimizer_kwargs': {'lr': 1e-4},
        'augmenter_kwargs': {
            # augmenter_spec is a comma-separated list of enabled
            # augmentations. Consult docstring for
            # imitation.augment.StandardAugmentations to see available
            # augmentations.
            "augmenter_spec": "translate,rotate,gaussian_blur",
        },
    }
    device = "auto"

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
    n_epochs = 10

    # how often should we save repL batch data?
    # (set to None to disable completely)
    repl_batch_save_interval = 1000

    _ = locals()
    del _


@represent_ex.named_config
def cosine_warmup_scheduler():
    algo_params = {
        "scheduler": LinearWarmupCosine,
        "scheduler_kwargs": {'warmup_epoch': 2, 'T_max': 10}
    }
    _ = locals()
    del _


@represent_ex.named_config
def ceb_breakout():
    env_id = 'BreakoutNoFrameskip-v4'
    train_from_expert = True
    algo = algos.FixedVarianceCEB
    batches_per_epoch = 5
    n_epochs = 1
    _ = locals()
    del _


@represent_ex.named_config
def expert_demos():
    dataset_configs = [{'type': 'demos'}]
    _ = locals()
    del _


@represent_ex.named_config
def random_demos():
    dataset_configs = [{'type': 'random'}]
    _ = locals()
    del _


def initialize_non_features_extractor(sb3_model):
    # This is a hack to get around the fact that you can't initialize only some
    # of the components of a SB3 policy upon creation, and we in fact want to
    # keep the loaded representation frozen, but orthogonally initalize other
    # components.
    sb3_model.policy.init_weights(sb3_model.policy.mlp_extractor, np.sqrt(2))
    sb3_model.policy.init_weights(sb3_model.policy.action_net, 0.01)
    sb3_model.policy.init_weights(sb3_model.policy.value_net, 1)
    return sb3_model


@represent_ex.main
def run(dataset_configs, algo, algo_params, seed, batches_per_epoch, n_epochs,
        torch_num_threads, repl_batch_save_interval, _config):
    # TODO fix to not assume FileStorageObserver always present

    log_dir = represent_ex.observers[0].dir
    if torch_num_threads is not None:
        torch.set_num_threads(torch_num_threads)

    if isinstance(algo, str):
        algo = getattr(algos, algo)

    # setup environment & dataset
    webdatasets, combined_meta = auto.load_wds_datasets(
        configs=dataset_configs)
    color_space = combined_meta['color_space']
    observation_space = combined_meta['observation_space']
    action_space = combined_meta['action_space']

    # callbacks for saving example batches
    def make_batch_saver(interval):
        return RepLSaveCallback(save_interval_batches=repl_batch_save_interval,
                                dest_dir=os.path.join(log_dir, 'batch_saves'),
                                color_space=color_space)
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

    # instantiate algo
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
        color_space=color_space,
        log_dir=log_dir,
        **algo_params)

    # setup model
    loss_record, most_recent_encoder_path = model.learn(
        datasets=webdatasets,
        batches_per_epoch=batches_per_epoch,
        n_epochs=n_epochs,
        callbacks=repl_callbacks,
        end_callbacks=repl_end_callbacks)

    return {
        'encoder_path': most_recent_encoder_path,
        # return average loss from final epoch for HP tuning
        'repl_loss': loss_record[-1],
        # Used for serialization validation testing in test_base_algos.py.
        'model': model,
    }


if __name__ == '__main__':
    represent_ex.observers.append(FileStorageObserver('runs/rep_learning_runs'))
    represent_ex.run_commandline()

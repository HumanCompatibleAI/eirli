#!/usr/bin/env python3
"""Run an IL algorithm in some selected domain."""
import logging
import os

from imitation.algorithms.bc import BC
from imitation.algorithms.adversarial import GAIL
from imitation.util.util import make_vec_env
import imitation.util.logger as imitation_logger
from sacred import Experiment
from sacred.observers import FileStorageObserver
from stable_baselines3.common.cmd_util import make_atari_env
import stable_baselines3.common.policies as sb3_pols
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.ppo import PPO
from stable_baselines3.common.utils import get_device

from il_representations.data import TransitionsMinimalDataset
from il_representations.envs.config import benchmark_ingredient
from il_representations.envs.atari_envs import load_dataset_atari
from il_representations.envs.dm_control_envs import load_dataset_dm_control
from il_representations.envs.magical_envs import load_dataset_magical
from il_representations.il.disc_rew_nets import ImageDiscrimNet

il_train_ex = Experiment('il_train', ingredients=[benchmark_ingredient])


@il_train_ex.config
def default_config():
    # choose between 'bc'/'gail'
    algo = 'bc'

    # bc config variables
    bc_n_epochs = 100

    # device to place all computations on
    dev_name = 'auto'


@il_train_ex.capture
def do_training_bc(venv_chans_first, dataset, out_dir, bc_n_epochs, dev_name):
    trainer = BC(observation_space=venv_chans_first.observation_space,
                 action_space=venv_chans_first.action_space,
                 policy_class=sb3_pols.ActorCriticCnnPolicy,
                 expert_data=dataset,
                 device=dev_name)

    logging.info("Beginning BC training")
    trainer.train(n_epochs=bc_n_epochs)

    final_path = os.path.join(out_dir, 'policy_final.pt')
    logging.info(f"Saving final policy to {final_path}")
    trainer.save_policy(final_path)


@il_train_ex.capture
def do_training_gail(venv_chans_first, dataset, dev_name):
    device = get_device(dev_name)
    # vec_env_chans_first = VecTransposeImage(vec_env_chans_last)
    discrim_net = ImageDiscrimNet(
        observation_space=venv_chans_first.observation_space,
        action_space=venv_chans_first.action_space)
    ppo_algo = PPO(
        policy=sb3_pols.ActorCriticCnnPolicy,
        env=venv_chans_first,
        # verbose=1 and tensorboard_log=False is a hack to work around SB3
        # issue #109.
        verbose=1,
        tensorboard_log=None,
        device=device,
    )
    trainer = GAIL(
        venv_chans_first,
        dataset,
        ppo_algo,
        discrim_kwargs=dict(discrim_net=discrim_net),
    )

    # trainer.train() doesn't work because SB3 tries to be clever about
    # automatically adding image transpose wrappers to vecenvs, but
    # `imitation` overwrites the vecenv later on without updating SB3's
    # internal data structures.
    #
    # Specifically, PPO notices that the given environment is not wrapped
    # in a `VecTransposeImage` wrapper, so it wraps it in one. Later on,
    # GAIL overwrites the wrapped env inside PPO with a new wrapped
    # environment that does _not_ apply VecTransposeImage. Hacky solution
    # is to make GAIL apply VecTranposeImage after wrapping the
    # environment. Non-hacky solution is to make SB3 algorithms not wrap
    # environments by default, or at least add a flag to `PPO` that
    # disables auto-wrapping of environments..

    # trainer.train(total_timesteps=2048)

    raise NotImplementedError(
        "GAIL doesn't yet work due to image transpose issues in imitation/SB3")


@il_train_ex.main
def train(algo, bc_n_epochs, benchmark, _config):
    # python built-in logging
    logging.basicConfig(level=logging.INFO)
    # `imitation` logging
    # FIXME(sam): I used this hack from run_rep_learner.py, but I don't
    # actually know the right way to write log files continuously in Sacred.
    log_dir = il_train_ex.observers[0].dir
    imitation_logger.configure(log_dir, ["stdout", "tensorboard"])

    if benchmark['benchmark_name'] == 'magical':
        gym_env_name_chans_first, dataset_dict = load_dataset_magical()
        human_readable_env_name = gym_env_name_chans_first
        venv_chans_first = make_vec_env(gym_env_name_chans_first,
                                        n_envs=1,
                                        parallel=False)
    elif benchmark['benchmark_name'] == 'dm_control':
        gym_env_name_chans_first, dataset_dict = load_dataset_dm_control()
        human_readable_env_name = gym_env_name_chans_first
        venv_chans_first = make_vec_env(gym_env_name_chans_first,
                                        n_envs=1,
                                        parallel=False)
    elif benchmark['benchmark_name'] == 'atari':
        dataset_dict = load_dataset_atari()
        human_readable_env_name = gym_env_name_chans_last \
            = benchmark['atari_env_id']
        venv_nhwc = VecFrameStack(make_atari_env(gym_env_name_chans_last), 4)
        venv_chans_first = VecTransposeImage(venv_nhwc)
    else:
        raise NotImplementedError(
            f"this code does not yet support "
            f"benchmark_name={benchmark['benchmark_name']!r}")

    dataset = TransitionsMinimalDataset(dataset_dict)

    logging.info(
        f"Loaded data for '{human_readable_env_name}', setting up env")

    logging.info(f"Setting up '{algo}' IL algorithm")

    if algo == 'bc':
        do_training_bc(dataset=dataset,
                       venv_chans_first=venv_chans_first,
                       out_dir=log_dir)

    elif algo == 'gail':
        do_training_gail(
            dataset=dataset,
            venv_chans_first=venv_chans_first,
            # FIXME(sam): whatever this is doing, it won't work
            # for Atari
            gym_env_name_chans_first=gym_env_name_chans_first)

    else:
        raise NotImplementedError(f"Can't handle algorithm '{algo}'")

    # FIXME(sam): make sure this always closes correctly, even when there's an
    # exception after creating it (could use try/catch or a context manager)
    venv_chans_first.close()

    return log_dir


if __name__ == '__main__':
    il_train_ex.observers.append(FileStorageObserver('il_train_runs'))
    il_train_ex.run_commandline()

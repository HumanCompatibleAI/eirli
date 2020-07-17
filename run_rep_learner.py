import sys, os
import gym

from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import algos
from algos.representation_learner import DEFAULT_HYPERPARAMS as rep_learner_params
from algos.representation_learner import RepresentationLearner
from sacred import Experiment
from sacred.observers import FileStorageObserver
from rl_baselines_zoo.utils import create_test_env
import numpy as np


represent_ex = Experiment('representation_learning')


@represent_ex.config
def default_config():
    env_id = 'BreakoutNoFrameskip-v4'
    algo = "SimCLR"
    n_envs = 1
    train_from_expert = True
    timesteps = 640
    pretrain_only = False
    pretrain_epochs = 50
    _ = locals()
    del _


def get_random_traj(env, timesteps):
    # Currently not designed for VecEnvs with n>1
    trajectory = {'states': [], 'actions': [], 'dones': []}
    obs = env.reset()
    for i in range(timesteps):
        trajectory['states'].append(obs.squeeze())
        action = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        obs, rew, dones, info = env.step(action)
        trajectory['actions'].append(action[0])
        trajectory['dones'].append(dones[0])
    return trajectory


@represent_ex.main
def run(env_id, seed, algo, n_envs, timesteps, train_from_expert,
           pretrain_only, pretrain_epochs, _config):

    # TODO fix this hacky nonsense
    log_dir = os.path.join(represent_ex.observers[0].dir, 'training_logs')
    os.mkdir(log_dir)
    #with TemporaryDirectory() as tmp_dir:
    if isinstance(algo, str):
        algo = dir(algos)[algo]

    is_atari = 'NoFrameskip' in env_id
    env = create_test_env(env_id, n_envs=n_envs, is_atari=is_atari,
                          stats_path=os.path.join(log_dir, 'stats'), seed=seed, log_dir=log_dir,
                          should_render=False)
    data = get_random_traj(env=env, timesteps=timesteps)

    # setup environment
    if is_atari:
        env = VecFrameStack(make_atari_env(env_id, n_envs, seed), 4)
    else:
        env = gym.make(env_id)
    assert issubclass(algo, RepresentationLearner)
    ## TODO allow passing in of kwargs here


    algo_params = {k: v for k, v in _config.items() if k in rep_learner_params.keys()}
    model = algo(env, log_dir=log_dir, **algo_params)

    # setup model
    model.learn(data)

    env.close()

    # Free memory
    del model


if __name__ == '__main__':
    represent_ex.observers.append(FileStorageObserver('rep_learning_runs'))
    represent_ex.run_commandline()

import os
import gym

from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.ppo import PPO
import algos
from algos.representation_learner import RepresentationLearner
from policy_interfacing import EncoderFeatureExtractor
from sacred import Experiment
from sacred.observers import FileStorageObserver
import numpy as np
import inspect

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
    representation_dim = 128
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
def run(env_id, seed, algo, n_envs, timesteps, representation_dim, train_from_expert,
           pretrain_only, pretrain_epochs, _config):

    # TODO fix this hacky nonsense
    log_dir = os.path.join(represent_ex.observers[0].dir, 'training_logs')
    os.mkdir(log_dir)
    #with TemporaryDirectory() as tmp_dir:
    if isinstance(algo, str):
        correct_algo_cls = None
        for algo_name, algo_cls in inspect.getmembers(algos):
            if algo == algo_name:
                correct_algo_cls = algo_cls
                break
        algo = correct_algo_cls

    is_atari = 'NoFrameskip' in env_id


    # setup environment
    if is_atari:
        env = VecFrameStack(make_atari_env(env_id, n_envs, seed), 4)
    else:
        env = gym.make(env_id)

    data = get_random_traj(env=env, timesteps=timesteps)
    assert issubclass(algo, RepresentationLearner)

    rep_learner_params = inspect.getfullargspec(RepresentationLearner.__init__).args
    algo_params = {k: v for k, v in _config.items() if k in rep_learner_params}
    model = algo(env, log_dir=log_dir, **algo_params)

    # setup model
    model.learn(data)

    encoder_feature_extractor_kwargs = {'features_dim': representation_dim, 'encoder': model.encoder}
    policy_kwargs = {'features_extractor_class': EncoderFeatureExtractor,
                     'features_extractor_kwargs': encoder_feature_extractor_kwargs }
    # encoder_policy = ActorCriticPolicy(observation_space=env.observation_space, action_space=env.action_space,
    #                                    lr_schedule=lambda x: 0.01, features_extractor_class=EncoderFeatureExtractor,
    #                                    features_extractor_kwargs=encoder_feature_extractor_kwargs)

    ppo_model = PPO(policy=ActorCriticPolicy, env=env, verbose=1, policy_kwargs=policy_kwargs)
    ppo_model.learn(total_timesteps=1000)
    env.close()

    # Free memory
    del model


if __name__ == '__main__':
    represent_ex.observers.append(FileStorageObserver('rep_learning_runs'))
    represent_ex.run_commandline()

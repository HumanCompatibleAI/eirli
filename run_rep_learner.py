import os
import gym
import torch
from glob import glob
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
    demo_timesteps = 640
    rl_training_timesteps = 1000
    pretrain_only = False
    pretrain_epochs = 50
    representation_dim = 128
    ppo_finetune = True
    _ = locals()
    del _


@represent_ex.named_config
def ceb_long_breakout():
    env_id = 'BreakoutNoFrameskip-v4'
    train_from_expert = True
    algo = algos.FixedVarianceCEB
    loss_constructor_kwargs = {'beta': 0.1}
    pretrain_epochs = 50
    demo_timesteps = None
    ppo_finetune=False
    _ = locals()
    del _

@represent_ex.named_config
def ceb_no_compression_long_breakout():
    env_id = 'BreakoutNoFrameskip-v4'
    train_from_expert = True
    algo = algos.FixedVarianceCEB
    loss_constructor_kwargs = {'beta': 0.0}
    pretrain_epochs = 50
    demo_timesteps = None
    ppo_finetune=False
    _ = locals()
    del _

@represent_ex.capture
def get_random_trajectories(env, demo_timesteps):
    # Currently not designed for VecEnvs with n>1
    trajectory = {'states': [], 'actions': [], 'dones': []}
    obs = env.reset()
    for i in range(demo_timesteps):
        trajectory['states'].append(obs.squeeze())
        action = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        obs, rew, dones, info = env.step(action)
        trajectory['actions'].append(action[0])
        trajectory['dones'].append(dones[0])
    return trajectory

@represent_ex.capture
def get_expert_trajectories(env_id, demo_timesteps):
    expert_data_loc = "/Users/cody/Data/expert_rollouts/"
    rollouts_path = f"{env_id}_rollouts_500_ts_100_traj.npy"
    full_rollouts_path = os.path.join(expert_data_loc, rollouts_path)
    trajectories = np.load(full_rollouts_path, allow_pickle=True)
    merged_trajectories = {'states': [], 'actions': [], 'dones': []}

    for ind, traj in enumerate(trajectories):
        for k in merged_trajectories.keys():
            merged_trajectories[k] += traj[k]
        if demo_timesteps is not None and len(merged_trajectories['states']) > demo_timesteps:
            for k in merged_trajectories.keys():
                merged_trajectories[k] = merged_trajectories[k][0:demo_timesteps]
            break
    if demo_timesteps is not None and len(merged_trajectories['states']) < demo_timesteps:
        raise Warning(f"Requested {demo_timesteps} timesteps, only was able to read in {len(merged_trajectories['states'])}")
    return merged_trajectories


def initialize_non_features_extractor(sb3_model):
    # This is a hack to get around the fact that you can't initialize only some of the components of a SB3 policy
    # upon creation, and we in fact want to keep the loaded representation frozen, but orthogonally initalize other
    # components.
    sb3_model.policy.init_weights(sb3_model.policy.mlp_extractor, np.sqrt(2))
    sb3_model.policy.init_weights(sb3_model.policy.action_net, 0.01)
    sb3_model.policy.init_weights(sb3_model.policy.value_net, 1)
    return sb3_model


@represent_ex.main
def run(env_id, seed, algo, n_envs, pretrain_epochs, rl_training_timesteps, representation_dim, ppo_finetune, train_from_expert, _config):

    # TODO fix to not assume FileStorageObserver always present
    log_dir = os.path.join(represent_ex.observers[0].dir, 'training_logs')
    os.mkdir(log_dir)

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
    if train_from_expert:
        data = get_expert_trajectories()
    else:
        data = get_random_trajectories(env=env)
    assert issubclass(algo, RepresentationLearner)

    rep_learner_params = inspect.getfullargspec(RepresentationLearner.__init__).args
    algo_params = {k: v for k, v in _config.items() if k in rep_learner_params}
    model = algo(env, log_dir=log_dir, **algo_params)

    # setup model
    model.learn(data)
    if ppo_finetune and not isinstance(model, algos.RecurrentCPC):
        encoder_checkpoint = model.encoder_checkpoints_path
        all_checkpoints = glob(os.path.join(encoder_checkpoint, '*'))
        latest_checkpoint = max(all_checkpoints, key=os.path.getctime)
        encoder_feature_extractor_kwargs = {'features_dim': representation_dim, 'encoder_path': latest_checkpoint}

        #TODO figure out how to not have to set `ortho_init` to False for the whole policy
        policy_kwargs = {'features_extractor_class': EncoderFeatureExtractor,
                         'features_extractor_kwargs': encoder_feature_extractor_kwargs,
                         'ortho_init': False}
        ppo_model = PPO(policy=ActorCriticPolicy, env=env, verbose=1, policy_kwargs=policy_kwargs)
        ppo_model = initialize_non_features_extractor(ppo_model)
        ppo_model.learn(total_timesteps=rl_training_timesteps)
        env.close()


if __name__ == '__main__':
    represent_ex.observers.append(FileStorageObserver('rep_learning_runs'))
    represent_ex.run_commandline()

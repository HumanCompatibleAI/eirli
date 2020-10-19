from il_representations.envs.config import benchmark_ingredient
from il_representations.utils import subset_all_dict_values
import os
import minerl
import numpy as np
from gym import Wrapper, spaces, Env, register
import time
import logging

MOCK_TO_REAL_LOOKUP = {
    'MinecraftTreechopMockEnv-v0': 'MineRLTreechopVectorObf-v0'
}
REAL_TO_MOCK_LOOKUP = {v:k for k,v in MOCK_TO_REAL_LOOKUP.items()}

@benchmark_ingredient.capture
def load_dataset_minecraft(minecraft_env_id, minecraft_data_root, n_traj=None, timesteps=None, chunk_length=100):
    if 'Mock' in minecraft_env_id:
        minecraft_env_id = MOCK_TO_REAL_LOOKUP[minecraft_env_id]
    data_iterator = minerl.data.make(environment=minecraft_env_id,
                                     data_dir=minecraft_data_root,
                                     max_recordings=n_traj)
    appended_trajectories = {'obs': [], 'acts': [], 'dones': []}
    start_time = time.time()
    for current_state, action, reward, next_state, done in data_iterator.batch_iter(batch_size=1,
                                                                                    num_epochs=1,
                                                                                    seq_len=chunk_length):
        appended_trajectories['obs'].append(MinecraftVectorWrapper.transform_obs(current_state)[0])
        appended_trajectories['acts'].append(MinecraftVectorWrapper.extract_action(action)[0])
        appended_trajectories['dones'].append(done[0])
    appended_trajectories['next_obs'] = appended_trajectories['obs'][1:]
    appended_trajectories['obs'] = appended_trajectories['obs'][0:-1]
    appended_trajectories['acts'] = appended_trajectories['acts'][0:-1]
    appended_trajectories['dones'] = appended_trajectories['dones'][0:-1]
    end_time = time.time()
    logging.info(f"Minecraft trajectory collection took {round(end_time - start_time, 2)} seconds to complete")
    merged_trajectories = {k: np.concatenate(v, axis=0) for k, v in appended_trajectories.items()}
    if timesteps is not None:
        merged_trajectories = subset_all_dict_values(merged_trajectories, timesteps)
    return merged_trajectories


def channels_first(el):
    if isinstance(el, np.ndarray):
        dimension_order = list(range(len(el.shape)))

        # Get final dimension, which is currently channels dimension
        final_ind = dimension_order[-1]
        # Switch with dimension which we want to contain channels dimension, the third from the end
        dimension_order[final_ind-2] = final_ind
        dimension_order[final_ind] = final_ind-2
        return np.transpose(el, tuple(dimension_order))

    elif isinstance(el, tuple):
        return (el[2], el[0], el[1])

    else:
        raise NotImplementedError("Input must be either array or tuple")


class MinecraftVectorWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert 'vector' in env.action_space.spaces.keys(), "Wrapper is only implemented to work with Vector Obfuscated envs"
        self.action_space = env.action_space.spaces['vector']
        pov_space = env.observation_space.spaces['pov']
        transposed_pov_space = spaces.Box(low=channels_first(pov_space.low),
                                          high=channels_first(pov_space.high),
                                          shape=channels_first(pov_space.shape),
                                          dtype=np.uint8)
        self.observation_space = transposed_pov_space

    @staticmethod
    def transform_obs(obs):
        return channels_first(obs['pov']).astype(np.uint8)

    @staticmethod
    def extract_action(action):
        return action['vector']

    @staticmethod
    def dictify_action(action):
        return {'vector': action}

    def step(self, action):
        obs, rew, dones, infos = self.env.step(MinecraftVectorWrapper.dictify_action(action))
        transformed_obs = MinecraftVectorWrapper.transform_obs(obs)
        return transformed_obs, rew, dones, infos

    def reset(self):
        obs = self.env.reset()
        return MinecraftVectorWrapper.transform_obs(obs)


class TestingEnvironment(Env):
    # Test environment for situations where we don't actually need to
    # interact with a Minecraft environment, only pull env/action space info from it
    def __init__(self, obs_shape=(64, 64, 3), action_shape=(64,), obs_high=255,
                obs_low=0, action_high=1.05, action_low=-1.05):
        super().__init__()
        self.observation_space = spaces.Dict({'pov': spaces.Box(shape=obs_shape, high=obs_high, low=obs_low)})
        self.action_space = spaces.Dict({'vector': spaces.Box(shape=action_shape, high=action_high, low=action_low)})
        self.steps_taken = 0

    def reset(self):
        steps_taken = 0
        return self.observation_space.sample()

    def step(self, action):
        self.steps_taken += 1
        if self.steps_taken >= 100:
            done = True
        else:
            done = False
        return self.observation_space.sample(), 0, done, dict()


def entry_point(**kwargs):
    # add in common kwargs
    return TestingEnvironment(**kwargs)

# frame skip 2
register('MinecraftTreechopMockEnv-v0',
         entry_point=entry_point)



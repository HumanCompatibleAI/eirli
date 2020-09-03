from il_representations.envs.config import benchmark_ingredient
import os
import minerl
import numpy as np
from gym import Wrapper, spaces


@benchmark_ingredient.capture
def load_dataset_minecraft(minecraft_env_id, minecraft_data_root, chunk_length=100):

    data_iterator = minerl.data.make(environment=minecraft_env_id,
                                     data_dir=minecraft_data_root)
    appended_trajectories = {'obs': [], 'acts': [], 'dones': []}
    for current_state, action, reward, next_state, done in data_iterator.batch_iter(batch_size=1,
                                                                                    num_epochs=1,
                                                                                    seq_len=chunk_length):
        appended_trajectories['obs'].append(MinecraftVectorWrapper.transform_obs(current_state)[0])
        appended_trajectories['acts'].append(MinecraftVectorWrapper.extract_action(action)[0])
        appended_trajectories['dones'].append(done[0])

    merged_trajectories = {k: np.concatenate(v, axis=0) for k, v in appended_trajectories.items()}
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









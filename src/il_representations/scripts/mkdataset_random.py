"""Make a repL WebDataset from random rollouts."""
import logging
import os
import sys
import time

from imitation.data import rollout
from imitation.policies import base as policy_base
import numpy as np
import sacred
from sacred import Experiment
from tqdm import tqdm

from il_representations.algos.utils import set_global_seeds
from il_representations.data.write_dataset import (get_meta_dict,
                                                   get_out_file_map,
                                                   write_trajectories)
from il_representations.envs import auto
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)

sacred.SETTINGS['CAPTURE_MODE'] = 'sys'  # workaround for sacred issue#740
mkdataset_random_ex = Experiment('mkdataset_random',
                                 ingredients=[
                                     env_cfg_ingredient, env_data_ingredient,
                                     venv_opts_ingredient
                                 ])


@mkdataset_random_ex.config
def default_config():
    # minimum number of timesteps to write (depending on how the vec env is set
    # up, we might write more than that)
    n_timesteps_min = 100000

    _ = locals()
    del _


@mkdataset_random_ex.main
def run(seed, env_data, env_cfg, n_timesteps_min):
    set_global_seeds(seed)
    logging.basicConfig(level=logging.INFO)

    out_file_map = get_out_file_map()
    out_file_path = os.path.join(out_file_map['random'], 'random.tgz')

    venv = auto.load_vec_env()
    policy = policy_base.RandomPolicy(venv.observation_space,
                                      venv.action_space)

    meta_dict = get_meta_dict()

    timestep_ctr = 0
    traj_ctr = 0

    def traj_iter():
        nonlocal timestep_ctr, traj_ctr
        if os.isatty(sys.stdout.fileno()):
            prog_bar = tqdm(total=n_timesteps_min, desc='steps')
        else:
            prog_bar = None
        while timestep_ctr < n_timesteps_min:
            # keys in dataset_dict: 'obs', 'next_obs', 'acts', 'infos', 'rews',
            # 'dones'
            # attributes of returned trajectories: obs, acts, infos, rews
            new_trajs = rollout.generate_trajectories(
                policy,
                venv,
                sample_until=rollout.min_timesteps(n_timesteps_min))
            for traj in new_trajs:
                obs = traj.obs[:-1]
                next_obs = traj.obs[1:]
                T = len(obs)
                dones = np.zeros((T, ), dtype='int64')
                dones[-1] = 1
                yield {
                    'obs': obs,
                    'next_obs': next_obs,
                    'acts': traj.acts,
                    'infos': traj.infos,
                    'rews': traj.rews,
                    'dones': dones,
                }

                assert T > 0  # to ensure forward progress
                timestep_ctr += T
                traj_ctr += 1
                if prog_bar is not None:
                    prog_bar.update(T)

    logging.info(
        f"Will write >={n_timesteps_min} timesteps to '{out_file_path}'")
    start = time.time()
    write_trajectories(out_file_path, meta_dict, traj_iter())
    elapsed = time.time() - start
    logging.info(
        f'Done, wrote {timestep_ctr} timesteps and {traj_ctr} trajectories '
        f'in {elapsed:.2f}s')


if __name__ == '__main__':
    mkdataset_random_ex.run_commandline()

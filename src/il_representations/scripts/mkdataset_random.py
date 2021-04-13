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
from il_representations.data.write_dataset import get_meta_dict, write_frames
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
def run(seed, env_data, env_cfg, n_timesteps_min, *,
        _max_steps_to_write_at_once=16384):
    set_global_seeds(seed)
    logging.basicConfig(level=logging.INFO)

    out_file_path = os.path.join(
        auto.get_data_dir(benchmark_name=env_cfg['benchmark_name'],
                          task_key=env_cfg['task_name'],
                          data_type='random'), 'random.tgz')

    venv = auto.load_vec_env()
    policy = policy_base.RandomPolicy(venv.observation_space,
                                      venv.action_space)

    meta_dict = get_meta_dict()

    timestep_ctr = 0
    traj_ctr = 0

    def frame_iter():
        nonlocal timestep_ctr, traj_ctr
        # keep generating trajectories until we meet or exceed the minimum time
        # step count
        while timestep_ctr < n_timesteps_min:
            n_to_generate = min(_max_steps_to_write_at_once,
                                n_timesteps_min - timestep_ctr)
            logging.info(f'Generating {n_to_generate} timesteps '
                         f'({timestep_ctr}/{n_timesteps_min} generated so '
                         'far)')

            # random demos
            new_trajs = rollout.generate_trajectories(
                policy,
                venv,
                sample_until=rollout.min_timesteps(n_to_generate))
            n_generated = sum(len(traj.acts) for traj in new_trajs)
            logging.info(f'Got {n_generated} steps, will write '
                         f'{len(new_trajs)} trajectories')
            if os.isatty(sys.stdout.fileno()):
                prog_bar = tqdm(total=n_generated, desc='steps')
            else:
                prog_bar = None
            for traj in new_trajs:
                obs = traj.obs[:-1]
                next_obs = traj.obs[1:]
                T = len(obs)
                assert T > 0, "empty trajectory?"
                dones = np.zeros((T, ), dtype='int64')
                dones[-1] = 1
                # yield a dictionary for each frame in the retrieved
                # trajectories
                for idx in range(T):
                    yield {
                        # Keys in dataset_dict: 'obs', 'next_obs', 'acts',
                        # 'infos', 'rews', 'dones'.
                        # Attributes of returned trajectories: obs, acts,
                        # infos, rews.
                        'obs': obs[idx],
                        'next_obs': next_obs[idx],
                        'acts': traj.acts[idx],
                        'infos': traj.infos[idx],
                        'rews': traj.rews[idx],
                        'dones': dones[idx],
                    }

                    timestep_ctr += 1
                    if prog_bar is not None:
                        prog_bar.update(1)

                traj_ctr += 1

    logging.info(
        f"Will write >={n_timesteps_min} timesteps to '{out_file_path}'")
    start = time.time()
    write_frames(out_file_path, meta_dict, frame_iter())
    elapsed = time.time() - start
    logging.info(
        f'Done, wrote {timestep_ctr} timesteps and {traj_ctr} trajectories '
        f'in {elapsed:.2f}s')


if __name__ == '__main__':
    mkdataset_random_ex.run_commandline()

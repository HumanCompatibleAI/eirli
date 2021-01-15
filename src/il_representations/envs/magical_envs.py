"""Importing this file automatically registers all relevant MAGICAL
environments."""

import collections
import logging
import os
import random
import shutil
from typing import List

import imitation.data.rollout as il_rollout
from imitation.util.util import make_vec_env
from magical import register_envs, saved_trajectories
from magical.evaluation import EvaluationProtocol
import numpy as np
import torch as th

from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)

register_envs()


def load_data(
        pickle_paths: List[str],
        preprocessor_name: str,
        remove_null_actions: bool = False,
):
    """Load MAGICAL data from pickle files."""

    # First we load pickles off disk and infer the env name from their content.
    # demo_trajectories will contain a list of `MAGICALTrajectory`
    # objects---these are essentially the same as imitation's `Trajectory`
    # class, except that the observation is a dictionary instead of an ndarray.
    env_name = None
    demo_trajectories = []
    for demo_dict in saved_trajectories.load_demos(pickle_paths):
        new_env_name = demo_dict['env_name']
        if env_name is None:
            env_name = new_env_name
        else:
            if env_name != new_env_name:
                raise ValueError(
                    f"supplied trajectory paths contain demos for multiple "
                    f"environments: {env_name}, {new_env_name} ")

        demo_trajectories.append(demo_dict['trajectory'])

    del new_env_name  # unused

    # Now we apply the supplied preprocessor, if any, to the loaded
    # trajectories. We'll probably apply the "LoRes4E" preprocessor to
    # everything, which replaces the default dict observation space with a
    # simple image observation space depicting only an egocentric view of the
    # environment. MAGICAL's default preprocessors are built into environment
    # names as a convenience, so we also update the environment name to include
    # the new preprocessor name.
    if preprocessor_name:
        demo_trajectories = saved_trajectories.preprocess_demos_with_wrapper(
            demo_trajectories,
            orig_env_name=env_name,
            preproc_name=preprocessor_name)

    # Finally we build a DictDataset for actions and observations.
    dataset_dict = collections.defaultdict(list)
    for trajectory in demo_trajectories:
        if isinstance(trajectory.obs, dict):
            # Without any preprocessing, MAGICAL observations are dicts
            # containing an 'ego' and 'allo' key for egocentric view and
            # allocentric view, respectively. We handle this case first.
            for key, value in trajectory.obs.items():
                # we clip off the last (terminal) time step, which doesn't
                # correspond to any action, and use it for next_obs instead
                dataset_dict[f'obs_{key}'].append(value[:-1])
                dataset_dict[f'next_obs_{key}'].append(value[1:])
        else:
            # Otherwise, observations should just be a flat ndarray
            assert isinstance(trajectory.obs, np.ndarray)
            # again clip off the terminal observation
            dataset_dict['obs'].append(trajectory.obs[:-1])
            dataset_dict['next_obs'].append(trajectory.obs[1:])
        dataset_dict['acts'].append(trajectory.acts)
        traj_t = len(trajectory.acts)
        dones_array = np.array([False] * (traj_t - 1) + [True], dtype='bool')
        dataset_dict['dones'].append(dones_array)

    # join together all the lists of ndarrays
    dataset_dict = {
        item_name: np.concatenate(array_list, axis=0)
        for item_name, array_list in dataset_dict.items()
    }

    if remove_null_actions:
        # remove all "stay still" actions
        chosen_mask = dataset_dict["acts"] != 0
        for key in dataset_dict.keys():
            dataset_dict[key] = dataset_dict[key][chosen_mask]

    return dataset_dict, env_name


@env_cfg_ingredient.capture
def get_env_name_magical(task_name, magical_preproc):
    # for MAGICAL, the 'task_name' is a prefix like MoveToCorner or
    # ClusterShape
    orig_env_name = task_name + '-Demo-v0'
    gym_env_name = saved_trajectories.splice_in_preproc_name(
        orig_env_name, magical_preproc)
    return gym_env_name


@env_data_ingredient.capture
def _get_magical_data_cfg(magical_demo_dirs, data_root):
    # workaround for Sacred issue #206
    return magical_demo_dirs, data_root


@env_cfg_ingredient.capture
def load_dataset_magical(
        task_name,
        magical_preproc,
        magical_remove_null_actions,
        n_traj=None):
    magical_demo_dirs, data_root = _get_magical_data_cfg()
    demo_dir = os.path.join(data_root, magical_demo_dirs[task_name])
    logging.info(
        f"Loading trajectory data for '{task_name}' from "
        f"'{demo_dir}'")
    demo_paths = [
        os.path.join(demo_dir, f) for f in os.listdir(demo_dir)
        if f.endswith('.pkl.gz')
    ]
    if not demo_paths:
        raise IOError(f"Could not find any demo pickle files in '{demo_dir}'")
    random.shuffle(demo_paths)
    if n_traj is not None:
        demo_paths = demo_paths[:n_traj]
    dataset_dict, loaded_env_name = load_data(
        demo_paths, preprocessor_name=magical_preproc,
        remove_null_actions=magical_remove_null_actions)
    gym_env_name = get_env_name_magical()
    assert loaded_env_name.startswith(gym_env_name.rsplit('-')[0])
    return dataset_dict


@env_cfg_ingredient.capture
def rewrite_dataset_magical(task_name, n_traj, new_data_root):
    # select the demo paths we want to load
    magical_demo_dirs, data_root = _get_magical_data_cfg()
    demo_dir = os.path.join(data_root, magical_demo_dirs[task_name])
    demo_paths = [
        os.path.join(demo_dir, f) for f in os.listdir(demo_dir)
        if f.endswith('.pkl.gz')
    ]
    if not demo_paths:
        raise IOError(f"Could not find any demo pickle files in '{demo_dir}'")
    if len(demo_paths) < n_traj:
        raise IOError(
            f"Requested n_traj={n_traj} trajectories, but only have "
            f"{len(demo_paths)} demo files")
    random.shuffle(demo_paths)
    if n_traj is not None:
        demo_paths = demo_paths[:n_traj]
    assert len(demo_paths) == n_traj

    # copy those paths over directly to the output directory
    out_dir = os.path.join(new_data_root, magical_demo_dirs[task_name])
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"Copying {len(demo_paths)} trajectory files to '{out_dir}':")
    for count, demo_path in enumerate(demo_paths, start=1):
        base_name = os.path.basename(demo_path)
        out_demo_path = os.path.join(out_dir, base_name)
        logging.info(
            f"  [{count}/{n_traj}] Copying '{demo_path}' -> '{out_demo_path}'")
        shutil.copy2(demo_path, out_demo_path)


class SB3EvaluationProtocol(EvaluationProtocol):
    """MAGICAL 'evaluation protocol' for Stable Baselines 3 policies."""

    # TODO: more docs, document __init__ in particular
    def __init__(self, policy, run_id, seed, video_writer=None, **kwargs):
        super().__init__(**kwargs)
        self._run_id = run_id
        self.policy = policy
        self.seed = seed
        self.video_writer = video_writer

    @property
    def run_id(self):
        """Identifier for this run in the dataframe produced by
        `.do_eval()`."""
        return self._run_id

    @venv_opts_ingredient.capture
    def obtain_scores(self, env_name, venv_parallel, n_envs):
        """Collect `self.n_rollouts` scores on environment `env_name`."""
        vec_env_chans_last = make_vec_env(env_name,
                                          n_envs=n_envs,
                                          seed=self.seed,
                                          parallel=venv_parallel)
        rng = np.random.RandomState(self.seed)
        trajectories = il_rollout.generate_trajectories(
            self.policy,
            vec_env_chans_last,
            sample_until=il_rollout.min_episodes(self.n_rollouts),
            rng=rng)
        scores = []
        for trajectory in trajectories[:self.n_rollouts]:
            scores.append(trajectory.infos[-1]['eval_score'])

            # optionally write the (scored) trajectory to video output
            if self.video_writer is not None:
                for obs in trajectory.obs:
                    self.video_writer.add_tensor(th.FloatTensor(obs) / 255.)

        return scores

"""Importing this file automatically registers all relevant MAGICAL
environments."""

import collections
from typing import List, Optional, Tuple

import imitation.data.dataset as il_dataset
from magical import register_envs, saved_trajectories
import numpy as np

register_envs()


def load_data(
    pickle_paths: List[str],
    preprocessor_name: Optional[str] = 'LoRes4E',
    transpose_observations=False,
) -> Tuple[str, il_dataset.Dataset]:
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
        env_name = saved_trajectories.splice_in_preproc_name(
            env_name, preprocessor_name)

    # Finally we build a DictDatset for actions and observations.
    dataset_dict = collections.defaultdict(list)
    # we will use obs_keys to decide which arrays we have to transpose, if
    # transpose_observations=True
    obs_keys = set()
    for trajectory in demo_trajectories:
        if isinstance(trajectory.obs, dict):
            # Without any preprocessing, MAGICAL observations are dicts
            # containing an 'ego' and 'allo' key for egocentric view and
            # allocentric view, respectively. We handle this case first.
            for key, value in trajectory.obs.items():
                # we clip off the last (terminal) time step, which doesn't
                # correspond to any action
                full_key = 'obs_' + key
                dataset_dict[full_key].append(value[:-1])
                obs_keys.add(full_key)
        else:
            # Otherwise, observations should just be a flat ndarray
            assert isinstance(trajectory.obs, np.ndarray)
            # again clip off the terminal observation
            dataset_dict['obs'].append(trajectory.obs[:-1])
            obs_keys.add('obs')
        dataset_dict['acts'].append(trajectory.acts)

    # join together all the lists of ndarrays
    dataset_dict = {
        item_name: np.concatenate(array_list, axis=0)
        for item_name, array_list in dataset_dict.items()
    }

    if transpose_observations:
        for obs_key in obs_keys:
            dataset_dict[obs_key] = np.moveaxis(dataset_dict[obs_key], -1, 1)

    dataset = il_dataset.RandomDictDataset(dataset_dict)

    return env_name, dataset

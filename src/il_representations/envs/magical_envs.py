"""Importing this file automatically registers all relevant MAGICAL
environments."""

import collections
import logging
import os
from typing import List, Tuple

import imitation.data.datasets as il_datasets
import imitation.data.types as il_types
import imitation.data.rollout as il_rollout
from imitation.util.util import make_vec_env
from magical import register_envs, saved_trajectories
from magical.evaluation import EvaluationProtocol
import numpy as np

from il_representations.envs.config import benchmark_ingredient

register_envs()


class TransitionsMinimalDataset(il_datasets.Dataset):
    """Exposes a dict {'obs': <observations ndarray>, 'acts'} as a dataset that
    enumerates `TransitionsMinimal` instances. Useful for interfacing with
    BC."""

    def __init__(self, data_map):
        assert data_map.keys() == {'obs', 'acts'}
        self.dict_dataset = il_datasets.RandomDictDataset(data_map)

    def sample(self, n_samples):
        dict_samples = self.dict_dataset.sample(n_samples)
        # we don't have infos dicts, so we insert some fake ones to make
        # TransitionsMinimal happy
        dummy_infos = np.asarray([{}] * n_samples, dtype='object')
        result = il_types.TransitionsMinimal(infos=dummy_infos, **dict_samples)
        assert len(result) == n_samples
        return result

    def size(self):
        return self.dict_dataset.size()


def load_data(
    pickle_paths: List[str],
    preprocessor_name: str,
    transpose_observations=False,
) -> Tuple[str, il_datasets.Dataset]:
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

    dataset = TransitionsMinimalDataset(dataset_dict)

    return env_name, dataset


@benchmark_ingredient.capture
def load_dataset_magical(magical_demo_dirs, magical_env_prefix,
                         magical_preproc):
    demo_dir = magical_demo_dirs[magical_env_prefix]
    logging.info(
        f"Loading trajectory data for '{magical_env_prefix}' from '{demo_dir}'"
    )
    demo_paths = [
        os.path.join(demo_dir, f) for f in os.listdir(demo_dir)
        if f.endswith('.pkl.gz')
    ]
    if not demo_paths:
        raise IOError(f"Could not find any demo pickle files in '{demo_dir}'")
    gym_env_name_chans_last, dataset = load_data(
        demo_paths,
        preprocessor_name=magical_preproc,
        transpose_observations=True)
    assert gym_env_name_chans_last.startswith(magical_env_prefix)
    return gym_env_name_chans_last, dataset


class SB3EvaluationProtocol(EvaluationProtocol):
    """MAGICAL 'evaluation protocol' for Stable Baselines 3 policies."""

    # TODO: more docs, document __init__ in particular
    def __init__(self, policy, run_id, seed, batch_size, **kwargs):
        super().__init__(**kwargs)
        self._run_id = run_id
        self.policy = policy
        self.seed = seed
        self.batch_size = batch_size

    @property
    def run_id(self):
        """Identifier for this run in the dataframe produced by
        `.do_eval()`."""
        return self._run_id

    def obtain_scores(self, env_name):
        """Collect `self.n_rollouts` scores on environment `env_name`."""
        vec_env_chans_last = make_vec_env(env_name,
                                          n_envs=self.batch_size,
                                          seed=self.seed,
                                          parallel=False)
        rng = np.random.RandomState(self.seed)
        trajectories = il_rollout.generate_trajectories(
            self.policy,
            vec_env_chans_last,
            sample_until=il_rollout.min_episodes(
                self.n_rollouts),
            rng=rng)
        scores = []
        for trajectory in trajectories[:self.n_rollouts]:
            scores.append(trajectory.infos[-1]['eval_score'])
        return scores

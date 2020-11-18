"""Make a repL WebDataset from random rollouts."""
import logging

import sacred
from sacred import Experiment

from il_representations.algos.utils import set_global_seeds
from il_representations.envs.config import (env_cfg_ingredient,
                                            venv_opts_ingredient)

sacred.SETTINGS['CAPTURE_MODE'] = 'sys'  # workaround for sacred issue#740
mkdataset_random_ex = Experiment('mkdataset_random',
                                 ingredients=[env_cfg_ingredient,
                                              venv_opts_ingredient])


@mkdataset_random_ex.config
def default_config():
    # overwrite the default destination
    custom_out_file_path = None

    _ = locals()
    del _


@mkdataset_random_ex.main
def run(seed, env_data, env_cfg, custom_out_file_path):
    set_global_seeds(seed)
    logging.basicConfig(level=logging.INFO)
    # TODO(sam): can implement this once rest of the pipeline works (doesn't
    # really have to be in initial PR)
    raise NotImplementedError("still need to implement this")


if __name__ == '__main__':
    mkdataset_random_ex.run_commandline()

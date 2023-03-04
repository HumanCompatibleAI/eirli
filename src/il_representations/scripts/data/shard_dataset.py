"""Shard an existing webdataset into separate datasets for each trajectory."""
import logging
import os
from typing import Iterator, List, Optional, Tuple

import sacred
from sacred import Experiment

from il_representations.algos.utils import set_global_seeds
from il_representations.data.read_dataset import load_ilr_datasets
from il_representations.data.write_dataset import write_frames

sacred.SETTINGS['CAPTURE_MODE'] = 'no'  # workaround for sacred issue#740
shard_dataset_ex = Experiment('shard_dataset', ingredients=[])


@shard_dataset_ex.config
def default_config():
    # old_data_root and new_data_root must be supplied
    old_data_root = None
    new_data_root = None

    _ = locals()
    del _


def trajectory_iter(frame_iter: Iterator[dict]) -> Iterator[List[dict]]:
    traj: List[dict] = []
    for frame in frame_iter:
        traj.append(frame)
        if frame['dones']:
            yield traj
            traj = []
    if traj:
        yield traj


@shard_dataset_ex.capture
def find_tar_zst_files(old_data_root: str) -> Iterator[Tuple[str, str, str]]:
    odr_abs = os.path.abspath(old_data_root)
    for root, dirs, files in os.walk(odr_abs, followlinks=True):
        for bn in files:
            if bn.endswith('.tar.zst') and not bn.startswith('.'):
                rel_dir = os.path.relpath(root, start=old_data_root)
                file_path = os.path.join(root, bn)
                yield file_path, rel_dir, bn


@shard_dataset_ex.main
def run(seed: int, old_data_root: Optional[str],
        new_data_root: Optional[str]):
    # XXX kind of bad to have this broken code checked in :P
    raise NotImplementedError(
        "this doesn't work yet, going to leave it until later to fix")

    if old_data_root is None or new_data_root is None:
        raise ValueError(
            "must supply old_data_root and new_data_root options to this "
            "Sacred experiment")

    set_global_seeds(seed)
    # python built-in logging
    logging.basicConfig(level=logging.INFO)

    for tar_zst_file, rel_dir, base_name in find_tar_zst_files():
        dataset = load_ilr_datasets([tar_zst_file])
        for traj_num, traj in enumerate(trajectory_iter(dataset)):
            # insert the trajectory number into the filename before the
            # extension (if any), but after rest of the name
            name_pre, name_post = os.path.splitext(base_name)
            new_name = f'{name_pre}-{traj_num:06d}{name_post}'
            out_file_path = os.path.join(new_data_root, rel_dir, new_name)
            if os.path.exists(out_file_path):
                # This is not a multiproc-safe way of checking for collisions,
                # but will hopefully catch coding errors where this script
                # accidentally writes out the same file twice.
                raise FileExistsError(
                    f"tried to overwrite existing path '{out_file_path}'")
            write_frames(out_file_path, dataset.meta, traj)


if __name__ == '__main__':
    # sacred is used only to parse args, and not to write logs, so we don't add
    # any observers
    shard_dataset_ex.run_commandline()

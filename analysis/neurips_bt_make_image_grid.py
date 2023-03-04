#!/usr/bin/env python3
"""This script takes a webdataset, selects `GRID_WIDTH_IN_IMAGES` random images
for each (discrete) action, then arranges them into a grid. Each row in the
grid corresponds to a distinct action, and each column is a randomly chosen
sample for that action."""

import collections
import logging
import os
import pathlib
import pprint
import random
from typing import Iterable, List, Optional, Sequence

from PIL import Image
import gym
import numpy as np
import sacred
from sacred import Experiment
import torch
import torch.nn.functional as F

from il_representations.algos.utils import set_global_seeds
from il_representations.envs import auto
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient)
from il_representations.script_utils import simplify_stacks

sacred.SETTINGS['CAPTURE_MODE'] = 'no'  # workaround for sacred issue#740
make_grid_ex = Experiment(
    'make_grid', ingredients=[env_cfg_ingredient, env_data_ingredient])


@make_grid_ex.config
def default_config():
    grid_width_images = 10
    # images are resized to be squares of size `IMAGE_SIZE_PX*IMAGE_SIZE_PX`
    image_size_px = 134
    # we put a separator between each image of this width (in pixels)
    separator_width_px = 2
    separator_color_u8 = (255, 255, 255)
    # where to write output?
    out_path = None
    dataset_config = {'type': 'demos'}
    _ = locals()
    del _


def choose_frames(frame_count: int, n_frames: int) -> List[int]:
    """Randomly choose an ordered subset of `n_frames` distinct frame indexes
    from the set `{0,1,…,frame_count-1}`. Return them as a list sorted from
    high to low."""
    chosen = np.random.choice(frame_count, size=(n_frames, ), replace=False)
    return sorted(chosen, reverse=True)


def resize_to_square(obs: np.ndarray, frame_size: int) -> np.ndarray:
    assert obs.shape[0] == 3 and obs.shape[1] == obs.shape[2], \
        f"expected square RGB image, CHW format, got shape {obs.shape}"
    tens = torch.from_numpy(obs)
    resized, = F.interpolate(tens[None].float(),
                             size=frame_size,
                             mode='bilinear',
                             align_corners=False).byte()
    return resized.numpy()


def select_frames(webdataset: Iterable[dict], frames_per_action: int,
                  frame_size: int) -> List[List[np.ndarray]]:
    """Select a list of `frames_per_action` frames for each action in the
    dataset. Returns a list of lists of frames. Each inner list in the outer
    list represents a separate row. Each frame in an inner list is a uint8
    ndarray of shape `frame_size*frame_size*n_chans`."""

    # first we do a single pass through the dataset to count the number of
    # frames with each action
    frame_count = collections.defaultdict(lambda: 0)
    for frame in webdataset:
        act = int(frame['acts'])
        frame_count[act] += 1

    print('Frame count:')
    for act, count in sorted(frame_count.items()):
        print(f'  {act}: {count}')

    # next we choose some frame indices for each action, based on the total
    # number of frames for each action
    chosen_frames = {
        act: choose_frames(total_count, frames_per_action)
        for act, total_count in frame_count.items()
    }

    # we can then extract the frames corresponding to the above indices
    frames_seen = collections.defaultdict(lambda: 0)
    returned_frames = collections.defaultdict(lambda: [])
    for frame in webdataset:
        # is this the next frame we want for this action?
        act = int(frame['acts'])
        idx = frames_seen[act]
        frames_seen[act] += 1
        if not (chosen_frames[act] and idx == chosen_frames[act][-1]):
            continue
        chosen_frames[act].pop()

        # here we drop all but the last image in each stack
        obs, = simplify_stacks(frame['obs'][None], keep_only_latest=True)
        # resize image to frame_size*frame_size
        obs_resize = resize_to_square(obs, frame_size)
        # convert to byte image with H*W*C format
        obs_u8 = obs_resize.astype('uint8')
        obs_hwc = np.transpose(obs_u8, (1, 2, 0))
        returned_frames[act].append(obs_hwc)

    # we return a list of lists of frames (we sort the dict by action so that
    # this function is deterministic, conditioned on RNG seed)
    frame_lists = [frames for act, frames in sorted(returned_frames.items())]
    for fl in frame_lists:
        random.shuffle(fl)

    return frame_lists


def make_grid(image_rows: List[List[np.ndarray]], padding: int,
              fill_col: Sequence[int]) -> np.ndarray:
    # figure out how big our output image will be
    n_rows = len(image_rows)
    n_images_per_row = len(image_rows[0])
    image_height, image_width = image_rows[0][0].shape[:2]
    out_width_px = (padding + image_width) * n_images_per_row + padding
    out_height_px = (padding + image_height) * n_rows + padding

    # make an empty image of the right fill colour
    out = np.zeros((out_height_px, out_width_px, 3),
                   dtype=image_rows[0][0].dtype)
    out[:, :] = fill_col

    # now place images in the empty image one at a time
    for row_idx in range(n_rows):
        row_start_h = padding + (padding + image_height) * row_idx
        for col_idx in range(n_images_per_row):
            col_start_w = padding + (padding + image_width) * col_idx
            img = image_rows[row_idx][col_idx]
            # h_slice and w_slice index into the 0th and 1st dimension of `out`
            # array, respectively
            h_slice = slice(row_start_h, row_start_h + image_height)
            w_slice = slice(col_start_w, col_start_w + image_width)
            out[h_slice, w_slice] = img

    return out


@make_grid_ex.main
def run(grid_width_images: int, image_size_px: int, separator_width_px: int,
        separator_color_u8: Sequence[int], out_path: Optional[str],
        dataset_config: dict, seed: int) -> None:
    set_global_seeds(seed)
    logging.getLogger().setLevel(logging.INFO)

    print('Supplied dataset config:')
    pprint.pprint(dataset_config)

    # we only support loading one dataset (hence the [dataset_config] thing)
    (webdataset, ), combined_meta = auto.load_wds_datasets(
        configs=[dataset_config])
    assert isinstance(combined_meta['action_space'], gym.spaces.Discrete)

    print("Collected metadata from loaded dataset:")
    pprint.pprint(combined_meta)

    # colllect some frames
    frame_lists = select_frames(webdataset,
                                frames_per_action=grid_width_images,
                                frame_size=image_size_px)

    # make a grid out of the frames
    grid = make_grid(frame_lists, padding=separator_width_px,
                     fill_col=separator_color_u8)

    # finally, save image
    img = Image.fromarray(grid)
    assert out_path is not None, "out_path argument must be supplied"
    out_path = pathlib.Path(out_path)
    os.makedirs(out_path.parent, exist_ok=True)
    img.save(out_path)

    print(f"Done, written to '{out_path}")


if __name__ == '__main__':
    make_grid_ex.run_commandline()

import torch
import sacred
import os
import numpy as np
from sacred import Experiment
from PIL import Image
from sacred.observers import FileStorageObserver
from stable_baselines3.common.utils import get_device

import il_representations.envs.auto as auto_env
from il_representations.envs.config import benchmark_ingredient
from il_representations.algos.pair_constructors import IdentityPairConstructor

from captum.attr import IntegratedGradients

interp_ex = Experiment('interp', ingredients=[benchmark_ingredient])


@interp_ex.config
def base_config():
    # Network settings
    model_path = ''
    network = None
    network_args = None
    device = None

    # Data settings
    benchmark_name = 'atari'
    imgs = [666, 888]  # index of each image in the dataset (int)
    assert all(isinstance(im, int) for im in imgs), 'imgs list should contain integers only'


@interp_ex.capture
def prepare_network(network, network_args, model_path, device):
    network = network(**network_args)
    network.eval()
    device = get_device("auto" if device is None else device)
    model_state_dict = torch.load(model_path, map_location=device)
    network.load_state_dict(model_state_dict)
    return network


@interp_ex.capture
def process_data(benchmark_name, imgs, device):
    img_list = []
    label_list = []
    data_dict = auto_env.load_dataset(benchmark_name)
    for img_idx in imgs:
        img = data_dict['obs'][img_idx]
        label = data_dict['acts'][img_idx]
        img = torch.FloatTensor(img).to(device).unsqueeze(dim=0)
        img_list.append(img)
        label_list.append(label)
    return img_list, label_list


@interp_ex.main
def run():
    # Load the network and images
    images, labels = process_data()
    network = prepare_network()

    for img in images:
        # Get policy prediction
        output = network(img)




if __name__ == '__main__':
    sacred.SETTINGS['CAPTURE_MODE'] = 'sys'
    interp_ex.observers.append(FileStorageObserver('runs/interpret_runs'))
    interp_ex.run_commandline()

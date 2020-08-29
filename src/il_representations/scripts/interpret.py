import torch
import sacred
import os
import numpy as np
import matplotlib.pyplot as plt

from sacred import Experiment
from PIL import Image
from sacred.observers import FileStorageObserver
from stable_baselines3.common.utils import get_device
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from captum.attr import IntegratedGradients, Saliency
from captum.attr import visualization as viz

import il_representations.envs.auto as auto_env
from il_representations.envs.config import benchmark_ingredient
from il_representations.algos.pair_constructors import IdentityPairConstructor
from il_representations.algos.model3l import NatureCnn

interp_ex = Experiment('interp', ingredients=[benchmark_ingredient])


@interp_ex.config
def base_config():
    # Network settings
    model_path = os.path.join(os.getcwd(), 'runs/downloads/nature_moco200_3l_seed555.pth')
    network = NatureCnn
    network_args = {'action_size': 4, 'obs_shape': [84, 84, 4], 'z_dim': 64, 'pretrained': True}
    device = None

    # Data settings
    benchmark_name = 'atari'
    imgs = [666]  # index of each image in the dataset (int)
    assert all(isinstance(im, int) for im in imgs), 'imgs list should contain integers only'
    show_imgs = False

    # Interpret settings
    saliency = True


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
        img /= 225
        img_list.append(img)
        label_list.append(label)
    return img_list, label_list


def save_img(img, save_name, save_dir, show=True):
    savefig_kwargs = {}
    if isinstance(img, torch.Tensor):
        if (img.shape[0]) == 4:
            img = img.permute(1, 2, 0)
    else:
        img = img[50:1150, 100:1100, :]
        plt.axis('off')
        savefig_kwargs = {'bbox_inches': 'tight', 'dpi': 150, 'pad_inches': 0}
    plt.imshow(img)
    if show:
        plt.show()
    plt.savefig(f'{save_dir}/{save_name}.png', **savefig_kwargs)
    plt.close()


def figure_2_numpy(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    image_shape = fig.get_size_inches() * fig.dpi
    channel = np.array([3])
    image_shape = np.concatenate((image_shape, channel)).astype(int)
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(image_shape)
    return image


def attribute_image_features(network, algorithm, input, label, **kwargs):
    network.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=label,
                                              **kwargs)

    return tensor_attributions


def saliency_(net, input, label):
    saliency = Saliency(net)
    grads = saliency.attribute(input, target=int(label))
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    return grads


@interp_ex.main
def run(show_imgs, saliency):
    # Load the network and images
    images, labels = process_data()
    network = prepare_network()

    for img, label in zip(images, labels):
        # Get policy prediction
        output = network(img)

        log_dir = interp_ex.observers[0].dir
        original_img = img[0].permute(1, 2, 0).numpy()
        save_img(img[0], 'original_image', log_dir, show=False)

        if saliency:
            saliency_grads = saliency_(network, img, label)
            saliency_viz = viz.visualize_image_attr(saliency_grads, original_img, method="blended_heat_map",
                                                    sign="absolute_value",
                                                    show_colorbar=True,
                                                    title="Overlayed Gradient Magnitudes")
            save_img(figure_2_numpy(saliency_viz[0]), 'saliency', log_dir, show=show_imgs)






if __name__ == '__main__':
    sacred.SETTINGS['CAPTURE_MODE'] = 'sys'
    interp_ex.observers.append(FileStorageObserver('runs/interpret_runs'))
    interp_ex.run_commandline()

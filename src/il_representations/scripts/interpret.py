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

from captum.attr import IntegratedGradients, Saliency, DeepLift, LayerConductance, LayerGradCam, LayerActivation, \
    LayerAttribution
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
    # Primary Attribution: Evaluates contribution of each input feature to the output of a model.
    saliency = False
    integrated_gradient = False
    deep_lift = False

    # Layer Attribution: Evaluates contribution of each neuron in a given layer to the output of the model.
    layer_conductance = False
    layer_gradcam = True


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
            img = img.permute(1, 2, 0).detach().numpy()
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


def attribute_image_features(network, algorithm, image, label, **kwargs):
    network.zero_grad()
    tensor_attributions = algorithm.attribute(image,
                                              target=label,
                                              **kwargs)
    return tensor_attributions


def saliency_(net, image, label, original_img, log_dir, show_imgs):
    saliency = Saliency(net)
    grads = saliency.attribute(image, target=label)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    saliency_viz = viz.visualize_image_attr(grads, original_img, method="blended_heat_map",
                                            sign="absolute_value",
                                            show_colorbar=True,
                                            title="Overlayed Gradient Magnitudes")
    save_img(figure_2_numpy(saliency_viz[0]), 'saliency', log_dir, show=show_imgs)
    return grads


def integrated_gradient_(net, image, label, original_img, log_dir, show_imgs):
    ig = IntegratedGradients(net)
    attr_ig, delta = attribute_image_features(net, ig, image, label, baselines=image * 0, return_convergence_delta=True)
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    ig_viz = viz.visualize_image_attr(ig, original_img, method="blended_heat_map", sign="all",
                                      show_colorbar=True,
                                      title="Overlayed Integrated Gradients")
    save_img(figure_2_numpy(ig_viz[0]), 'integrated_gradients', log_dir, show=show_imgs)
    return attr_ig


def deep_lift_(net, image, label, original_img, log_dir, show_imgs):
    dl = DeepLift(net)
    attr_dl = attribute_image_features(net, dl, image, label, baselines=image * 0)
    attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    dl_viz = viz.visualize_image_attr(dl, original_img, method="blended_heat_map", sign="all",
                                      show_colorbar=True,
                                      title="Overlayed DeepLift")
    save_img(figure_2_numpy(dl_viz[0]), 'deep_lift', log_dir, show=show_imgs)
    return attr_dl


def layer_conductance_(net, layer, image, label):
    layer_cond = LayerConductance(net, layer)
    attribution = layer_cond.attribute(image, target=label)  # Shape: torch.Size([1, 32, 20, 20])
    return attribution


def layer_gradcam_(net, layer, image, label, original_img, log_dir, show_imgs):
    lgc = LayerGradCam(net, layer)
    gc_attr = lgc.attribute(image, target=label)
    upsampled_gc_attr = LayerAttribution.interpolate(gc_attr, image.shape[2:])
    lg_viz = viz.visualize_image_attr_multiple(upsampled_gc_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
                                               original_image=original_img,
                                               signs=["all", "positive", "negative"],
                                               methods=["original_image", "blended_heat_map", "blended_heat_map"])
    save_img(figure_2_numpy(lg_viz[0]), 'layer_gradcam', log_dir, show=show_imgs)

@interp_ex.main
def run(show_imgs, saliency, integrated_gradient, deep_lift, layer_conductance, layer_gradcam):
    # Load the network and images
    images, labels = process_data()
    network = prepare_network()

    for img, label in zip(images, labels):
        # Get policy prediction
        output = network(img)
        original_img = img[0].permute(1, 2, 0).numpy()
        label = int(label)
        img.requires_grad = True

        log_dir = interp_ex.observers[0].dir
        save_img(img[0], 'original_image', log_dir, show=False)

        if saliency:
            saliency_(network, img, label, original_img, log_dir, show_imgs)

        if integrated_gradient:
            integrated_gradient_(network, img.contiguous(), label, original_img, log_dir, show_imgs)

        if deep_lift:
            deep_lift_(network, img, label, original_img, log_dir, show_imgs)

        if layer_conductance:  # TODO: How to interpret this?
            lc = layer_conductance_(network, network.encoder[0], img, label)

        if layer_gradcam:
            layer_gradcam_(network, network.encoder[4], img, label, original_img, log_dir, show_imgs)


if __name__ == '__main__':
    sacred.SETTINGS['CAPTURE_MODE'] = 'sys'
    interp_ex.observers.append(FileStorageObserver('runs/interpret_runs'))
    interp_ex.run_commandline()

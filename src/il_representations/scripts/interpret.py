import torch
import sacred
import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sacred import Experiment
from PIL import Image
from sacred.observers import FileStorageObserver
from stable_baselines3.common.utils import get_device
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from captum.attr import IntegratedGradients, Saliency, DeepLift, LayerConductance, LayerGradCam, LayerActivation, \
    LayerAttribution, LayerIntegratedGradients, LayerGradientXActivation
from captum.attr import visualization as viz

import il_representations.envs.auto as auto_env
from il_representations.envs.config import benchmark_ingredient
from il_representations.algos.pair_constructors import IdentityPairConstructor
from il_representations.algos.model3l import NatureCnn

interp_ex = Experiment('interp', ingredients=[benchmark_ingredient])


@interp_ex.config
def base_config():
    # Network settings
    model_path = os.path.join(os.getcwd(), 'runs/downloads/moco_3l_seed888.pth')
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
    layer_gradcam = False
    layer_activation = False
    layer_gradxact = True


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
    dl_viz = viz.visualize_image_attr(attr_dl, original_img, method="blended_heat_map", sign="all",
                                      show_colorbar=True,
                                      title="Overlayed DeepLift")
    save_img(figure_2_numpy(dl_viz[0]), 'deep_lift', log_dir, show=show_imgs)
    return attr_dl


def layer_conductance_(net, layer, image, label, log_dir, show_imgs=True, columns=10):
    layer_cond = LayerConductance(net, layer)
    attribution = layer_cond.attribute(image, n_steps=100, attribute_to_layer_input=True, target=label)
    attribution = attribution[0]
    if len(attribution.shape) == 2:  # Attribution has one dimension only - usually seen in linear layers.
        l_weight = layer.weight
        plot_linear_layer_attributions(attribution, l_weight, log_dir)
    elif len(attribution.shape) == 4:  # Attribution has three dimensions - usually seen in convolution layers.
        attribution = attribution[0]
        num_channels = attribution.shape[0]
        show_img_grid(attribution, math.ceil(num_channels/columns), columns, log_dir, 'layer_conductance',
                      'layer_conductance', show_imgs)

    return attribution


def layer_gradcam_(net, layer, image, label, original_img, log_dir, show_imgs):
    lgc = LayerGradCam(net, layer)
    gc_attr = lgc.attribute(image, target=label)
    upsampled_gc_attr = LayerAttribution.interpolate(gc_attr, image.shape[2:])  # Shape [1, 1, 84, 84]
    lg_viz_pos = viz.visualize_image_attr(upsampled_gc_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
                                      original_img, method="blended_heat_map", sign="positive",
                                      show_colorbar=True,
                                      title="Overlayed DeepLift (Positive)")
    lg_viz_neg = viz.visualize_image_attr(upsampled_gc_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
                                      original_img, method="blended_heat_map", sign="negative",
                                      show_colorbar=True,
                                      title="Overlayed DeepLift (Negative)")
    save_img(figure_2_numpy(lg_viz_pos[0]), 'layer_gradcam_pos', log_dir, show=show_imgs)
    save_img(figure_2_numpy(lg_viz_neg[0]), 'layer_gradcam_neg', log_dir, show=show_imgs)


def layer_act_(net, layer, algo, algo_name, image, label, log_dir, attr_kwargs=None, show_imgs=True, columns=10):
    layer_a = algo(net, layer)
    a_attr = layer_a.attribute(image, **attr_kwargs)[0]
    if len(a_attr.shape) == 1:  # Attribution has one dimension only - usually seen in linear layers.
        l_weight = layer.weight
        plot_linear_layer_attributions(a_attr, l_weight, log_dir)
    elif len(a_attr.shape) == 3:  # Attribution has three dimensions - usually seen in convolution layers.
        num_channels = a_attr.shape[0]
        layer_info = str(layer)
        img_title = f'{algo_name} of {layer_info}'
        show_img_grid(a_attr, math.ceil(num_channels/columns), columns, log_dir, algo_name,
                      img_title, show_imgs)


def show_img_grid(imgs, rows, columns, save_dir, save_name, img_title, show):
    fig = plt.figure()
    plt.title(img_title, verticalalignment='baseline')
    plt.axis('off')
    for i in range(len(imgs)):
        img = imgs[i]
        if isinstance(img, torch.Tensor):
            img = img.detach().numpy()
        img = cv2.resize(img, dsize=(40, 40), interpolation=cv2.INTER_CUBIC)
        fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        plt.imshow(img)
    plt.savefig(f'{save_dir}/{save_name}.png', dpi=400)
    if show:
        plt.show()
    plt.close(fig)


def plot_linear_layer_attributions(lc_attr_test, layer_weight, save_dir):
    plt.figure(figsize=(15, 8))

    x_axis_data = np.arange(lc_attr_test.shape[1])

    y_axis_lc_attr_test = lc_attr_test.mean(0).detach().numpy()
    y_axis_lc_attr_test = y_axis_lc_attr_test / np.linalg.norm(y_axis_lc_attr_test, ord=1)

    y_axis_lin_weight = layer_weight[0].detach().numpy()
    y_axis_lin_weight = y_axis_lin_weight / np.linalg.norm(y_axis_lin_weight, ord=1)

    width = 0.25
    legends = ['Attributions', 'Weights']
    x_axis_labels = ['Neuron {}'.format(i) for i in range(len(y_axis_lin_weight))]

    ax = plt.subplot()
    ax.set_title('Aggregated neuron importances and learned weights in the indicated linear layer of the model')

    ax.bar(x_axis_data + width, y_axis_lc_attr_test, width, align='center', alpha=0.5, color='red')
    ax.bar(x_axis_data + 2 * width, y_axis_lin_weight, width, align='center', alpha=0.5, color='green')
    plt.legend(legends, loc=2, prop={'size': 20})
    ax.autoscale_view()
    plt.tight_layout()

    ax.set_xticks(x_axis_data + 0.5)
    ax.set_xticklabels(x_axis_labels)

    plt.show()

    plt.savefig(f'{save_dir}/layer_conductance.png')


@interp_ex.main
def run(show_imgs, saliency, integrated_gradient, deep_lift, layer_conductance, layer_gradcam, layer_gradxact,
        layer_activation):
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

        if layer_conductance:
            # layer_conductance_(network, network.mlp_pi_decoder[0], img, label, log_dir)
            layer_conductance_(network, network.encoder[4], img, label, log_dir)

        if layer_gradcam:
            # GradCAM is usually applied to the last convolutional layer in the network.
            layer_gradcam_(network, network.encoder[4], img, label, original_img, log_dir, show_imgs,)

        if layer_gradxact:
            layer_act_(network, network.encoder[0], LayerGradientXActivation, 'layer_GradXActivation',
                       img, label, log_dir, attr_kwargs={'target': label})

        if layer_activation:
            layer_act_(network, network.encoder[0], LayerActivation, 'layer_Activation',
                       img, label, log_dir, attr_kwargs={})


if __name__ == '__main__':
    sacred.SETTINGS['CAPTURE_MODE'] = 'sys'
    interp_ex.observers.append(FileStorageObserver('runs/interpret_runs'))
    interp_ex.run_commandline()

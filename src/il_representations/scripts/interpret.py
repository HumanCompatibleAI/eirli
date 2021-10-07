import torch
import sacred
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from pathlib import Path

from gym import spaces
from sacred import Experiment
from sacred.observers import FileStorageObserver
from stable_baselines3.common.utils import get_device
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from captum.attr import IntegratedGradients, Saliency, DeepLift
from captum.attr import visualization as viz
from stable_baselines3.common.preprocessing import preprocess_obs

from il_representations.scripts.il_train import make_policy
from il_representations.utils import TensorFrameWriter
import il_representations.envs.auto as auto_env
from il_representations.data.read_dataset import InterleavedDataset
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)

sacred.SETTINGS['CAPTURE_MODE'] = 'sys'
interp_ex = Experiment('interp', ingredients=[
                                            env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient
])

@interp_ex.config
def base_config():
    # Network setting
    encoder_path = ''

    # Data settings
    device = get_device("auto")
    save_video = False
    length = 2
    save_image = True  # If true, {length} number of images will be saved.
    dataset_configs = [{'type': 'demos'}]

    # interp_algos = [
    #     # Primary Attribution: Evaluates contribution of each input feature to the output of a model.
    #     'saliency',
    #     'integrated_gradient',
    #     'deep_lift',
    # ]
    chosen_algo = 'integrated_gradient'


class InterpAlgos:
    def __init__(self):
        self._algos = {}

    def get(self, name):
        return self._algos[name]

    def register(self, f):
        self._algos[f.__name__] = f


interp_algos = InterpAlgos()


@interp_ex.capture
def save_img(save_name, save_dir, image=None):
    if image is not None:
        plt.imshow(image)
    plt.axis('off')
    plt.savefig(f'{save_dir}/{save_name}.png', bbox_inches='tight')
    plt.close()


def figure_2_tensor(fig):
    """
    Captum's visualize_image_attr method returns matplotlib.pyplot.figure object. To process and plot figures,
    we need to convert them to torch tensors first.
    """
    canvas = FigureCanvas(fig)
    canvas.draw()
    image_shape = fig.get_size_inches() * fig.dpi
    channel = np.array([3])
    image_shape = np.concatenate((image_shape, channel)).astype(int)
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(image_shape)
    image = torch.Tensor(image.copy())
    return image


def attribute_image_features(network, algorithm, image, label, **kwargs):
    network.zero_grad()
    tensor_attributions = algorithm.attribute(image,
                                              target=label,
                                              **kwargs)
    return tensor_attributions


@interp_algos.register
def saliency(net, tensor_image, label):
    saliency = Saliency(net)
    grads = saliency.attribute(tensor_image, target=label)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    saliency_viz = viz.visualize_image_attr(grads,
                                            tensor_image[0].permute(1, 2, 0).detach().cpu().numpy(),
                                            method="blended_heat_map",
                                            sign="absolute_value",
                                            show_colorbar=False)
    return figure_2_tensor(saliency_viz[0])


@interp_algos.register
def integrated_gradient(net, tensor_image, label):
    ig = IntegratedGradients(net)
    attr_ig, delta = attribute_image_features(net, ig, tensor_image, label,
                                              baselines=tensor_image * 0,
                                              return_convergence_delta=True, )
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    ig_viz = viz.visualize_image_attr(attr_ig,
                                      tensor_image[0].permute(1, 2, 0).detach().cpu().numpy(),
                                      method="blended_heat_map",
                                      sign="all",
                                      show_colorbar=True,
                                      title="Overlayed Integrated Gradients")
    return figure_2_tensor(ig_viz[0])


@interp_algos.register
def deep_lift(net, tensor_image, label):
    dl = DeepLift(net)
    attr_dl = attribute_image_features(net, dl, tensor_image, label,
                                       baselines=tensor_image * 0,)
    attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    dl_viz = viz.visualize_image_attr(attr_dl,
                                      tensor_image[0].permute(1, 2, 0).detach().cpu().numpy(),
                                      method="blended_heat_map",
                                      sign="all",
                                      show_colorbar=True,
                                      title="Overlayed DeepLift")
    return figure_2_tensor(dl_viz[0])


@interp_ex.capture
def prepare_network(combined_meta, encoder_path, device):
    policy = make_policy(encoder_path=encoder_path,
                         policy_continue_path=None,
                         observation_space=combined_meta['observation_space'],
                         action_space=combined_meta['action_space'],
                         ortho_init=False,
                         log_std_init=0.0,
                         postproc_arch=(),
                         freeze_pol_encoder=True,
                         encoder_kwargs={},
                         algo='bc',
                         lr_schedule=None)
    # TODO: make this from action space
    return LoadedPolicy(policy, use_first_action=False).to(device)


class LoadedPolicy(nn.Module):
    """A wrapper class for loaded policy so it has the same interface as required
    by Captum. Specifically, most of our saved policies are ActorCriticCnnPolicy,
    and its forward() function outputs a tuple of (actions, values, log_prob).
    In Captum we only need its actions.
    """
    def __init__(self, policy, use_first_action):
        super().__init__()
        self.policy = policy
        self.use_first_action = use_first_action

    def forward(self, x):
        # TODO: make this compatible with both DMC and procgen.
        action = self.policy.action_net(self.policy.features_extractor(x/255.))
        if self.use_first_action:
            # Sometimes the action has several components, like in DMC tasks.
            # We use the first component by default.
            action = action[:, 0].unsqueeze(dim=0)
        return action


@interp_ex.capture
def get_data(env_cfg):
    dataset = auto_env.load_dict_dataset(benchmark_name=env_cfg['benchmark_name'],
                                         task_name=env_cfg['task_name'])
    return dataset['obs'], dataset['acts']

@interp_ex.main
def run(chosen_algo, save_video, filename, dataset_configs, save_image,
        save_original_image, device, length):
    # setup environment & dataset
    datasets, combined_meta = auto_env.load_wds_datasets(configs=dataset_configs)
    observation_space = combined_meta['observation_space']

    network = prepare_network(combined_meta)
    images, labels = get_data()

    log_dir = interp_ex.observers[0].dir
    filename = chosen_algo

    if save_video:
        video_writer = TensorFrameWriter(f"{log_dir}/{filename}.mp4",
                                         'RGB',
                                         fps=8,
                                         adjust_axis=False,
                                         make_grid=False)

    for itr, (image, label) in enumerate(zip(images, labels)):
        if itr > length:
            break
        tensor_image = torch.Tensor(image).unsqueeze(dim=0).to(device)
        tensor_label = int(label)

        # For continuous space actions, we don't need to provide a label since
        # Captum assumes the provided label stands for "class_num" and is an
        # integer.
        action_space = combined_meta['action_space']
        if not isinstance(action_space, spaces.Discrete):
            tensor_label = None
        interp_algo_func = interp_algos.get(chosen_algo)

        interpreted_img = interp_algo_func(network, tensor_image, tensor_label)

        if save_video:
            video_writer.add_tensor(preprocess_obs(interpreted_img,
                                                   observation_space,
                                                   normalize_images=True))
        if save_image:
            Path(f'{log_dir}/images').mkdir(parents=True, exist_ok=True)
            save_img(save_name=f'{filename}_{itr}',
                     save_dir=f'{log_dir}/images')

            if save_original_image:
                channel_per_frame = 3 if combined_meta['color_space'] == \
                                    'RGB' else 1

                # Most loaded images are frame stacked by 3 or 4 frames. Here
                # we are visually "stacking" them by taking 1/3 or 1/4 pixels
                # from each of them, saving images that are exactly seen
                # by a network. For example saved images, see Figure 4 of the
                # EIRLI paper.
                processed_image = []
                i = 0
                while i < len(image):
                    processed_image.append(1.0/float(channel_per_frame) * \
                        image[i:i+channel_per_frame].astype(float))
                    i += channel_per_frame
                processed_image = np.sum(processed_image, axis=0).astype(int)
                # Move channel to the last axis for saving images.
                processed_image = np.transpose(processed_image, (1, 2, 0))
                save_img(save_name=f'{filename}_{itr}_original',
                        save_dir=f'{log_dir}/images',
                        image=processed_image)
        plt.close('all')

    if save_video:
        video_writer.close()


if __name__ == '__main__':
    sacred.SETTINGS['CAPTURE_MODE'] = 'no'
    interp_ex.observers.append(FileStorageObserver('runs/interpret_runs'))
    interp_ex.run_commandline()

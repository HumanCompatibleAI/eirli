from algos import *
from gym.spaces import Discrete, Box
from sacred import Experiment
from sacred.observers import FileStorageObserver
from algos.utils import gaussian_blur

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18


class MockGymEnv(object):
    """A mock Gym env for a supervised learning dataset pretending to be an RL
    task. Action space is set to Discrete(1), observation space corresponds to
    the original supervised learning task.
    """
    def __init__(self, obs_space):
        self.observation_space = obs_space
        self.action_space = Discrete(1)

    def seed(self, seed):
        pass

    def close(self):
        pass


def transform_to_rl(dataset):
    """Transforms the input supervised learning dataset into an "RL dataset", by
    adding dummy 'actions' (always 0) and 'dones' (always False), and pretending
    that everything is from the same 'trajectory'.
    """
    states = [img for img, label in dataset]
    data_dict = {
        'states': states,
        'actions': [0.0] * len(states),
        'dones': [False] * len(states),
    }
    return data_dict


class LinearHead(nn.Module):
    def __init__(self, encoder, output_dim):
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim
        self.layer = nn.Linear(encoder.representation_dim, output_dim)

    def forward(self, x):
        encoding = self.encoder.encode_context(x, None).loc.detach()
        return self.layer(encoding)


def train_classifier(classifier, dataset, num_epochs, device):
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(classifier.layer.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print('[Epoch %d, Batch %3d] Average loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0


def evaluate_classifier(classifier, dataset, device):
    testloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy: %d %%' % (100 * correct / total))


def representation_learning(algo, trainset, device, log_dir, config):
    print('Creating model for representation learning')

    if isinstance(algo, str):
        algo = globals()[algo]
    assert issubclass(algo, RepresentationLearner)

    env = MockGymEnv(Box(low=-1.0, high=1.0, shape=(32, 32, 3), dtype=np.float32))
    rep_learning_augmentations = [
        transforms.Lambda(torch.tensor),
        transforms.ToPILImage(),
        transforms.Pad(4),
        transforms.RandomCrop(16),
        transforms.Pad(8),
        # SimCLR doesn't use blur for CIFAR-10
    ]
    # Note that the resnet18 model used here has an architecture meant for
    # ImageNet, not CIFAR-10. The SimCLR implementation uses a version
    # specialized for CIFAR, see https://github.com/google-research/simclr/blob/37ad4e01fb22e3e6c7c4753bd51a1e481c2d992e/resnet.py#L531
    model = algo(
        env, log_dir=log_dir, batch_size=config['rep_batch_size'], representation_dim=1000, device=device, shuffle_batches=True,
        encoder_kwargs={'architecture_module_cls': lambda *args: resnet18()},
        augmenter_kwargs={'augmentations': rep_learning_augmentations},
        optimizer_kwargs={'lr': 1e-3, 'weight_decay': 1e-4},
        loss_calculator_kwargs={'temp': 0.5},
    )

    print('Train representation learner')
    rep_learning_data = transform_to_rl(trainset)
    model.learn(rep_learning_data, config['pretrain_epochs'])
    env.close()
    return model


cifar_ex = Experiment('cifar')


@cifar_ex.config
def default_config():
    seed = 1
    algo = SimCLR
    data_dir = 'cifar10/'
    pretrain_epochs = 1000
    finetune_epochs = 100
    rep_batch_size = 512
    _ = locals()
    del _


@cifar_ex.main
def run(seed, algo, data_dir, pretrain_epochs, finetune_epochs, rep_batch_size, _config):
    # TODO fix this hacky nonsense
    log_dir = os.path.join(cifar_ex.observers[0].dir, 'training_logs')
    os.mkdir(log_dir)
    ## TODO allow passing in of kwargs here
    #trainloader = torch.utils.data.DataLoader(
    #    trainset, batch_size=opt.batch_size_train, shuffle=True, num_workers=2)

    # Load in data
    os.makedirs(data_dir, exist_ok=True)
    transformations = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(transformations)
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = representation_learning(algo, trainset, device, log_dir, _config)

    print('Train linear head')
    classifier = LinearHead(model.encoder, 10).to(device)
    train_classifier(classifier, trainset, num_epochs=finetune_epochs, device=device)

    print('Evaluate accuracy on test set')
    evaluate_classifier(classifier, testset, device=device)


if __name__ == '__main__':
    cifar_ex.observers.append(FileStorageObserver('cifar_runs'))
    cifar_ex.run_commandline()

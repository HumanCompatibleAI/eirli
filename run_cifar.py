from algos import *
from gym.spaces import Discrete, Box
from sacred import Experiment
from sacred.observers import FileStorageObserver
from algos.utils import gaussian_blur

import numpy as np
import os
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
from algos.utils import LinearWarmupCosine


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


def train_classifier(classifier, data_dir, num_epochs, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(32, interpolation=PIL.Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        # No color jitter or grayscale for finetuning
        # SimCLR doesn't use blur for CIFAR-10
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(classifier.layer.parameters(), lr=0.2, momentum=0.9, weight_decay=0.0, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs} with lr {optimizer.param_groups[0]['lr']}")
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

        scheduler.step()


def evaluate_classifier(classifier, data_dir, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
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


def representation_learning(algo, data_dir, num_epochs, device, log_dir):
    print('Creating model for representation learning')

    if isinstance(algo, str):
        algo = globals()[algo]
    assert issubclass(algo, RepresentationLearner)

    rep_learning_augmentations = [
        transforms.Lambda(torch.tensor),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(32, interpolation=PIL.Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # SimCLR doesn't use blur for CIFAR-10
    ]
    env = MockGymEnv(Box(low=0.0, high=1.0, shape=(3, 32, 32), dtype=np.float32))
    # Note that the resnet18 model used here has an architecture meant for
    # ImageNet, not CIFAR-10. The SimCLR implementation uses a version
    # specialized for CIFAR, see https://github.com/google-research/simclr/blob/37ad4e01fb22e3e6c7c4753bd51a1e481c2d992e/resnet.py#L531
    # It seems that SimCLR does not include the final fully connected layer for ResNets, so we set it to the identity.
    resnet_without_fc = resnet18()
    resnet_without_fc.fc = torch.nn.Identity()
    model = algo(
        env, log_dir=log_dir, batch_size=512, representation_dim=512, projection_dim=128,
        device=device, normalize=False, shuffle_batches=True,
        encoder_kwargs={'architecture_module_cls': lambda *args: resnet_without_fc},
        augmenter_kwargs={'augmentations': rep_learning_augmentations},
        optimizer_kwargs={'lr': 2.0, 'weight_decay': 1e-4},
        scheduler=LinearWarmupCosine,
        scheduler_kwargs={'warmup_epoch': 10, 'T_max': num_epochs},
        loss_calculator_kwargs={'temp': 0.5},
    )

    print('Train representation learner')
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    rep_learning_data = transform_to_rl(trainset)
    model.learn(rep_learning_data, num_epochs)
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
    _ = locals()
    del _


@cifar_ex.main
def run(seed, algo, data_dir, pretrain_epochs, finetune_epochs, _config):
    # TODO fix this hacky nonsense
    log_dir = os.path.join(cifar_ex.observers[0].dir, 'training_logs')
    os.mkdir(log_dir)
    os.makedirs(data_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = representation_learning(algo, data_dir, pretrain_epochs, device, log_dir)

    print('Train linear head')
    classifier = LinearHead(model.encoder, 10).to(device)
    train_classifier(classifier, data_dir, num_epochs=finetune_epochs, device=device)

    print('Evaluate accuracy on test set')
    evaluate_classifier(classifier, data_dir, device=device)


if __name__ == '__main__':
    cifar_ex.observers.append(FileStorageObserver('cifar_runs'))
    cifar_ex.run_commandline()

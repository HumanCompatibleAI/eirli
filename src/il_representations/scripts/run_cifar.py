from gym.spaces import Discrete, Box
from sacred import Experiment
from sacred.observers import FileStorageObserver
from il_representations import algos
from il_representations.algos.optimizers import LARS
from il_representations.algos.utils import LinearWarmupCosine
from imitation.augment.color import ColorSpace
from math import ceil

import numpy as np
import os
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet50


class MockGymEnv(object):
    """A mock Gym env for a supervised learning dataset pretending to be an RL
    task. Action space is set to Discrete(1), observation space corresponds to
    the original supervised learning task.
    """
    def __init__(self, obs_space):
        self.observation_space = obs_space
        self.action_space = Discrete(1)
        self.color_space = ColorSpace.RGB

    def seed(self, seed):
        pass

    def close(self):
        pass


def transform_to_rl(dataset):
    """Transforms the input supervised learning dataset into an "RL dataset", by
    adding dummy 'actions' (always 0) and 'dones' (always False), and pretending
    that everything is from the same 'trajectory'.
    """
    obs = [img for img, label in dataset]
    data_dict = {
        'obs': obs,
        'acts': [0.0] * len(obs),
        'dones': [False] * len(obs),
    }
    return data_dict


class LinearHead(nn.Module):
    def __init__(self, encoder, encoder_dim, output_dim):
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim
        self.layer = nn.Linear(encoder_dim, output_dim)

    def forward(self, x):
        encoding = self.encoder.encode_context(x, None).loc.detach()
        return self.layer(encoding)


def train_classifier(classifier, data_dir, num_epochs, device):
    transform = transforms.Compose([
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


def representation_learning(algo, data_dir, device, log_dir, config):
    print('Train representation learner')
    if isinstance(algo, str):
        algo = getattr(algos, algo)
    assert issubclass(algo, algos.RepresentationLearner)

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
    augmenter_kwargs = {
        "augmenter_spec": "translate,flip_lr,color_jitter_ex,gray",
        "color_space": env.color_space,

        # (Cynthia) Here I'm using augmenter_func because I want our settings
        # to be as close to SimCLR as possible
        "augmenter_func": rep_learning_augmentations
    }
    optimizer_kwargs = {
        "lr": 3e-4
    }

    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    rep_learning_data = transform_to_rl(trainset)
    num_examples = len(rep_learning_data)
    num_epochs = config['pretrain_epochs']
    batch_size = config['pretrain_batch_size']
    num_steps = num_epochs * int(ceil(num_examples / batch_size))

    # Modify resnet according to SimCLR paper Appendix B.9
    simclr_resnet = resnet50()
    simclr_resnet.fc = torch.nn.Identity()
    simclr_resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
    simclr_resnet.maxpool = torch.nn.Identity()

    model = algo(
        observation_space=env.observation_space,
        action_space=env.action_space,
        log_dir=log_dir,
        batch_size=batch_size,
        representation_dim=config['representation_dim'],
        projection_dim=config['projection_dim'],
        device=device,
        normalize=False,
        shuffle_batches=True,
        color_space=ColorSpace.RGB,
        save_interval=config['pretrain_save_interval'],
        encoder_kwargs={'obs_encoder_cls': lambda *args: simclr_resnet},
        augmenter_kwargs=augmenter_kwargs,
        optimizer=torch.optim.Adam,
        optimizer_kwargs=optimizer_kwargs,
        scheduler=LinearWarmupCosine,
        scheduler_kwargs={'warmup_epoch': 10, 'total_epochs': num_epochs},
        loss_calculator_kwargs={'temp': config['pretrain_temperature']},
    )

    # TODO: Check batches per epoch
    model.learn(rep_learning_data, 1000, num_epochs)
    env.close()
    return model


cifar_ex = Experiment('cifar')


@cifar_ex.config
def default_config():
    seed = 1
    algo = 'SimCLR'
    data_dir = 'cifar10/'
    pretrain_epochs = 1000
    finetune_epochs = 100
    representation_dim = 512
    projection_dim = 128
    pretrain_lr = 3e-4
    pretrain_weight_decay = 1e-4
    pretrain_momentum = 0.9
    pretrain_batch_size = 512
    pretrain_save_interval = 100
    pretrain_temperature = 0.5
    _ = locals()
    del _


@cifar_ex.main
def run(seed, algo, data_dir, pretrain_epochs, finetune_epochs, representation_dim, _config):
    # TODO fix this hacky nonsense
    log_dir = os.path.join(cifar_ex.observers[0].dir, 'training_logs')
    os.mkdir(log_dir)
    os.makedirs(data_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = representation_learning(algo, data_dir, device, log_dir, _config)

    print('Train linear head')
    classifier = LinearHead(model.encoder, representation_dim, output_dim=10).to(device)
    train_classifier(classifier, data_dir, num_epochs=finetune_epochs, device=device)

    print('Evaluate accuracy on test set')
    evaluate_classifier(classifier, data_dir, device=device)


if __name__ == '__main__':
    cifar_ex.observers.append(FileStorageObserver('cifar_runs'))
    cifar_ex.run_commandline()

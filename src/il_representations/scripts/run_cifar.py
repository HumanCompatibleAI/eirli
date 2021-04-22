import numpy as np
import os
import PIL
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet50
from torchvision.datasets import CIFAR10
from PIL import Image

from tqdm import tqdm
from math import ceil
import time
from sacred import Experiment
from sacred.observers import FileStorageObserver
from il_representations import algos
from il_representations.algos.utils import LinearWarmupCosine
from il_representations.envs.auto import load_wds_datasets
from il_representations.envs.config import (env_cfg_ingredient,
                                            env_data_ingredient,
                                            venv_opts_ingredient)


cifar_ex = Experiment('cifar', ingredients=[
                                    env_cfg_ingredient, env_data_ingredient,
                                    venv_opts_ingredient
                                ])


class LinearHead(nn.Module):
    def __init__(self, encoder, encoder_dim, output_dim):
        super().__init__()
        self.encoder = encoder
        self.encoder.fc = nn.Linear(2048, output_dim)

    def forward(self, x):
        return self.encoder(x)


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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(classifier.encoder.parameters(), lr=3e-4)
    # optimizer = optim.Adam(classifier.encoder.fc.parameters(), lr=3e-4, momentum=0.9, weight_decay=0.0, nesterov=True)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    progress_dict = {'loss': [], 'train_acc': [], 'test_acc': []}

    start_time = time.time()

    for epoch in range(num_epochs):
        loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()

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
            train_acc_meter.update(accuracy(outputs, labels)[0].item())
            loss_meter.update(loss.item())
            running_loss += loss.item()

            if i % 20 == 19:    # print every 20 mini-batches
                hours, rem = divmod(time.time() - start_time, 3600)
                minutes, seconds = divmod(rem, 60)
                print(f"[{int(hours)}:{int(minutes)}:{int(seconds)}] "
                      f"Epoch {epoch}, Batch {i} "
                      f"Average loss: {loss_meter.avg} "
                      f"Average acc: {train_acc_meter.avg} "
                      f"Running loss: {running_loss / 20}")
                running_loss = 0.0

        #scheduler.step()
        test_acc = evaluate_classifier(testloader, classifier, device)

        progress_dict['loss'].append(loss_meter.avg)
        progress_dict['train_acc'].append(train_acc_meter.avg)
        progress_dict['test_acc'].append(test_acc)

        with open('./progress.json', 'w') as f:
            json.dump(progress_dict, f)


def evaluate_classifier(classifier, data_dir, device):
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    total = 0
    test_acc_meter = AverageMeter()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            test_acc_meter.update(accuracy(outputs, labels)[0].item())

    return test_acc_meter.avg

class SimCLRModel(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLRModel, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        # Temporarily add an extra layer to be closer to our model implementation
        self.f = nn.Sequential(*self.f)
        # # projection head
        # self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
        #                        nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        #out = self.g(feature)
        return F.normalize(feature, dim=-1) #, F.normalize(out, dim=-1)


def representation_learning(algo, device, log_dir, config):
    print('Train representation learner')
    if isinstance(algo, str):
        algo = getattr(algos, algo)
    assert issubclass(algo, algos.RepresentationLearner)

    rep_learning_augmentations = transforms.Compose([
        transforms.Lambda(lambda x: x.cpu().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(32), #, interpolation=PIL.Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # SimCLR doesn't use blur for CIFAR-10
    ])

    rep_learning_data, combined_meta = load_wds_datasets([{}])
    augmenter_kwargs = {
        "augmenter_spec": "translate,flip_lr,color_jitter_ex,gray",
        "color_space": combined_meta['color_space'],

        # (Cynthia) Here I'm using augmenter_func because I want our settings
        # to be as close to SimCLR as possible
        "augment_func": rep_learning_augmentations
    }
    optimizer_kwargs = {
        "lr": 1e-3,
        "weight_decay": 1e-6
    }

    # This is currently erroneously 1
    num_epochs = config['pretrain_epochs']
    batch_size = config['pretrain_batch_size']
    batches_per_epoch = config['pretrain_batches_per_epoch']

    # Modify resnet according to SimCLR paper Appendix B.9
    # simclr_resnet = resnet50()
    # simclr_resnet.fc = torch.nn.Linear(2048, config['representation_dim'])
    # simclr_resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
    # simclr_resnet.maxpool = torch.nn.Identity()

    model = algo(
        observation_space=combined_meta['observation_space'],
        action_space=combined_meta['action_space'],
        log_dir=log_dir,
        batch_size=batch_size,
        representation_dim=config['representation_dim'],
        projection_dim=config['projection_dim'],
        device=device,
        normalize=False,
        shuffle_batches=True,
        color_space=combined_meta['color_space'],
        save_interval=config['pretrain_save_interval'],
        encoder_kwargs={'obs_encoder_cls': lambda *args: SimCLRModel()},
        decoder_kwargs={'projection_architecture': [{'output_dim': 512}]},
        augmenter_kwargs=augmenter_kwargs,
        optimizer=torch.optim.Adam,
        optimizer_kwargs=optimizer_kwargs,
        #scheduler=LinearWarmupCosine,
        scheduler_kwargs={'warmup_epoch': 2, 'total_epochs': num_epochs},
        loss_calculator_kwargs={'temp': config['pretrain_temperature'],
                                'use_repo_loss': config['use_repo_loss']},
        log_interval=10,
        calc_log_interval=10
    )
    _, encoder_checkpoint_path = model.learn(rep_learning_data, batches_per_epoch, num_epochs)
    print("Representation Learning trained!")
    pretrained_model = torch.load(encoder_checkpoint_path)
    return pretrained_model


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, k, num_classes, temperature, epoch):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, num_classes, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100

## data handling class copied from SimCLR implementation
class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

@cifar_ex.config
def default_config():
    seed = 1
    algo = 'SimCLR'
    data_dir = 'cifar10/'
    pretrain_epochs = 1000
    pretrain_batches_per_epoch = 390
    finetune_epochs = 100
    representation_dim = 2048 ## TODO change back
    projection_dim = 128
    pretrain_lr = 3e-4
    pretrain_weight_decay = 1e-4
    pretrain_momentum = 0.9
    pretrain_batch_size = 512
    pretrain_save_interval = 100
    pretrain_temperature = 0.5
    pretrained_model = None
    use_repo_loss = False
    _ = locals()
    del _


@cifar_ex.main
def run(seed, algo, data_dir, pretrain_epochs, finetune_epochs, representation_dim,
        pretrained_model, pretrain_batch_size, _config):
    # TODO fix this hacky nonsense
    if pretrained_model is None:
        log_dir = os.path.join(cifar_ex.observers[0].dir, 'training_logs')
        os.mkdir(log_dir)
        os.makedirs(data_dir, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = representation_learning(algo, device, log_dir, _config)

    else:
        model = torch.load(pretrained_model)

    test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    memory_data = CIFAR10Pair(root='data', train=True, transform=test_transform, download=True)
    memory_loader = torch.utils.data.DataLoader(memory_data, batch_size=pretrain_batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_data = CIFAR10Pair(root='data', train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=pretrain_batch_size, shuffle=False, num_workers=16, pin_memory=True)


    # KNN testing from SimCLR repo for comparison
    test(model.network, memory_loader, test_loader, k=200, num_classes=10, temperature=_config['pretrain_temperature'], epoch=-1)
    # print('Train linear head')
    # classifier = LinearHead(model.network, representation_dim, output_dim=10).to(device)
    # train_classifier(classifier, data_dir, num_epochs=finetune_epochs, device=device)

    # print('Evaluate accuracy on test set')
    # evaluate_classifier(classifier, data_dir, device=device)


if __name__ == '__main__':
    cifar_ex.observers.append(FileStorageObserver('runs/cifar_runs'))
    cifar_ex.run_commandline()

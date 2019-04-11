import os
import numpy as np
import torch
import sys
import pickle
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
import time
import re
import torchvision.datasets as dset


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class AverageMeter(object):

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def get_loader(args):
    # return training loader, testing loader and set block number of resnet.
    if args.dataset == 'cifar10':
        args.classes_num = 10
        return get_cifar_loader(args)
    elif args.dataset == 'cifar100':
        args.classes_num = 100
        return get_cifar100_loader(args)
    elif args.dataset == 'imageNet':
        args.classes_num = 1000
        return get_imageNet_loader(args)
    else:
        raise NotImplementedError


def get_cifar_loader(args):
    # Set the data loader
    train_transform, test_transform = _data_transforms_cifar10()
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    return train_queue, test_queue


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def get_imageNet_loader(args):
    traindir = os.path.join(args.data, 'imageNet', 'train')
    valdir = os.path.join(args.data, 'imageNet', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader


def get_cifar100_loader(args):
    train_transform, test_transform = _data_transforms_cifar100()

    train_data = datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR100(root=args.data, train=False, download=True, transform=test_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    return train_queue, test_queue


def _data_transforms_cifar100():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    return train_transform, test_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6


def save_checkpoint(model, is_best, save, n_epoch):
    filename = os.path.join(save, 'last.pth')
    state = {
        'net': model.state_dict(),
        'n_epoch': n_epoch
    }
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'best.pth')
        shutil.copyfile(filename, best_filename)


def load(model, args):
    model_path = args.load
    ckpt = torch.load(model_path)
    try:
        net_dic = ckpt['net']
    except:
        model.load_state_dict(ckpt)
        return

    model.load_state_dict(net_dic)
    args.epoch_start = ckpt['n_epoch'] + 1


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def getTime(seconds):
    t = int(seconds)

    day = t // 86400
    hour = (t - (day * 86400)) // 3600
    minit = (t - ((day * 86400) + (hour * 3600))) // 60
    seconds = t - ((day * 86400) + (hour * 3600) + (minit * 60))
    return "{} days {} hours {} minutes {} seconds remaining.".format(day, hour, minit, seconds)

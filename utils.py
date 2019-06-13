import os
import time
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
from typing import Tuple
from collections.abc import Iterable, MappingView
import matplotlib
import model.compress as compress

from meter import AverageMeter

matplotlib.use('Agg')


def create_exp_dir(path, scripts_to_save=None):
    r"""
    create experiment directory for check point and scripts (optional)


    Args:
        path: path of directory, the check point will in ./path/
        scripts_to_save: scripts to save
    """
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


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
    train_data = datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

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
        num_workers=args.workers, pin_memory=True)

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
    except KeyError:
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


def q_table_dct_gen(q_list=None):
    assert len(q_list) == 15, "q_list must be 15 values form low to high band"
    if type(q_list) is not torch.Tensor:
        q_list = torch.tensor(q_list, dtype=torch.get_default_dtype())
    q_table = torch.ones(8, 8)
    if q_list is not None:
        for i in range(8):
            q_table[i, :] = q_list[i:i + 8]

    return q_table


def gen_seg_dict(k, maximum, len_key=False):
    """
    Generate code length dictionary for unsigned sparse-exponential-Golomb

    Args:
        k (int): k for exponential-Golomb
        maximum (int): Maximum for range
        len_key (bool): code for key or length for key

    Returns:
        code_length_dict or length_code_dict for signed sparse-exponential-Golomb
    """
    assert k >= 0, "k must >= 0"
    assert maximum >= 0, "maximum must >= 0"
    length_code_dict = {1: {0}}
    if maximum != 0:
        pitch = 2 ** k
        integer_cur = pitch
        len_cur = k + 2
        length_code_dict[len_cur] = set(range(1, pitch + 1))

        while integer_cur < maximum:
            pitch *= 2
            len_cur += 2
            length_code_dict[len_cur] = set(range(integer_cur + 1, integer_cur + pitch + 1))
            integer_cur += pitch

    if len_key:
        return length_code_dict
    else:
        code_length_dict = {}
        for length in length_code_dict:
            for code in length_code_dict[length]:
                if code < maximum:
                    code_length_dict[code] = length

        return code_length_dict


def gen_signed_seg_dict(k, maximum, len_key=False):
    r"""
    Generate code length dictionary for signed version sparse-exponential-Golomb

    Args:
        k (int): k for exponential-Golomb
        maximum (int): Maximum for range
        len_key (bool): code for key or length for key

    Returns:
        code_length_dict or length_code_dict  for signed sparse-exponential-Golomb
    """
    assert k >= 0, "k must >= 0"
    maximum = 2 * abs(maximum)
    length_code_dict = {1: {0}}
    if maximum != 0:
        set_list = []
        pitch = 2 ** k
        integer_cur = pitch
        for i in range(1, pitch + 1):
            if (i % 2) != 0:
                set_list.append(-((i + 1) // 2))
            else:
                set_list.append((i + 1) // 2)

        len_cur = k + 2
        length_code_dict[len_cur] = set(set_list)

        while integer_cur < maximum:
            set_list = []
            pitch *= 2
            len_cur += 2
            for i in range(integer_cur + 1, integer_cur + pitch + 1):
                if (i % 2) != 0:
                    set_list.append(-((i + 1) // 2))
                else:
                    set_list.append((i + 1) // 2)

            integer_cur += pitch
            length_code_dict[len_cur] = set(set_list)

    if len_key:
        return length_code_dict
    else:
        code_length_dict = {}
        for length in length_code_dict:
            for code in length_code_dict[length]:
                if code in range(-maximum // 2, maximum // 2):
                    code_length_dict[code] = length
        return code_length_dict


def stream2bit_cnt(in_stream, code_length_dict, conti=False, dual_conti=False):
    """
    Bit cnt for given stream with dictionary of code length

    Args:
        in_stream (torch.Tensor):  Input stream
        code_length_dict(dict): code_length_dict, key(int) is code_length, value is iterable codes before encoding
        conti(bool): The continuous sequence of number
        dual_conti(bool): The sequence

    Returns:
        Total length (bits)
    """
    eps = 10 ** -5
    assert ((in_stream % 1) < eps).max(), "in_stream need to be integers"
    assert not (dual_conti and conti), "Only one or none of setting can be true"

    if len(in_stream.size()) > 1:
        in_stream = in_stream.view(-1)

    size = in_stream.size(0)
    total_cnt = 0
    total_len = 0
    for code_len in code_length_dict:
        if conti:
            maximum = max(code_length_dict[code_len])
            minimum = min(code_length_dict[code_len])
            match = (in_stream <= (maximum + eps)) & (in_stream >= (minimum - eps))
            current_cnt = match.sum().item()
            total_cnt += current_cnt
            total_len += (current_cnt * code_len)
        elif dual_conti:
            max_pos = -2 ** 40
            min_pos = 2 ** 40
            max_neg = -2 ** 40
            min_neg = 2 ** 40
            for integer in code_length_dict[code_len]:
                if integer >= 0:
                    max_pos = max(max_pos, integer)
                    min_pos = min(min_pos, integer)
                else:
                    max_neg = max(max_neg, integer)
                    min_neg = min(min_neg, integer)

            match_pos = (in_stream <= (max_pos + eps)) & (in_stream >= (min_pos - eps))
            match_neg = (in_stream <= (max_neg + eps)) & (in_stream >= (min_neg - eps))
            match = match_pos | match_neg
            current_cnt = match.sum().item()
            total_cnt += current_cnt
            total_len += (current_cnt * code_len)
        else:
            for integer in code_length_dict[code_len]:
                match = (in_stream - integer).abs() < 10 ** -5
                current_cnt = match.sum().item()
                total_cnt += current_cnt
                total_len += (current_cnt * code_len)
                if total_cnt >= size:
                    break
        if total_cnt >= size:
            break

    assert total_cnt == size, \
        "{}, {} Not all element in in_stream encoded".format(total_cnt, size)
    return total_len


def infer_base(data_queue, model, handler_list=None):
    model.eval()
    end = time.time()
    total_step = len(data_queue)
    data_time = AverageMeter()
    batch_time = AverageMeter()

    with torch.no_grad():
        for step, (x, target) in enumerate(data_queue):
            time.time()
            data_time.update(time.time() - end)

            x = Variable(x).cuda()
            target = target.cuda()

            result = model(x)

            for handler in handler_list:
                handler.update_batch(result, (x, target))

            batch_time.update(time.time() - end)
            end = time.time()

            time_remain = getTime((total_step - step) * batch_time.avg)
            print('Time: {:.4f} Data: {:.4f} Time remaining: {}'.format(
                batch_time.avg, data_time.avg, time_remain))


def imagenet_model_graph_mapping(pretrain_dic, ns):
    mapped_dic = {}
    for k in pretrain_dic:
        if k[:5] == "layer":
            layer_cnt = 0
            for i in range(int(k[5]) - 1):
                layer_cnt += ns[i - 1]  # index of layer in pretrain model start in 1 ...
            layer_cnt += int(k[7])
            if k[8:20] == ".downsample.":
                k_new = "stages.layers.{}".format(layer_cnt) + k[8:]
            else:
                k_new = "stages.layers.{}.block".format(layer_cnt) + k[8:]
        else:
            k_new = k

        mapped_dic[k_new] = pretrain_dic[k]

    return mapped_dic


def iterable_l2(x):
    if isinstance(x, torch.Tensor):
        x = x.cuda()
        return (x * x).sum()

    elif isinstance(x, Iterable):
        l2 = 0
        for x_e in x:
            l2 += iterable_l2(x_e)
        return l2


def iterable_l1(x):
    if isinstance(x, torch.Tensor):
        return x.cuda().abs().sum()

    elif isinstance(x, Iterable):
        l1 = 0
        for x_e in x:
            l1 += iterable_l1(x_e)
        return l1


def piecewise_linear(x, length_code_dict):
    if isinstance(x, torch.Tensor):
        l1 = 0
        key_gap = 2
        for key in length_code_dict.keys():
            if key > 1 + key_gap:
                code_min = min(length_code_dict[key - key_gap])
                set_size = len(length_code_dict[key])  # number of element in set of code in same code length
                pos = (x + code_min) / set_size * 2 * key_gap
                neg = (-x + code_min) / set_size * 2 * key_gap

                l1 = l1 + torch.clamp(torch.max(pos, neg), min=0, max=key_gap)

    elif isinstance(x, Iterable):
        l1 = 0
        for x_e in x:
            l1 += iterable_l1(x_e)
        return l1


def dwt_channel_weighted_loss(x, channel_weight):
    r"""
    Loss dedicated to dwt
    With different weight (amount of penalty) for different channels

    Args:
        x (list[Tuple[torch.Tensor, list[torch.Tensor]]]): list of DWT result in each layers
        channel_weight (list[torch.Tensor], MappingView[torch.Tensor]): list of channel_weight in each layers

    Returns:
        loss
    """
    loss = 0
    for x_layer, channel_weight_layer in zip(x, channel_weight):
        _, xh = x_layer
        # match dim
        channel_weight_layer = channel_weight_layer.unsqueeze(dim=-1)
        channel_weight_layer = channel_weight_layer.unsqueeze(dim=-1)
        channel_weight_layer = channel_weight_layer.unsqueeze(dim=-1)
        xh0_cuda = xh[0].cuda()
        # channel_weight_layer = compress.random_round(channel_weight_layer, rand_factor=0.05)
        loss += (channel_weight_layer * xh0_cuda * xh0_cuda).sum()  # highest frequency only

    return loss


def bipolar(x, th):
    return -((x - th).abs().sum())


def get_param_names(model, name_tail):
    names = []
    for name, _ in model.named_parameters():
        if name[-len(name_tail):] == name_tail:
            names.append(name)
    return names


def get_params(model, name_tail):
    params = []
    for name, param in model.named_parameters():
        if name[-len(name_tail):] == name_tail:
            params.append(param)
    return params


def optimizer_setting_separator(model, settings):
    params = [{}]
    for setting in settings:
        temp = setting.copy()
        temp.pop('setting_names')
        params.append(temp)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        default = True
        for indx, setting in enumerate(settings):
            if name in setting['setting_names']:
                if 'params' in params[indx + 1].keys():
                    params[indx + 1]['params'].append(param)
                else:
                    params[indx + 1]['params'] = [param]
                default = False
        if default:
            if 'params' in params[0].keys():
                params[0]['params'].append(param)
            else:
                params[0]['params'] = [param]
    return params


def freq_select2channel_weight(freq_select, ratio=0.5, tau=5):
    r"""

    Args:
        freq_select(torch.Tensor): freq_select from MaskCompressDWT
        ratio(float): Retaining ratio
        tau(float): Temperature of hyperbolic tangent

    Returns:
        channel_weight(Tensor)
    """
    assert len(freq_select.size()) == 1, \
        "freq_select's size need to be (channel, ), but get {}".format(freq_select.size())
    if abs(ratio - 0.5) < 0.000001:
        # return F.leaky_relu(((freq_select - freq_select.median()) * tau).tanh())
        return F.relu(((freq_select - freq_select.median()) * tau).tanh())
    else:
        kth = int(freq_select.size(0) * ratio)
        if kth != 0:
            kthvalue, _ = torch.kthvalue(freq_select, kth)
        else:
            kthvalue = 0
        # return F.leaky_relu(((freq_select - kthvalue) * tau).tanh())
        return F.relu(((freq_select - kthvalue) * tau).tanh())


if __name__ == '__main__':
    pass

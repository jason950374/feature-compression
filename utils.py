import os
import numpy as np
import torch
import torch.utils.data
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.datasets as dset
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class AverageMeter:
    """
    On-line Average Meter
    """

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


class HistMeter:
    eps = 10 ** -5

    def __init__(self, codes=None):
        self.hist = {}
        # __init__ also used in reset. If codes=None, use old self.codes
        if codes is not None:
            self.codes = codes

        try:
            for code in self.codes:
                self.hist[code] = 0
        except AttributeError:
            self.codes = None

    def reset(self, codes=None):
        self.__init__(codes)

    def update(self, in_stream):
        cnt = 0
        if len(in_stream.size()) > 1:
            in_stream = in_stream.view(-1)
        size = in_stream.size(0)

        for code in self.hist:
            match = (in_stream <= (code + self.eps)) & (in_stream >= (code - self.eps))
            cnt += match.sum()
            self.hist[code] += match.sum()

        assert cnt == size, \
            "{}vs. {}: Some code in in_stream not find in self.hist".format(cnt, size)

    def get_bit_cnt(self, code_length_dict):
        r"""
            Bit cnt for given histogram with dictionary of code length

            Args:
                    hist (dict):  Input histogram dictionary (code, cnt)
                    code_length_dict (dict): code_length_dict, key(int) is code_length, value is iterable codes before
                        encoding

            Note:
                    Different from bins of numpy.histogram, hist and code has same length.
                    Normally, code_list[i] can be
                    :math:`\frac{bins[i]+bins[i+1]}{2}`

            Returns:
                    Total length (bits)
            """
        total_len = 0

        for code in self.hist:
            total_len += self.hist[code] * code_length_dict[code]

        return total_len

    def plt_hist(self, plt_fn=None):
        if plt_fn is None:
            plt_fn = plt

        codes = np.asarray(list(self.hist.keys()))
        codes.sort()
        hist = []
        for code in codes:
            hist.append(self.hist[code])
        plt_fn.bar(codes, hist, width=1)


def create_exp_dir(path, scripts_to_save=None):
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


def get_q_range(train_queue, model, do_print=True):
    model.eval()

    feature_maps = None
    fm_transforms = None
    with torch.no_grad():
        for step, (x, target) in enumerate(train_queue):
            x = Variable(x).cuda()

            _, feature_maps_batch, fm_transforms_batch = model(x)

            if feature_maps is not None:
                feature_maps_org = feature_maps
                feature_maps = []
                for feature_map, feature_map_batch in zip(feature_maps_org, feature_maps_batch):
                    feature_maps.append(torch.cat((feature_map, feature_map_batch)))
            else:
                feature_maps = feature_maps_batch

            # concatenate mini-batch into whole data set
            if len(fm_transforms_batch) != 0:
                if fm_transforms is not None:
                    fm_transforms_org = fm_transforms
                    fm_transforms = []
                    for fm_transform, fm_transform_batch in zip(fm_transforms_org, fm_transforms_batch):
                        if type(fm_transform_batch) is tuple:
                            XL = torch.cat((fm_transform[0], fm_transform_batch[0]))
                            XH = []
                            for XH_level, XH_level_batch in zip(fm_transform[1], fm_transform_batch[1]):
                                XH.append(torch.cat((XH_level, XH_level_batch)))

                            fm_transforms.append((XL, XH))
                        else:
                            fm_transforms.append(torch.cat((fm_transform, fm_transform_batch)))
                else:
                    fm_transforms = fm_transforms_batch

        # Result of fm_transforms will not returned by this function
        # So this section only need to be executed when do_print=True
        if (fm_transforms is not None) and do_print:
            maximum = []
            minimum = []
            for fm_transform in fm_transforms:
                max_layer = -2 ** 40
                min_layer = 2 ** 40
                # DWT
                if type(fm_transform) is tuple:
                    XL, XH = fm_transform
                    max_layer = max(max_layer, XL.cuda().max().item())
                    min_layer = min(min_layer, XL.cuda().min().item())
                    for xh in XH:
                        max_layer = max(max_layer, xh.cuda().max().item())
                        min_layer = min(min_layer, xh.cuda().min().item())
                # DCT
                else:
                    max_layer = max(max_layer, fm_transform.cuda().max().item())
                    min_layer = min(min_layer, fm_transform.cuda().min().item())

                maximum.append(max_layer)
                minimum.append(min_layer)

            print("==============================================================")
            for indx, min_layer, max_layer in zip(range(len(minimum)), minimum, maximum):
                print("transform range {}: ({}, {})".format(indx, min_layer, max_layer))
            print("==============================================================")

        maximum_fm = []
        minimum_fm = []
        for feature_map in feature_maps:
            max_layer = -2 ** 40
            min_layer = 2 ** 40
            max_cur = feature_map.cuda().max()
            if max_layer < max_cur:
                max_layer = max_cur
            min_cur = feature_map.cuda().min()
            if min_layer > min_cur:
                min_layer = min_cur
            maximum_fm.append(max_layer)
            minimum_fm.append(min_layer)

        if do_print:
            for indx, min_layer, max_layer in zip(range(len(minimum_fm)), minimum_fm, maximum_fm):
                print("range {}: ({}, {})".format(indx, min_layer, max_layer))
            print("==============================================================")

    return minimum_fm, maximum_fm


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

    Return:
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
                code_length_dict[code] = length

        return code_length_dict


def gen_signed_seg_dict(k, maximum, len_key=False):
    r"""
    Generate code length dictionary for signed version sparse-exponential-Golomb

    Args:
        k (int): k for exponential-Golomb
        maximum (int): Maximum for range
        len_key (bool): code for key or length for key

    Return:
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
                code_length_dict[code] = length
        return code_length_dict


def stream2bit_cnt(in_stream, code_length_dict, conti=False, dual_conti=False):
    """
    Bit cnt for given stream with dictionary of code length

    :param in_stream:  Input stream
    :param code_length_dict: code_length_dict, key(int) is code_length, value is iterable codes before encoding
    :param conti: The continuous sequence of number
    :param dual_conti: The sequence
    :return: Total length (bits)
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


def infer_base(train_queue, model, handler_list=None):
    model.eval()

    with torch.no_grad():
        for step, (x, target) in enumerate(train_queue):
            x = Variable(x).cuda()
            target = target.cuda()

            result = model(x)

            for handler in handler_list:
                handler.update_batch(result, (x, target))


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


if __name__ == '__main__':
    pass

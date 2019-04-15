import numpy as np
import torch
import torch.nn as nn
import glob
import utils
import argparse
import logging
import sys
import os
import time
import random
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model.ResNet import ResNetCifar
from model.compress import CompressDCT, CompressDWT, QuantiUnsign
from torch.autograd import Variable

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--data', type=str, default='/home/jason/data/',
                    help='location of the data corpus relative to home')
parser.add_argument('--workers', type=int, default=4, help='workers for data loader')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=-1, help='random seed')
parser.add_argument('--depth', type=int, default=20, help='Depth of base resnet model')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes of now dataset.')
parser.add_argument('--load', type=str, default="")

args = parser.parse_args()
args.save = 'ckpts/test_{}_resnet{}_{}'.format(args.dataset, args.depth, time.strftime("%m%d_%H%M%S"))

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
batch_time = utils.AverageMeter()
data_time = utils.AverageMeter()

if args.dataset == 'cifar10':
    args.num_classes = 10
    args.usage_weight = 2
    utils.multiply_adds = 2
elif args.dataset == 'cifar100':
    args.num_classes = 100
    args.usage_weight = 2
    utils.multiply_adds = 2
elif args.dataset == 'imageNet':
    args.num_classes = 1000
    args.weight_decay = 1e-4
    args.workers = 64
    args.usage_weight = 1
    utils.multiply_adds = 1
else:
    raise NotImplementedError(
        '{} dataset is not supported. Only support cifar10, cifar100 and imageNet.'.format(args.dataset))

Q_table_dct = torch.tensor([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
], dtype=torch.float)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True

    cudnn.benchmark = True

    logging.info("args = %s", args)

    # Set the data loader
    train_queue, test_queue = utils.get_loader(args)

    # Build up the network
    # model = nn.DataParallel(ResNetCifar().cuda())
    model = ResNetCifar(args.depth, args.classes_num).cuda()

    utils.load(model, args)

    # Insert compress_block after load since compress_block not include in training phase in this case
    # _, maximum_fm = get_q_range(train_queue, model)
    maximum_fm = [5.2, 6.7, 5.3, 5.8, 6.7, 7.6, 4.6, 5.7, 36]  # quick test for this ckpts
    compress_list = compress_list_gen(args.depth, maximum_fm)

    model.compress_replace(compress_list)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    test_acc, test_acc_5 = infer(test_queue, model)

    logging.info('[Test] acc %.2f%%;', test_acc)


def infer(test_queue, model):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    feature_maps = None
    fm_transforms = None
    with torch.no_grad():
        for step, (x, target) in enumerate(test_queue):
            x = Variable(x).cuda()
            target = Variable(target).cuda(async=True)

            logits, feature_maps_batch, fm_transforms_batch = model(x)  # TODO clean up

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = x.size(0)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
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

        if fm_transforms is not None:
            zero_cnt = 0
            size_flat = 0
            maximun = -2 ** 40
            minimun = 2 ** 40
            for layer_num, fm_transform in enumerate(fm_transforms):
                if type(fm_transform) is tuple:
                    XL, XH = fm_transform
                    plt.hist(XL.view(-1))
                    plt.savefig('{}/Layer{}_XL.png'.format(args.save, layer_num))
                    plt.clf()

                    for i, xh in enumerate(XH):
                        plt.hist(xh.view(-1))
                        plt.savefig('{}/Layer{}_XH_{}.png'.format(args.save, layer_num, i))
                        plt.clf()

                    zero_cnt += (XL.cuda().abs() < 10 ** -10).sum().item()
                    size_flat += XL.view(-1).size(0)
                    max_cur = XL.cuda().max()
                    if maximun < max_cur:
                        maximun = max_cur
                    min_cur = XL.cuda().min()
                    if minimun > min_cur:
                        minimun = min_cur
                    for xh in XH:
                        zero_cnt += (xh.cuda().abs() < 10 ** -10).sum().item()
                        size_flat += xh.view(-1).size(0)
                        max_cur = xh.cuda().max()
                        if maximun < max_cur:
                            maximun = max_cur
                        min_cur = xh.cuda().min()
                        if minimun > min_cur:
                            minimun = min_cur
                else:
                    zero_cnt += (fm_transform.cuda().abs() < 10 ** -10).sum().item()
                    size_flat += fm_transform.view(-1).size(0)
                    plt.hist(fm_transform.view(-1))
                    plt.savefig('{}/Layer{}_DCT.png'.format(args.save, layer_num))
                    plt.clf()
                    max_cur = fm_transform.cuda().max()
                    if maximun < max_cur:
                        maximun = max_cur
                    min_cur = fm_transform.cuda().min()
                    if minimun > min_cur:
                        minimun = min_cur

            logging.info("==============================================================")
            logging.info("fm_transforms == 0: {}".format(zero_cnt))
            logging.info("fm_transforms size: {}".format(size_flat))
            logging.info("transform sparsity: {}".format(zero_cnt / size_flat))
            logging.info("transform range ({}, {})".format(minimun, maximun))
            logging.info("==============================================================")

        zero_cnt = 0
        size_flat = 0
        maximun = -2 ** 40
        minimun = 2 ** 40
        for feature_map in feature_maps:
            zero_cnt += (feature_map.cuda().abs() < 10 ** -10).sum().item()
            size_flat += feature_map.view(-1).size(0)
            max_cur = feature_map.cuda().max()
            if maximun < max_cur:
                maximun = max_cur
            min_cur = feature_map.cuda().min()
            if minimun > min_cur:
                minimun = min_cur

        logging.info("feature_maps == 0: {}".format(zero_cnt))
        logging.info("feature_maps size: {}".format(size_flat))
        logging.info("sparsity: {}".format(zero_cnt / size_flat))
        logging.info("range ({}, {})".format(minimun, maximun))
        logging.info("==============================================================")

    return top1.avg, top5.avg


def get_q_range(train_queue, model):
    model.eval()

    feature_maps = None
    fm_transforms = None
    with torch.no_grad():
        for step, (x, target) in enumerate(train_queue):
            x = Variable(x).cuda()

            _, feature_maps_batch, fm_transforms_batch = model(x)  # TODO clean up

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

        maximun = []
        minimun = []
        if fm_transforms is not None:
            for fm_transform in fm_transforms:
                max_layer = -2 ** 40
                min_layer = 2 ** 40
                # DWT
                if type(fm_transform) is tuple:
                    XL, XH = fm_transform
                    max_cur = XL.cuda().max()
                    if max_layer < max_cur:
                        max_layer = max_cur
                    min_cur = XL.cuda().min()
                    if min_layer > min_cur:
                        min_layer = min_cur
                    for xh in XH:
                        max_cur = xh.cuda().max()
                        if max_layer < max_cur:
                            max_layer = max_cur
                        min_cur = xh.cuda().min()
                        if min_layer > min_cur:
                            min_layer = min_cur
                # DCT
                else:
                    max_cur = fm_transform.cuda().max()
                    if max_layer < max_cur:
                        max_layer = max_cur
                    min_cur = fm_transform.cuda().min()
                    if min_layer > min_cur:
                        min_layer = min_cur

                maximun.append(max_layer)
                minimun.append(min_layer)

            print("==============================================================")
            for indx, min_layer, max_layer in zip(range(len(minimun)), minimun, maximun):
                print("transform range {}: ({}, {})".format(indx, min_layer, max_layer))
            print("==============================================================")

        maximun_fm = []
        minimun_fm = []
        for feature_map in feature_maps:
            max_layer = -2 ** 40
            min_layer = 2 ** 40
            max_cur = feature_map.cuda().max()
            if max_layer < max_cur:
                max_layer = max_cur
            min_cur = feature_map.cuda().min()
            if min_layer > min_cur:
                min_layer = min_cur
            maximun_fm.append(max_layer)
            minimun_fm.append(min_layer)

        for indx, min_layer, max_layer in zip(range(len(minimun_fm)), minimun_fm, maximun_fm):
            print("range {}: ({}, {})".format(indx, min_layer, max_layer))
        print("==============================================================")

    return minimun_fm, maximun_fm


def compress_list_gen(depth, maximum_fm):
    encoder_list = []
    decoder_list = []
    for i in range((depth - 2) // 2):
        q_factor = maximum_fm[i] / 255

        maximum_fm = [5.2, 6.7, 5.3, 5.8, 6.7, 7.6, 4.6, 5.7, 36]
        if i < 8:
            q_table_dwt = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float)
            q_list_dct = [150, 50, 50, 50, 50, 50, 50, 50,
                          100, 100, 100, 100, 100, 100, 200]
        else:
            q_table_dwt = torch.tensor([10**6, 10**6, 10**6, 1], dtype=torch.float)
            q_list_dct = [150, 10**6, 10**6, 10**6, 10**6, 10**6, 10**6, 10**6,
                          10**6, 10**6, 10**6, 10**6, 10**6, 10**6, 10**6]
        q_table_dwt = q_table_dwt * 255 / maximum_fm[i]

        encoder_seq = [
            QuantiUnsign(bit=8, q_factor=q_factor).cuda(),
            CompressDCT(q_table_dct_gen(q_list_dct)).cuda(),
            # CompressDWT(level=3, q_table=q_table_dwt).cuda()
        ]
        decoder_seq = [
            # CompressDWT(level=3, q_table=q_table_dwt, is_encoder=False).cuda(),
            CompressDCT(q_table_dct_gen(q_list_dct), is_encoder=False).cuda(),
            QuantiUnsign(bit=8, q_factor=q_factor, is_encoder=False).cuda()
        ]

        encoder_list.append(nn.Sequential(*encoder_seq))
        decoder_list.append(nn.Sequential(*decoder_seq))
    return encoder_list, decoder_list


def q_table_dct_gen(q_list=None):
    assert len(q_list) == 15, "q_list must be 15 values form low to high band"
    if type(q_list) is not torch.Tensor:
        q_list = torch.tensor(q_list, dtype=torch.float)
    q_table = torch.ones(8, 8)
    if q_list is not None:
        for i in range(8):
            q_table[i, :] = q_list[i:i + 8]

    return q_table


if __name__ == '__main__':
    main()

import numpy as np
import torch
import glob
import utils
import argparse
import logging
import sys
import os
import time
import random
import torch.backends.cudnn as cudnn
import functional.dct as dct
from model.ResNet import ResNetCifar
from torch.autograd import Variable

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--data', type=str, default='/home/jason/data/', help='location of the data corpus relative to home')
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

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    utils.load(model, args)

    test_acc, test_acc_5, feature_maps = infer(test_queue, model)

    logging.info('[Test] acc %.2f%%;', test_acc)


def infer(test_queue, model):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    feature_maps = None
    feature_maps_dct = None
    with torch.no_grad():
        for step, (x, target) in enumerate(test_queue):
            x = Variable(x).cuda()
            target = Variable(target).cuda(async=True)

            logits, feature_maps_batch, feature_maps_dct_batch = model(x)

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

            if feature_maps_dct is not None:
                feature_maps_dct_org = feature_maps_dct
                feature_maps_dct = []
                for feature_map_dct, feature_map_dct_batch in zip(feature_maps_dct_org, feature_maps_dct_batch):
                    feature_maps_dct.append(torch.cat((feature_map_dct, feature_map_dct_batch)))
            else:
                feature_maps_dct = feature_maps_dct_batch

        zero_cnt = 0
        size_flat = 0
        for feature_map_dct in feature_maps_dct:
            zero_cnt += (feature_map_dct.cuda().abs() < 10 ** -10).sum().item()
            size_flat += feature_map_dct.size(0)*feature_map_dct.size(1)*feature_map_dct.size(2)*feature_map_dct.size(3)

        print("==============================================================")
        print("feature_maps_dct == 0: ", zero_cnt)
        print("feature_maps_dct size: ", size_flat)
        print("DCT sparsity: ", zero_cnt / size_flat)
        print("==============================================================")

        zero_cnt = 0
        size_flat = 0
        for feature_map in feature_maps:
            zero_cnt += (feature_map.cuda().abs() < 10 ** -10).sum().item()
            size_flat += feature_map.size(0) * feature_map.size(1) * feature_map.size(2) * feature_map.size(3)

        print("feature_maps == 0: ", zero_cnt)
        print("feature_maps size: ", size_flat)
        print("sparsity: ", zero_cnt / size_flat)
        print("==============================================================")

    return top1.avg, top5.avg, feature_maps


if __name__ == '__main__':
    main()

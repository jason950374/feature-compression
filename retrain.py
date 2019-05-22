import numpy as np
import torch.cuda
import torch.nn
import torch.optim
import glob

import meter
import utils
import argparse
import logging
import sys
import os
import time
import random
import torch.backends.cudnn as cudnn
from model.ResNet import ResNetCifar, resnet18
from torch.autograd import Variable
from model.compress import *

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
# parser.add_argument('--data', type=str, default='/home/jason/data/',
#                        help='location of the data corpus relative to home')
parser.add_argument('--data', type=str, default='/home/gasoon/datasets',
                    help='location of the data corpus relative to home')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate for batch_size=256')
parser.add_argument('--workers', type=int, default=4, help='workers for data loader')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs_test', type=int, default=0, help='num of epochs begin to test')
parser.add_argument('--seed', type=int, default=-1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--depth', type=int, default=20, help='Depth of base resnet model')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes of now dataset.')
parser.add_argument('--load', type=str, default="")
parser.add_argument('--epoch_start', type=int, default=0, help='Epoch number begin')
parser.add_argument('--wavelet', type=str, default="db1", help='Mother wavelet for DWT')
parser.add_argument('--k', type=int, default=0, help="k for exponential-Golomb")
parser.add_argument('--l1_coe', type=float, default=0, help="coefficient of L1 regularizer for sparsity")
parser.add_argument('--bit', type=int, default=8, help="coefficient of L1 regularizer for sparsity")

args = parser.parse_args()
# args.save = 'ckpts/retrain_{}_{}'.format(args.wavelet, args.load[6:-9])
args.save = 'ckpts/retrain_wavelet_{}_relinkBP_no_transform'.format(args.wavelet)
args.learning_rate = args.learning_rate * args.batch_size / 256


utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
batch_time = meter.AverageMeter()
data_time = meter.AverageMeter()

if args.dataset == 'cifar10':
    args.num_classes = 10
    args.epochs = 50
    args.usage_weight = 2
    utils.multiply_adds = 2
elif args.dataset == 'cifar100':
    args.num_classes = 100
    args.epochs = 50
    args.usage_weight = 2
    utils.multiply_adds = 2
elif args.dataset == 'imageNet':
    args.num_classes = 1000
    args.epochs = 30
    args.weight_decay = 1e-4
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
    # Build up the network
    if args.dataset == 'cifar10':
        # model = nn.DataParallel(ResNetCifar().cuda())
        model = ResNetCifar(args.depth, args.classes_num).cuda()
    elif args.dataset == 'imageNet':
        # model = nn.DataParallel(resnet18().cuda())
        if args.depth == 18:
            model = resnet18().cuda()
        else:
            raise NotImplementedError(
                'Depth:{} is not supported.'.format(args.depth))
    else:
        raise NotImplementedError(
            '{} dataset is not supported. Only support cifar10, cifar100 and imageNet.'.format(args.dataset))

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    if args.dataset == 'imageNet':
        net_dic = torch.load(args.load)
        net_dic_fix = utils.imagenet_model_graph_mapping(net_dic, [2, 2, 2, 2])
        model.load_state_dict(net_dic_fix)
    else:
        utils.load(model, args)

    # quick test for this ckpts: cifar10_resnet20_0409_184724
    maximum_fm = [5.2, 6.7, 5.3, 5.8, 6.7, 7.6, 4.6, 5.7, 36]
    # quick test for pretrain resnet18
    # maximum_fm = [11, 15.5, 14, 11.5, 8.5, 14, 11.5, 101]
    compress_list = compress_list_gen(maximum_fm, args.wavelet, args.bit)

    model.compress_replace(compress_list)

    utils.save_checkpoint(model, False, args.save, 0)

    # Set the optimizer
    settings = [{'setting_names': utils.get_param_names(model, 'transform_matrix'),
                 'lr': args.learning_rate,
                 'weight_decay': 0,
                 'momentum': 0
                 }]
    params = utils.optimizer_setting_separator(model, settings)

    optimizer = torch.optim.SGD(
        # params,
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Set the objective function
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    best_test_acc = 0.0

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        milestones = [15, 30]
    elif args.dataset == 'imageNet':
        milestones = [10, 20]
    else:
        milestones = None

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
    length_code_dict = utils.gen_signed_seg_dict(args.k, 2 ** (args.bit-1), len_key=True)

    for epoch in range(0, args.epochs):
        scheduler.step()
        logging.info('[Train] Epoch = %d , LR = %e', epoch, scheduler.get_lr()[0])
        is_best = False
        train(train_queue, model, criterion, optimizer, epoch, args, length_code_dict)

        # Evaluate the test accuracy
        if epoch > args.epochs_test:
            this_acc, _ = infer(test_queue, model)

            if this_acc > best_test_acc:
                best_test_acc = this_acc
                is_best = True

            logging.info(
                '[Test] Epoch:%d/%d acc %.2f%%; best %.2f%%', epoch, args.epochs, this_acc, best_test_acc)

            logging.info('Saved into %s', args.save)

            utils.save_checkpoint(model, is_best, args.save, epoch)
            logging.info('============================================================================')


def train(train_queue, model, criterion, optimizer, cur_epoch, args, length_code_dict, warm_up=False):
    objs = meter.AverageMeter()
    top1 = meter.AverageMeter()
    top5 = meter.AverageMeter()
    model.train(True)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.BatchNorm1d):
            module.eval()
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()
        if isinstance(module, torch.nn.modules.BatchNorm3d):
            module.eval()

    total_step = len(train_queue)

    end = time.time()

    total_epoch = 5 if warm_up else args.epochs
    suffix = 'Warm Up' if warm_up else 'Train'

    for step, (x, target) in enumerate(train_queue):
        data_time.update(time.time() - end)

        x = Variable(x).cuda()
        target = Variable(target).cuda(async=True)

        # Forward propagation
        logits, _, fm_transforms = model(x)

        loss = criterion(logits, target)
        if args.l1_coe > 1e-20:
            l1 = utils.iterable_weighted_l1(fm_transforms, length_code_dict)
            loss += l1 * args.l1_coe

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        model.update()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = x.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.report_freq == 0:
            logging.info('[%s] Epoch:%d/%d Step:%d/%d Loss:%.2e Top1_acc:%.1f%% Top5_acc:%.1f%%',
                         suffix, cur_epoch, total_epoch, step, total_step, objs.avg, top1.avg, top5.avg)
            time_remain = utils.getTime((((args.epochs - cur_epoch) * total_step) - step) * batch_time.avg)
            logging.info('[{}] Time: {:.4f} Data: {:.4f} Time remaining: {}'.format(
                    suffix, batch_time.avg, data_time.avg, time_remain))


def infer(test_queue, model):
    top1 = meter.AverageMeter()
    top5 = meter.AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (x, target) in enumerate(test_queue):
            x = Variable(x).cuda()
            target = Variable(target).cuda(async=True)

            logits, _, _ = model(x)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = x.size(0)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    return top1.avg, top5.avg


# TODO put into utils?
def compress_list_gen(maximum_fm, wavelet='db1', bit=8):
    compress_list = []
    channel = [16, 16, 16, 32, 32, 32, 64, 64, 64]
    for i in range(len(maximum_fm) - 1):
        q_factor = maximum_fm[i] / (2 ** bit - 1)

        q_table_dwt = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.get_default_dtype())
        # q_table_dwt = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.get_default_dtype())
        q_list_dct = [25, 25, 25, 25, 25, 25, 25, 25,
                      25, 25, 25, 25, 25, 25, 25]

        q_table_dwt = q_table_dwt * 255 / maximum_fm[i]

        compress_seq = [
            # Transform(channel[i]).cuda(),
            QuantiUnsign(bit=bit, q_factor=q_factor, is_shift=False).cuda(),
            FtMapShiftNorm(),
            # CompressDCT(q_table=utils.q_table_dct_gen(q_list_dct)).cuda(),
            CompressDWT(level=3, bit=bit, q_table=q_table_dwt, wave=wavelet).cuda()
        ]

        compress_list.append(Compress(BypassSequential(*compress_seq)))

    q_factor = maximum_fm[-1] / (2 ** bit - 1)
    q_table_dwt = torch.tensor([10 ** 6, 10 ** 6, 10 ** 6, 1], dtype=torch.get_default_dtype())
    q_list_dct = [25, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6,
                  10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6]

    q_table_dwt = q_table_dwt * 255 / maximum_fm[-1]

    compress_seq = [
        QuantiUnsign(bit=bit, q_factor=q_factor).cuda(),
        # CompressDCT(q_table=utils.q_table_dct_gen(q_list_dct)).cuda(),
        CompressDWT(level=3, bit=bit, q_table=q_table_dwt, wave='haar').cuda()
    ]

    compress_list.append(Compress(BypassSequential(*compress_seq)))

    return compress_list


if __name__ == '__main__':
    main()

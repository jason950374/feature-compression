import numpy as np
import torch.cuda
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

from compress_setups import compress_list_gen_branch
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
parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate for batch_size=256')
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
parser.add_argument('--wavelet', type=str, default="db2", help='Mother wavelet for DWT')
parser.add_argument('--k', type=int, default=0, help="k for exponential-Golomb")
parser.add_argument('--l1_coe', type=float, default=0, help="coefficient of L1 regularizer for sparsity")
parser.add_argument('--bit', type=int, default=8, help="coefficient of L1 regularizer for sparsity")
parser.add_argument('--norm_mode', type=str, default='l1', help="l1, l2, sum, otherwise no normalize")
parser.add_argument('--rand_factor', type=float, default=0, help="rand_factor")
parser.add_argument('--tauMask', type=float, default=2, help="tau for softmin in MaskCompressDWT")
parser.add_argument('--tauLoss', type=float, default=2, help="tau for tanh in sparsity loss")
parser.add_argument('--retainRatio', type=float, default=0.75, help="retaining ratio for MaskCompressDWT")
parser.add_argument('--biloss_coe', type=float, default=1e-3, help="bipolar Loss for s")
parser.add_argument('--gamma', type=float, default=0.1, help="gamma")

args = parser.parse_args()
# args.save = 'ckpts/retrain_{}_{}'.format(args.wavelet, args.load[6:-9])
args.save = 'ckpts/retrain_tauLoss_long_{}_biLoss_{}_lr_{}_spLoss_{}_retainRatio_{}_{}'.format(
    args.tauLoss, args.biloss_coe, args.learning_rate, args.l1_coe, args.retainRatio, time.strftime("%m%d_%H%M%S"))
# args.save = 'ckpts/retrain_long_lr_{}_{}'.format(args.learning_rate, time.strftime("%m%d_%H%M%S"))

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
    args.epochs = 80
    args.usage_weight = 2
    utils.multiply_adds = 2
elif args.dataset == 'cifar100':
    args.num_classes = 100
    args.epochs = 50
    args.usage_weight = 2
    utils.multiply_adds = 2
elif args.dataset == 'imageNet':
    args.num_classes = 1000
    args.epochs = 50
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
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    cudnn.benchmark = True

    logging.info("args = %s", args)

    # Set the data loader
    train_queue, test_queue = utils.get_loader(args)

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

    if args.dataset == 'cifar10':
        # quick test for this ckpts: cifar10_resnet20_0409_184724
        maximum_fm_branch = [5.2, 6.7, 5.3, 5.8, 6.7, 7.6, 4.6, 5.7, 36]
        dwt_coe_branch = [0.018, 0.021, 0.022, 0.020, 0.019, 0.017, 0.018, 0.014, 0.047]
        maximum_fm_block = [2.7, 2.9, 2.7, 2.5, 2.3, 2.3, 2.7, 2.7, 3.4]
        channel = [16, 16, 16, 32, 32, 32, 64, 64, 64]
    elif args.dataset == 'imageNet':
        # quick test for pretrain resnet18
        maximum_fm_branch = [11, 15.5, 14, 11.5, 8.5, 14, 11.5, 101]
        dwt_coe_branch = [0.010, 0.013, 0.0094, 0.012, 0.010, 0.0061, 0.0069, 0.036]
        channel = [64, 64, 128, 128, 256, 256, 512, 512]
    else:
        raise NotImplementedError(
            '{} dataset is not supported. Only support cifar10, cifar100 and imageNet.'.format(args.dataset))

    compress_list = compress_list_gen_branch(channel, maximum_fm_branch, args.wavelet, args.bit,
                                             norm_mode=args.norm_mode,
                                             retain_ratio=args.retainRatio,
                                             tau_mask=args.tauMask,
                                             dwt_coe_branch=dwt_coe_branch)

    model.compress_replace_branch(compress_list)

    utils.save_checkpoint(model, False, args.save, 0)

    # Set the optimizer

    settings = [
        {
            'setting_names': utils.get_param_names(model, 'freq_select'),
            'lr': 0, # args.learning_rate,
            'weight_decay': 0,
            'momentum': 0
        },
        {
            'setting_names': utils.get_param_names(model, 'transform_matrix'),
            'lr': 0,  # args.learning_rate,
            'weight_decay': 0,
            'momentum': 0
        },
    ]
    params = utils.optimizer_setting_separator(model, settings)

    optimizer = torch.optim.SGD(
        params,
        # model.parameters(),
        # lr=args.learning_rate,
        lr=0,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Set the objective function
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    best_test_acc = 0.0

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        milestones = [40, 70]
    elif args.dataset == 'imageNet':
        milestones = [20, 40]
    else:
        milestones = None

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
    length_code_dict = utils.gen_signed_seg_dict(args.k, 2 ** (args.bit - 1), len_key=True)
    # lr_list = [0, 0.0001, 0.001, 0.01, 0.1]
    '''
    for epoch in range(0, 10):
        # optimizer.param_groups[0]['initial_lr'] = args.learning_rate * lr_list[epoch]
        # optimizer.param_groups[0]['lr'] = args.learning_rate * lr_list[epoch]
        is_best = False
        logging.info('[Warm] Epoch = %d', epoch)
        train(train_queue, model, criterion, optimizer, epoch, args, milestones, length_code_dict)

        for compress in model.stages.compress:
            indx = compress.compress.module[-1].reorder()
            compress.compress.separate.reorder(indx)
        model.update()
        this_acc, _ = infer(test_queue, model)

        if this_acc > best_test_acc:
            best_test_acc = this_acc
            is_best = True

        logging.info(
            '[Test] Epoch:%d/%d acc %.2f%%; best %.2f%%', epoch, args.epochs, this_acc, best_test_acc)

        logging.info('Saved into %s', args.save)

        utils.save_checkpoint(model, is_best, args.save, epoch)
        logging.info('============================================================================')'''

    optimizer.param_groups[0]['initial_lr'] = args.learning_rate
    optimizer.param_groups[0]['lr'] = args.learning_rate

    for epoch in range(0, args.epochs):
        scheduler.step()
        if epoch <= 5:
            optimizer.param_groups[2]['initial_lr'] = args.learning_rate * (10 ** (epoch - 5))
            optimizer.param_groups[2]['lr'] = args.learning_rate * (10 ** (epoch - 5))
        logging.info('[Train] Epoch = %d , LR = %e', epoch, scheduler.get_lr()[0])
        is_best = False
        train(train_queue, model, criterion, optimizer, epoch, args, milestones, length_code_dict)

        # Evaluate the test accuracy
        if epoch > args.epochs_test:
            if args.l1_coe > 1e-20:
                for compress in model.stages.compress:
                    indx = compress.compress.module[-1].reorder()
                    compress.compress.separate.reorder(indx)
            else:
                for compress in model.stages.compress:
                    indx = compress.compress[-1].reorder()
                    compress.compress[0].reorder(indx)
            model.update()
            this_acc, _ = infer(test_queue, model)

            if this_acc > best_test_acc:
                best_test_acc = this_acc
                is_best = True

            logging.info(
                '[Test] Epoch:%d/%d acc %.2f%%; best %.2f%%', epoch, args.epochs, this_acc, best_test_acc)

            logging.info('Saved into %s', args.save)

            utils.save_checkpoint(model, is_best, args.save, epoch)
            logging.info('============================================================================')


def train(train_queue, model, criterion, optimizer, cur_epoch, args, milestones, length_code_dict, warm_up=False):
    biloss_meter = meter.AverageMeter()
    classloss_meter = meter.AverageMeter()
    sp_loss_meter = meter.AverageMeter()
    detloss_meter = meter.AverageMeter()
    top1 = meter.AverageMeter()
    top5 = meter.AverageMeter()
    model.train(True)
    total_step = len(train_queue)

    end = time.time()

    total_epoch = 5 if warm_up else args.epochs
    suffix = 'Warm Up' if warm_up else 'Train'
    biloss_coe = 0
    if cur_epoch < 5:
        biloss_coe = args.biloss_coe * (10 ** (cur_epoch - 5))
    else:
        biloss_coe = args.biloss_coe

    for step, (x, target) in enumerate(train_queue):
        data_time.update(time.time() - end)

        x = Variable(x).cuda()
        target = Variable(target).cuda(async=True)

        # Forward propagation
        logits, _, fm_transforms, _, _ = model(x)

        loss = 0
        classloss = criterion(logits, target)
        loss += classloss
        n = x.size(0)

        if args.l1_coe > 1e-20:
            # l1 = utils.iterable_l1(fm_transforms)
            freq_selects = [compress.compress.module[-1].freq_select.detach() for compress in model.stages.compress]
            channel_weight = [utils.freq_select2channel_weight(freq_select, ratio=args.retainRatio, tau=args.tauLoss)
                              for freq_select in freq_selects]

            sp_loss = utils.dwt_channel_weighted_loss(fm_transforms, channel_weight) * args.l1_coe
            loss += sp_loss

            detloss = 0
            # margin = 0.01
            margin = 0.1
            for compress in model.stages.compress:
                m = compress.compress.separate.transform_matrix
                m = m / m.abs().sum(dim=1, keepdim=True)
                det = torch.det(m)
                factor = det.abs().detach() * m.size(0)
                detloss += ((F.relu(margin - det.abs()) / factor) * args.batch_size * 1e-2)
            loss += detloss

            biloss = 0
            for compress in model.stages.compress:
                if abs(args.retainRatio - 0.5) < 0.000001:
                    threshold = compress.compress.module[-1].freq_select.median()
                else:
                    kth = int(compress.compress.module[-1].freq_select.size(0) * args.retainRatio)
                    threshold, _ = torch.kthvalue(compress.compress.module[-1].freq_select, kth)
                biloss += utils.bipolar(compress.compress.module[-1].freq_select,
                                        threshold.detach())
            loss += (biloss * args.biloss_coe)

            sp_loss_meter.update(sp_loss.item(), n)
            detloss_meter.update(detloss.item() if isinstance(detloss, torch.Tensor) else detloss, n)
        else:
            biloss = 0
            for compress in model.stages.compress:
                if abs(args.retainRatio - 0.5) < 0.000001:
                    threshold = compress.compress[-1].freq_select.median()
                else:
                    kth = int(compress.compress[-1].freq_select.size(0) * args.retainRatio)
                    threshold, _ = torch.kthvalue(compress.compress[-1].freq_select, kth)
                    threshold2, _ = torch.kthvalue(compress.compress[-1].freq_select, kth + 1)
                    threshold = (threshold + threshold2) / 2
                biloss += utils.bipolar(compress.compress[-1].freq_select,
                                        threshold.detach())
            loss += (biloss * biloss_coe)

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        param_trm = utils.get_params(model, "transform_matrix")
        nn.utils.clip_grad_norm_(param_trm, 0.1)
        optimizer.step()

        model.update()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

        biloss_meter.update(biloss.item(), n)
        classloss_meter.update(classloss.item(), n)

        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.report_freq == 0:
            logging.info(
                '[%s] Epoch:%d/%d Step:%d/%d biloss:%.2e classloss:%.2e sp_loss:%.2e detloss:%.2e Top1_acc:%.1f%% Top5_acc:%.1f%%',
                suffix, cur_epoch, total_epoch, step, total_step, biloss_meter.avg, classloss_meter.avg,
                sp_loss_meter.avg,
                detloss_meter.avg, top1.avg, top5.avg)
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

            logits, _, _, _, _ = model(x)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = x.size(0)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    return top1.avg, top5.avg


if __name__ == '__main__':
    main()

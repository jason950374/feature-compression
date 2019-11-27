from .utils import load_dataset_n_pretrain_model, infer_in_train
import torch.cuda
import torch.optim
import glob
from utils import utils, meter
import argparse
import logging
import sys
import os
import time

from .compress_setups import compress_list_gen_branch
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
parser.add_argument('--tauLoss', type=float, default=2, help="tau for tanh in sparsity loss")
parser.add_argument('--gamma', type=float, default=0.1, help="gamma for lr_scheduler")

args = parser.parse_args()
args.save = 'ckpt_new_cifar/retrain_lr_{}_{}_{}'.format(args.learning_rate, args.wavelet, time.strftime("%m%d_%H%M%S"))

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

    logging.info("args = %s", args)

    train_queue, test_queue, model = load_dataset_n_pretrain_model(args)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    if args.epoch_start == 0:
        if args.dataset == 'imageNet':
            net_dic = torch.load(args.load)
            net_dic_fix = utils.imagenet_model_graph_mapping(net_dic, [2, 2, 2, 2])
            model.load_state_dict(net_dic_fix)
        else:
            utils.load(model, args)
        args.epoch_start = 0

    freq_selects = None
    if args.dataset == 'cifar10':
        # quick test for this ckpts: cifar10_resnet20_0409_184724
        maximum_fm_branch = [5.2, 6.7, 5.3, 5.8, 6.7, 7.6, 4.6, 5.7, 36]
        dwt_coe_branch = [0.018, 0.021, 0.022, 0.020, 0.019, 0.017, 0.018, 0.014, 0.047]
        maximum_fm_block = [2.7, 2.9, 2.7, 2.5, 2.3, 2.3, 2.7, 2.7, 3.4]
        dwt_coe_block = [0.015, 0.017, 0.024, 0.030, 0.019, 0.012, 0.029, 0.014, 0.0060]
        channel = [16, 16, 16, 32, 32, 32, 64, 64, 64]
    elif args.dataset == 'imageNet':
        # quick test for pretrain resnet18
        maximum_fm_branch = [11, 15.5, 14, 11.5, 8.5, 14, 11.5, 101]
        dwt_coe_branch = [0.010, 0.013, 0.0094, 0.012, 0.010, 0.0061, 0.0069, 0.036]
        maximum_fm_block = []
        dwt_coe_block = []
        channel = [64, 64, 128, 128, 256, 256, 512, 512]
    else:
        raise NotImplementedError(
            '{} dataset is not supported. Only support cifar10, cifar100 and imageNet.'.format(args.dataset))

    compress_list_branch = compress_list_gen_branch(channel, maximum_fm_branch, args.wavelet, args.bit,
                                                    norm_mode=args.norm_mode,
                                                    dwt_coe_branch=dwt_coe_branch)

    '''compress_list_block = compress_list_gen_block(channel, maximum_fm_block, args.wavelet, args.bit,
                                                  norm_mode=args.norm_mode,
                                                  dwt_coe_block=dwt_coe_block)'''

    model.compress_replace_branch(compress_list_branch)
    # model.compress_replace_inblock(compress_list_block)

    utils.save_checkpoint(model, False, args.save)

    # Set the optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    if args.epoch_start != 0:
        utils.load(model, args, optimizer)

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
    model.update()

    for epoch in range(0, args.epochs):
        logging.info('[Train] Epoch = %d , LR = %e', epoch, scheduler.get_lr()[0])
        is_best = False
        train(train_queue, model, criterion, optimizer, epoch, args, milestones, length_code_dict)
        scheduler.step()

        # Evaluate the test accuracy
        if epoch > args.epochs_test:
            this_acc, _ = infer_in_train(test_queue, model)

            if this_acc > best_test_acc:
                best_test_acc = this_acc
                is_best = True

            logging.info(
                '[Test] Epoch:%d/%d acc %.2f%%; best %.2f%%', epoch, args.epochs, this_acc, best_test_acc)

            logging.info('Saved into %s', args.save)

            utils.save_checkpoint(model, is_best, args.save, optimizer)
            logging.info('============================================================================')


def train(train_queue, model, criterion, optimizer, cur_epoch, args, milestones, length_code_dict, warm_up=False):
    classloss_meter = meter.AverageMeter()
    top1 = meter.AverageMeter()
    top5 = meter.AverageMeter()
    model.train(True)
    total_step = len(train_queue)

    end = time.time()

    total_epoch = 5 if warm_up else args.epochs
    suffix = 'Warm Up' if warm_up else 'Train'

    for step, (x, target) in enumerate(train_queue):
        data_time.update(time.time() - end)

        x = Variable(x).cuda()
        target = Variable(target).cuda(async=True)

        # Forward propagation
        logits, _, fm_transforms, _, _ = model(x)

        classloss = criterion(logits, target)
        loss = classloss
        n = x.size(0)

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        param_trm = utils.get_params(model, "transform_matrix")
        nn.utils.clip_grad_norm_(param_trm, 0.1)
        optimizer.step()

        model.update()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        classloss_meter.update(classloss.item(), n)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.report_freq == 0:
            logging.info(
                '[%s] Epoch:%d/%d Step:%d/%d classloss:%.2e Top1_acc:%.1f%% Top5_acc:%.1f%%',
                suffix, cur_epoch, total_epoch, step, total_step, classloss_meter.avg,
                top1.avg, top5.avg)
            time_remain = utils.getTime((((args.epochs - cur_epoch) * total_step) - step) * batch_time.avg)
            logging.info('[{}] Time: {:.4f} Data: {:.4f} Time remaining: {}'.format(
                suffix, batch_time.avg, data_time.avg, time_remain))


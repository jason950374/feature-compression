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
from model.ResNet import ResNetCifar
from torch.autograd import Variable

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--data', type=str, default='/home/jason/data/', help='location of the data corpus relative to home')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate for batch_size=256')
parser.add_argument('--workers', type=int, default=4, help='workers for data loader')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=350, help='num of training epochs')
parser.add_argument('--epochs_test', type=int, default=0, help='num of epochs begin to test')
parser.add_argument('--seed', type=int, default=-1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--depth', type=int, default=20, help='Depth of base resnet model')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes of now dataset.')
parser.add_argument('--load', type=str, default="")
parser.add_argument('--epoch_start', type=int, default=0, help='Epoch number begin')

args = parser.parse_args()
args.save = 'ckpts/{}_resnet{}_{}'.format(args.dataset, args.depth, time.strftime("%m%d_%H%M%S"))
args.learning_rate = args.learning_rate * args.batch_size / 256 / 1024


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
    args.epochs = 350
    args.usage_weight = 2
    utils.multiply_adds = 2
elif args.dataset == 'cifar100':
    args.num_classes = 100
    args.epochs = 350
    args.usage_weight = 2
    utils.multiply_adds = 2
elif args.dataset == 'imageNet':
    args.num_classes = 1000
    args.epochs = 100
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

    utils.save_checkpoint(model, False, args.save, 0)

    # Set the optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Set milestones for the learning rate scheduler
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        args.milestones = [150, 250, 350]
    elif args.dataset == 'imageNet':
        args.milestones = [30, 60, 90]
    else:
        args.milestones = None

    warm_up_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0, 1, 2, 3, 4], gamma=4)

    if args.load:
        utils.load(model, args)

    # Set the objective function
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    best_test_acc = 0.0

    if args.epoch_start == 0:
        for epoch in range(5):  # warm up
            logging.info('[Warm up] Epoch = %d , LR = %e', epoch, warm_up_scheduler.get_lr()[0])
            train(train_queue, model, criterion, optimizer, epoch, args, warm_up=True)
            warm_up_scheduler.step()

    def rescale_lr(opt):
        for param_group in opt.param_groups:
                param_group['initial_lr'] = args.learning_rate * 1024
                param_group['lr'] = args.learning_rate * 1024

    rescale_lr(optimizer)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones)

    # if restore from ckpt, step scheduler to current epoch
    for epoch in range(args.epoch_start):
        scheduler.step()

    for epoch in range(args.epoch_start, args.epochs):
        scheduler.step()  # Update learning rate at the beginning of each epoch
        logging.info('[Train] Epoch = %d , LR = %e', epoch, scheduler.get_lr()[0])
        is_best = False
        train(train_queue, model, criterion, optimizer, epoch, args,)

        # Evaluate the test accuracy
        if epoch > args.epochs_test:
            test_acc, test_acc_5 = infer(test_queue, model)

            this_acc = test_acc

            if this_acc > best_test_acc:
                best_test_acc = this_acc
                is_best = True

            logging.info(
                '[Test] Epoch:%d/%d acc %.2f%%; best %.2f%%', epoch, args.epochs, this_acc, best_test_acc)

            logging.info('Saved into %s', args.save)

            utils.save_checkpoint(model, is_best, args.save, epoch)
            logging.info('============================================================================')


def train(train_queue, model, criterion, optimizer, cur_epoch, args, warm_up=False):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train(True)
    total_step = len(train_queue)

    end = time.time()

    total_epoch = 5 if warm_up else args.epochs
    suffix = 'Warm Up' if warm_up else 'Train'

    for step, (x, target) in enumerate(train_queue):
        time.time()
        data_time.update(time.time() - end)

        x = Variable(x).cuda()
        target = Variable(target).cuda(async=True)

        # Forward propagation
        logits, _ = model(x)
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

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
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (x, target) in enumerate(test_queue):
            x = Variable(x).cuda()
            target = Variable(target).cuda(async=True)

            logits, _ = model(x)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = x.size(0)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    return top1.avg, top5.avg


if __name__ == '__main__':
    main()

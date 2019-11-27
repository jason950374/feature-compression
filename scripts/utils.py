import random

import numpy as np
import torch.cuda
from torch.autograd import Variable
from torch.backends import cudnn as cudnn

from model import ResNetCifar, resnet18
from utils import utils, meter


def load_dataset_n_pretrain_model(args):
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True

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

    return train_queue, test_queue, model


def infer_in_train(test_queue, model):
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
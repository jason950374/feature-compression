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
from model.compress import *
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
], dtype=torch.get_default_dtype())

# torch.set_default_dtype(torch.float64)


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
    # _, maximum_fm = utils.get_q_range(train_queue, model)
    maximum_fm = [5.2, 6.7, 5.3, 5.8, 6.7, 7.6, 4.6, 5.7, 36]  # quick test for this ckpts
    compress_list = compress_list_gen(args.depth, maximum_fm)

    model.compress_replace(compress_list)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    acc_hd = utils.HandlerAcc()
    fm_hd = utils.HandlerFm()
    # tr_hd = utils.HandlerDCT_Fm()
    tr_hd = utils.HandlerDWT_Fm()
    # tr_hd = utils.HandlerQuanti()

    u_code_length_dict = utils.gen_signed_seg_dict(0, 128)
    code_length_dict = utils.gen_seg_dict(0, 256)
    # code_length_dict = {8: set(range(-128, 127))}
    utils.infer_base(test_queue, model, [acc_hd, fm_hd, tr_hd])
    fm_hd.print_result(print_fn=logging.info)
    tr_hd.print_result(print_fn=logging.info, save=args.save, code_length_dict=u_code_length_dict)
    # tr_hd.print_result(print_fn=logging.info, code_length_dict=code_length_dict)
    acc_hd.print_result(print_fn=logging.info)


def compress_list_gen(depth, maximum_fm):
    encoder_list = []
    decoder_list = []
    for i in range(((depth - 2) // 2) - 1):
        q_factor = maximum_fm[i] / 255

        q_table_dwt = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.get_default_dtype())
        q_list_dct = [150, 50, 50, 50, 50, 50, 50, 50,
                      100, 100, 100, 100, 100, 100, 200]

        q_table_dwt = q_table_dwt * 255 / maximum_fm[i]

        encoder_seq = [
            QuantiUnsign(bit=8, q_factor=q_factor, is_shift=False).cuda(),
            FtMapShiftNorm(),
            # CompressDCT(utils.q_table_dct_gen(q_list_dct)).cuda(),
            CompressDWT(level=3, q_table=q_table_dwt, wave='haar').cuda()
        ]
        decoder_seq = [
            CompressDWT(level=3, q_table=q_table_dwt, wave='haar', is_encoder=False).cuda(),
            # CompressDCT(utils.q_table_dct_gen(q_list_dct), is_encoder=False).cuda(),
            FtMapShiftNorm(is_encoder=False),
            QuantiUnsign(bit=8, q_factor=q_factor, is_encoder=False, is_shift=False).cuda()
        ]

        encoder_list.append(BypassSequential(*encoder_seq))
        decoder_list.append(BypassSequential(*decoder_seq, is_encoder=False))

    q_factor = maximum_fm[-1] / 255
    q_table_dwt = torch.tensor([10 ** 6, 10 ** 6, 10 ** 6, 1], dtype=torch.get_default_dtype())
    # q_table_dwt = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.get_default_dtype())
    q_list_dct = [150, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6,
                  10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6]

    q_table_dwt = q_table_dwt * 255 / maximum_fm[-1]

    encoder_seq = [
        QuantiUnsign(bit=8, q_factor=q_factor).cuda(),
        # CompressDCT(utils.q_table_dct_gen(q_list_dct)).cuda(),
        CompressDWT(level=3, q_table=q_table_dwt, wave='haar').cuda()
    ]
    decoder_seq = [
        CompressDWT(level=3, q_table=q_table_dwt, wave='haar', is_encoder=False).cuda(),
        # CompressDCT(utils.q_table_dct_gen(q_list_dct), is_encoder=False).cuda(),
        QuantiUnsign(bit=8, q_factor=q_factor, is_encoder=False).cuda()
    ]

    encoder_list.append(BypassSequential(*encoder_seq))
    decoder_list.append(BypassSequential(*decoder_seq, is_encoder=False))

    return encoder_list, decoder_list


if __name__ == '__main__':
    main()

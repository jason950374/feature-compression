import numpy as np
import torch.cuda
import glob
import infer_result_handler
import meter
import utils
import argparse
import logging
import sys
import os
import time
import random
import torch.backends.cudnn as cudnn

from compress_setups import compress_list_gen_branch, compress_list_gen_block
from model.ResNet import ResNetCifar, resnet18
from model.compress import *
import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser("infer")
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
# parser.add_argument('--data', type=str, default='/home/jason/data/',
#                     help='location of the data corpus relative to home')
parser.add_argument('--data', type=str, default='/home/gasoon/datasets',
                    help='location of the data corpus relative to home')
parser.add_argument('--workers', type=int, default=4, help='workers for data loader')
parser.add_argument('--report_freq', type=float, default=400, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=-1, help='random seed')
parser.add_argument('--depth', type=int, default=20, help='Depth of base resnet model')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes of now dataset.')
parser.add_argument('--load', type=str, default="")
parser.add_argument('--wavelet', type=str, default="db2", help='Mother wavelet for DWT')
parser.add_argument('--k', type=int, default=1, help="k for exponential-Golomb")
parser.add_argument('--bit', type=int, default=8, help="coefficient of L1 regularizer for sparsity")
parser.add_argument('--norm_mode', type=str, default='l1', help="coefficient of L1 regularizer for sparsity")
parser.add_argument('--retainRatio', type=float, default=0.5, help="retaining ratio for MaskCompressDWT")

args = parser.parse_args()
args.save = 'ckpts/test_{}_resnet{}_{}'.format(args.dataset, args.depth, time.strftime("%m%d_%H%M%S"))


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
    args.usage_weight = 2
    utils.multiply_adds = 2
elif args.dataset == 'cifar100':
    args.num_classes = 100
    args.usage_weight = 2
    utils.multiply_adds = 2
elif args.dataset == 'imageNet':
    args.num_classes = 1000
    args.usage_weight = 1
    utils.multiply_adds = 1
else:
    raise NotImplementedError(
        '{} dataset is not supported. Only support cifar10, cifar100 and imageNet.'.format(args.dataset))

# torch.set_default_dtype(torch.float64)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    if args.seed >= 0:
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

    '''
    if args.dataset == 'imageNet':
        net_dic = torch.load(args.load)
        net_dic_fix = utils.imagenet_model_graph_mapping(net_dic, [2, 2, 2, 2])
        model.load_state_dict(net_dic_fix)
    else:
        utils.load(model, args)'''

    # Insert compress_block after load since compress_block not include in training phase in this case
    '''
    hd_maximum_fm = infer_result_handler.HandlerFm(print_fn=logging.info, print_sparsity=False)
    utils.infer_base(train_queue, model, [hd_maximum_fm, acc_hd])
    hd_maximum_fm.print_result()
    maximum_fm_branch = hd_maximum_fm.max_mins_branch.copy()
    maximum_fm_block = hd_maximum_fm.max_mins_block.copy()
    print([maxMinMeter.max for maxMinMeter in maximum_fm_branch])
    print([maxMinMeter.min for maxMinMeter in maximum_fm_branch])
    print([maxMinMeter.max for maxMinMeter in maximum_fm_block])
    print([maxMinMeter.min for maxMinMeter in maximum_fm_block])'''

    # quick test for this ckpts: cifar10_resnet20_0409_184724
    maximum_fm_branch = [5.2, 6.7, 5.3, 5.8, 6.7, 7.6, 4.6, 5.7, 36]
    maximum_fm_block = [2.7, 2.9, 2.7, 2.5, 2.3, 2.3, 2.7, 2.7, 3.4]
    channel = [16, 16, 16, 32, 32, 32, 64, 64, 64]

    # quick test for pretrain resnet18
    # maximum_fm = [11, 15.5, 14, 11.5, 8.5, 14, 11.5, 101]
    # channel = [64, 64, 128, 128, 256, 256, 512, 512]

    compress_list_branch = compress_list_gen_branch(channel, maximum_fm_branch, args.wavelet, args.bit,
                                                    norm_mode=args.norm_mode,
                                                    retain_ratio=args.retainRatio)

    compress_list_block = compress_list_gen_block(channel, maximum_fm_block, args.wavelet, args.bit,
                                                  norm_mode=args.norm_mode,
                                                  retain_ratio=args.retainRatio)

    utils.load(model, args)
    model.compress_replace_branch(compress_list_branch)
    model.compress_replace_inblock(compress_list_block)

    '''
    for compress in model.stages.compress:
        # m = compress.compress.separate.transform_matrix.data
        # compress.compress.separate.transform_matrix.data = torch.eye(m.size(0)).cuda()
        indx = compress.compress.module[-1].reorder()
        compress.compress.separate.reorder(indx)
    model.update()'''

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    code_length_dict = utils.gen_signed_seg_dict(args.k, 2 ** (args.bit-1))
    u_code_length_dict = utils.gen_seg_dict(args.k, 2 ** args.bit)
    acc_hd = infer_result_handler.HandlerAcc(print_fn=logging.info)
    fm_hd = infer_result_handler.HandlerFm(print_fn=logging.info)
    # tr_hd = infer_result_handler.HandlerDCT_Fm(print_fn=logging.info, save=args.save, code_length_dict=code_length_dict)
    tr_hd = infer_result_handler.HandlerDWT_Fm(print_fn=logging.info, save=args.save, code_length_dict=code_length_dict)
    # tr_hd = infer_result_handler.HandlerMaskedDWT_Fm(print_fn=logging.info, save=args.save, code_length_dict=code_length_dict)
    # tr_hd = infer_result_handler.HandlerQuanti(print_fn=logging.info, code_length_dict=u_code_length_dict)
    # tr_hd = infer_result_handler.HandlerTrans(print_fn=logging.info)
    # handler_list = [fm_hd, tr_hd, acc_hd]
    handler_list = [tr_hd, acc_hd]

    utils.infer_base(test_queue, model, handler_list)

    for handler in handler_list:
        handler.print_result()
    '''
    for k in range(2, 6):
        logging.info("===========   {}   ===========".format(k))
        # code_length_dict = utils.gen_signed_seg_dict(k, 2 ** (args.bit-1))
        # tr_hd.set_config(code_length_dict)
        u_code_length_dict = utils.gen_seg_dict(k, 256)
        tr_hd.set_config(u_code_length_dict)
        tr_hd.print_result()'''


if __name__ == '__main__':
    main()


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
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=-1, help='random seed')
parser.add_argument('--depth', type=int, default=20, help='Depth of base resnet model')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes of now dataset.')
parser.add_argument('--load', type=str, default="")
parser.add_argument('--wavelet', type=str, default="db1", help='Mother wavelet for DWT')
parser.add_argument('--k', type=int, default=1, help="k for exponential-Golomb")
parser.add_argument('--bit', type=int, default=8, help="coefficient of L1 regularizer for sparsity")

args = parser.parse_args()
args.save = 'ckpts/test_{}_resnet{}_{}_k{}_{}'.format(args.dataset, args.depth,
                                                      args.wavelet, args.k, time.strftime("%m%d_%H%M%S"))


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

    if args.dataset == 'imageNet':
        net_dic = torch.load(args.load)
        net_dic_fix = utils.imagenet_model_graph_mapping(net_dic, [2, 2, 2, 2])
        model.load_state_dict(net_dic_fix)
    else:
        utils.load(model, args)

    # Insert compress_block after load since compress_block not include in training phase in this case
    # hd_maximum_fm = infer_result_handler.HandlerFm(print_fn=logging.info, print_sparsity=False)
    # utils.infer_base(train_queue, model, [hd_maximum_fm])
    # hd_maximum_fm.print_result()
    # maximum_fm = hd_maximum_fm.maximums.copy()

    # quick test for this ckpts: cifar10_resnet20_0409_184724
    maximum_fm = [5.2, 6.7, 5.3, 5.8, 6.7, 7.6, 4.6, 5.7, 36]
    # quick test for pretrain resnet18
    # maximum_fm = [11, 15.5, 14, 11.5, 8.5, 14, 11.5, 101]

    compress_list = compress_list_gen(maximum_fm, args.wavelet, args.bit)

    model.compress_replace(compress_list)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    code_length_dict = utils.gen_signed_seg_dict(args.k, 2 ** (args.bit-1))
    u_code_length_dict = utils.gen_seg_dict(args.k, 2 ** args.bit)
    acc_hd = infer_result_handler.HandlerAcc(print_fn=logging.info)
    # fm_hd = infer_result_handler.HandlerFm(print_fn=logging.info)
    # tr_hd = infer_result_handler.HandlerDCT_Fm(print_fn=logging.info, save=args.save, code_length_dict=code_length_dict)
    # tr_hd = infer_result_handler.HandlerDWT_Fm(print_fn=logging.info, save=args.save, code_length_dict=code_length_dict)
    tr_hd = infer_result_handler.HandlerDWT_Fm(print_fn=logging.info, save=args.save, code_length_dict=code_length_dict)
    # tr_hd = infer_result_handler.HandlerQuanti(print_fn=logging.info, code_length_dict=u_code_length_dict)
    # tr_hd = infer_result_handler.HandlerTrans(print_fn=logging.info)
    handler_list = [tr_hd, acc_hd]

    utils.infer_base(test_queue, model, handler_list)

    for handler in handler_list:
        handler.print_result()

    """
    for k in range(1, 5):
        logging.info("===========   {}   ===========".format(k))
        code_length_dict = utils.gen_signed_seg_dict(k, 2 ** (args.bit-1))
        # u_code_length_dict = utils.gen_seg_dict(k, 256)
        tr_hd.set_config(code_length_dict)
        tr_hd.print_result()"""


def compress_list_gen(maximum_fm, wavelet='db1', bit=8):
    compress_list = []
    channel = [16, 16, 16, 32, 32, 32, 64, 64, 64]
    x_size = [32, 32, 32, 16, 16, 16, 8, 8, 8]

    for i in range(len(maximum_fm) - 1):
        q_factor = maximum_fm[i] / (2 ** bit - 1)

        q_table_dwt = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.get_default_dtype())
        q_list_dct = [25, 25, 25, 25, 25, 25, 25, 25,
                      25, 25, 25, 25, 25, 25, 25]

        q_table_dwt = q_table_dwt * 255 / maximum_fm[i]

        compress_seq = [
            # Transform(channel[i], init_value=U.t() if i < 3 else None).cuda(),
            # Transform(channel[i]).cuda(),
            QuantiUnsign(bit=bit, q_factor=q_factor, is_shift=False).cuda(),
            FtMapShiftNorm(),
            # CompressDCT(q_table=utils.q_table_dct_gen(q_list_dct)).cuda(),
            CompressDWT(level=3, bit=bit, q_table=q_table_dwt, wave=wavelet).cuda()
            # AdaptiveDWT(x_size[i], level=1, bit=bit, q_table=q_table_dwt).cuda()
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


#!/usr/bin/env python

import argparse
import os
import os.path as osp

import fcn
import numpy as np
import skimage.io
import torch
from torch.autograd import Variable
import model.FCN as FCN
import torchfcn
import tqdm
import infer_result_handler
import utils
import time
import logging
import sys

from compress_setups import compress_list_gen_block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', help='Model path')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    save = 'ckpts_voc/test_fcn_{}'.format(time.strftime("%m%d_%H%M%S"))
    utils.create_exp_dir(save)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    root = osp.expanduser('~/data')
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.VOC2011ClassSeg(
            root, split='seg11valid', transform=True),
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)

    n_class = len(val_loader.dataset.class_names)

    if osp.basename(model_file).startswith('fcn32s'):
        model = FCN.FCN32s(n_class=21)
    elif osp.basename(model_file).startswith('fcn16s'):
        model = FCN.FCN16s(n_class=21)
    elif osp.basename(model_file).startswith('fcn8s'):
        if osp.basename(model_file).startswith('fcn8s-atonce'):
            model = FCN.FCN8sAtOnce(n_class=21)
        else:
            model = FCN.FCN8s(n_class=21)
    else:
        raise ValueError
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    model_data = torch.load(model_file)
    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])

    code_length_dict = utils.gen_signed_seg_dict(1, 2 ** (8 - 1))
    u_code_length_dict = utils.gen_seg_dict(1, 2 ** 8)
    hd_maximum_fm = infer_result_handler.HandlerFm(print_fn=logging.info, print_sparsity=False)
    hd_dwt_fm = infer_result_handler.HandlerDWT_Fm(print_fn=logging.info,
                                                   code_length_dict=code_length_dict,
                                                   save=save)
    hd_quan_fm = infer_result_handler.HandlerQuanti(print_fn=logging.info, save=save,
                                                    code_length_dict=u_code_length_dict)
    hd_quad_tree = infer_result_handler.HandlerQuadTree(print_fn=logging.info)

    hd_list = [hd_maximum_fm, hd_quan_fm, hd_quad_tree]

    channel = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    a = 1
    maximum_fm = [1100 * a, 4800 * a, 8800 * a, 13200 * a, 20500 * a, 20100 * a, 17200 * a, 18800 * a, 7400 * a,
                  4500 * a, 2400 * a, 1200 * a, 490 * a]
    compress_list_block = compress_list_gen_block(channel, maximum_fm, bit=8)
    model.compress_replace(compress_list_block)

    model.eval()

    print('==> Evaluating with VOC2011ClassSeg seg11valid')
    visualizations = []
    label_trues, label_preds = [], []
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(val_loader),
                                                   total=len(val_loader),
                                                   ncols=80, leave=False):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            score, fms, tr_fms = model(data)

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()

            for hd in hd_list:
                hd.update_batch((score, fms, tr_fms, None, None))

            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = val_loader.dataset.untransform(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class,
                        label_names=val_loader.dataset.class_names)
                    visualizations.append(viz)
    metrics = torchfcn.utils.label_accuracy_score(
        label_trues, label_preds, n_class=n_class)
    metrics = np.array(metrics)
    metrics *= 100

    for hd in hd_list:
        hd.print_result()

    logging.info('''\
Accuracy: {0}
Accuracy Class: {1}
Mean IU: {2}
FWAV Accuracy: {3}'''.format(*metrics))

    viz = fcn.utils.get_tile_image(visualizations)
    skimage.io.imsave('{}/viz_evaluate.png'.format(save), viz)


if __name__ == '__main__':
    main()

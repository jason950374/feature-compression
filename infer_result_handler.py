import torch
from matplotlib import pyplot as plt

from utils import AverageMeter, HistMeter, accuracy


class InferResultHandler:
    def update_batch(self, result, inputs=None):
        raise NotImplementedError

    def print_result(self):
        raise NotImplementedError

    def reset(self):
        self.__init__()

    def set_config(self, *args):
        raise NotImplementedError


class HandlerAcc(InferResultHandler):
    def __init__(self, print_fn=None):
        # State
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

        # IO
        if print_fn is not None:
            self.print_fn = print_fn
        else:
            self.print_fn = print

    def update_batch(self, result, inputs=None):
        assert inputs is not None, "HandlerAcc need target label"
        (x, target) = inputs
        logits_batch, _, _ = result
        prec1, prec5 = accuracy(logits_batch, target, topk=(1, 5))
        n = logits_batch.size(0)
        self.top1.update(prec1.item(), n)
        self.top5.update(prec5.item(), n)

    def print_result(self):
        self.print_fn('[Test] acc %.2f%%;', self.top1.avg)

    def set_config(self, *args):
        pass


class HandlerFm(InferResultHandler):
    def __init__(self, print_fn=None):
        # State
        self.zero_cnt = 0
        self.size_flat = 0
        self.maximum = -2 ** 40
        self.minimum = 2 ** 40

        # IO
        if print_fn is not None:
            self.print_fn = print_fn
        else:
            self.print_fn = print

    def update_batch(self, result, inputs=None):
        _, feature_maps_batch, _ = result
        for feature_map_batch in feature_maps_batch:
            self.zero_cnt += (feature_map_batch.cuda().abs() < 10 ** -10).sum().item()
            self.size_flat += feature_map_batch.view(-1).size(0)
            self.maximum = max(self.maximum, feature_map_batch.cuda().max())
            self.minimum = min(self.minimum, feature_map_batch.cuda().min().item())

    def print_result(self):
        self.print_fn("==============================================================")
        self.print_fn("feature_maps == 0: {}".format(self.zero_cnt))
        self.print_fn("feature_maps size: {}".format(self.size_flat))
        self.print_fn("sparsity: {}".format(self.zero_cnt / self.size_flat))
        self.print_fn("range ({}, {})".format(self.minimum, self.maximum))
        self.print_fn("==============================================================")

    def set_config(self, *args):
        pass


class HandlerQuanti(InferResultHandler):
    def __init__(self, print_fn=None, code_length_dict=None):
        # State
        self.states_updated = False
        self.zero_cnt = 0
        self.size_flat = 0
        self.maximum = -2 ** 40
        self.minimum = 2 ** 40
        self.code_len = 0
        if code_length_dict is not None:
            self.hist_meter = HistMeter(code_length_dict)

        # IO
        if print_fn is not None:
            self.print_fn = print_fn
        else:
            self.print_fn = print

        # Config
        self.code_length_dict = code_length_dict

    def update_batch(self, result, inputs=None):
        _, _, fm_transforms_batch = result
        for feature_map_batch in fm_transforms_batch:
            self.zero_cnt += (feature_map_batch.cuda().abs() < 10 ** -10).sum().item()
            self.size_flat += feature_map_batch.view(-1).size(0)
            self.maximum = max(self.maximum, feature_map_batch.cuda().max())
            self.minimum = min(self.minimum, feature_map_batch.cuda().min().item())
            self.hist_meter.update(feature_map_batch.cuda())
        self.states_updated = False

    def print_result(self):
        if (self.code_length_dict is not None) or (not self.states_updated):
            self.code_len = self.hist_meter.get_bit_cnt(self.code_length_dict)
        self.states_updated = True
        self.print_fn("==============================================================")
        self.print_fn("feature_maps == 0: {}".format(self.zero_cnt))
        self.print_fn("feature_maps size: {}".format(self.size_flat))
        self.print_fn("sparsity: {}".format(self.zero_cnt / self.size_flat))
        self.print_fn("range ({}, {})".format(self.minimum, self.maximum))
        self.print_fn("code length {}".format(self.code_len))
        self.print_fn("Compress rate {}/{}= {}".format(self.size_flat * 8, self.code_len,
                                                       self.size_flat * 8 / self.code_len))
        self.print_fn("==============================================================")

    def set_config(self, code_length_dict=None):
        self.code_length_dict = code_length_dict
        if (self.code_length_dict is not None) or (not self.states_updated):
            self.code_len = self.hist_meter.get_bit_cnt(self.code_length_dict)
        self.states_updated = True


class HandlerDWT_Fm(InferResultHandler):
    def __init__(self, print_fn=None, plt_fn=None, save="", code_length_dict=None):
        self.states_updated = False
        self.zero_cnt = 0
        self.size_flat = 0
        self.maximum = -2 ** 40
        self.minimum = 2 ** 40
        self.code_len = 0
        self.hist_meters = []

        # IO
        if print_fn is not None:
            self.print_fn = print_fn
        else:
            self.print_fn = print

        if plt_fn is not None:
            self.plt_fn = plt_fn
        else:
            self.plt_fn = plt

        self.save = save

        # Config
        self.code_length_dict = code_length_dict

    def update_batch(self, result, inputs=None):
        _, _, fm_transforms_batch = result
        assert len(fm_transforms_batch) != 0, "Nothing in fm_transforms_batch!!!"
        self.states_updated = False

        for layer_num, feature_map_batch in enumerate(fm_transforms_batch):
            assert type(feature_map_batch) is tuple, "fm_transform_batch is not tuple, make sure it's DWT"

            XL, XH = feature_map_batch

            self.zero_cnt += (XL.cuda().abs() < 10 ** -10).sum().item()
            self.size_flat += XL.view(-1).size(0)
            self.maximum = max(self.maximum, XL.cuda().max().item())
            self.minimum = min(self.minimum, XL.cuda().min().item())
            if len(self.hist_meters) <= layer_num:
                self.hist_meters.append((HistMeter(self.code_length_dict), []))

            self.hist_meters[layer_num][0].update(XL.cuda())

            for i, xh in enumerate(XH):
                self.zero_cnt += (xh.cuda().abs() < 10 ** -10).sum().item()
                self.size_flat += xh.view(-1).size(0)
                self.maximum = max(self.maximum, xh.cuda().max().item())
                self.minimum = min(self.minimum, xh.cuda().min().item())

                if len(self.hist_meters[layer_num][1]) <= i:
                    self.hist_meters[layer_num][1].append(HistMeter(self.code_length_dict))

                self.hist_meters[layer_num][1][i].update(xh.cuda())

    def print_result(self):
        if not self.states_updated:
            assert self.hist_meters is not None, "Please update before print"
            self.code_len = 0
            for layer_num, hist_layer in enumerate(self.hist_meters):
                xl_hist, xh_hists = hist_layer
                xl_hist.plt_hist(plt_fn=self.plt_fn)
                self.plt_fn.savefig('{}/Layer{}_XL.png'.format(self.save, layer_num))
                self.plt_fn.clf()

                if self.code_length_dict is not None:
                    self.code_len += xl_hist.get_bit_cnt(self.code_length_dict)

                for i, xh_hist in enumerate(xh_hists):
                    xh_hist.plt_hist(plt_fn=self.plt_fn)
                    self.plt_fn.savefig('{}/Layer{}_XH_{}.png'.format(self.save, layer_num, i))
                    self.plt_fn.clf()
                    if self.code_length_dict is not None:
                        self.code_len += xh_hist.get_bit_cnt(self.code_length_dict)

            self.states_updated = True

        self.print_fn("==============================================================")
        self.print_fn("fm_transforms == 0: {}".format(self.zero_cnt))
        self.print_fn("fm_transforms size: {}".format(self.size_flat))
        self.print_fn("transform sparsity: {}".format(self.zero_cnt / self.size_flat))
        self.print_fn("transform range ({}, {})".format(self.minimum, self.maximum))
        self.print_fn("code length {}".format(self.code_len))
        self.print_fn("Compress rate {}/{}= {}".format(self.size_flat * 8, self.code_len,
                                                       self.size_flat * 8 / self.code_len))
        self.print_fn("==============================================================")

    def set_config(self, code_length_dict=None):
        self.states_updated = False
        self.code_length_dict = code_length_dict


# TODO fix memory eater
class HandlerDCT_Fm(InferResultHandler):
    def __init__(self, print_fn=None, plt_fn=None, save="", code_length_dict=None):
        self.fm_transforms = None
        self.states_updated = True
        self.zero_cnt = 0
        self.size_flat = 0
        self.maximum = -2 ** 40
        self.minimum = 2 ** 40
        self.code_len = 0

        # IO
        if print_fn is not None:
            self.print_fn = print_fn
        else:
            self.print_fn = print

        if plt_fn is not None:
            self.plt_fn = plt_fn
        else:
            self.plt_fn = plt

        self.save = save

        # Config
        self.code_length_dict = code_length_dict

    def update_batch(self, result, inputs=None):
        _, _, fm_transforms_batch = result
        assert len(fm_transforms_batch) != 0, "Nothing in fm_transforms_batch!!!"
        self.states_updated = False
        if self.fm_transforms is not None:
            fm_transforms = []
            for fm_transform, fm_transform_batch in zip(self.fm_transforms, fm_transforms_batch):
                if type(fm_transform_batch) is tuple:
                    XL = torch.cat((fm_transform[0], fm_transform_batch[0]))
                    XH = []
                    for XH_level, XH_level_batch in zip(fm_transform[1], fm_transform_batch[1]):
                        XH.append(torch.cat((XH_level, XH_level_batch)))

                    fm_transforms.append((XL, XH))
                else:
                    fm_transforms.append(torch.cat((fm_transform, fm_transform_batch)))
        else:
            fm_transforms = fm_transforms_batch

        self.fm_transforms = fm_transforms

    def print_result(self):
        assert self.fm_transforms is not None, "Please update before print"

        if not self.states_updated:
            self.code_len = 0
            for layer_num, fm_transform in enumerate(self.fm_transforms):
                self.zero_cnt += (fm_transform.cuda().abs() < 10 ** -10).sum().item()
                self.size_flat += fm_transform.view(-1).size(0)
                if self.code_length_dict is not None:
                    self.code_len += stream2bit_cnt(fm_transform.cuda(), self.code_length_dict)
                self.plt_fn.hist(fm_transform.view(-1), bins=255)
                self.plt_fn.savefig('{}/Layer{}_DCT.png'.format(self.save, layer_num))
                self.plt_fn.clf()
                max_cur = fm_transform.cuda().max()
                if self.maximum < max_cur:
                    self.maximum = max_cur
                min_cur = fm_transform.cuda().min()
                if self.minimum > min_cur:
                    self.minimum = min_cur

        self.print_fn("==============================================================")
        self.print_fn("fm_transforms == 0: {}".format(self.zero_cnt))
        self.print_fn("fm_transforms size: {}".format(self.size_flat))
        self.print_fn("transform sparsity: {}".format(self.zero_cnt / self.size_flat))
        self.print_fn("transform range ({}, {})".format(self.minimum, self.maximum))
        if self.code_length_dict is not None:
            self.print_fn("code length {}".format(self.code_len))
            self.print_fn("Compress rate {}/{}= {}".format(self.size_flat * 8, self.code_len,
                                                           self.size_flat * 8 / self.code_len))
        self.print_fn("==============================================================")

    def set_config(self, code_length_dict=None):
        self.states_updated = False
        self.code_length_dict = code_length_dict

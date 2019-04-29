import torch
from matplotlib import pyplot as plt

from utils import AverageMeter, accuracy, get_bit_cnt


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
        self.feature_maps = None
        self.states_updated = True
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
        self.states_updated = False
        if self.feature_maps is not None:
            feature_maps = []
            for feature_map, feature_map_batch in zip(self.feature_maps, feature_maps_batch):
                feature_maps.append(torch.cat((feature_map, feature_map_batch)))
        else:
            feature_maps = feature_maps_batch

        self.feature_maps = feature_maps

    def print_result(self):
        assert self.feature_maps is not None, "Please update before print"

        if not self.states_updated:
            for feature_map in self.feature_maps:
                self.zero_cnt += (feature_map.cuda().abs() < 10 ** -10).sum().item()
                self.size_flat += feature_map.view(-1).size(0)
                self.maximum = max(self.maximum, feature_map.cuda().max())
                self.minimum = min(self.minimum, feature_map.cuda().min().item())
            self.states_updated = True

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

        # Config
        self.code_length_dict = code_length_dict

    def update_batch(self, result, inputs=None):
        _, _, fm_transforms_batch = result
        self.states_updated = False
        if self.fm_transforms is not None:
            feature_maps = []
            for feature_map, feature_map_batch in zip(self.fm_transforms, fm_transforms_batch):
                feature_maps.append(torch.cat((feature_map, feature_map_batch)))
        else:
            feature_maps = fm_transforms_batch

        self.fm_transforms = feature_maps

    def print_result(self):
        assert self.fm_transforms is not None, "Please update before print"

        if not self.states_updated:
            self.code_len = 0
            for feature_map in self.fm_transforms:
                self.zero_cnt += (feature_map.cuda().abs() < 10 ** -10).sum().item()
                self.size_flat += feature_map.view(-1).size(0)
                self.maximum = max(self.maximum, feature_map.cuda().max())
                self.minimum = min(self.minimum, feature_map.cuda().min().item())
                if self.code_length_dict is not None:
                    self.code_len += get_bit_cnt(feature_map.cuda(), self.code_length_dict)
            self.states_updated = True

        self.print_fn("==============================================================")
        self.print_fn("feature_maps == 0: {}".format(self.zero_cnt))
        self.print_fn("feature_maps size: {}".format(self.size_flat))
        self.print_fn("sparsity: {}".format(self.zero_cnt / self.size_flat))
        self.print_fn("range ({}, {})".format(self.minimum, self.maximum))
        if self.code_length_dict is not None:
            self.print_fn("code length {}".format(self.code_len))
            self.print_fn("Compress rate {}/{}= {}".format(self.size_flat * 8, self.code_len,
                                                           self.size_flat * 8 / self.code_len))
        self.print_fn("==============================================================")

    def set_config(self, code_length_dict=None):
        self.states_updated = False
        self.code_length_dict = code_length_dict


class HandlerDWT_Fm(InferResultHandler):
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
                assert type(fm_transform_batch) is tuple, "fm_transform_batch is not tuple, make sure it's DWT"
                XL = torch.cat((fm_transform[0], fm_transform_batch[0]))
                XH = []
                for XH_level, XH_level_batch in zip(fm_transform[1], fm_transform_batch[1]):
                    XH.append(torch.cat((XH_level, XH_level_batch)))

                fm_transforms.append((XL, XH))

        else:
            fm_transforms = fm_transforms_batch

        self.fm_transforms = fm_transforms

    def print_result(self):
        assert self.fm_transforms is not None, "Please update before print"

        if not self.states_updated:
            self.code_len = 0
            for layer_num, fm_transform in enumerate(self.fm_transforms):
                assert type(fm_transform) is tuple, "fm_transform_batch is not tuple, make sure it's DWT"
                XL, XH = fm_transform
                self.plt_fn.hist(XL.view(-1), bins=255)
                self.plt_fn.savefig('{}/Layer{}_XL.png'.format(self.save, layer_num))
                self.plt_fn.clf()

                for i, xh in enumerate(XH):
                    self.plt_fn.hist(xh.view(-1), bins=255)
                    self.plt_fn.savefig('{}/Layer{}_XH_{}.png'.format(self.save, layer_num, i))
                    self.plt_fn.clf()

                self.zero_cnt += (XL.cuda().abs() < 10 ** -10).sum().item()
                self.size_flat += XL.view(-1).size(0)
                self.maximum = max(self.maximum, XL.cuda().max().item())
                self.minimum = min(self.minimum, XL.cuda().min().item())
                if self.code_length_dict is not None:
                    self.code_len += get_bit_cnt(XL.cuda(), self.code_length_dict, dual_conti=True)

                for xh in XH:
                    self.zero_cnt += (xh.cuda().abs() < 10 ** -10).sum().item()
                    self.size_flat += xh.view(-1).size(0)
                    self.maximum = max(self.maximum, xh.cuda().max().item())
                    self.minimum = min(self.minimum, xh.cuda().min().item())
                    if self.code_length_dict is not None:
                        self.code_len += get_bit_cnt(xh.cuda(), self.code_length_dict, dual_conti=True)

            self.states_updated = True

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
                    self.code_len += get_bit_cnt(fm_transform.cuda(), self.code_length_dict)
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

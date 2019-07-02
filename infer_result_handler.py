import torch

from utils import accuracy
from meter import *


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
    def __init__(self, print_fn=print):
        # State
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

        # IO
        self.print_fn = print_fn

    def update_batch(self, result, inputs=None):
        assert inputs is not None, "HandlerAcc need target label"
        (x, target) = inputs
        logits_batch, _, _, _, _ = result
        prec1, prec5 = accuracy(logits_batch, target, topk=(1, 5))
        n = logits_batch.size(0)
        self.top1.update(prec1.item(), n)
        self.top5.update(prec5.item(), n)

    def print_result(self):
        self.print_fn('[Test] acc %.2f%%;, Top5 acc %.2f%%', self.top1.avg, self.top5.avg)

    def set_config(self, *args):
        pass


class HandlerFm(InferResultHandler):
    def __init__(self, print_fn=print, print_sparsity=True, print_range_all=False, print_range_layer=True):
        # State
        self.states_updated = False
        self.zero_cnt = 0
        self.size_flat = 0
        self.max_mins_branch = []
        self.max_mins_block = []
        self.max = -2 ** 40
        self.min = 2 ** 40

        # IO
        self.print_fn = print_fn

        # config
        self.is_print_sparsity = print_sparsity
        self.is_print_range_all = print_range_all
        self.is_print_range_layer = print_range_layer

    def update_batch(self, result, inputs=None):
        _, feature_maps_branch, _, feature_maps_block, _ = result
        for layer_num, (feature_map_branch, feature_map_block) in \
                enumerate(zip(feature_maps_branch, feature_maps_block)):
            feature_map_branch_cuda = feature_map_branch.cuda()
            feature_map_block_cuda = feature_map_block.cuda()
            self.zero_cnt += (feature_map_branch_cuda.abs() < 10 ** -10).sum().item()
            self.zero_cnt += (feature_map_block_cuda.abs() < 10 ** -10).sum().item()
            self.size_flat += element_cnt(feature_map_branch)
            self.size_flat += element_cnt(feature_map_block)
            if len(self.max_mins_branch) <= layer_num:
                self.max_mins_branch.append(MaxMinMeter())
                self.max_mins_block.append(MaxMinMeter())
            self.max_mins_branch[layer_num].update(feature_map_branch_cuda)
            self.max_mins_block[layer_num].update(feature_map_block_cuda)

        self.states_updated = False

    def print_result(self):
        self.print_fn("==============================================================")
        if self.is_print_sparsity:
            self._print_sparsity()
        if self.is_print_range_all:
            self._print_range_all()
        if self.is_print_range_layer:
            self._print_range_layer()
        self.print_fn("==============================================================")

    def _print_sparsity(self):
        self.print_fn("feature_maps == 0: {}".format(self.zero_cnt))
        self.print_fn("feature_maps size: {}".format(self.size_flat))
        self.print_fn("sparsity: {}".format(self.zero_cnt / self.size_flat))

    def _print_range_all(self):
        if not self.states_updated:
            for layer_num, meter in enumerate(self.max_mins_branch):
                self.max = max(self.max, meter.max)
                self.min = min(self.min, meter.min)
        self.print_fn("range: ({}, {})".format(self.min, self.max))

    def _print_range_layer(self):
        for layer_num, meter in enumerate(self.max_mins_branch):
            self.print_fn("range in branch of layer{}:  ({}, {})".format(layer_num, meter.max, meter.min))
        for layer_num, meter in enumerate(self.max_mins_block):
            self.print_fn("range in block of layer{}:  ({}, {})".format(layer_num, meter.max, meter.min))

    def set_config(self, print_sparsity=True, print_range_all=False, print_range_layer=True):
        self.is_print_sparsity = print_sparsity
        self.is_print_range_all = print_range_all
        self.is_print_range_layer = print_range_layer


class HandlerQuanti(InferResultHandler):
    def __init__(self, print_fn=print, code_length_dict=None, print_sparsity=True, print_range_all=False,
                 print_range_layer=True):
        # State
        self.states_updated = False
        self.zero_cnt = 0
        self.size_flat = 0
        self.max_mins_branch = []
        self.max_mins_block = []
        self.max = -2 ** 40
        self.min = 2 ** 40
        self.code_len = 0
        if code_length_dict is not None:
            self.hist_meter_branch = HistMeter(code_length_dict)
            self.hist_meter_block = HistMeter(code_length_dict)

        # IO
        self.print_fn = print_fn

        # Config
        self.code_length_dict = code_length_dict
        self.print_sparsity = print_sparsity
        self.print_range_all = print_range_all
        self.print_range_layer = print_range_layer

    def update_batch(self, result, inputs=None):
        _, _, fm_transforms_branch, _, fm_transforms_block = result
        for layer_num, (fm_transform_branch, fm_transform_block) in enumerate(zip(fm_transforms_branch, fm_transforms_block)):
            fm_transform_branch_cuda = fm_transform_branch.cuda()
            fm_transform_block_cuda = fm_transform_block.cuda()

            self.zero_cnt += (fm_transform_branch_cuda.abs() < 10 ** -10).sum().item()
            self.zero_cnt += (fm_transform_block_cuda.abs() < 10 ** -10).sum().item()

            self.size_flat += element_cnt(fm_transform_branch)
            self.size_flat += element_cnt(fm_transform_block)

            if len(self.max_mins_branch) <= layer_num:
                self.max_mins_branch.append(MaxMinMeter())
                self.max_mins_block.append(MaxMinMeter())

            self.max_mins_branch[layer_num].update(fm_transform_branch_cuda)
            self.max_mins_block[layer_num].update(fm_transform_block_cuda)
            self.hist_meter_branch.update(fm_transform_branch_cuda)
            self.hist_meter_block.update(fm_transform_block_cuda)
        self.states_updated = False

    def print_result(self):
        if not self.states_updated:
            if self.code_length_dict is not None:
                self.code_len = self.hist_meter_branch.get_bit_cnt(self.code_length_dict)
                self.code_len += self.hist_meter_block.get_bit_cnt(self.code_length_dict)
            if self.print_range_all:
                for layer_num, meter in enumerate(self.max_mins_branch):
                    self.max = max(self.max, meter.max)
                    self.min = min(self.min, meter.min)
            self.states_updated = True

        self.print_fn("==============================================================")
        if self.print_sparsity:
            self.print_fn("feature_maps == 0: {}".format(self.zero_cnt))
            self.print_fn("feature_maps size: {}".format(self.size_flat))
            self.print_fn("sparsity: {}".format(self.zero_cnt / self.size_flat))
        if self.print_range_all:
            self.print_fn("range: ({}, {})".format(self.min, self.max))
        if self.print_range_layer:
            for layer_num, meter in enumerate(self.max_mins_branch):
                self.print_fn("range in branch of layer{}:  ({}, {})".format(layer_num, meter.max, meter.min))
            for layer_num, meter in enumerate(self.max_mins_block):
                self.print_fn("range in block of layer{}:  ({}, {})".format(layer_num, meter.max, meter.min))
        self.print_fn("code length {}".format(self.code_len))
        self.print_fn("Compress rate {}/{} = {}".format(self.size_flat * 8, self.code_len,
                                                        self.size_flat * 8 / self.code_len))
        self.print_fn("==============================================================")

    def set_config(self, code_length_dict=None, print_sparsity=True, print_range_all=False, print_range_layer=True):
        self.code_length_dict = code_length_dict
        if self.code_length_dict is not None:
            self.code_len = self.hist_meter_branch.get_bit_cnt(self.code_length_dict)
            self.code_len += self.hist_meter_block.get_bit_cnt(self.code_length_dict)
        self.states_updated = True
        self.print_sparsity = print_sparsity
        self.print_range_all = print_range_all
        self.print_range_layer = print_range_layer


class HandlerTrans(InferResultHandler):
    def __init__(self, print_fn=print, print_sparsity=True, print_range_all=False,
                 print_range_layer=True):
        # State
        self.states_updated = False
        self.zero_cnt = 0
        self.size_flat = 0
        self.max_mins = []
        self.max = -2 ** 40
        self.min = 2 ** 40
        self.code_len = 0

        # IO
        self.print_fn = print_fn

        # Config
        self.print_sparsity = print_sparsity
        self.print_range_all = print_range_all
        self.print_range_layer = print_range_layer

    def update_batch(self, result, inputs=None):
        _, _, fm_transforms_batch, _, _ = result
        for layer_num, fm_transform_batch in enumerate(fm_transforms_batch):
            fm_transform_batch_cuda = fm_transform_batch.cuda()
            self.zero_cnt += (fm_transform_batch_cuda.abs() < 10 ** -10).sum().item()
            self.size_flat += element_cnt(fm_transform_batch)
            if len(self.max_mins) <= layer_num:
                self.max_mins.append(MaxMinMeter())
            self.max_mins[layer_num].update(fm_transform_batch_cuda)
        self.states_updated = False

    def print_result(self):
        if not self.states_updated:
            if self.print_range_all:
                for layer_num, meter in enumerate(self.max_mins):
                    self.max = max(self.max, meter.max)
                    self.min = min(self.min, meter.min)
            self.states_updated = True

        self.print_fn("==============================================================")
        if self.print_sparsity:
            self.print_fn("feature_maps == 0: {}".format(self.zero_cnt))
            self.print_fn("feature_maps size: {}".format(self.size_flat))
            self.print_fn("sparsity: {}".format(self.zero_cnt / self.size_flat))
        if self.print_range_all:
            self.print_fn("range: ({}, {})".format(self.min, self.max))
        if self.print_range_layer:
            for layer_num, meter in enumerate(self.max_mins):
                self.print_fn("range in layer{}:  ({}, {})".format(layer_num, meter.max, meter.min))
        self.print_fn("==============================================================")

    def set_config(self, print_sparsity=True, print_range_all=False, print_range_layer=True):
        self.print_sparsity = print_sparsity
        self.print_range_all = print_range_all
        self.print_range_layer = print_range_layer


class HandlerDWT_Fm(InferResultHandler):
    def __init__(self, print_fn=print, save="", code_length_dict=None, print_sparsity=True, print_range_all=False,
                 print_range_layer=True, is_inblock=False, is_branch=True):
        self.states_updated = False
        self.zero_cnt = 0
        self.size_flat = 0
        self.max_mins_branch = []
        self.max_mins_block = []
        self.max = -2 ** 40
        self.min = 2 ** 40
        self.code_len = 0
        self.hist_meters_branch = []
        self.hist_meters_block = []
        self.max_ch = []
        self.min_ch = []
        self.is_inblock = is_inblock
        self.is_branch = is_branch

        # IO
        self.print_fn = print_fn

        self.save = save

        # Config
        self.code_length_dict = code_length_dict
        self.is_print_sparsity = print_sparsity
        self.is_print_range_all = print_range_all
        self.is_print_range_layer = print_range_layer

    def update_batch(self, result, inputs=None):
        _, _, fm_transforms_branch, _, fm_transforms_block = result
        if self.is_branch:
            assert len(fm_transforms_branch) != 0, "Nothing in fm_transforms_branch!!!"
            for layer_num, fm_transform_branch in enumerate(fm_transforms_branch):
                self._update_dwt(fm_transform_branch, layer_num, self.max_mins_branch, self.hist_meters_branch)
            for layer_num, meter in enumerate(self.max_mins_branch):
                self.max = max(self.max, meter.max)
                self.min = min(self.min, meter.min)
        if self.is_inblock:
            assert len(fm_transforms_block) != 0, "Nothing in fm_transforms_block!!!"
            for layer_num, fm_transform_block in enumerate(fm_transforms_block):
                self._update_dwt(fm_transform_block, layer_num, self.max_mins_block, self.hist_meters_block)
            for layer_num, meter in enumerate(self.max_mins_block):
                self.max = max(self.max, meter.max)
                self.min = min(self.min, meter.min)
        self.states_updated = False

    def _update_dwt(self, feature_map_dwt, layer_num, max_mins, hist_meters):
        assert type(feature_map_dwt) is tuple, "feature_map_dwt is not tuple, make sure it's DWT"

        XL, XH, size_cur = feature_map_dwt
        XL_cuda = XL.cuda()

        if len(max_mins) <= layer_num:
            max_mins.append(MaxMinMeter())

        if len(hist_meters) <= layer_num:
            hist_meters.append((HistMeter(self.code_length_dict), []))

        self.zero_cnt += (XL_cuda.abs() < 10 ** -10).sum().item()
        self.size_flat += element_cnt(XL)
        max_mins[layer_num].update(XL_cuda)
        hist_meters[layer_num][0].update(XL_cuda)

        for i, xh in enumerate(XH):
            xh_cuda = xh.cuda()
            lh, hl, hh = torch.unbind(xh_cuda, dim=2)
            if (xh.size(-1) * 2) != size_cur[-1]:
                assert (xh.size(-1) * 2) - 1 == size_cur[-1], \
                    "Size mismatch {}, {}".format((xh.size(-1) * 2) - 1, size_cur[-1])

                assert hl[..., -1].abs().max() < 1e-10, "Wrong padding"
                hl = hl[..., :-1]
                assert hh[..., -1].abs().max() < 1e-10, "Wrong padding"
                hh = hh[..., :-1]
            if (xh.size(-2) * 2) != size_cur[-2]:
                assert (xh.size(-2) * 2) - 1 == size_cur[-2], \
                    "Size mismatch {}, {}".format((xh.size(-1) * 2) - 1, size_cur[-2])

                assert lh[..., -1, :].abs().max() < 1e-10, "Wrong padding"
                lh = lh[..., :-1, :]
                assert hh[..., -1, :].abs().max() < 1e-10, "Wrong padding"
                hh = hh[..., :-1, :]

            for h in [lh, hl, hh]:
                self.zero_cnt += (h.abs() < 10 ** -10).sum().item()
                self.size_flat += element_cnt(h)
                max_mins[layer_num].update(h)

                if len(hist_meters[layer_num][1]) <= i:
                    hist_meters[layer_num][1].append(HistMeter(self.code_length_dict))

                hist_meters[layer_num][1][i].update(h)

            size_cur = (xh.size(-2), xh.size(-1))

    def print_result(self):
        if not self.states_updated:
            self.code_len = 0
            figure = plt.figure()
            ax = figure.add_subplot(1, 1, 1)

            if self.is_branch:
                self._update_after_config(figure, ax, self.hist_meters_branch, 'branch_')

            if self.is_inblock:
                self._update_after_config(figure, ax, self.hist_meters_block, 'block_')

            plt.close(figure)

            self.states_updated = True

        self.print_fn("==============================================================")
        '''
        for layer, max_c, min_c in zip(range(len(self.max_ch)), self.max_ch, self.min_ch):
            self.print_fn("---------- Layer {} -----------".format(layer))
            self.print_fn(max_c)
            self.print_fn(min_c)
            self.print_fn("")'''
        '''
        for layer_num, meters_layer in enumerate(self.hist_meters):
            zero_cnt = meters_layer[0].hist[0]
            size_total = meters_layer[0].size_total
            for meter in meters_layer[1]:
                zero_cnt += meter.hist[0]
                size_total += meter.size_total
            self.print_fn("zero count in layer{}:  {}".format(layer_num, zero_cnt))
            self.print_fn("sparsity in layer{}:  {}".format(layer_num, zero_cnt / size_total))'''
        if self.is_print_sparsity:
            self._print_sparsity()
        if self.is_print_range_all:
            self.print_fn("range: ({}, {})".format(self.min, self.max))
        if self.is_print_range_layer:
            self._print_range_layer()
        self.print_fn("code length {}".format(self.code_len))
        self.print_fn("Compress rate {}/{}= {}".format(self.size_flat * 8, self.code_len,
                                                       self.size_flat * 8 / self.code_len))
        self.print_fn("==============================================================")

    def _update_after_config(self, figure, ax, hist_list, prefix=''):
        assert hist_list is not None, "Please update before print"
        for layer_num, hist_layer in enumerate(hist_list):
            xl_hist, xh_hists = hist_layer
            xl_hist.plt_hist(plt_fn=ax)
            figure.savefig('{}/{}_Layer{}_XL.png'.format(self.save, prefix, layer_num))
            ax.cla()

            if self.code_length_dict is not None:
                self.code_len += xl_hist.get_bit_cnt(self.code_length_dict)

            for i, xh_hist in enumerate(xh_hists):
                xh_hist.plt_hist(plt_fn=ax)
                figure.savefig('{}/{}_Layer{}_XH_{}.png'.format(self.save, prefix, layer_num, i))
                ax.cla()
                if self.code_length_dict is not None:
                    self.code_len += xh_hist.get_bit_cnt(self.code_length_dict)

    def _print_sparsity(self):
        self.print_fn("fm_transforms == 0: {}".format(self.zero_cnt))
        self.print_fn("fm_transforms size: {}".format(self.size_flat))
        self.print_fn("transform sparsity: {}".format(self.zero_cnt / self.size_flat))

    def _print_range_layer(self):
        for layer_num, meter in enumerate(self.max_mins_branch):
            self.print_fn("range in branch of layer{}:  ({}, {})".format(layer_num, meter.max, meter.min))
        for layer_num, meter in enumerate(self.max_mins_block):
            self.print_fn("range in block of layer{}:  ({}, {})".format(layer_num, meter.max, meter.min))

    def set_config(self, code_length_dict=None, print_sparsity=True, print_range_all=False,
                   print_range_layer=True):
        self.states_updated = False
        self.code_length_dict = code_length_dict
        self.is_print_sparsity = print_sparsity
        self.is_print_range_all = print_range_all
        self.is_print_range_layer = print_range_layer


class HandlerMaskedDWT_Fm(InferResultHandler):
    def __init__(self, print_fn=print, save="", code_length_dict=None, print_sparsity=True, print_range_all=False,
                 print_range_layer=True, is_inblock=False, is_branch=True):
        self.states_updated = False
        self.zero_cnt = 0
        self.size_flat = 0
        self.max_mins = []
        self.max = -2 ** 40
        self.min = 2 ** 40
        self.code_len = 0
        self.hist_meters = []
        self.max_ch = []
        self.min_ch = []

        # IO
        self.print_fn = print_fn

        self.save = save

        # Config
        self.code_length_dict = code_length_dict
        self.print_sparsity = print_sparsity
        self.print_range_all = print_range_all
        self.print_range_layer = print_range_layer

    def update_batch(self, result, inputs=None):
        _, _, fm_transforms_batch, _, _ = result
        assert len(fm_transforms_batch) != 0, "Nothing in fm_transforms_batch!!!"
        self.states_updated = False

        for layer_num, feature_map_batch in enumerate(fm_transforms_batch):
            assert type(feature_map_batch) is tuple, "fm_transform_batch is not tuple, make sure it's DWT"
            x_dwt_remain, x_dwt_masked = feature_map_batch

            self._update_dwt(x_dwt_remain, layer_num, self.max_mins, self.hist_meters, masked=False)
            self._update_dwt(x_dwt_masked, layer_num, self.max_mins, self.hist_meters, masked=True)

    def _update_dwt(self, feature_map_dwt, layer_num, max_mins, hist_meters, masked=False):
        assert type(feature_map_dwt) is tuple, "feature_map_dwt is not tuple, make sure it's DWT"

        XL, XH, size_cur = feature_map_dwt
        if masked:
            size_cur = (XH[0].size(-2), XH[0].size(-1))
            XH = XH[1:]
        XL_cuda = XL.cuda()

        if len(max_mins) <= layer_num:
            max_mins.append(MaxMinMeter())

        if len(hist_meters) <= layer_num:
            hist_meters.append((HistMeter(self.code_length_dict), []))

        self.zero_cnt += (XL_cuda.abs() < 10 ** -10).sum().item()
        self.size_flat += element_cnt(XL)
        max_mins[layer_num].update(XL_cuda)
        hist_meters[layer_num][0].update(XL_cuda)

        for i, xh in enumerate(XH):
            xh_cuda = xh.cuda()
            lh, hl, hh = torch.unbind(xh_cuda, dim=2)
            if (xh.size(-1) * 2) != size_cur[-1]:
                assert (xh.size(-1) * 2) - 1 == size_cur[-1], \
                    "Size mismatch {}, {}".format((xh.size(-1) * 2) - 1, size_cur[-1])

                assert hl[..., -1].abs().max() < 1e-10, "Wrong padding"
                hl = hl[..., :-1]
                assert hh[..., -1].abs().max() < 1e-10, "Wrong padding"
                hh = hh[..., :-1]
            if (xh.size(-2) * 2) != size_cur[-2]:
                assert (xh.size(-2) * 2) - 1 == size_cur[-2], \
                    "Size mismatch {}, {}".format((xh.size(-1) * 2) - 1, size_cur[-2])

                assert lh[..., -1, :].abs().max() < 1e-10, "Wrong padding"
                lh = lh[..., :-1, :]
                assert hh[..., -1, :].abs().max() < 1e-10, "Wrong padding"
                hh = hh[..., :-1, :]

            for h in [lh, hl, hh]:
                self.zero_cnt += (h.abs() < 10 ** -10).sum().item()
                self.size_flat += element_cnt(h)
                max_mins[layer_num].update(h)

                if len(hist_meters[layer_num][1]) <= i:
                    hist_meters[layer_num][1].append(HistMeter(self.code_length_dict))

                hist_meters[layer_num][1][i].update(h)

            size_cur = (xh.size(-2), xh.size(-1))

    def print_result(self):
        if not self.states_updated:
            if self.print_range_all:
                for layer_num, meter in enumerate(self.max_mins):
                    self.max = max(self.max, meter.max)
                    self.min = min(self.min, meter.min)

            self.code_len = 0
            figure = plt.figure()
            ax = figure.add_subplot(1, 1, 1)

            self._update_after_config(figure, ax, self.hist_meters)

            self.states_updated = True

        self.print_fn("==============================================================")
        if self.print_sparsity:
            self.print_fn("fm_transforms == 0: {}".format(self.zero_cnt))
            self.print_fn("fm_transforms size: {}".format(self.size_flat))
            self.print_fn("transform sparsity: {}".format(self.zero_cnt / self.size_flat))
        if self.print_range_all:
            self.print_fn("range: ({}, {})".format(self.min, self.max))
        if self.print_range_layer:
            for layer_num, meter in enumerate(self.max_mins):
                self.print_fn("range in layer{}:  ({}, {})".format(layer_num, meter.max, meter.min))
        self.print_fn("code length {}".format(self.code_len))
        self.print_fn("Compress rate {}/{}= {}".format(self.size_flat * 8, self.code_len,
                                                       self.size_flat * 8 / self.code_len))
        self.print_fn("==============================================================")

    def _update_after_config(self, figure, ax, hist_list, prefix=''):
        assert hist_list is not None, "Please update before print"
        for layer_num, hist_layer in enumerate(hist_list):
            xl_hist, xh_hists = hist_layer
            xl_hist.plt_hist(plt_fn=ax)
            figure.savefig('{}/{}_Layer{}_XL.png'.format(self.save, prefix, layer_num))
            ax.cla()

            if self.code_length_dict is not None:
                self.code_len += xl_hist.get_bit_cnt(self.code_length_dict)

            for i, xh_hist in enumerate(xh_hists):
                xh_hist.plt_hist(plt_fn=ax)
                figure.savefig('{}/{}_Layer{}_XH_{}.png'.format(self.save, prefix, layer_num, i))
                ax.cla()
                if self.code_length_dict is not None:
                    self.code_len += xh_hist.get_bit_cnt(self.code_length_dict)

    def set_config(self, code_length_dict=None, print_sparsity=True, print_range_all=False,
                   print_range_layer=True):
        self.states_updated = False
        self.code_length_dict = code_length_dict
        self.print_sparsity = print_sparsity
        self.print_range_all = print_range_all
        self.print_range_layer = print_range_layer


class HandlerDCT_Fm(InferResultHandler):
    def __init__(self, print_fn=print, save="", code_length_dict=None):
        # states
        self.states_updated = True
        self.zero_cnt = 0
        self.size_flat = 0
        self.maximum = -2 ** 40
        self.minimum = 2 ** 40
        self.code_len = 0
        self.hist_meters = []

        # IO
        self.print_fn = print_fn

        self.save = save

        # Config
        self.code_length_dict = code_length_dict

    def update_batch(self, result, inputs=None):
        _, _, fm_transforms_batch, _, _ = result
        assert len(fm_transforms_batch) != 0, "Nothing in fm_transforms_batch!!!"
        self.states_updated = False
        for layer_num, fm_transform_batch in enumerate(fm_transforms_batch):
            self.zero_cnt += (fm_transform_batch.cuda().abs() < 10 ** -10).sum().item()
            self.size_flat += element_cnt(fm_transform_batch)

            max_cur = fm_transform_batch.cuda().max()
            min_cur = fm_transform_batch.cuda().min()

            self.maximum = max(self.maximum, max_cur)
            self.minimum = min(self.minimum, min_cur)

            if len(self.hist_meters) <= layer_num:
                self.hist_meters.append(HistMeter(self.code_length_dict))

            self.hist_meters[layer_num].update(fm_transform_batch.cuda())

    def print_result(self):
        if not self.states_updated:
            self.code_len = 0
            figure = plt.figure()
            ax = figure.add_subplot(1, 1, 1)

            for layer_num, hist_layer in enumerate(self.hist_meters):
                hist_layer.plt_hist(plt_fn=ax)
                figure.savefig('{}/Layer{}_DCT.png'.format(self.save, layer_num))
                ax.cla()
                self.code_len += hist_layer.get_bit_cnt(self.code_length_dict)

            plt.close(figure)

        self.print_fn("==============================================================")
        self.print_fn("fm_transforms == 0: {}".format(self.zero_cnt))
        self.print_fn("fm_transforms size: {}".format(self.size_flat))
        self.print_fn("transform sparsity: {}".format(self.zero_cnt / self.size_flat))
        self.print_fn("transform range ({}, {})".format(self.minimum, self.maximum))
        self.print_fn("code length {}".format(self.code_len))
        self.print_fn("Compress rate {}/{} = {}".format(self.size_flat * 8, self.code_len,
                                                        self.size_flat * 8 / self.code_len))
        self.print_fn("==============================================================")

    def set_config(self, code_length_dict=None):
        self.states_updated = False
        self.code_length_dict = code_length_dict

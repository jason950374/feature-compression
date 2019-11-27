import numpy as np
import torch
import json
from matplotlib import pyplot as plt


class AverageMeter:
    """
    On-line Average Meter
    """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class MaxMinMeter:
    _init_max = -2 ** 40
    _init_min = 2 ** 40

    def __init__(self):
        self.max = MaxMinMeter._init_max
        self.min = MaxMinMeter._init_min

    def reset(self):
        self.__init__()

    def update(self, in_stream):
        layer_max = in_stream.max().item()
        layer_min = in_stream.min().item()
        self.max = max(self.max, layer_max)
        self.min = min(self.min, layer_min)


class HistMeter:
    r"""
        If the in_stream is integer,it can be implemented with torch.bincount

    """
    eps = 10 ** -3
    _init_max = -2 ** 40
    _init_min = 2 ** 40

    def __init__(self, codes=None):
        self.hist = {}
        self.size_total = 0
        # __init__ also used in reset. If codes=None, use old self.codes
        if codes is not None:
            self.codes = codes

        # gen code length tensor

        self.max = self._init_max
        self.min = self._init_min
        self.hist = None

    def reset(self, codes=None):
        self.__init__(codes)

    def update(self, in_stream):
        r"""

        Args:
            in_stream(torch.Tensor): Input data

        """
        with torch.no_grad():
            assert (((in_stream % 1) >= (1 - HistMeter.eps)) | ((in_stream % 1) < HistMeter.eps)).min(), \
                "in_stream need to be integers"
            pos_mask = (in_stream >= 0).long()
            neg_mask = (in_stream < 0).long()
            in_stream_org = in_stream
            in_stream = (in_stream + HistMeter.eps).long() * pos_mask + \
                        (in_stream - HistMeter.eps).long() * neg_mask
            in_stream = in_stream.view(-1)
            min_in = int(in_stream.min())
            max_in = int(in_stream.max())
            if self.hist is None:
                in_stream = in_stream - min_in
                self.hist = torch.bincount(in_stream)
                self.max = max_in
                self.min = min_in
            else:
                right_extd = max(max_in - self.max, 0)
                left_extd = max(self.min - min_in, 0)
                if right_extd + left_extd > 0:
                    org_hist = self.hist
                    self.hist = torch.zeros([(self.max + right_extd + 1) - (self.min - left_extd)])
                    self.hist = self.hist.to(device=org_hist.device, dtype=org_hist.dtype)
                    self.hist[left_extd:(left_extd + self.max - self.min + 1)] = org_hist
                    self.max = self.max + right_extd
                    self.min = self.min - left_extd
                in_stream = in_stream - self.min
                self.hist = self.hist + torch.bincount(in_stream, minlength=self.max - self.min + 1)

    def get_bit_cnt(self, code_length_dict):
        r"""
            Bit cnt for given histogram with dictionary of code length

            Args:
                    hist (dict):  Input histogram dictionary (code, cnt)
                    code_length_dict (dict): code_length_dict, key(int) is code_length, value is iterable codes before
                        encoding

            Note:
                    Different from bins of numpy.histogram, hist and code has same length.
                    Normally, code_list[i] can be
                    :math:`\frac{bins[i]+bins[i+1]}{2}`

            Returns:
                    Total length (bits)
            """
        with torch.no_grad():
            weight = []
            for code in range(self.min, self.max + 1):
                weight.append(code_length_dict[code])

            weight = torch.tensor(weight).to(device=self.hist.device, dtype=self.hist.dtype)

            total_len = (self.hist * weight).sum()

        return total_len

    def save_hist_json(self, filename):
        with open(filename, 'w') as f:
            json.dump({'hist': self.hist.tolist(),
                       'max': self.max,
                       'min': self.min,
                       }, f)

    def plt_hist(self, plt_fn=None):
        if plt_fn is None:
            plt_fn = plt

        plt_fn.bar(range(self.min, self.max + 1), self.hist.cpu(), width=1)


def element_cnt(x):
    cnt = 1
    for s in x.size():
        cnt *= s
    return cnt

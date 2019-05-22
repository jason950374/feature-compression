import numpy as np
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
    eps = 10 ** -4
    # eps = 0.49999999999

    def __init__(self, codes=None):
        self.hist = {}
        self.size_total = 0
        # __init__ also used in reset. If codes=None, use old self.codes
        if codes is not None:
            self.codes = codes

        try:
            for code in self.codes:
                self.hist[code] = 0
        except AttributeError:
            self.codes = None

    def reset(self, codes=None):
        self.__init__(codes)

    def update(self, in_stream):
        assert (((in_stream % 1) >= (1 - HistMeter.eps)) | ((in_stream % 1) < HistMeter.eps)).min(), \
            "in_stream need to be integers"
        cnt = 0
        if len(in_stream.size()) > 1:
            in_stream = in_stream.view(-1)
        size = in_stream.size(0)

        for code in self.hist:
            match = (in_stream < (code + HistMeter.eps)) & (in_stream >= (code - HistMeter.eps))
            cnt += int(match.sum().item())
            self.hist[code] += match.sum().item()

        self.size_total += size
        assert cnt == size, \
            "{} vs. {}: Some code in in_stream not find in self.hist".format(cnt, size)

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
        total_len = 0

        for code in self.hist:
            total_len += self.hist[code] * code_length_dict[code]

        return total_len

    def plt_hist(self, plt_fn=None, tight=True):
        if plt_fn is None:
            plt_fn = plt

        codes = np.asarray(list(self.hist.keys()))
        codes.sort()
        xmin = codes[0]
        xmax = codes[-1]

        hist = []
        if tight:
            for code in codes:
                if self.hist[code] == 0:
                    xmin = code
                else:
                    break

            for code in reversed(codes):
                if self.hist[code] == 0:
                    xmax = code
                else:
                    break

            assert xmax > xmin, "Nothing??"

            for code in range(xmin, xmax + 1):
                hist.append(self.hist[code])

            plt_fn.bar(range(xmin, xmax + 1), hist, width=1)

        else:
            for code in codes:
                if code != 0:
                    hist.append(self.hist[code])
                else:
                    hist.append(0)

            plt_fn.bar(codes, hist, width=1)

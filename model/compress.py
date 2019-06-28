import pywt
import torch.nn as nn
import torch
import torch.nn.functional as F
import functional as my_f
from collections import OrderedDict

from functional.dwt import lowlevel


def random_round(x, rand_factor=0.):
    assert rand_factor > -10 ** -10, "rand_factor must be positive"
    x_org = x
    # if rand_factor == 0, do not do torch.rand_like to speed up? (not sure the effectiveness)
    if rand_factor > 10 ** -10:
        rand = torch.rand_like(x) - 0.5
        x = torch.round(x + rand * rand_factor)
    else:
        x = torch.round(x)

    return x.detach() + x_org - x_org.detach()


def softmin_round(x, min_int, max_int, tau=1):
    """

    Args:
        x (torch.Tensor): input
        min_int (int):
        max_int (int):
        tau (float):

    Returns:
        rounded x (torch.Tensor)
    """
    int_set = torch.arange(min_int, max_int+1, step=1.)
    in_device = x.get_device()
    if in_device >= 0:  # On gpu
        int_set = int_set.to(in_device)
    x_repeat = x.unsqueeze(-1)
    x_repeat = x_repeat.repeat_interleave(int_set.size(0), dim=-1)
    # distant = (x_repeat - int_set).abs() * tau
    distant = ((x_repeat - int_set) ** 2) * tau
    weight = F.softmin(distant, dim=-1)
    return (weight * int_set).sum(-1)


class EncoderDecoderPair(nn.Module):
    is_bypass = False

    def __init__(self):
        super(EncoderDecoderPair, self).__init__()

    def forward(self, x, is_encoder=True):
        raise NotImplementedError

    def update(self):
        pass


class CompressDCT(EncoderDecoderPair):
    """
        Compress with DCT and Q table
        Args:
            bit (int): Bits after quantization
            q_table: Quantization table
    """
    def __init__(self, bit=8, q_table=None):
        super(CompressDCT, self).__init__()
        self.bit = bit
        if q_table is None:
            self.register_buffer('q_table', torch.ones(8, 8))
        else:
            assert q_table.size() == (8, 8)
            self.register_buffer('q_table', q_table)

    # TODO  and BP path
    def forward(self, x, is_encoder=True):
        assert len(x.size()) == 4, "Dimension of x need to be 4, which corresponds to (N, C, H, W)"
        N, C, H, W = x.size()
        q_table = self.q_table.repeat(N, C, 1, 1)
        r_h = H % 8
        r_w = W % 8

        if is_encoder:
            fm_transform = torch.zeros_like(x)
            for i_h in range(H // 8):
                for i_w in range(W // 8):
                    x_dct = my_f.dct_2d(x[..., 8*i_h:8*(i_h+1), 8*i_w:8*(i_w+1)])
                    x_dct = torch.round((x_dct / q_table))
                    x_dct = x_dct.clamp(-2 ** (self.bit - 1), 2 ** (self.bit-1) - 1)
                    fm_transform[..., 8 * i_h:8 * (i_h + 1), 8 * i_w:8 * (i_w + 1)] = x_dct

                if r_w != 0:
                    q_table_w = q_table[..., :r_w]
                    for i in range(r_w):
                        q_table_w[..., i] = q_table[..., i * 8 // r_w]

                    x_dct = my_f.dct_2d(x[..., 8*i_h:8*(i_h+1), -r_w:])
                    x_dct = torch.round((x_dct / q_table_w))
                    x_dct = x_dct.clamp(-2 ** (self.bit - 1), 2 ** (self.bit-1) - 1)
                    fm_transform[..., 8 * i_h:8 * (i_h + 1), -r_w:] = x_dct

            if r_h != 0:
                for i_w in range(W // 8):
                    q_table_h = q_table[..., :r_h, :]
                    for i in range(r_h):
                        q_table_h[..., i, :] = q_table[..., i * 8 // r_h, :]

                    x_dct = my_f.dct_2d(x[..., -r_h:, 8 * i_w:8 * (i_w + 1)])
                    x_dct = torch.round((x_dct / q_table_h))
                    x_dct = x_dct.clamp(-2 ** (self.bit - 1), 2 ** (self.bit-1) - 1)
                    fm_transform[..., -r_h:, 8 * i_w:8 * (i_w + 1)] = x_dct
                if r_w != 0:
                    q_table_h_w = q_table[..., :r_h, :r_w]
                    for i in range(r_h):
                        for j in range(r_w):
                            q_table_h_w[..., i, j] = q_table[..., i * 8 // r_h, j * 8 // r_w]

                    x_dct = my_f.dct_2d(x[..., -r_h:, -r_w:])
                    x_dct = torch.round((x_dct / q_table_h_w))
                    x_dct = x_dct.clamp(-2 ** (self.bit - 1), 2 ** (self.bit-1) - 1)
                    fm_transform[..., -r_h:, -r_w:] = x_dct
            return fm_transform
        else:
            fm_transform = x
            x = torch.ones_like(x)
            for i_h in range(H // 8):
                for i_w in range(W // 8):
                    x_dct = fm_transform[..., 8 * i_h:8 * (i_h + 1), 8 * i_w:8 * (i_w + 1)]
                    x[..., 8 * i_h:8 * (i_h + 1), 8 * i_w:8 * (i_w + 1)] = my_f.idct_2d(x_dct * q_table)

                if r_w != 0:
                    x_dct = fm_transform[..., 8 * i_h:8 * (i_h + 1), -r_w:]
                    x[..., 8 * i_h:8 * (i_h + 1), -r_w:] = my_f.idct_2d(x_dct * q_table[..., -r_w:])

            if r_h != 0:
                for i_w in range(W // 8):
                    x_dct = fm_transform[..., -r_h:, 8 * i_w:8 * (i_w + 1)]
                    x[..., -r_h:, 8 * i_w:8 * (i_w + 1)] = my_f.idct_2d(x_dct * q_table[..., -r_h:, :])
                if r_w != 0:
                    x_dct = fm_transform[..., -r_h:, -r_w:]
                    x[..., -r_h:, -r_w:] = my_f.idct_2d(x_dct * q_table[..., -r_h:, -r_w:])
            return x


class CompressDWT(EncoderDecoderPair):
    """
    Compress with DWT and Q table
    """
    def __init__(self, level=1, bit=8, q_table=None, wave='haar', rand_factor=0.):
        """
        Args:
            level (int): Level of DWT
            bit (int): Bits after quantization
            q_table (list, tuple): Quantization table, scale is fine to coarse
            wave (str, pywt.Wavelet, tuple or list): Mother wavelet
            rand_factor:
        """
        super(CompressDWT, self).__init__()
        self.bit = bit
        if q_table is None:
            self.register_buffer('q_table', torch.ones(level))
        else:
            assert q_table.size() == (level + 1,)
            self.register_buffer('q_table', q_table)
        self.level = level
        self.rand_factor = rand_factor
        self.DWT = my_f.DWT(J=level, wave=wave, mode='periodization', separable=False)
        self.IDWT = my_f.IDWT(wave=wave, mode='periodization', separable=False)

    def forward(self, x, quanti=True, is_encoder=True):
        if is_encoder:
            assert len(x.size()) == 4, "Dimension of x need to be 4, which corresponds to (N, C, H, W)"
            ll, xh, size = self.DWT(x)
            ll = ll / self.q_table[-1]
            if quanti:
                ll = random_round(ll, self.rand_factor)
                ll = ll.clamp(-2 ** (self.bit - 1), 2 ** (self.bit - 1) - 1)
            for i in range(self.level):
                xh[i] = xh[i] / self.q_table[i]
                if quanti:
                    xh[i] = random_round(xh[i], self.rand_factor)
                    xh[i] = xh[i].clamp(-2 ** (self.bit - 1), 2 ** (self.bit - 1) - 1)

            return ll, xh, size

        else:
            assert len(x) == 3, "Must be tuple include LL, Hs and size"
            ll, xh_org, size = x
            ll = ll * self.q_table[-1]
            xh = []
            for i in range(self.level):
                xh.append(xh_org[i] * self.q_table[i])

            x = self.IDWT((ll, xh, size))

            return x


class AdaptiveDWT(EncoderDecoderPair):
    r"""
    Compress with DWT and Q table
    The wave will select adaptively
    """

    def __init__(self, x_size, level=1, bit=8, q_table=None, rand_factor=0.):
        """

        Args:
            x_size: x_size. If x's H or W is odd number, need x_size to guarantee the reconstruction has same size
            level (int): Level of DWT
            bit (int): Bits after quantization
            q_table (list, tuple): Quantization table, scale is fine to coarse
            rand_factor:
        """
        super(AdaptiveDWT, self).__init__()
        self.bit = bit
        if q_table is None:
            self.register_buffer('q_table', torch.ones(level))
        else:
            assert q_table.size() == (level + 1,)
            self.register_buffer('q_table', q_table)
        self.level = level
        self.rand_factor = rand_factor
        self.size = None

        size_cur = int(x_size)
        for i in range(level):
            wave_num = max(size_cur // 4, 1)
            size_cur = size_cur // 2
            wave = pywt.Wavelet("db" + str(wave_num))
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
            filts = lowlevel.prep_filt_afb2d_nonsep(h0_col, h1_col, h0_row, h1_row)
            filts = nn.Parameter(filts, requires_grad=False)
            self.register_parameter("h" + str(i), filts)

            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
            filts = lowlevel.prep_filt_sfb2d_nonsep(g0_col, g1_col, g0_row, g1_row)
            filts = nn.Parameter(filts, requires_grad=False)
            self.register_parameter("g" + str(i), filts)

    def forward(self, x, is_encoder=True):
        if is_encoder:
            assert len(x.size()) == 4, "Dimension of x need to be 4, which corresponds to (N, C, H, W)"
            self.size = x.size()
            yh = []
            ll = x

            # Do a multilevel transform
            for i in range(self.level):
                # Do 1 level of the transform
                h = self.__getattr__("h" + str(i))
                y = lowlevel.afb2d_nonsep(ll, h, 'periodization')

                # Separate the low and bandpasses
                s = y.shape
                y = y.reshape(s[0], -1, 4, s[-2], s[-1])
                ll = y[:, :, 0].contiguous()
                yh_cur = y[:, :, 1:].contiguous() / self.q_table[i]
                yh_cur = random_round(yh_cur, self.rand_factor)
                yh.append(yh_cur)

            ll = ll / self.q_table[-1]
            ll = random_round(ll, self.rand_factor)
            return ll, yh

        else:
            assert len(x) == 2, "Must be tuple include LL and Hs"
            yl, yh = x
            yl = yl * self.q_table[-1]
            ll = yl

            # Do a multilevel inverse transform
            for i, h in enumerate(yh[::-1]):
                h = h * self.q_table[self.level - i - 1]

                # 'Unpad' added dimensions
                if ll.shape[-2] > h.shape[-2]:
                    ll = ll[..., :-1, :]
                if ll.shape[-1] > h.shape[-1]:
                    ll = ll[..., :-1]

                # Do the synthesis filter banks
                c = torch.cat((ll[:, :, None], h), dim=2)
                g = self.__getattr__("g" + str(self.level - i - 1))
                ll = lowlevel.sfb2d_nonsep(c, g, 'periodization')

            # 'Unpad' added dimensions
            if ll.shape[-2] > self.size[-2]:
                ll = ll[..., :-1, :]
            if ll.shape[-1] > self.size[-1]:
                ll = ll[..., :-1]

            return ll


class MaskCompressDWT(EncoderDecoderPair):
    def __init__(self, kwargs_dwt, channel_num, is_adaptive=False, ratio=0.5, softmin_tau=1):
        """
        Args:
            kwargs_dwt: keyword args for dwt
            channel_num (int): number of channels
            is_adaptive (bool): use CompressDWT or AdaptiveDWT
            ratio (float): retain ratio
            softmin_tau (float): temperature of softmin in softmin_round
        """
        super(MaskCompressDWT, self).__init__()
        self.compressDWT = AdaptiveDWT(**kwargs_dwt) if is_adaptive else CompressDWT(**kwargs_dwt)
        self.softmin_tau = softmin_tau
        self.freq_select = nn.Parameter(torch.ones(channel_num))
        # self.freq_select = nn.Parameter(torch.cat([torch.ones(channel_num // 2),
        #                                            torch.zeros(channel_num // 2)]))
        self.norm = channel_num * (1 - ratio)
        self.channel_num = channel_num
        self.ratio = ratio

    def forward(self, x, quanti=True, is_encoder=True):
        if self.training or (self.ratio == 1):
            if is_encoder:
                x = self.compressDWT(x, quanti=quanti, is_encoder=True)

                return x
            else:
                ll, h_list, size = x

                # masking
                freq_select_clamp = self.freq_select.clamp(0, 1)
                freq_select_norm = (freq_select_clamp / freq_select_clamp.sum()) * self.norm
                # freq_select_norm = (freq_select_clamp - freq_select_clamp.min()) / \
                #                    (freq_select_clamp.max() - freq_select_clamp.min() + 1e-10)
                if abs(self.ratio - 0.5) < 0.000001:
                    threshold = freq_select_norm.detach().median()
                else:
                    kth = int(freq_select_norm.size(0) * self.ratio)
                    if kth != 0:
                        threshold, _ = torch.kthvalue(freq_select_norm.detach(), kth)
                    else:
                        threshold = 0

                freq_select_norm = freq_select_norm.unsqueeze(dim=-1)
                freq_select_norm = freq_select_norm.unsqueeze(dim=-1)
                freq_select_norm = freq_select_norm.unsqueeze(dim=-1)

                mask_hard = (1 - (freq_select_norm > threshold).float())
                mask_soft = (1 - softmin_round(freq_select_norm, 0, 1, self.softmin_tau))
                mask = mask_hard.detach() + mask_soft - mask_soft.detach()
                h_list[0] = h_list[0] * mask

                x = self.compressDWT((ll, h_list, size), is_encoder=False)
                return x
        else:  # eval
            if is_encoder:
                assert self.channel_num == x.size(1), \
                    "Channel mismatch {}, {}".format(self.channel_num, x.size(1))

                remain_num = int(self.channel_num * self.ratio)
                mask_num = self.channel_num - remain_num
                x_remain, x_masked = x.split([remain_num, mask_num], dim=1)
                x_dwt_remain = self.compressDWT(x_remain, is_encoder=True)
                x_dwt_masked = self.compressDWT(x_masked, is_encoder=True)
                x_dwt_masked[1][0].zero_()
                return x_dwt_remain, x_dwt_masked
            else:
                x_dwt_remain, x_dwt_masked = x
                x_remain = self.compressDWT(x_dwt_remain, is_encoder=False)
                x_masked = self.compressDWT(x_dwt_masked, is_encoder=False)
                x = torch.cat([x_remain, x_masked], dim=1)

                return x

    def update(self):
        self.compressDWT.update()
        with torch.no_grad():
            freq_select_clamp = self.freq_select.clamp(0, 1)
            freq_select_norm = (freq_select_clamp / freq_select_clamp.sum()) * self.norm
            self.freq_select.data = freq_select_norm

    def reorder(self):
        with torch.no_grad():
            _, indx = self.freq_select.sort()
        return indx


class QuantiUnsign(EncoderDecoderPair):
    r"""
    Quantization with scaling factor and bit of unsigned integer

    Args:
        bit (int): Bits of integer after quantization
        q_factor (float): scale factor to match the range  after quantization
        is_shift (bool): Shift range of value from unsigned to symmetric signed
    """
    def __init__(self, bit=8, q_factor=1., is_shift=False):
        super(QuantiUnsign, self).__init__()
        self.bit = bit
        self.q_factor = q_factor
        self.is_shift = is_shift

    def forward(self, x, quanti=True, is_encoder=True):
        if is_encoder:
            x /= self.q_factor
            if quanti:
                x = random_round(x)
                x = x.clamp(0, 2 ** self.bit - 1)
            if self.is_shift:
                x = x - 2 ** (self.bit - 1)
            return x
        else:
            if self.is_shift:
                x = x + 2 ** (self.bit - 1)
            x = x * self.q_factor
            return x


class FtMapShiftNorm(EncoderDecoderPair):
    """
            Shift whole feature map with mean
        """
    is_bypass = True

    def __init__(self):
        super(FtMapShiftNorm, self).__init__()

    def forward(self, x, quanti=True, is_encoder=True):
        r"""

        Args:
            x ( torch.Tensor, tuple[torch.Tensor]):
            is_encoder (bool):

        Returns:

        """
        if is_encoder:
            x_mean = x
            for i in range(1, len(x_mean.size())):
                x_mean = x_mean.mean(i, keepdim=True)
            x = x - x_mean
            return x, x_mean
        else:
            x, x_mean = x
            x = x + x_mean

            return x


class Transform(EncoderDecoderPair):
    r"""
    Apply transform_matrix to pixels and maintain inverse_matrix for decoder

    The dimension of transform_matrix is (input channel, input channel) which maps feature map to
    same channel dimension

    Args:
        channel_num (int): The input channel number
    """
    def __init__(self, channel_num, norm_mode='l1', init_value=None):
        super(Transform, self).__init__()
        self.transform_matrix = nn.Parameter(torch.Tensor(channel_num, channel_num))
        # self.register_buffer('transform_matrix_reorder', None)
        self.register_buffer('transform_matrix_reorder', torch.Tensor(channel_num, channel_num))
        self.register_buffer('inverse_matrix', torch.Tensor(channel_num, channel_num))
        # self.register_buffer('inverse_matrix_reorder', None)
        self.register_buffer('inverse_matrix_reorder', torch.Tensor(channel_num, channel_num))
        self.norm_mode = norm_mode

        if init_value is None:
            # nn.init.uniform_(self.transform_matrix, 0, 1)
            # nn.init.kaiming_normal_(self.transform_matrix)
            # with torch.no_grad():
            #     self.transform_matrix /= self.transform_matrix.sum(dim=1)

            nn.init.eye_(self.transform_matrix)
        else:
            with torch.no_grad():
                self.transform_matrix.data = init_value.clone()

        self.update()

    def forward(self, x, is_encoder=True):
        if is_encoder:
            if self.training:
                weight = self.transform_matrix.unsqueeze(-1)
                weight = weight.unsqueeze(-1)
                if self.norm_mode == 'sum':
                    weight_norm = weight / weight.sum(dim=1, keepdim=True)  # sum
                elif self.norm_mode == 'l1':
                    weight_norm = weight / weight.abs().sum(dim=1, keepdim=True)  # L1
                elif self.norm_mode == 'l2':
                    weight_norm = weight / (weight ** 2).sum(dim=1, keepdim=True)  # L2
                else:
                    weight_norm = weight
                x_tr_de_w = F.conv2d(x, weight_norm.detach())
                x_tr_de_x = F.conv2d(x.detach(), weight_norm)
                # x_tr_de_x = F.conv2d(x, weight_norm)

                return x_tr_de_w, x_tr_de_x
            else:
                weight = self.transform_matrix_reorder.unsqueeze(-1)
                weight = weight.unsqueeze(-1)
                x = F.conv2d(x, weight)
                return x, x.clone()

        else:
            if self.training:
                weight = self.inverse_matrix.unsqueeze(-1)
            else:
                weight = self.inverse_matrix_reorder.unsqueeze(-1)
            weight = weight.unsqueeze(-1)
            x = F.conv2d(x, weight)

            return x

    def update(self):
        """
        Call update after transform_matrix is modified (e.g. after optimizer updated).
        """
        with torch.no_grad():
            if self.norm_mode == 'sum':
                self.transform_matrix.data /= self.transform_matrix.sum(dim=1, keepdim=True)  # sum
            elif self.norm_mode == 'l1':
                self.transform_matrix.data /= self.transform_matrix.abs().sum(dim=1, keepdim=True)  # L1
            elif self.norm_mode == 'l2':
                self.transform_matrix.data /= (self.transform_matrix ** 2).sum(dim=1, keepdim=True)  # L2
            # self.inverse_matrix.data = torch.pinverse(self.transform_matrix)
        self.inverse_matrix.data = torch.inverse(self.transform_matrix)

    def reorder(self, indx):
        with torch.no_grad():
            self.transform_matrix_reorder.data = self.transform_matrix[indx, :]
            self.inverse_matrix_reorder.data = torch.inverse(self.transform_matrix_reorder)


class DownSampleBranch(nn.Sequential):
    r"""
    DownSample Branch. Given transforms will apply to both paths
    Input need to be EncoderDecoderPair to ensure have both encode decode path

    Args:
            *args  (List[EncoderDecoderPair], Tuple[EncoderDecoderPair]): Sequence of EncoderDecoderPair
    """
    def __init__(self, *args):
        super(DownSampleBranch, self).__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                assert isinstance(module, EncoderDecoderPair), "BypassSequential only accept EncoderDecoderPair"
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                assert isinstance(module, EncoderDecoderPair), "BypassSequential only accept EncoderDecoderPair"
                self.add_module(str(idx), module)

    def forward(self, x, is_encoder=True):
        if is_encoder:
            x_org, x_dn = x.chunk(2, dim=-1)
            x_dn = F.avg_pool2d(x_dn, 2)

            return x_org, x_dn
        else:
            x_org, x_dn = x

            x_up = F.interpolate(x_dn, scale_factor=2)

            x = torch.cat((x_org, x_up), dim=1)

            return x


class DualPath(nn.Sequential):
    def __init__(self, separate, module):
        super(DualPath, self).__init__()
        self.separate = separate
        self.module = module

    def forward(self, x, is_encoder=True):
        if is_encoder:
            x_path_0, x_path_1 = self.separate(x)
            x_path_0 = self.module(x_path_0, is_encoder=True)
            # x_path_1 = self.module(x_path_1, quanti=(not self.training), is_encoder=True)
            x_path_1 = self.module(x_path_1, quanti=True, is_encoder=True)

            return x_path_0, x_path_1
        else:
            x, _ = x
            x = self.module(x, is_encoder=False)  # TODO only one pass?
            x = self.separate(x, is_encoder=False)
            return x

    def update(self):
        for module in self._modules.values():
            module.update()


class BypassSequential(nn.Sequential):
    r"""
    Sequential with bypass path
    Input need to be EncoderDecoderPair to ensure have both encode decode path
    The sequence of decoder will reverse automatically

    Args:
            *args  (List[EncoderDecoderPair], Tuple[EncoderDecoderPair]): Sequence of EncoderDecoderPair
    """
    def __init__(self, *args):
        super(BypassSequential, self).__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                assert isinstance(module, EncoderDecoderPair), "BypassSequential only accept EncoderDecoderPair"
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                assert isinstance(module, EncoderDecoderPair), "BypassSequential only accept EncoderDecoderPair"
                self.add_module(str(idx), module)

    def forward(self, x, quanti=True, is_encoder=True):
        if is_encoder:
            bypass_stack = []
            for module in self._modules.values():
                x = module(x, quanti, is_encoder=True)
                if module.is_bypass:
                    bypass = x[1]
                    x = x[0]

                    bypass_stack.append(bypass)
            return x, bypass_stack
        else:
            x, bypass_stack = x
            for indx, module in enumerate(reversed(self._modules.values())):
                if module.is_bypass:
                    x = module((x, bypass_stack[-indx]), is_encoder=False)
                else:
                    x = module(x, is_encoder=False)

        return x

    def update(self):
        for module in self._modules.values():
            module.update()


class Compress(nn.Module):
    """
    Forward do both encode and decode

    Args:
        compress (nn.Module): compress module, need to have both encode and decode ability

    Returns:
        x, fm_transforms
    """
    def __init__(self, compress):
        super(Compress, self).__init__()
        self.compress = compress

    def forward(self, x):
        fm_transforms = self.compress(x, is_encoder=True)
        x = self.compress(fm_transforms, is_encoder=False)

        return x, fm_transforms[0]  # [1][0] # TODO ugly

    def update(self):
        self.compress.update()

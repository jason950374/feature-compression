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


class EncoderDecoderPair(nn.Module):
    is_bypass = False
    bypass_indx = 1

    def __init__(self):
        super(EncoderDecoderPair, self).__init__()

    def forward(self, *inputs):
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
                    X = my_f.dct_2d(x[..., 8*i_h:8*(i_h+1), 8*i_w:8*(i_w+1)])
                    X = torch.round((X / q_table))
                    X = X.clamp(-2 ** (self.bit - 1), 2 ** (self.bit-1) - 1)
                    fm_transform[..., 8 * i_h:8 * (i_h + 1), 8 * i_w:8 * (i_w + 1)] = X

                if r_w != 0:
                    q_table_w = q_table[..., :r_w]
                    for i in range(r_w):
                        q_table_w[..., i] = q_table[..., i * 8 // r_w]

                    X = my_f.dct_2d(x[..., 8*i_h:8*(i_h+1), -r_w:])
                    X = torch.round((X / q_table_w))
                    X = X.clamp(-2 ** (self.bit - 1), 2 ** (self.bit-1) - 1)
                    fm_transform[..., 8 * i_h:8 * (i_h + 1), -r_w:] = X

            if r_h != 0:
                for i_w in range(W // 8):
                    q_table_h = q_table[..., :r_h, :]
                    for i in range(r_h):
                        q_table_h[..., i, :] = q_table[..., i * 8 // r_h, :]

                    X = my_f.dct_2d(x[..., -r_h:, 8 * i_w:8 * (i_w + 1)])
                    X = torch.round((X / q_table_h))
                    X = X.clamp(-2 ** (self.bit - 1), 2 ** (self.bit-1) - 1)
                    fm_transform[..., -r_h:, 8 * i_w:8 * (i_w + 1)] = X
                if r_w != 0:
                    q_table_h_w = q_table[..., :r_h, :r_w]
                    for i in range(r_h):
                        for j in range(r_w):
                            q_table_h_w[..., i, j] = q_table[..., i * 8 // r_h, j * 8 // r_w]

                    X = my_f.dct_2d(x[..., -r_h:, -r_w:])
                    X = torch.round((X / q_table_h_w))
                    X = X.clamp(-2 ** (self.bit - 1), 2 ** (self.bit-1) - 1)
                    fm_transform[..., -r_h:, -r_w:] = X
            return fm_transform
        else:
            fm_transform = x
            x = torch.ones_like(x)
            for i_h in range(H // 8):
                for i_w in range(W // 8):
                    X = fm_transform[..., 8 * i_h:8 * (i_h + 1), 8 * i_w:8 * (i_w + 1)]
                    x[..., 8 * i_h:8 * (i_h + 1), 8 * i_w:8 * (i_w + 1)] = my_f.idct_2d(X * q_table)

                if r_w != 0:
                    X = fm_transform[..., 8 * i_h:8 * (i_h + 1), -r_w:]
                    x[..., 8 * i_h:8 * (i_h + 1), -r_w:] = my_f.idct_2d(X * q_table[..., -r_w:])

            if r_h != 0:
                for i_w in range(W // 8):
                    X = fm_transform[..., -r_h:, 8 * i_w:8 * (i_w + 1)]
                    x[..., -r_h:, 8 * i_w:8 * (i_w + 1)] = my_f.idct_2d(X * q_table[..., -r_h:, :])
                if r_w != 0:
                    X = fm_transform[..., -r_h:, -r_w:]
                    x[..., -r_h:, -r_w:] = my_f.idct_2d(X * q_table[..., -r_h:, -r_w:])
            return x


class CompressDWT(EncoderDecoderPair):
    """
            Compress with DWT and Q table
            Args:
                    level (int): Level of DWT
                    bit (int): Bits after quantization
                    q_table (list, tuple): Quantization table, scale is fine to coarse
                    wave (str, pywt.Wavelet, tuple or list): Mother wavelet
        """
    def __init__(self, level=1, bit=8, q_table=None, wave='haar', rand_factor=0.):
        super(CompressDWT, self).__init__()
        self.bit = bit
        if q_table is None:
            self.register_buffer('q_table', torch.ones(level))
        else:
            assert q_table.size() == (level + 1,)
            self.register_buffer('q_table', q_table)
        self.level = level
        self.rand_factor = rand_factor
        self.size = None
        self.DWT = my_f.DWT(J=level, wave=wave, mode='periodization', separable=False)
        self.IDWT = my_f.IDWT(wave=wave, mode='periodization', separable=False)

    def forward(self, x, is_encoder=True):
        if is_encoder:
            assert len(x.size()) == 4, "Dimension of x need to be 4, which corresponds to (N, C, H, W)"
            self.size = x.size()
            XL, XH = self.DWT(x)
            XL = XL / self.q_table[-1]
            XL = random_round(XL, self.rand_factor)
            XL = XL.clamp(-2 ** (self.bit - 1), 2 ** (self.bit - 1) - 1)
            for i in range(self.level):
                XH[i] = XH[i] / self.q_table[i]
                XH[i] = random_round(XH[i], self.rand_factor)
                XH[i] = XH[i].clamp(-2 ** (self.bit - 1), 2 ** (self.bit - 1) - 1)

            return XL, XH

        else:
            assert len(x) == 2, "Must be tuple include LL and Hs"
            XL, XH_org = x
            XL = XL * self.q_table[-1]
            XH = []
            for i in range(self.level):
                XH.append(XH_org[i] * self.q_table[i])

            x = self.IDWT((XL, XH), self.size)

            return x


class AdaptiveDWT(EncoderDecoderPair):
    """
                Compress with DWT and Q table
                The wave will select adaptively
                Args:
                        level (int): Level of DWT
                        bit (int): Bits after quantization
                        q_table (list, tuple): Quantization table, scale is fine to coarse
            """

    def __init__(self, x_size, level=1, bit=8, q_table=None, rand_factor=0.):
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


class QuantiUnsign(EncoderDecoderPair):
    """
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

    def forward(self, x, is_encoder=True):
        if is_encoder:
            x /= self.q_factor
            x = torch.round(x).detach() + x - x.detach()
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
    bypass_indx = 1

    def __init__(self):
        super(FtMapShiftNorm, self).__init__()

    def forward(self, x, is_encoder=True):
        """
                First dimension of input seem as batch
                    For encoder: arg is input
                    For decoder: arg is both input and mean
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
    """
        Apply transform_matrix to pixels and maintain inverse_matrix for decoder

        The dimension of transform_matrix is (input channel, input channel) which maps feature map to
        same channel dimension

        Args:
            channel_num (int): The input channel number
    """
    def __init__(self, channel_num, init_value=None):
        super(Transform, self).__init__()
        self.transform_matrix = nn.Parameter(torch.Tensor(channel_num, channel_num))
        self.register_buffer('inverse_matrix', torch.Tensor(channel_num, channel_num))

        # nn.init.kaiming_normal_(self.transform_matrix)
        if init_value is None:
            # nn.init.uniform_(self.transform_matrix, 0, 1)
            # with torch.no_grad():
            #     self.transform_matrix /= self.transform_matrix.sum(dim=1)

            nn.init.eye_(self.transform_matrix)
        else:
            with torch.no_grad():
                self.transform_matrix.data = init_value.clone()

        self.update()

    def forward(self, x, is_encoder=True):
        if is_encoder:
            weight = self.transform_matrix.unsqueeze(-1)
            weight = weight.unsqueeze(-1)
            x_tr = F.conv2d(x, weight)

            return x_tr

        else:
            weight = self.inverse_matrix.unsqueeze(-1)
            weight = weight.unsqueeze(-1)
            x = F.conv2d(x, weight)

            return x

    def update(self):
        """
        Call update after transform_matrix is modified (e.g. after optimizer updated).
        """
        self.inverse_matrix.data = torch.pinverse(self.transform_matrix)


class DownSampleBranch(nn.Sequential):
    """
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


class BypassSequential(nn.Sequential):
    """
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

    def forward(self, x, is_encoder=True):
        if is_encoder:
            bypass_stack = []
            for module in self._modules.values():
                x = module(x, is_encoder=True)
                if module.is_bypass:
                    bypass = x[module.bypass_indx:]
                    x = x[:module.bypass_indx]

                    if len(bypass) == 1:
                        bypass = bypass[0]
                    if len(x) == 1:
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

                Return:
                    x, fm_transforms
            """
    def __init__(self, compress):
        super(Compress, self).__init__()
        self.compress = compress

    def forward(self, x):
        fm_transforms = self.compress(x, is_encoder=True)
        x = self.compress(fm_transforms, is_encoder=False)
        fm_transforms, bypass_stack = fm_transforms  # TODO clean up
        return x, fm_transforms

    def update(self):
        self.compress.update()

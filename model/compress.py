import torch.nn as nn
import torch
import functional as my_f
from collections import OrderedDict


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

    def __init__(self, is_encoder=True):
        super(EncoderDecoderPair, self).__init__()
        self.is_encoder = is_encoder

    def __init_single__(self, is_encoder):
        self.is_encoder = is_encoder

    def forward(self, *inputs):
        raise NotImplementedError


class CompressDCT(EncoderDecoderPair):
    """
        Compress with DCT and Q table
        Args:
            bit (int): Bits after quantization
            q_table: Quantization table
            is_encoder (bool):  True if is encoder for CompressDCT. False if is decoder for CompressDCT
    """
    def __init__(self, bit=8, q_table=None, is_encoder=True):
        super(CompressDCT, self).__init__(is_encoder)
        self.bit = bit
        if q_table is None:
            self.register_buffer('q_table', torch.ones(8, 8))
        else:
            assert q_table.size() == (8, 8)
            self.register_buffer('q_table', q_table)

    # TODO  and BP path
    def forward(self, x):
        assert len(x.size()) == 4, "Dimension of x need to be 4, which corresponds to (N, C, H, W)"
        N, C, H, W = x.size()
        q_table = self.q_table.repeat(N, C, 1, 1)
        r_h = H % 8
        r_w = W % 8
        if self.is_encoder:
            fm_transform = torch.zeros_like(x)
            for i_h in range(H // 8):
                for i_w in range(W // 8):
                    X = my_f.dct_2d(x[..., 8*i_h:8*(i_h+1), 8*i_w:8*(i_w+1)])
                    X = torch.round((X / q_table))
                    X = X.clamp(-2 ** (self.bit - 1), 2 ** (self.bit-1) - 1)
                    fm_transform[..., 8 * i_h:8 * (i_h + 1), 8 * i_w:8 * (i_w + 1)] = X

                if r_w != 0:
                    X = my_f.dct_2d(x[..., 8*i_h:8*(i_h+1), -r_w:])
                    X = torch.round((X / q_table[..., -r_w:]))
                    X = X.clamp(-2 ** (self.bit - 1), 2 ** (self.bit-1) - 1)
                    fm_transform[..., 8 * i_h:8 * (i_h + 1), -r_w:] = X

            if r_h != 0:
                for i_w in range(W // 8):
                    X = my_f.dct_2d(x[..., -r_h:, 8 * i_w:8 * (i_w + 1)])
                    X = torch.round((X / q_table[..., -r_h:, :]))
                    X = X.clamp(-2 ** (self.bit - 1), 2 ** (self.bit-1) - 1)
                    fm_transform[..., -r_h:, 8 * i_w:8 * (i_w + 1)] = X
                if r_w != 0:
                    X = my_f.dct_2d(x[..., -r_h:, -r_w:])
                    X = torch.round((X / q_table[..., -r_h:, -r_w:]))
                    X = X.clamp(-2 ** (self.bit - 1), 2 ** (self.bit-1) - 1)
                    fm_transform[..., -r_h:, -r_w:] = X
            return fm_transform
        else:
            fm_transform = x
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
                    is_encoder (bool):  True if is encoder for CompressDWT. False if is decoder for CompressDWT
        """
    def __init__(self, level=1, bit=8, q_table=None, wave='haar', rand_factor=0., is_encoder=True):
        super(CompressDWT, self).__init__(is_encoder)
        self.bit = bit
        if q_table is None:
            self.register_buffer('q_table', torch.ones(level))
        else:
            assert q_table.size() == (level + 1,)
            self.register_buffer('q_table', q_table)
        self.level = level
        self.rand_factor = rand_factor
        if is_encoder:
            self.DWT = my_f.DWT(J=level, wave=wave, mode='periodization', separable=False)
        else:
            self.IDWT = my_f.IDWT(wave=wave, mode='periodization', separable=False)

    def forward(self, x):
        if self.is_encoder:
            assert len(x.size()) == 4, "Dimension of x need to be 4, which corresponds to (N, C, H, W)"
            XL, XH = self.DWT(x)
            XL = XL / self.q_table[-1]
            # XL = torch.round(XL).detach() + XL - XL.detach()
            XL = random_round(XL, self.rand_factor)
            XL = XL.clamp(-2 ** (self.bit - 1), 2 ** (self.bit - 1) - 1)
            for i in range(self.level):
                XH[i] = XH[i] / self.q_table[i]
                # XH[i] = torch.round(XH[i]).detach() + XH[i] - XH[i].detach()
                XH[i] = random_round(XH[i], self.rand_factor)
                XH[i] = XH[i].clamp(-2 ** (self.bit - 1), 2 ** (self.bit - 1) - 1)

            return XL, XH

        else:
            assert len(x) == 2, "Must be tuple include LL and Hs"
            XL, XH = x
            XL = XL * self.q_table[-1]
            for i in range(self.level):
                XH[i] = XH[i] * self.q_table[i]

            x = self.IDWT((XL, XH))

            return x


class QuantiUnsign(EncoderDecoderPair):
    """
                Quantization with scaling factor and bit of unsigned integer

                Args:
                    bit (int): Bits of integer after quantization
                    q_factor (float): scale factor to match the range  after quantization
                    is_shift (bool): Shift range of value from unsigned to symmetric signed
                    is_encoder (bool):  True if is encoder for QuantiUnsign. False if is decoder for QuantiUnsign
            """
    def __init__(self, bit=8, q_factor=1., is_shift=False, is_encoder=True):
        super(QuantiUnsign, self).__init__(is_encoder)
        self.bit = bit
        self.q_factor = q_factor
        self.is_shift = is_shift

    def forward(self, x):
        if self.is_encoder:
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
            Args:
                is_encoder(bool):  True if is encoder for FtMapShiftNorm. False if is decoder for FtMapShiftNorm
        """
    is_bypass = True
    bypass_indx = 1

    def __init__(self, is_encoder=True):
        super(FtMapShiftNorm, self).__init__(is_encoder)

    def forward(self, x):
        """
                First dimension of input seem as batch
                    For encoder: arg is input
                    For decoder: arg is both input and mean
            """
        if self.is_encoder:
            x_mean = x
            for i in range(1, len(x_mean.size())):
                x_mean = x_mean.mean(i, keepdim=True)
            x = x - x_mean
            return x, x_mean
        else:
            x, x_mean = x
            x = x + x_mean
            return x


class BypassSequential(nn.Sequential, EncoderDecoderPair):
    """
                Sequential with bypass path
                Input need to be EncoderDecoderPair to ensure
                Args:
                        *args  (List[EncoderDecoderPair], Tuple[EncoderDecoderPair]): Sequence of EncoderDecoderPair
                        is_encoder (bool):  True if is encoder for FtMapShiftNorm.
                                                    False if is decoder for FtMapShiftNorm
            """
    def __init__(self, *args, is_encoder=True):
        super(BypassSequential, self).__init__()
        super(nn.Sequential, self).__init_single__(is_encoder)

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                assert isinstance(module, EncoderDecoderPair), "BypassSequential only accept EncoderDecoderPair"
                assert module.is_encoder == is_encoder, "is_encoder of sub module does not match"
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                assert isinstance(module, EncoderDecoderPair), "BypassSequential only accept EncoderDecoderPair"
                assert module.is_encoder == is_encoder, "is_encoder of sub module does not match"
                self.add_module(str(idx), module)

    def forward(self, x):
        if self.is_encoder:
            bypass_stack = []
            for module in self._modules.values():
                x = module(x)
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
            for indx, module in enumerate(self._modules.values()):
                if module.is_bypass:
                    x = module((x, bypass_stack[-indx]))
                else:
                    x = module(x)
            return x

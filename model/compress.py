import torch.nn as nn
import torch
import functional as my_f


class EncoderDecoderPair(nn.Module):
    def __init__(self, is_encoder):
        super(EncoderDecoderPair, self).__init__()
        self.is_encoder = is_encoder

    def forward(self, *input):
        raise NotImplementedError


# TODO clamp not included yet
class CompressDCT(EncoderDecoderPair):
    """
        Compress with DCT and Q table
        :param q_table: Quantization table
    """
    def __init__(self, q_table=None, is_encoder=True):
        super(CompressDCT, self).__init__(is_encoder)
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
                    fm_transform[..., 8 * i_h:8 * (i_h + 1), 8 * i_w:8 * (i_w + 1)] = X

                if r_w != 0:
                    X = my_f.dct_2d(x[..., 8*i_h:8*(i_h+1), -r_w:])
                    X = torch.round((X / q_table[..., -r_w:]))
                    fm_transform[..., 8 * i_h:8 * (i_h + 1), -r_w:] = X

            if r_h != 0:
                for i_w in range(W // 8):
                    X = my_f.dct_2d(x[..., -r_h:, 8 * i_w:8 * (i_w + 1)])
                    X = torch.round((X / q_table[..., -r_h:, :]))
                    fm_transform[..., -r_h:, 8 * i_w:8 * (i_w + 1)] = X
                if r_w != 0:
                    X = my_f.dct_2d(x[..., -r_h:, -r_w:])
                    X = torch.round((X / q_table[..., -r_h:, -r_w:]))
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

# TODO clamp not included yet
class CompressDWT(EncoderDecoderPair):
    """
            Compress with DWT and Q table
            :param level: Level of DWT
            :param q_table: Quantization table, scale is fine to coarse
            :param wave: mother wavelet
        """
    def __init__(self, level=1, q_table=None, wave='haar', is_encoder=True):
        super(CompressDWT, self).__init__(is_encoder)
        if q_table is None:
            self.register_buffer('q_table', torch.ones(level))
        else:
            assert q_table.size() == (level,)
            self.register_buffer('q_table', q_table)
        self.level = level
        if is_encoder:
            self.DWT = my_f.DWT(J=level, wave=wave, mode='no_pad', separable=False)
        else:
            self.IDWT = my_f.IDWT(wave=wave, mode='no_pad', separable=False)

    # TODO  and BP path
    def forward(self, x):
        if self.is_encoder:
            assert len(x.size()) == 4, "Dimension of x need to be 4, which corresponds to (N, C, H, W)"
            XL, XH = self.DWT(x)
            XL = torch.round(XL * 1)
            for i in range(self.level):
                XH[i] = torch.round(XH[i] / self.q_table[i])

            return XL, XH

        else:
            assert len(x) == 2, "Must be tuple include LL and Hs"
            XL, XH = x
            XL = XL / 1
            for i in range(self.level):
                XH[i] = XH[i] * self.q_table[i]

            x = self.IDWT((XL, XH))

            return x


class QuantiUnsign(EncoderDecoderPair):
    """
                Quantization with scaling factor and bit of unsigned integer

                :param bit: Bits of integer after quantization
                :param q_factor: scale factor to match the range  after quantization
            """
    def __init__(self, bit=8, q_factor=1, is_encoder=True):
        super(QuantiUnsign, self).__init__(is_encoder)
        self.bit = bit
        self.q_factor = q_factor

    # TODO  and BP path
    def forward(self, x):
        if self.is_encoder:
            x /= self.q_factor
            x = torch.round(x)
            x = x.clamp(0, 2 ** self.bit - 1)
            return x
        else:
            x = x * self.q_factor

            return x

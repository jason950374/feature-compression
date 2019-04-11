import torch.nn as nn
import torch
import functional as my_f


# TODO clamp not included yet
class CompressDCT(nn.Module):
    """
        Compress with DCT and Q table
        :param q_table: Quantization table
    """
    def __init__(self, q_table=None):
        super(CompressDCT, self).__init__()
        if q_table is None:
            self.register_buffer('q_table', torch.ones(8, 8))
        else:
            assert q_table.size() == (8, 8)
            self.register_buffer('q_table', q_table)

    # TODO  and BP path
    def forward(self, x):
        assert len(x.size()) == 4, "Dimension of x need to be 4, which corresponds to (N, C, H, W)"
        X = my_f.dct(x)
        N, C, H, W = x.size()
        # TODO remove debugging assert & original design
        q_table_org = torch.cat([torch.cat([self.q_table.unsqueeze(0)] * C).unsqueeze(0)] * N)
        q_table = self.q_table.repeat(N, C, 1, 1)
        assert (q_table_org - q_table).abs().max() < 10-100, (q_table_org - q_table).abs().max()
        r_h = H % 8
        r_w = W % 8
        for i_h in range(H // 8):
            for i_w in range(W // 8):
                X = my_f.dct_2d(x[..., 8*i_h:8*(i_h+1), 8*i_w:8*(i_w+1)])
                X = torch.round((X / q_table))
                x[..., 8 * i_h:8 * (i_h + 1), 8 * i_w:8 * (i_w + 1)] = my_f.idct_2d(X* q_table)

            if r_w != 0:
                X = my_f.dct_2d(x[..., 8*i_h:8*(i_h+1), -r_w:])
                X = torch.round((X / q_table[..., -r_w:]))
                x[..., 8 * i_h:8 * (i_h + 1), -r_w:] = my_f.idct_2d(X* q_table[..., -r_w:])

        if r_h != 0:
            for i_w in range(W // 8):
                X = my_f.dct_2d(x[..., -r_h:, 8 * i_w:8 * (i_w + 1)])
                X = torch.round((X / q_table[..., -r_h:, :]))
                x[..., -r_h:, 8 * i_w:8 * (i_w + 1)] = my_f.idct_2d(X* q_table[..., -r_h:, :])
            if r_w != 0:
                X = my_f.dct_2d(x[..., -r_h:, -r_w:])
                X = torch.round((X / q_table[..., -r_h:, -r_w:]))
                x[..., -r_h:, -r_w:] = my_f.idct_2d(X * q_table[..., -r_h:, -r_w:])

        return x


class CompressDWT(nn.Module):
    """
            Compress with DCT and Q table
            :param q_table: Quantization table, scale is fine to coarse
        """

    def __init__(self, level=1, q_table=None, wave='haar'):
        super(CompressDWT, self).__init__()
        if q_table is None:
            self.register_buffer('q_table', torch.ones(level))
        else:
            assert q_table.size() == (level,)
            self.register_buffer('q_table', q_table)
        self.level = level
        self.DWT = my_f.DWT(J=level, wave=wave, mode='no_pad', separable=False)
        self.IDWT = my_f.IDWT(wave=wave, mode='no_pad', separable=False)

    # TODO  and BP path
    def forward(self, x):
        assert len(x.size()) == 4, "Dimension of x need to be 4, which corresponds to (N, C, H, W)"
        XL, XH = self.DWT(x)
        XL = torch.round(XL*10) / 10
        for i in range(self.level):
            XH[i] = torch.round(XH[i] / self.q_table[i])

        # reconstruct
        for i in range(self.level):
            XH[i] = XH[i] * self.q_table[i]

        x = self.IDWT((XL, XH))

        return x

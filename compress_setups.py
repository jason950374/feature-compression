import torch
from model.compress import *


def quant(bit, q_factor):
    return Compress(QuantiUnsign(bit=bit, q_factor=q_factor, is_shift=False).cuda()).cuda()


def dwt(q_table_dwt, wavelet, bit, q_factor, shift=True):
    compress_seq = [
        QuantiUnsign(bit=bit, q_factor=q_factor, is_shift=False).cuda(),
        FtMapShiftNorm() if shift else None,
        CompressDWT(level=len(q_table_dwt)-1, bit=bit, q_table=q_table_dwt, wave=wavelet).cuda()
    ]
    if not shift:
        compress_seq.remove(None)
    seq = BypassSequential(*compress_seq)
    return Compress(seq).cuda()


def with_tran(channel, q_table_dwt, wavelet, bit, q_factor, norm_mode, shift=True):
    tr = Transform_seperate(channel, norm_mode=norm_mode).cuda()
    compress_seq = [
        QuantiUnsign(bit=bit, q_factor=q_factor, is_shift=False).cuda(),
        FtMapShiftNorm() if shift else None,
        CompressDWT(level=3, bit=bit, q_table=q_table_dwt, wave=wavelet).cuda()
    ]
    if not shift:
        compress_seq.remove(None)

    seq = BypassSequential(*compress_seq)
    pair = DualPath(tr, seq)
    return Compress(pair).cuda()


def with_mask_M(channel, q_table_dwt, wavelet, bit, q_factor, norm_mode, retain_ratio, tau_mask, shift=True):
    tr = Transform_seperate(channel, norm_mode=norm_mode).cuda()
    compress_seq = [
        QuantiUnsign(bit=bit, q_factor=q_factor, is_shift=False).cuda(),
        FtMapShiftNorm() if shift else None,
        MaskCompressDWT({"level": 3, "bit": bit, "q_table": q_table_dwt, "wave": wavelet},
                        channel, ratio=retain_ratio, softmin_tau=tau_mask).cuda()
    ]
    if not shift:
        compress_seq.remove(None)

    seq = BypassSequential(*compress_seq)
    pair = DualPath(tr, seq)
    return Compress(pair).cuda()


def with_mask(channel, q_table_dwt, wavelet, bit, q_factor, norm_mode, retain_ratio, tau_mask, shift=True):
    compress_seq = [
        Transform(channel, norm_mode=norm_mode).cuda(),
        QuantiUnsign(bit=bit, q_factor=q_factor, is_shift=False).cuda(),
        FtMapShiftNorm() if shift else None,
        MaskCompressDWT({"level": 3, "bit": bit, "q_table": q_table_dwt, "wave": wavelet},
                        channel, ratio=retain_ratio, softmin_tau=tau_mask).cuda()
    ]
    if not shift:
        compress_seq.remove(None)

    seq = BypassSequential(*compress_seq)
    return Compress(seq).cuda()


def compress_list_gen_branch(channel, maximum_fm, wavelet='db2', bit=8, norm_mode='l1', retain_ratio=0.5, tau_mask=1):
    compress_list = []
    for i in range(len(maximum_fm) - 1):
        q_factor = maximum_fm[i] / (2 ** bit - 1)

        q_table_dwt = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.get_default_dtype())
        q_list_dct = [25, 25, 25, 25, 25, 25, 25, 25,
                      25, 25, 25, 25, 25, 25, 25]

        q_table_dwt = q_table_dwt * (2 ** bit - 1) / maximum_fm[i]

        # c = quant(bit, q_factor)
        # c = dwt(q_table_dwt, wavelet, bit, q_factor)
        # c = with_mask_M(channel[i], q_table_dwt, wavelet, bit, q_factor, norm_mode, retain_ratio, tau_mask)
        c = with_mask(channel[i], q_table_dwt, wavelet, bit, q_factor, norm_mode, retain_ratio, tau_mask)
        compress_list.append(c)

    q_factor = maximum_fm[-1] / (2 ** bit - 1)
    q_table_dwt = torch.tensor([10 ** 6, 10 ** 6, 10 ** 6, 1], dtype=torch.get_default_dtype())
    q_list_dct = [25, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6,
                  10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6]

    q_table_dwt = q_table_dwt * (2 ** bit - 1) / maximum_fm[-1]

    # c = quant(bit, q_factor)
    # c = dwt(q_table_dwt, wavelet, bit, q_factor, shift=False)
    #c = with_mask_M(channel[-1], q_table_dwt, 'haar', bit, q_factor, norm_mode, retain_ratio, tau_mask, shift=False)
    c = with_mask(channel[-1], q_table_dwt, 'haar', bit, q_factor, norm_mode, retain_ratio, tau_mask, shift=False)
    compress_list.append(c)

    return compress_list


def compress_list_gen_block(channel, maximum_fm, wavelet='db2', bit=8, norm_mode='l1', retain_ratio=0.5, tau_mask=1):
    compress_list = []
    for i in range(len(maximum_fm)):
        q_factor = maximum_fm[i] / (2 ** bit - 1)

        # if i == 0:
        #     q_table_dwt = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.get_default_dtype())
        # else:
        q_table_dwt = torch.tensor([0.001, 0.01, 0.01, 0.01, 0.01, 0.1], dtype=torch.get_default_dtype())
        q_list_dct = [25, 25, 25, 25, 25, 25, 25, 25,
                      25, 25, 25, 25, 25, 25, 25]

        q_table_dwt = q_table_dwt * (2 ** bit - 1)  # / maximum_fm[i]

        c = quant(bit, q_factor)
        # c = dwt(q_table_dwt, wavelet, bit, q_factor)
        # c = with_mask(channel[i], q_table_dwt, wavelet, bit, q_factor, norm_mode, retain_ratio, tau_mask)
        compress_list.append(c)

    return compress_list

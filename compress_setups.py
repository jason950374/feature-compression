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


def with_mask(channel, q_table_dwt, wavelet, bit, q_factor, norm_mode, retain_ratio, tau_mask, freq_select=None, shift=True):
    compress_seq = [
        Transform(channel, norm_mode=norm_mode).cuda(),
        QuantiUnsign(bit=bit, q_factor=q_factor, is_shift=False).cuda(),
        FtMapShiftNorm() if shift else None,
        MaskCompressDWT({"level": 3, "bit": bit, "q_table": q_table_dwt, "wave": wavelet},
                        channel, freq_select=freq_select, ratio=retain_ratio, softmin_tau=tau_mask).cuda()
    ]
    if not shift:
        compress_seq.remove(None)

    seq = BypassSequential(*compress_seq)
    return Compress(seq).cuda()


def compress_list_gen_branch(channel, maximum_fm, wavelet='db2', bit=8, dwt_coe_branch=None, norm_mode='l1',
                             retain_ratio=0.5, tau_mask=1, freq_selects=None):
    compress_list = []
    for i in range(len(maximum_fm) - 1):
        q_factor = maximum_fm[i] / (2 ** bit - 1)

        q_table_dwt = torch.tensor([1, 1, 1, 1], dtype=torch.get_default_dtype())
        # q_table_dwt = torch.tensor([0.5, 0.25, 0.125, 0.125], dtype=torch.get_default_dtype())
        q_list_dct = [25, 25, 25, 25, 25, 25, 25, 25,
                      25, 25, 25, 25, 25, 25, 25]

        q_table_dwt = q_table_dwt * (2 ** bit - 1)
        if dwt_coe_branch is not None:
            q_table_dwt *= dwt_coe_branch[i]

        freq_select = freq_selects[i] if freq_selects is not None else None

        # c = quant(bit, q_factor)
        # c = dwt(q_table_dwt, wavelet, bit, q_factor, shift=False)
        # c = with_mask_M(channel[i], q_table_dwt, wavelet, bit, q_factor, norm_mode, retain_ratio, tau_mask)
        c = with_mask(channel[i], q_table_dwt, wavelet, bit, q_factor, norm_mode, retain_ratio, tau_mask,
                      freq_select=freq_select)
        compress_list.append(c)

    q_factor = maximum_fm[-1] / (2 ** bit - 1)
    q_table_dwt = torch.tensor([10 ** 6, 10 ** 6, 10 ** 6, 1], dtype=torch.get_default_dtype())
    q_list_dct = [25, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6,
                  10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6, 10 ** 6]

    q_table_dwt = q_table_dwt * (2 ** bit - 1)
    if dwt_coe_branch is not None:
        q_table_dwt *= dwt_coe_branch[-1]

    freq_select = freq_selects[-1] if freq_selects is not None else None
    # c = quant(bit, q_factor)
    # c = dwt(q_table_dwt, wavelet, bit, q_factor, shift=False)
    # c = with_mask_M(channel[-1], q_table_dwt, 'haar', bit, q_factor, norm_mode, retain_ratio, tau_mask, shift=False)
    c = with_mask(channel[-1], q_table_dwt, 'haar', bit, q_factor, norm_mode, retain_ratio, tau_mask,
                  freq_select=freq_select, shift=False)
    compress_list.append(c)

    return compress_list


def compress_list_gen_block(channel, maximum_fm, wavelet='db2', bit=8, dwt_coe_block=None, norm_mode='l1',
                            retain_ratio=0.5, tau_mask=1,
                            freq_selects=None):
    compress_list = []
    for i in range(len(maximum_fm)):
        q_factor = maximum_fm[i] / (2 ** bit - 1)

        '''if i == 0:
            q_table_dwt = torch.tensor([1, 1, 1, 1], dtype=torch.get_default_dtype())
        else:'''
        q_table_dwt = torch.tensor([1, 1, 1, 1], dtype=torch.get_default_dtype())
        q_list_dct = [25, 25, 25, 25, 25, 25, 25, 25,
                      25, 25, 25, 25, 25, 25, 25]

        q_table_dwt = q_table_dwt * (2 ** bit - 1)

        freq_select = freq_selects[i] if freq_selects is not None else None
        if dwt_coe_block is not None:
            q_table_dwt *= dwt_coe_block[i]

        # c = quant(bit, q_factor)
        c = dwt(q_table_dwt, wavelet, bit, q_factor)
        # c = with_mask_M(channel[i], q_table_dwt, wavelet, bit, q_factor, norm_mode, retain_ratio, tau_mask)
        # c = with_mask(channel[i], q_table_dwt, wavelet, bit, q_factor, norm_mode, retain_ratio, tau_mask,
        #               freq_select=freq_select)
        compress_list.append(c)

    return compress_list

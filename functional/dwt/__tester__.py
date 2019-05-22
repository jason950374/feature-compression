import torch
from functional.dwt import DWTForward, DWTInverse
from functional.dwt.lowlevel import *

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    '''
    x = torch.Tensor(10, 64, 9, 9).cuda()
    # x = torch.Tensor(1, 1, 2, 2).cuda()
    # x.normal_(0, 1) * 255
    x.uniform_(0, 255)
    # x.fill_(1)

    dwt = DWTForward(J=3, wave='db7', mode='per', separable=True).cuda()
    dwti = DWTInverse(wave='db7', mode='per', separable=True).cuda()
    X = dwt(x)
    x_reconstruct = dwti(X, x.size())
    error = torch.abs(x - x_reconstruct)
    assert error.max().item() < 1e-10, (error.mean(), error.max())

    dwt_nonsep = DWTForward(J=3, wave='db7', mode='per', separable=False).cuda()
    dwti_nonsep = DWTInverse(wave='db7', mode='per', separable=False).cuda()
    X_nonsep = dwt_nonsep(x)
    x_reconstruct = dwti_nonsep(X_nonsep, x.size())
    error = torch.abs(x - x_reconstruct)
    assert error.max().item() < 1e-10, (error.mean(), error.max())
    
    error = torch.abs(X[0] - X_nonsep[0])
    assert error.max().item() < 1e-10, (error.mean(), error.max())
    for H, H_nonsep in zip(X[1], X_nonsep[1]):
        error = torch.abs(H - H_nonsep)
        assert error.max().item() < 1e-10, (error.mean(), error.max())'''

    B = 1
    C = 1
    N = 5
    x = torch.Tensor(B, C, 1, N).cuda()
    x.uniform_(0, 255)
    # x.fill_(1)
    wave = pywt.Wavelet('db2')
    '''------------------------------------------------------------------'''
    # X = afb1d(x, wave.dec_lo, wave.dec_hi, mode='per')
    h_lo = torch.tensor(np.copy(np.array(wave.dec_lo).ravel()[::-1])).cuda()
    h_hi = torch.tensor(np.copy(np.array(wave.dec_hi).ravel()[::-1])).cuda()
    L = h_lo.numel()
    shape = [1, 1, 1, 1]
    shape[-1] = L
    if h_lo.shape != tuple(shape):
        h_lo = h_lo.reshape(*shape)
    if h_hi.shape != tuple(shape):
        h_hi = h_hi.reshape(*shape)
    h_hi = torch.cat([h_hi] * C, dim=0)
    h_lo = torch.cat([h_lo] * C, dim=0)

    L2 = L // 2
    N2 = N // 2
    x_ext = x
    x_ext = roll(x_ext, -L2, dim=-1)
    if x.shape[-1] % 2 == 1:
        x_ext = torch.cat((x_ext, x_ext[..., -1:]), dim=3)
        N2 += 1

    pad = (0, L - 1)
    ll = F.conv2d(x_ext, h_lo, padding=pad, stride=(1, 2), groups=C)
    h = F.conv2d(x_ext, h_hi, padding=pad, stride=(1, 2), groups=C)

    ll[:, :, :, :L2] += ll[:, :, :, N2:N2 + L2]
    h[:, :, :, :L2] += h[:, :, :, N2:N2 + L2]
    ll = ll[:, :, :, :N2]
    h = h[:, :, :, :N2]
    ll_temp = ll.clone()
    ll_temp[..., -1] = 0.5 * ll_temp[..., -1] + 0.5 * h[..., -1]
    h_temp = h[..., :-1]

    '''------------------------------------------------------------------'''
    # x_reconstruct = sfb1d(ll, h, wave.rec_lo, wave.rec_hi, mode='per')
    g_lo = torch.tensor(np.copy(np.array(wave.rec_lo).ravel())).cuda()
    g_hi = torch.tensor(np.copy(np.array(wave.rec_hi).ravel())).cuda()

    # If g aren't in the right shape, make them so
    if g_lo.shape != tuple(shape):
        g_lo = g_lo.reshape(*shape)
    if g_hi.shape != tuple(shape):
        g_hi = g_hi.reshape(*shape)

    g_lo = torch.cat([g_lo] * C, dim=0)
    g_hi = torch.cat([g_hi] * C, dim=0)

    x_l = F.conv_transpose2d(ll, g_lo, stride=(1, 2), groups=C)
    x_h = F.conv_transpose2d(h, g_hi, stride=(1, 2), groups=C)
    x_l_temp = F.conv_transpose2d(ll_temp, g_lo, stride=(1, 2), groups=C)
    h_temp = torch.cat([h_temp, ll_temp[..., -1:]], dim=-1)
    x_h_temp = F.conv_transpose2d(h_temp, g_hi, stride=(1, 2), groups=C)

    N_temp = N + (N % 2)

    x_h[..., :L - 2] += x_h[..., N_temp:N_temp + L - 2]
    x_l[..., :L - 2] += x_l[..., N_temp:N_temp + L - 2]
    x_l_temp[..., :L - 2] += x_l_temp[..., N - 1:N - 1 + L - 2]
    x_h_temp[..., :L - 2] += x_h_temp[..., N - 1:N - 1 + L - 2]
    x_reconstruct = x_l + x_h
    x_reconstruct_temp = x_l_temp + x_h_temp

    x_reconstruct = x_reconstruct[..., :N_temp]
    x_reconstruct_temp = x_reconstruct_temp[..., :N_temp]
    x_reconstruct = roll(x_reconstruct, 1 + (N % 2) - L // 2, dim=-1)
    x_reconstruct_temp = roll(x_reconstruct_temp, 1 + (N % 2) - L // 2, dim=-1)
    # x_reconstruct_temp[..., L2:] = x_reconstruct_temp[..., L+1:]
    x_reconstruct = torch.cat([x_reconstruct[..., :L2-1], x_reconstruct[..., L2:]], dim=-1)

    error = torch.abs(x - x_reconstruct)
    assert error.max().item() < 1e-10, (error.mean(), error.max())


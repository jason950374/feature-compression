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

    x = torch.Tensor(16, 11).cuda()



import torch
import torch.nn.functional as F
import numpy as np
from functional.dwt.utils import reflect
import pywt


def roll(x, n, dim, make_even=False):
    while n < 0:
        n = x.size(dim) + n

    if make_even and x.size(dim) % 2 == 1:
        end = 1
    else:
        end = 0

    if dim == 0:
        return torch.cat((x[-n:], x[:-n + end]), dim=0)
    elif dim == 1:
        return torch.cat((x[:, -n:], x[:, :-n + end]), dim=1)
    elif dim == 2 or dim == -2:
        return torch.cat((x[:, :, -n:], x[:, :, :-n + end]), dim=2)
    elif dim == 3 or dim == -1:
        return torch.cat((x[:, :, :, -n:], x[:, :, :, :-n + end]), dim=3)


def mypad(x, pad, mode='constant', value=0):
    if mode == 'symmetric':
        # Horizontal only
        if pad[0] == 0 and pad[1] == 0:
            m1, m2 = pad[2], pad[3]
            l = x.shape[-2]
            xe = reflect(np.arange(-m1, l + m2, dtype='int32'), -0.5, l - 0.5)
            return x[:, :, xe]
        # Vertical only
        elif pad[2] == 0 and pad[3] == 0:
            m1, m2 = pad[0], pad[1]
            l = x.shape[-1]
            xe = reflect(np.arange(-m1, l + m2, dtype='int32'), -0.5, l - 0.5)
            return x[:, :, :, xe]
        # Both
        else:
            m1, m2 = pad[0], pad[1]
            l1 = x.shape[-1]
            xe_row = reflect(np.arange(-m1, l1 + m2, dtype='int32'), -0.5, l1 - 0.5)
            m1, m2 = pad[2], pad[3]
            l2 = x.shape[-2]
            xe_col = reflect(np.arange(-m1, l2 + m2, dtype='int32'), -0.5, l2 - 0.5)
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:, :, i, j]
    elif mode == 'constant' or mode == 'reflect' or mode == 'replicate':
        return F.pad(x, pad, mode, value)
    elif mode == 'zero':
        return F.pad(x, pad)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


def afb1d(x, h0, h1, mode='zero', dim=-1):
    """ 1D analysis filter bank (along one dimension only) of an image

    Inputs:
        x (tensor): 4D input with the last two dimensions the spatial input
        af (tensor) - analysis low and highpass filters. Should have shape
        (2, 1, h, 1) or (2, 1, 1, w).
        dim (int) - dimension of filtering. d=2 is for a vertical filter (called
        column filtering but filters across the rows). d=3 is for a horizontal
        filter, (called row filtering but filters across the columns).

    Returns:
        lo, hi: lowpass and highpass subbands
    """
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    N = x.shape[d]
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                          dtype=torch.get_default_dtype(), device=x.device)
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]),
                          dtype=torch.get_default_dtype(), device=x.device)
    L = h0.numel()
    L2 = L // 2
    shape = [1, 1, 1, 1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    if mode == 'per' or mode == 'periodization':
        if x.shape[dim] % 2 == 1:
            if d == 2:
                x = torch.cat((x, x[:, :, -1:]), dim=2)
            else:
                x = torch.cat((x, x[:, :, :, -1:]), dim=3)
            N += 1

        x = roll(x, -L2, dim=d)
        pad = (L - 1, 0) if d == 2 else (0, L - 1)
        lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)

        N2 = N // 2

        if d == 2:
            if L2 < N2:
                lohi[:, :, :L2] += lohi[:, :, N2:N2 + L2]
            else:
                for shift in range(N2, L2, N2):
                    lohi[:, :, :N2] += lohi[:, :, shift: shift + N2]
                res = L2 % N2
                if res != 0:
                    lohi[:, :, :res] += lohi[:, :, -res:]
                else:
                    lohi[:, :, :N2] += lohi[:, :, -N2:]
            lohi = lohi[:, :, :N2]
        else:
            if L2 < N2:
                lohi[:, :, :, :L2] += lohi[:, :, :, N2:N2 + L2]
            else:
                for shift in range(N2, L2, N2):
                    lohi[:, :, :, :N2] += lohi[:, :, :, shift:shift + N2]
                res = L2 % N2
                if res != 0:
                    lohi[:, :, :, :res] += lohi[:, :, :, -res:]
                else:
                    lohi[:, :, :, :N2] += lohi[:, :, :, -N2:]
            lohi = lohi[:, :, :, :N2]
    else:
        # Calculate the pad size
        outsize = pywt.dwt_coeff_len(N, L, mode=mode)
        p = 2 * (outsize - 1) - N + L
        if mode == 'zero':
            # Sadly, pytorch only allows for same padding before and after, if
            # we need to do more padding after for odd length signals, have to
            # prepad
            if p % 2 == 1:
                pad = (0, 0, 0, 1) if d == 2 else (0, 1, 0, 0)
                x = F.pad(x, pad)
            pad = (p // 2, 0) if d == 2 else (0, p // 2)
            # Calculate the high and lowpass
            lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        elif mode == 'symmetric' or mode == 'reflect':
            pad = (0, 0, p // 2, (p + 1) // 2) if d == 2 else (p // 2, (p + 1) // 2, 0, 0)
            x = mypad(x, pad=pad, mode=mode)
            lohi = F.conv2d(x, h, stride=s, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return lohi


def sfb1d(lo, hi, g0, g1, mode='zero', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    d = dim % 4
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                          dtype=torch.get_default_dtype(), device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.copy(np.array(g1).ravel()),
                          dtype=torch.get_default_dtype(), device=lo.device)
    L = g0.numel()
    shape = [1, 1, 1, 1]
    shape[d] = L
    N = 2 * lo.shape[d]
    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    s = (2, 1) if d == 2 else (1, 2)
    g0 = torch.cat([g0] * C, dim=0)
    g1 = torch.cat([g1] * C, dim=0)
    if mode == 'per' or mode == 'periodization':
        y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + \
            F.conv_transpose2d(hi, g1, stride=s, groups=C)
        if d == 2:
            if L - 2 < N:
                y[:, :, :L - 2] += y[:, :, N:N + L - 2]
            else:
                for shift in range(N, L - 2, N):
                    y[:, :, :N] += y[:, :, shift:shift + N]
                res = (L - 2) % N
                if res > 0:
                    y[:, :, :res] += y[:, :, -res:]
                else:
                    y[:, :, :N] += y[:, :, -N:]
            y = y[:, :, :N]
        else:
            if L - 2 < N:
                y[:, :, :, :L - 2] += y[:, :, :, N:N + L - 2]
            else:
                for shift in range(N, L - 2, N):
                    y[:, :, :, :N] += y[:, :, :, shift:shift + N]
                res = (L - 2) % N
                if res > 0:
                    y[:, :, :, :res] += y[:, :, :, -res:]
                else:
                    y[:, :, :, :N] += y[:, :, :, -N:]
            y = y[:, :, :, :N]
        y = roll(y, 1 - L // 2, dim=dim)
    else:
        if mode == 'zero' or mode == 'symmetric' or mode == 'reflect':
            pad = (L - 2, 0) if d == 2 else (0, L - 2)
            y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + \
                F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return y


def afb2d(x, filts, mode='zero'):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`

    Inputs:
        x (torch.Tensor): Input to decompose
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.

    Returns:
        y: Tensor of shape (N, C, 4, H, W)
    """
    C = x.shape[1]
    tensorize = [not isinstance(f, torch.Tensor) for f in filts]
    if len(filts) == 2:
        h0, h1 = filts
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                h0, h1, device=x.device)
        else:
            h0_col = h0
            h0_row = h0.transpose(2, 3)
            h1_col = h1
            h1_row = h1.transpose(2, 3)
    elif len(filts) == 4:
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                *filts, device=x.device)
        else:
            h0_col, h1_col, h0_row, h1_row = filts
    else:
        raise ValueError("Unknown form for input filts")

    lohi = afb1d(x, h0_row, h1_row, mode=mode, dim=3)
    y = afb1d(lohi, h0_col, h1_col, mode=mode, dim=2)

    return y


def afb2d_nonsep(x, filts, mode='zero'):
    """ Does a 1 level 2d wavelet decomposition of an input. Doesn't do separate
    row and column filtering.

    Inputs:
        x (torch.Tensor): Input to decompose
        filts (list or torch.Tensor): If a list is given, should be the low and
            highpass filter banks. If a tensor is given, it should be of the
            form created by
            :py:func:`pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d_nonsep`
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.

    Returns:
        y: Tensor of shape (N, C, 4, H, W)
    """
    C = x.shape[1]
    Ny = x.shape[2]
    Nx = x.shape[3]

    # Check the filter inputs
    if isinstance(filts, (tuple, list)):
        if len(filts) == 2:
            filts = prep_filt_afb2d_nonsep(filts[0], filts[1], device=x.device)
        else:
            filts = prep_filt_afb2d_nonsep(
                filts[0], filts[1], filts[2], filts[3], device=x.device)

    f = torch.cat([filts] * C, dim=0)
    Ly = f.size(2)
    Lx = f.size(3)

    if mode == 'periodization' or mode == 'per':
        if x.shape[2] % 2 == 1:
            pad_odd = torch.cat((x[:, :, -Ly//2:],
                             torch.zeros_like(x[:, :, :1]),
                             x[:, :, :Ly//2 - 1]), dim=2) * filts[-1, 0, :, :1]
            pad_odd = -pad_odd.sum(dim=2, keepdim=True) / filts[-1, 0, Ly//2, :1]
            x = torch.cat((x, pad_odd), dim=2)
            Ny += 1
        if x.shape[3] % 2 == 1:
            pad_odd = torch.cat((x[:, :, :, -Ly // 2:],
                                 torch.zeros_like(x[:, :, :, :1]),
                                 x[:, :, :, :Ly // 2 - 1]), dim=3) * filts[-1, 0, 0, :]
            pad_odd = -pad_odd.sum(dim=3, keepdim=True) / filts[-1, 0, Ly // 2, :1]
            x = torch.cat((x, pad_odd), dim=3)
            Nx += 1
        pad = (Ly - 1, Lx - 1)
        stride = (2, 2)
        x = roll(roll(x, -Ly // 2, dim=2), -Lx // 2, dim=3)
        y = F.conv2d(x, f, padding=pad, stride=stride, groups=C)
        if Ly < Ny:
            y[:, :, :Ly // 2] += y[:, :, Ny // 2:Ny // 2 + Ly // 2]
        else:
            for shift in range(Ny // 2, Ly // 2, Ny // 2):
                y[:, :, :Ny // 2] += y[:, :, shift:shift + Ny // 2]
            res = (Ly // 2) % (Ny // 2)
            if res != 0:
                y[:, :, :res] += y[:, :, -res:]
            else:
                y[:, :, :Ny // 2] += y[:, :, -Ny // 2:]

        if Lx < Nx:
            y[:, :, :, :Lx // 2] += y[:, :, :, Nx // 2:Nx // 2 + Lx // 2]
        else:
            for shift in range(Nx // 2, Lx // 2, Nx // 2):
                y[:, :, :, :Nx // 2] += y[:, :, :, shift:shift + Nx // 2]
            res = (Lx // 2) % (Nx // 2)
            if res != 0:
                y[:, :, :, :res] += y[:, :, :, -res:]
            else:
                y[:, :, :, :Nx // 2] += y[:, :, :, -Nx // 2:]

        y = y[:, :, :Ny // 2, :Nx // 2]
    elif mode == 'zero' or mode == 'symmetric' or mode == 'reflect':
        # Calculate the pad size
        out1 = pywt.dwt_coeff_len(Ny, Ly, mode=mode)
        out2 = pywt.dwt_coeff_len(Nx, Lx, mode=mode)
        p1 = 2 * (out1 - 1) - Ny + Ly
        p2 = 2 * (out2 - 1) - Nx + Lx
        if mode == 'zero':
            # Sadly, pytorch only allows for same padding before and after, if
            # we need to do more padding after for odd length signals, have to
            # prepad
            if p1 % 2 == 1 and p2 % 2 == 1:
                x = F.pad(x, (0, 1, 0, 1))
            elif p1 % 2 == 1:
                x = F.pad(x, (0, 0, 0, 1))
            elif p2 % 2 == 1:
                x = F.pad(x, (0, 1, 0, 0))
            # Calculate the high and lowpass
            y = F.conv2d(x, f, padding=(p1 // 2, p2 // 2), stride=2, groups=C)
        else:  # mode: 'symmetric' or 'reflect':
            pad = (p2 // 2, (p2 + 1) // 2, p1 // 2, (p1 + 1) // 2)
            x = mypad(x, pad=pad, mode=mode)
            y = F.conv2d(x, f, stride=2, groups=C)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

    return y


def sfb2d(ll, lh, hl, hh, filts, mode='zero'):
    """ Does a single level 2d wavelet reconstruction of wavelet coefficients.
    Does separate row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.sfb1d`

    Inputs:
        ll (torch.Tensor): lowpass coefficients
        lh (torch.Tensor): horizontal coefficients
        hl (torch.Tensor): vertical coefficients
        hh (torch.Tensor): diagonal coefficients
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.
    """
    tensorize = [not isinstance(x, torch.Tensor) for x in filts]
    if len(filts) == 2:
        g0, g1 = filts
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(g0, g1)
        else:
            g0_col = g0
            g0_row = g0.transpose(2, 3)
            g1_col = g1
            g1_row = g1.transpose(2, 3)
    elif len(filts) == 4:
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(*filts)
        else:
            g0_col, g1_col, g0_row, g1_row = filts
    else:
        raise ValueError("Unknown form for input filts")

    lo = sfb1d(ll, lh, g0_col, g1_col, mode=mode, dim=2)
    hi = sfb1d(hl, hh, g0_col, g1_col, mode=mode, dim=2)
    y = sfb1d(lo, hi, g0_row, g1_row, mode=mode, dim=3)

    return y


def sfb2d_nonsep(coeffs, filts, mode='zero'):
    """ Does a single level 2d wavelet reconstruction of wavelet coefficients.
    Does not do separable filtering.

    Inputs:
        coeffs (torch.Tensor): tensor of coefficients of shape (N, C, 4, H, W)
            where the third dimension indexes across the (ll, lh, hl, hh) bands.
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d_nonsep`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d_nonsep`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.
    """
    C = coeffs.shape[1]
    Ny = coeffs.shape[-2]
    Nx = coeffs.shape[-1]

    # Check the filter inputs - should be in the form of a torch tensor, but if
    # not, tensorize it here.
    if isinstance(filts, (tuple, list)):
        if len(filts) == 2:
            filts = prep_filt_sfb2d_nonsep(filts[0], filts[1], device=coeffs.device)
        elif len(filts) == 4:
            filts = prep_filt_sfb2d_nonsep(
                filts[0], filts[1], filts[2], filts[3], device=coeffs.device)
        else:
            raise ValueError("Unkown form for input filts")
    f = torch.cat([filts] * C, dim=0)
    Ly = f.size(2)
    Lx = f.size(3)
    x = coeffs.reshape(coeffs.shape[0], -1, coeffs.shape[-2], coeffs.shape[-1])
    if mode == 'periodization' or mode == 'per':
        ll = F.conv_transpose2d(x, f, groups=C, stride=2)
        if 2 * Ny > Ly - 2:
            ll[:, :, :Ly - 2] += ll[:, :, 2 * Ny:2 * Ny + Ly - 2]
        else:
            for shift in range(2 * Ny, Ly - 2, 2 * Ny):
                ll[:, :, :2 * Ny] += ll[:, :, shift:shift + 2 * Ny]
            res = (2 * Ny + Ly - 2) % (2 * Ny)
            if res != 0:
                ll[:, :, :res] += ll[:, :, -res:]
            else:
                ll[:, :, :2 * Ny] += ll[:, :, -2 * Ny:]

        if 2 * Nx > Lx - 2:
            ll[:, :, :, :Lx - 2] += ll[:, :, :, 2 * Nx:2 * Nx + Lx - 2]
        else:
            for shift in range(2 * Nx, Ly - 2, 2 * Nx):
                ll[:, :, :, :2 * Nx] += ll[:, :, :, shift:shift + 2 * Nx]
            res = (2 * Nx + Ly - 2) % (2 * Nx)
            if res != 0:
                ll[:, :, :, :res] += ll[:, :, :, -res:]
            else:
                ll[:, :, :, :2 * Nx] += ll[:, :, :, -2 * Nx:]

        ll = ll[:, :, :2 * Ny, :2 * Nx]
        ll = roll(roll(ll, 1 - Ly // 2, dim=2), 1 - Lx // 2, dim=3)
    elif mode == 'symmetric' or mode == 'zero' or mode == 'reflect':
        pad = (Ly - 2, Lx - 2)
        ll = F.conv_transpose2d(x, f, padding=pad, groups=C, stride=2)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

    return ll.contiguous()


def prep_filt_afb2d_nonsep(h0_col, h1_col, h0_row=None, h1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the afb2d_nonsep function.
    In particular, makes 2d point spread functions, and mirror images them in
    preparation to do torch.conv2d.

    Inputs:
        h0_col (array-like): low pass column filter bank
        h1_col (array-like): high pass column filter bank
        h0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        h1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        filts: (4, 1, h, w) tensor ready to get the four subbands
    """
    h0_col = np.array(h0_col).ravel()
    h1_col = np.array(h1_col).ravel()
    if h0_row is None:
        h0_row = h0_col
    if h1_row is None:
        h1_row = h1_col
    ll = np.outer(h0_col, h0_row)
    lh = np.outer(h1_col, h0_row)
    hl = np.outer(h0_col, h1_row)
    hh = np.outer(h1_col, h1_row)
    filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                      hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]], axis=0)
    filts = torch.tensor(filts, dtype=torch.get_default_dtype(), device=device)
    return filts


def prep_filt_sfb2d_nonsep(g0_col, g1_col, g0_row=None, g1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the sfb2d_nonsep function.
    In particular, makes 2d point spread functions. Does not mirror image them
    as sfb2d_nonsep uses conv2d_transpose which acts like normal convolution.

    Inputs:
        g0_col (array-like): low pass column filter bank
        g1_col (array-like): high pass column filter bank
        g0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        g1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        filts: (4, 1, h, w) tensor ready to combine the four subbands
    """
    g0_col = np.array(g0_col).ravel()
    g1_col = np.array(g1_col).ravel()
    if g0_row is None:
        g0_row = g0_col
    if g1_row is None:
        g1_row = g1_col
    ll = np.outer(g0_col, g0_row)
    lh = np.outer(g1_col, g0_row)
    hl = np.outer(g0_col, g1_row)
    hh = np.outer(g1_col, g1_row)
    filts = np.stack([ll[None], lh[None], hl[None], hh[None]], axis=0)
    filts = torch.tensor(filts, dtype=torch.get_default_dtype(), device=device)
    return filts


def prep_filt_sfb2d(g0_col, g1_col, g0_row=None, g1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the sfb2d function.  In
    particular, makes the tensors the right shape. It does not mirror image them
    as as sfb2d uses conv2d_transpose which acts like normal convolution.

    Inputs:
        g0_col (array-like): low pass column filter bank
        g1_col (array-like): high pass column filter bank
        g0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        g1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        (g0_col, g1_col, g0_row, g1_row)
    """
    g0_col = np.array(g0_col).ravel()
    g1_col = np.array(g1_col).ravel()
    t = torch.get_default_dtype()
    if g0_row is None:
        g0_row = g0_col
    if g1_row is None:
        g1_row = g1_col
    g0_col = torch.tensor(g0_col, device=device, dtype=t).reshape((1, 1, -1, 1))
    g1_col = torch.tensor(g1_col, device=device, dtype=t).reshape((1, 1, -1, 1))
    g0_row = torch.tensor(g0_row, device=device, dtype=t).reshape((1, 1, 1, -1))
    g1_row = torch.tensor(g1_row, device=device, dtype=t).reshape((1, 1, 1, -1))

    return g0_col, g1_col, g0_row, g1_row


def prep_filt_afb2d(h0_col, h1_col, h0_row=None, h1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.

    Inputs:
        h0_col (array-like): low pass column filter bank
        h1_col (array-like): high pass column filter bank
        h0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        h1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        (h0_col, h1_col, h0_row, h1_row)
    """
    h0_col = np.array(h0_col[::-1]).ravel()
    h1_col = np.array(h1_col[::-1]).ravel()
    t = torch.get_default_dtype()
    if h0_row is None:
        h0_row = h0_col
    else:
        h0_row = np.array(h0_row[::-1]).ravel()
    if h1_row is None:
        h1_row = h1_col
    else:
        h1_row = np.array(h1_row[::-1]).ravel()
    h0_col = torch.tensor(h0_col, device=device, dtype=t).reshape((1, 1, -1, 1))
    h1_col = torch.tensor(h1_col, device=device, dtype=t).reshape((1, 1, -1, 1))
    h0_row = torch.tensor(h0_row, device=device, dtype=t).reshape((1, 1, 1, -1))
    h1_row = torch.tensor(h1_row, device=device, dtype=t).reshape((1, 1, 1, -1))

    return h0_col, h1_col, h0_row, h1_row

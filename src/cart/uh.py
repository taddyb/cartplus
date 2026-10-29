"""GR4J unit hydrographs (Perrin et al. 2003) and their causal convolution."""
import torch

L1, L2 = 6, 13                                                                                # UH buffer lengths (X4 <= 6)


def _sc1(t, x4):
    """S-curve of UH1: 0 for t <= 0, (t/X4)^2.5 for 0 < t < X4, 1 afterwards."""
    tt = t.clamp(min=0)
    return torch.where(t <= 0, torch.zeros_like(tt / x4), torch.where(t < x4, (tt / x4) ** 2.5, torch.ones_like(tt / x4)))


def _sc2(t, x4):
    """S-curve of UH2: 0.5 (t/X4)^2.5 for t <= X4, 1 - 0.5 (2 - t/X4)^2.5 for X4 < t < 2 X4, 1 afterwards."""
    tt = t.clamp(min=0)
    a = 0.5 * (tt / x4) ** 2.5
    b = 1 - 0.5 * (2 - tt / x4).clamp(min=0) ** 2.5
    return torch.where(t <= 0, torch.zeros_like(a), torch.where(t <= x4, a, torch.where(t < 2 * x4, b, torch.ones_like(a))))


def ordinates(x4):
    """x4 [N] -> UH1 ordinates [N, L1], UH2 ordinates [N, L2]."""
    x4 = x4[..., None]
    j1 = torch.arange(1, L1 + 1, dtype=x4.dtype, device=x4.device); j2 = torch.arange(1, L2 + 1, dtype=x4.dtype, device=x4.device)
    return _sc1(j1, x4) - _sc1(j1 - 1, x4), _sc2(j2, x4) - _sc2(j2 - 1, x4)


def uh_conv(x, o):
    """Q(t) = sum_i o_i x(t - i) as one depthwise causal convolution: x [..., T], o [..., L]."""
    shp, L = x.shape, o.shape[-1]
    xi = x.reshape(1, -1, shp[-1])
    y = torch.nn.functional.conv1d(torch.nn.functional.pad(xi, (L - 1, 0)), o.reshape(-1, 1, L).flip(-1).to(x.dtype), groups=xi.shape[1])
    return y.reshape(shp)

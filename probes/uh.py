"""Gauge step of distributed CaRT+: area-weighted sum of divide runoff, and a gamma unit hydrograph at the gauge.

Author: tbindas

What it does:
  membership()   the matrix [gauges, divides] of area fractions of one chunk.
  aggregate()    gauge runoff = sum over the gauge's divides of fraction_i * runoff_i.
  UH_gamma()     gamma unit hydrograph weights from the shape `a` and the time scale `b`.
  UH_conv()      causal convolution of a series with a unit hydrograph.
  UHHead         a small network: scaled divide attributes -> `a` in [0, 2.9] and `b` in [0, 6.5] per divide (sigmoid).
  route()        the whole gauge step: sum to the gauge, gauge `a` and `b` as area-weighted means, UH_gamma with
                 lenF = 15, UH_conv.

Experimental purpose: distributed CaRT+ must use the gauge step of the distributed dHBV run (decisions 4 to 8 of the
brief of 2026-11-11), so that the two distributed models differ in the runoff model and not in the routing.

Sources. UH_gamma and UH_conv are copies of hydroDL/model/rnn.py:1239-1281 (repository water_loss,
dPLHBVrelease-master/hydroDL-dev). The training path does not import hydroDL. Differences from the source lines:
  UH_gamma  `t = t.cuda(aa.device)` -> `t = t.to(aa)` (same device, and the dtype of `a`, so the float64 tests can use it).
            The unused line `w = torch.zeros([lenF, m[1], m[2]])` is removed.
  UH_conv   only viewmode 1 exists in the source; the argument is removed.
The order (sum to the gauge, then the unit hydrograph), the parameter ranges, the area-weighted mean of `a` and `b`
and lenF = 15 follow hydroDL/model/HydroModels/waterlossv18_1.py:291, 450, 464-475.
"""
import torch
import torch.nn.functional as F
from torch import nn

ROUT_RANGE = ((0.0, 2.9), (0.0, 6.5))                                                         # routscaLst: ranges of a and b
LEN_F = 15                                                                                    # length of the unit hydrograph, days
HIDDEN = 32


def UH_conv(x, UH):
    """x [batch, var, time], UH [batch, var, uhLen] -> [batch, var, time].
    Q(t) = integral x(tau) UH(t - tau) dtau; conv1d correlates, so the UH is flipped. The batch goes into conv groups."""
    mm = x.shape; nb = mm[0]
    m = UH.shape[-1]
    padd = m - 1
    xx = x.view([1, nb, mm[-1]])
    w = UH.view([nb, 1, m])
    groups = nb
    y = F.conv1d(xx, torch.flip(w, [2]), groups=groups, padding=padd, stride=1, bias=None)
    y = y[:, :, 0:-padd]
    return y.view(mm)


def UH_gamma(a, b, lenF=10):
    """a, b [time (the same value at each step), batch, var] -> UH weights [lenF, batch, var], sum 1 over the first axis"""
    m = a.shape
    aa = F.relu(a[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.1                                # minimum 0.1
    theta = F.relu(b[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.5                             # minimum 0.5
    t = torch.arange(0.5, lenF * 1.0).view([lenF, 1, 1]).repeat([1, m[1], m[2]])
    t = t.to(aa)
    denom = (aa.lgamma().exp()) * (theta ** aa)
    mid = t ** (aa - 1)
    right = torch.exp(-t / theta)
    w = 1 / denom * mid * right
    w = w / w.sum(0)                                                                           # scale to 1 for each UH
    return w


class UHHead(nn.Module):
    """scaled divide attributes [N, F] -> (a [N] in [0, 2.9], b [N] in [0, 6.5]).
    Linear(F, 32) - GELU - Linear(32, 2) - sigmoid - range. The last layer starts at zero weights and zero bias, so each
    divide starts at the middle of the ranges (a = 1.45, b = 3.25) and the start does not depend on the random stream."""

    def __init__(self, n_attr, dev):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_attr, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 2)).to(dev)
        with torch.no_grad():
            self.body[2].weight.zero_(); self.body[2].bias.zero_()

    def forward(self, xs):
        s = torch.sigmoid(self.body(xs))
        (a0, a1), (b0, b1) = ROUT_RANGE
        return a0 + s[:, 0] * (a1 - a0), b0 + s[:, 1] * (b1 - b0)


def membership(gauges, rows, gmap, frac, dev, dtype=torch.float32):
    """-> M [len(gauges), len(rows)]: M[j, i] = area fraction of divide rows[i] in gauge gauges[j] (0 where not a member).
    rows must be ascending and hold each divide of the gauges."""
    import numpy as np
    rows = np.asarray(rows)
    M = np.zeros((len(gauges), len(rows)))
    for j, g in enumerate(gauges):
        pos = np.searchsorted(rows, gmap[g])
        assert (rows[pos] == gmap[g]).all(), "a divide of the gauge is not in the chunk"
        M[j, pos] = frac[g]
    return torch.from_numpy(M).to(device=dev, dtype=dtype)


def aggregate(M, Qd):
    """gauge runoff [gauges, time] = M [gauges, divides] @ divide runoff [divides, time]"""
    return M @ Qd


def route(M, Qd, a_div, b_div):
    """the gauge step. M [G, D] fractions, Qd [D, T] divide runoff, a_div, b_div [D] divide UH parameters.
    -> (Q at the gauge after the unit hydrograph [G, T], summed runoff before it [G, T], gauge a [G], gauge b [G])"""
    Qg = aggregate(M, Qd)
    a = M @ a_div                                                                              # area-weighted mean (fractions sum to 1)
    b = M @ b_div
    routa = a.repeat(LEN_F, 1).unsqueeze(-1)                                                   # [lenF, G, 1]
    routb = b.repeat(LEN_F, 1).unsqueeze(-1)
    UH = UH_gamma(routa, routb, lenF=LEN_F).permute([1, 2, 0])                                 # [G, 1, lenF]
    Qr = UH_conv(Qg.unsqueeze(1), UH)[:, 0, :]
    return Qr, Qg, a, b

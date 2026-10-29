"""HBV snow routine (HBV-light, Seibert & Vis 2012) as a fused Triton recursion with stored-state adjoint
(cart.kernels.adjoint.backward_k), used by the regional HBV (baselines.hbv_regional). Order of operations as baselines.dhbv (hydrodl2
HBV), plus the HBV-light snowfall correction SFCF. Per day, mean air temperature T:
    snow   = SFCF P [T < TT],  rain = P [T >= TT]
    SP    += snow
    melt   = min(CFMAX max(T - TT, 0), SP);           SP -= melt, MW += melt
    refr   = min(CFR CFMAX max(TT - T, 0), MW);       SP += refr, MW -= refr
    tosoil = max(MW - CWH SP, 0);                     MW -= tosoil
    liquid input = rain + tosoil
States SP (dry snow) and MW (liquid water in the pack). Mass: sum(rain + SFCF snow) = sum(liquid) + d(SP + MW); SFCF != 1 is an input
correction (as in HBV), booked separately.
Parameters and ranges: TT [-2.5, 2.5] C, CFMAX [0.5, 10] mm/C/d, CFR [0, 0.1], CWH [0, 0.2] (hydrodl2 / baselines.dhbv), SFCF [0.5, 1.5].
"""

import numpy as np
import torch
import triton
import triton.language as tl

from cart.kernels.adjoint import backward_k

BLOCK = 32
NAMES = ["TT", "CFMAX", "SFCF", "CFR", "CWH"]
LO = np.array([-2.5, 0.5, 0.5, 0.0, 0.0])
HI = np.array([2.5, 10.0, 1.5, 0.1, 0.2])


@triton.jit
def _hbvsnow_fwd(p_ptr, t_ptr, tt_ptr, cf_ptr, sf_ptr, cr_ptr, cw_ptr, x0_ptr, X_ptr, q_ptr, N, T, BLOCK: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    tt = tl.load(tt_ptr + n, mask=m, other=0.0); cf = tl.load(cf_ptr + n, mask=m, other=0.0); sfc = tl.load(sf_ptr + n, mask=m, other=1.0)
    cr = tl.load(cr_ptr + n, mask=m, other=0.0); cw = tl.load(cw_ptr + n, mask=m, other=0.0)
    sp = tl.load(x0_ptr + n * 2 + 0, mask=m, other=0.0); mw = tl.load(x0_ptr + n * 2 + 1, mask=m, other=0.0)
    for t in range(T):
        p = tl.load(p_ptr + n * T + t, mask=m, other=0.0)
        ta = tl.load(t_ptr + n * T + t, mask=m, other=0.0)
        cold = ta < tt
        rain = tl.where(cold, 0.0, p)
        sp = sp + tl.where(cold, sfc * p, 0.0)
        melt = tl.minimum(tl.maximum(cf * (ta - tt), 0.0), sp)
        mw = mw + melt
        sp = sp - melt
        refr = tl.minimum(tl.maximum(cr * cf * (tt - ta), 0.0), mw)
        sp = sp + refr
        mw = mw - refr
        tos = tl.maximum(mw - cw * sp, 0.0)
        mw = mw - tos
        tl.store(X_ptr + (n * T + t) * 2 + 0, sp, mask=m)
        tl.store(X_ptr + (n * T + t) * 2 + 1, mw, mask=m)
        tl.store(q_ptr + n * T + t, rain + tos, mask=m)


def hbvsnow_step(xp, p, ta, tt, cf, sfc, cr, cw):
    sp, mw = xp
    cold = ta < tt
    rain = torch.where(cold, torch.zeros_like(p), p)
    sp = sp + torch.where(cold, sfc * p, torch.zeros_like(p))
    melt = torch.minimum(torch.clamp(cf * (ta - tt), min=0.0), sp)
    mw = mw + melt
    sp = sp - melt
    refr = torch.minimum(torch.clamp(cr * cf * (tt - ta), min=0.0), mw)
    sp = sp + refr
    mw = mw - refr
    tos = torch.clamp(mw - cw * sp, min=0.0)
    mw = mw - tos
    return [sp, mw], [rain + tos]


class HBVSnow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, ta, x0, tt, cf, sfc, cr, cw):
        N, T = p.shape
        a = [v.detach().float().contiguous() for v in (p, ta, tt, cf, sfc, cr, cw, x0)]
        X = torch.empty(N, T, 2, device=p.device); q = torch.empty(N, T, device=p.device)
        _hbvsnow_fwd[(triton.cdiv(N, BLOCK),)](*a, X, q, N, T, BLOCK=BLOCK)
        ctx.save_for_backward(a[0], a[1], a[-1], X, *a[2:7])
        return q, X

    @staticmethod
    def backward(ctx, gq, gX):
        p, ta, x0, X, *prm = ctx.saved_tensors
        gi, gp = backward_k(hbvsnow_step, X, x0, None, [gq], [p, ta], prm)
        return (None, None, None, *gp)


def hbv_snow(p, ta, prm, x0=None):
    """p, ta [N, T]; prm dict TT, CFMAX, SFCF, CFR, CWH of [N] -> liquid input [N, T], states [N, T, 2] (SP, MW)"""
    N = p.shape[0]
    x0 = torch.zeros(N, 2, device=p.device) if x0 is None else x0
    return HBVSnow.apply(p, ta, x0, *(prm[k] for k in NAMES))


def hbv_snow_np(P, Ta, prm):
    """float64 numpy, P, Ta [T, N]; prm dict of [N] -> liquid input [T, N], final (SP, MW)"""
    tt, cf, sfc, cr, cw = (np.asarray(prm[k], np.float64) for k in NAMES)
    sp = np.zeros(P.shape[1]); mw = np.zeros(P.shape[1]); out = np.zeros(P.shape)
    for t in range(P.shape[0]):
        cold = Ta[t] < tt
        rain = np.where(cold, 0.0, P[t])
        sp = sp + np.where(cold, sfc * P[t], 0.0)
        melt = np.minimum(np.maximum(cf * (Ta[t] - tt), 0.0), sp); mw = mw + melt; sp = sp - melt
        refr = np.minimum(np.maximum(cr * cf * (tt - Ta[t]), 0.0), mw); sp = sp + refr; mw = mw - refr
        tos = np.maximum(mw - cw * sp, 0.0); mw = mw - tos
        out[t] = rain + tos
    return out, (sp, mw)

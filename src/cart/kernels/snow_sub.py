"""Five-band degree-day snowpack with sublimation to the atmosphere (a boundary flux N_k -> atmosphere), fused Triton recursion with
stored-state adjoint (adjoint.backward_k). Per band and day, after snowfall and before melt: sublimation u_k = min(ks E, N_k), E the
demand (PET); ks = 0 gives the plain degree-day band snowpack.
Outputs: liquid input (rain + melt) and area-weighted sublimation; states N_k."""
import numpy as np
import torch
import triton
import triton.language as tl
from .adjoint import backward_k

BLOCK = 32


@triton.jit
def _bsub_fwd(p_ptr, t_ptr, e_ptr, dT_ptr, a_ptr, tt_ptr, cf_ptr, sf_ptr, ks_ptr, x0_ptr, X_ptr, q_ptr, u_ptr, N, T, K: tl.constexpr, BLOCK: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    tt = tl.load(tt_ptr + n, mask=m, other=0.0); cf = tl.load(cf_ptr + n, mask=m, other=0.0); sf = tl.load(sf_ptr + n, mask=m, other=1.0)
    ks = tl.load(ks_ptr + n, mask=m, other=0.0)
    for t in range(T):
        p = tl.load(p_ptr + n * T + t, mask=m, other=0.0)
        ta = tl.load(t_ptr + n * T + t, mask=m, other=0.0)
        e = tl.load(e_ptr + n * T + t, mask=m, other=0.0)
        liq = tl.zeros_like(p); sub = tl.zeros_like(p)
        for k in tl.static_range(K):
            dT = tl.load(dT_ptr + n * K + k, mask=m, other=0.0); a = tl.load(a_ptr + n * K + k, mask=m, other=0.0)
            if t == 0:
                s = tl.load(x0_ptr + n * K + k, mask=m, other=0.0)
            else:
                s = tl.load(X_ptr + (n * T + t - 1) * K + k, mask=m, other=0.0)
            tk = ta + dT
            cold = tk < tt
            s = s + tl.where(cold, sf * p, 0.0)
            u = tl.minimum(ks * e, s)
            s = s - u
            mlt = tl.minimum(tl.maximum(cf * (tk - tt), 0.0), s)
            s = s - mlt
            liq += a * (tl.where(cold, 0.0, p) + mlt)
            sub += a * u
            tl.store(X_ptr + (n * T + t) * K + k, s, mask=m)
        tl.store(q_ptr + n * T + t, liq, mask=m)
        tl.store(u_ptr + n * T + t, sub, mask=m)


def bsub_step(xp, p, ta, e, *prm):
    K = len(xp); tt, cf, sf, ks = prm[:4]; dT = prm[4:4 + K]; a = prm[4 + K:4 + 2 * K]
    xs, liq, sub = [], torch.zeros_like(p), torch.zeros_like(p)
    for k in range(K):
        tk = ta + dT[k]; cold = tk < tt
        s = xp[k] + torch.where(cold, sf * p, torch.zeros_like(p))
        u = torch.minimum(ks * e, s); s = s - u
        mlt = torch.minimum(torch.clamp(cf * (tk - tt), min=0.0), s); s = s - mlt
        liq = liq + a[k] * (torch.where(cold, torch.zeros_like(p), p) + mlt); sub = sub + a[k] * u
        xs.append(s)
    return xs, [liq, sub]


class BandsSub(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, ta, e, dT, a, tt, cf, sf, ks, x0):
        N, T = p.shape; K = dT.shape[1]
        v = [t_.detach().float().contiguous() for t_ in (p, ta, e, dT, a, tt, cf, sf, ks, x0)]
        X = torch.empty(N, T, K, device=p.device); q = torch.empty(N, T, device=p.device); u = torch.empty(N, T, device=p.device)
        _bsub_fwd[(triton.cdiv(N, BLOCK),)](*v, X, q, u, N, T, K=K, BLOCK=BLOCK)
        ctx.save_for_backward(*v, X)
        return q, u, X

    @staticmethod
    def backward(ctx, gq, gu, gX):
        p, ta, e, dT, a, tt, cf, sf, ks, x0, X = ctx.saved_tensors
        K = dT.shape[1]
        prm = [tt, cf, sf, ks] + [dT[:, k] for k in range(K)] + [a[:, k] for k in range(K)]
        gi, gp = backward_k(bsub_step, X, x0, gX if gX is not None else None, [gq, gu], [p, ta, e], prm)
        gdT = torch.stack([g if g is not None else torch.zeros_like(tt) for g in gp[4:4 + K]], 1)
        return None, None, gi[2], gdT, None, gp[0], gp[1], gp[2], gp[3], None


def snow_bands_sub(p, ta, e, dT, a, tt, cf, sf, ks, x0=None):
    """p, ta, e [N, T]; dT, a [N, K]; tt, cf, sf, ks [N] -> liquid [N, T], sublimation [N, T], SWE per band [N, T, K]"""
    x0 = torch.zeros_like(dT) if x0 is None else x0
    return BandsSub.apply(p, ta, e, dT, a, tt, cf, sf, ks, x0)

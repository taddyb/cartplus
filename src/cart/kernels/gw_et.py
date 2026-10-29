"""Groundwater store with evapotranspiration from groundwater (riparian / phreatophyte uptake), a boundary flux W -> atmosphere, as a
fused Triton recursion with stored-state adjoint (adjoint.backward_k).

    dW/dt = G - a W^2 - e_W,   e_W = lam * Er * W / (W + W0)
Er = residual atmospheric demand after interception and soil evaporation (>= 0), lam in [0, 1) the share of it met from groundwater at
large storage, W0 the steady storage of the store (scale of the Michaelis-Menten saturation). Recession: -dln Qs/dt gains 2 e_W / W =
2 lam Er / (W + W0), i.e. the observed rise of the low-flow recession rate with PET (dk/dE ~ lam / W0 at W ~ W0).
Daily update (operator split, in the order inflow -> uptake -> release):
    Wm = W + G;   e_W = Wm (1 - exp(-lam Er / (Wm + W0)))   (frozen-rate exact for the uptake; e_W <= Wm, >= 0)
    W1 = Wm - e_W;   W+ = W1 / (1 + a W1),   Qs = W1 - W+   (exact quadratic release, as grstores.quad_store)
lam = 0 reproduces quad_store exactly. Outputs Qs, e_W [N, T]; states W [N, T].
"""

import numpy as np
import torch
import triton
import triton.language as tl

from .adjoint import backward_k

BLOCK = 32


@triton.jit
def _gwet_fwd(g_ptr, er_ptr, a_ptr, lam_ptr, w0_ptr, x0_ptr, X_ptr, q_ptr, e_ptr, N, T, BLOCK: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    a = tl.load(a_ptr + n, mask=m, other=0.0)
    lam = tl.load(lam_ptr + n, mask=m, other=0.0)
    w0 = tl.load(w0_ptr + n, mask=m, other=1.0)
    W = tl.load(x0_ptr + n, mask=m, other=0.0)
    for t in range(T):
        g = tl.load(g_ptr + n * T + t, mask=m, other=0.0)
        er = tl.load(er_ptr + n * T + t, mask=m, other=0.0)
        Wm = W + g
        x = lam * er / (Wm + w0)
        ew = Wm * tl.where(x < 1e-3, x - 0.5 * x * x + x * x * x / 6.0, 1.0 - tl.exp(-x))
        W1 = Wm - ew
        Wn = W1 / (1.0 + a * W1)
        tl.store(X_ptr + n * T + t, Wn, mask=m)
        tl.store(q_ptr + n * T + t, W1 - Wn, mask=m)
        tl.store(e_ptr + n * T + t, ew, mask=m)
        W = Wn


def gwet_step(xp, g, er, a, lam, w0):
    (W,) = xp
    Wm = W + g
    x = lam * er / (Wm + w0)
    ew = Wm * torch.where(x < 1e-3, x - 0.5 * x * x + x * x * x / 6.0, 1.0 - torch.exp(-x))
    W1 = Wm - ew
    Wn = W1 / (1.0 + a * W1)
    return [Wn], [W1 - Wn, ew]


class GwEt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, er, x0, a, lam, w0):
        N, T = g.shape
        v = [t_.detach().float().contiguous() for t_ in (g, er, a, lam, w0, x0)]
        X = torch.empty(N, T, device=g.device); Q = torch.empty(N, T, device=g.device); Ew = torch.empty(N, T, device=g.device)
        _gwet_fwd[(triton.cdiv(N, BLOCK),)](*v, X, Q, Ew, N, T, BLOCK=BLOCK)
        ctx.save_for_backward(v[0], v[1], v[5], X, v[2], v[3], v[4])
        return Q, Ew, X

    @staticmethod
    def backward(ctx, gq, ge, gX):
        g, er, x0, X, a, lam, w0 = ctx.saved_tensors
        gi, gp = backward_k(gwet_step, X[..., None], x0[:, None], gX[..., None] if gX is not None else None, [gq, ge], [g, er], [a, lam, w0])
        return (gi[0], gi[1], None, *gp)


def gw_et(G, Er, a, lam, W0, x0):
    """G, Er [N, T]; a, lam, W0, x0 [N] -> Qs, e_W, W [N, T]"""
    return GwEt.apply(G, Er, x0, a, lam, W0)

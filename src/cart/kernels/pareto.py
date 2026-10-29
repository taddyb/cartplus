"""CaRT production store with the partition of rain derived from a Pareto distribution of point storage capacities (Moore 1985 PDM,
cmin = 0), as a fused Triton recursion with stored-state adjoint (adjoint.backward_k). Evaporation and percolation as GR4J.
Evaporation: es = C (1 - d^be) te / (1 + d^(be-1) te), te = tanh(en/C): the GR4J daily integral with the PDM evaporation ratio
1 - (1 - s)^be (Moore eq. 8) in place of s(2 - s); be = 2 reproduces GR4J exactly.

Point capacities c ~ F(c) = 1 - (1 - c/cmax)^b, mean capacity = store capacity C  =>  cmax = (b + 1) C.
All points with capacity below the critical capacity C* are full; S(C*) = C [1 - (1 - C*/cmax)^(b+1)].  A net rain depth pn raises
C* by pn (capped at cmax); the water entering the store is exactly
    ps = S(C* + pn) - S(C*) = C [d - max(d^(1/(b+1)) - pn/cmax, 0)^(b+1)],   d = 1 - S/C,
and the rest pn - ps is saturation-excess runoff. Infiltrating fraction at the start of rain = d^(b/(b+1)): b -> 0 all rain enters
until full, b large -> runoff fraction grows quickly with S. (GR4J: 1 - s^2, not a member of this family; b ~ 0.7 matches it at s = 0.5.)
DAY (S at the start):  pn = (p - e)+, en = (e - p)+, ic = min(p, e);  ps as above;  es = S (2 - s) tanh(en/C) / (1 + (1 - s) tanh(en/C));
    S2 = S - es + ps;  G = S2 [1 - (1 + kq (S2/C)^4)^(-1/4)];  S+ = S2 - G.   Outputs D = pn - ps, G, AET = es + ic.
"""

import numpy as np
import torch
import triton
import triton.language as tl

from .adjoint import backward_k

BLOCK = 32


@triton.jit
def _pp_fwd(p_ptr, e_ptr, c_ptr, k_ptr, b_ptr, be_ptr, x0_ptr, X_ptr, d_ptr, g_ptr, a_ptr, N, T, BLOCK: tl.constexpr, NET: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    Cc = tl.load(c_ptr + n, mask=m, other=1.0)
    kq = tl.load(k_ptr + n, mask=m, other=0.0)
    b = tl.load(b_ptr + n, mask=m, other=1.0)
    be = tl.load(be_ptr + n, mask=m, other=2.0)
    S = tl.load(x0_ptr + n, mask=m, other=0.0)
    cm = (b + 1.0) * Cc
    for t in range(T):
        p = tl.load(p_ptr + n * T + t, mask=m, other=0.0)
        e = tl.load(e_ptr + n * T + t, mask=m, other=0.0)
        if NET:
            pn = tl.maximum(p - e, 0.0)
            en = tl.maximum(e - p, 0.0)
            ic = tl.minimum(p, e)
        else:
            pn = p
            en = e
            ic = p * 0.0
        s = tl.minimum(tl.maximum(S / Cc, 0.0), 1.0)
        d = tl.maximum(1.0 - s, 1e-7)
        z = tl.exp(tl.log(d) / (b + 1.0))
        z2 = tl.maximum(z - pn / cm, 0.0)
        v = tl.where(z2 > 0.0, tl.exp((b + 1.0) * tl.log(tl.maximum(z2, 1e-30))), 0.0)
        ps = tl.minimum(tl.maximum(Cc * (d - v), 0.0), pn)
        te = 1.0 - 2.0 / (tl.exp(2.0 * en / Cc) + 1.0)
        dbe = tl.exp(be * tl.log(d))
        es = tl.minimum(Cc * (1.0 - dbe) * te / (1.0 + dbe / d * te), S)
        S2 = S - es + ps
        q = S2 / Cc
        r4 = kq * q * q * q * q
        g = S2 * (1.0 - tl.exp(-0.25 * tl.log(1.0 + r4)))
        S = S2 - g
        tl.store(X_ptr + n * T + t, S, mask=m)
        tl.store(d_ptr + n * T + t, pn - ps, mask=m)
        tl.store(g_ptr + n * T + t, g, mask=m)
        tl.store(a_ptr + n * T + t, es + ic, mask=m)


def pp_step(xp, p, e, Cc, kq, b, be, net=True):
    (S,) = xp
    if net:
        pn, en, ic = torch.relu(p - e), torch.relu(e - p), torch.minimum(p, e)
    else:
        pn, en, ic = p, e, torch.zeros_like(p)
    s = (S / Cc).clamp(0.0, 1.0)
    d = (1.0 - s).clamp(min=1e-7)
    cm = (b + 1.0) * Cc
    z = torch.exp(torch.log(d) / (b + 1.0))
    z2 = torch.relu(z - pn / cm)
    v = torch.where(z2 > 0.0, torch.exp((b + 1.0) * torch.log(z2.clamp(min=1e-30))), torch.zeros_like(z2))
    ps = torch.minimum((Cc * (d - v)).clamp(min=0.0), pn)
    te = torch.tanh(en / Cc)
    dbe = torch.exp(be * torch.log(d))
    es = torch.minimum(Cc * (1.0 - dbe) * te / (1.0 + dbe / d * te), S)
    S2 = S - es + ps
    g = S2 * (1.0 - torch.exp(-0.25 * torch.log1p(kq * (S2 / Cc) ** 4)))
    return [S2 - g], [pn - ps, g, es + ic]


class ParetoProd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, e, Cc, kq, b, be, x0, net=True):
        N, T = p.shape
        ctx.net = net
        v = [t_.detach().float().contiguous() for t_ in (p, e, Cc, kq, b, be, x0)]
        X, D, G, A = (torch.empty(N, T, device=p.device) for _ in range(4))
        _pp_fwd[(triton.cdiv(N, BLOCK),)](*v, X, D, G, A, N, T, BLOCK=BLOCK, NET=net)
        ctx.save_for_backward(*v, X)
        return D, G, A, X

    @staticmethod
    def backward(ctx, gD, gG, gA, gX):
        p, e, Cc, kq, b, be, x0, X = ctx.saved_tensors
        step = pp_step if ctx.net else (lambda xp, p_, e_, *prm: pp_step(xp, p_, e_, *prm, net=False))
        gi, gp = backward_k(step, X[..., None], x0[:, None], gX[..., None] if gX is not None else None, [gD, gG, gA], [p, e], [Cc, kq, b, be])
        return gi[0], gi[1], gp[0], gp[1], gp[2], gp[3], None, None


def pareto_prod(p, e, Cc, kq, b, S0, be=None, net=True):
    """p, e [N, T]; Cc, kq, b, S0 [N]; be [N] evaporation exponent (None = 2 = GR4J) -> D, G, AET, S [N, T]"""
    be = torch.full_like(b, 2.0) if be is None else be
    return ParetoProd.apply(p, e, Cc, kq, b, be, S0, net)

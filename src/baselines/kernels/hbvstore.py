"""Fused GPU recursions of HBV-9 (hbv9_core.py, HBV-light without snow; Seibert & Vis 2012) with stored-state adjoints, for regional training.

HBV-9 is a chain of three scalar recursions, each depending only on the one above it:
  soil   s = sm_{t-1}:  F = P r^BETA, r = clamp(s/FC, 1e-6, 1);  s1 = s + P - F;  ea = min(E min(s1/(LP FC), 1), s1);  sm_t = s1 - ea
  upper  u = suz_{t-1}: u1 = u + F_t;  perc = min(PERC, u1);  u2 = u1 - perc;  q0 = K0 max(u2 - UZL, 0);  q1 = K1 u2;  suz_t = u2 - q0 - q1
  lower  slz_t = (1 - K2)(slz_{t-1} + perc_t),  q2 = K2 (slz_{t-1} + perc_t)            (linear: lin_rec.lin_rec)
  routing  Q_t = sum_i c_i qgw_{t-i}, qgw = q0 + q1 + q2, c = MAXBAS triangle (hbv9_core.weights)   (cart.uh.uh_conv)
Order and formulas as hbv9_core.run_numpy (the float64 reference implementation).

Forward: one Triton kernel per nonlinear recursion (soil, upper), storing the state series. Backward (stored-state adjoint): the
elementwise one-day step is re-evaluated in torch on the stored previous states; its local partials
a_t = dx_t/dx_{t-1} and c_t = sum_k g_k,t d out_k,t / dx_{t-1} drive the scalar adjoint lambda_t = g_x,t + c_{t+1} + a_{t+1} lambda_{t+1}
(lin_rec._adjoint), and the parameter / input gradients are one vectorised VJP of sum_t [lambda_t x_t + sum_k g_k,t out_k,t].
"""

import torch
import triton
import triton.language as tl

from cart.uh import uh_conv
from . import hbv9_core as hbv9
from .lin_rec import BLOCK, _adjoint, lin_rec


@triton.jit
def _soil_fwd(p_ptr, e_ptr, fc_ptr, lp_ptr, be_ptr, s0_ptr, s_ptr, f_ptr, N, T, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    fc = tl.load(fc_ptr + offs, mask=m, other=1.0)
    lp = tl.load(lp_ptr + offs, mask=m, other=1.0)
    be = tl.load(be_ptr + offs, mask=m, other=1.0)
    s = tl.load(s0_ptr + offs, mask=m, other=0.0)
    for t in range(T):
        p = tl.load(p_ptr + offs * T + t, mask=m, other=0.0)
        e = tl.load(e_ptr + offs * T + t, mask=m, other=0.0)
        r = tl.minimum(tl.maximum(s / fc, 1e-6), 1.0)
        f = p * tl.exp(be * tl.log(r))
        s1 = s + p - f
        ea = tl.minimum(e * tl.minimum(s1 / (lp * fc), 1.0), s1)
        s = s1 - ea
        tl.store(s_ptr + offs * T + t, s, mask=m)
        tl.store(f_ptr + offs * T + t, f, mask=m)


@triton.jit
def _upper_fwd(f_ptr, pc_ptr, uz_ptr, k0_ptr, k1_ptr, u0_ptr, u_ptr, perc_ptr, q_ptr, N, T, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    pc = tl.load(pc_ptr + offs, mask=m, other=0.0)
    uz = tl.load(uz_ptr + offs, mask=m, other=0.0)
    k0 = tl.load(k0_ptr + offs, mask=m, other=0.0)
    k1 = tl.load(k1_ptr + offs, mask=m, other=0.0)
    u = tl.load(u0_ptr + offs, mask=m, other=0.0)
    for t in range(T):
        f = tl.load(f_ptr + offs * T + t, mask=m, other=0.0)
        u1 = u + f
        pe = tl.minimum(pc, u1)
        u2 = u1 - pe
        q = k0 * tl.maximum(u2 - uz, 0.0) + k1 * u2
        u = u2 - q
        tl.store(u_ptr + offs * T + t, u, mask=m)
        tl.store(perc_ptr + offs * T + t, pe, mask=m)
        tl.store(q_ptr + offs * T + t, q, mask=m)


def soil_step(s, p, e, fc, lp, be):
    r = (s / fc).clamp(1e-6, 1.0)
    f = p * r ** be
    s1 = s + p - f
    ea = torch.minimum(e * torch.clamp(s1 / (lp * fc), max=1.0), s1)
    return s1 - ea, (f,)


def upper_step(u, f, pc, uz, k0, k1):
    u1 = u + f
    pe = torch.minimum(pc, u1)
    u2 = u1 - pe
    q = k0 * torch.clamp(u2 - uz, min=0.0) + k1 * u2
    return u2 - q, (pe, q)


def _stored_backward(step, X, x0, gx, gouts, inputs, params):
    """generic stored-state adjoint of x_t, outs_t = step(x_{t-1}, inputs_t, params); returns grads of inputs and params"""
    N, T = X.shape
    xp = torch.cat([x0[:, None], X[:, :-1]], 1).detach().requires_grad_(True)
    ins = [v.detach().requires_grad_(True) for v in inputs]
    prm = [v.detach()[:, None].expand(N, T).contiguous().requires_grad_(True) for v in params]
    with torch.enable_grad():
        x, outs = step(xp, *ins, *prm)
        a = torch.autograd.grad(x.sum(), xp, retain_graph=True)[0]
        go = sum((g * o).sum() for g, o in zip(gouts, outs) if g is not None)
        c = torch.autograd.grad(go, xp, retain_graph=True, allow_unused=True)[0] if torch.is_tensor(go) else None
        c = torch.zeros_like(X) if c is None else c
        gy = (torch.zeros_like(X) if gx is None else gx) + torch.cat([c[:, 1:], torch.zeros_like(c[:, :1])], 1)
        a_next = torch.cat([a[:, 1:], torch.zeros_like(a[:, :1])], 1).contiguous()
        lam = torch.empty_like(X)
        _adjoint[(triton.cdiv(N, BLOCK),)](a_next.float(), gy.contiguous().float(), torch.zeros(N, device=X.device), lam, N, T, BLOCK=BLOCK)
        total = (lam.to(x.dtype) * x).sum() + (go if torch.is_tensor(go) else 0.0)
        gr = torch.autograd.grad(total, ins + prm, allow_unused=True)
    gi = [g for g in gr[:len(ins)]]
    gp = [g.sum(1) if g is not None else None for g in gr[len(ins):]]
    return gi, gp


class Soil(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, e, fc, lp, be, s0):
        N, T = p.shape
        args = [v.detach().float().contiguous() for v in (p, e, fc, lp, be, s0)]
        S, Fx = torch.empty_like(args[0]), torch.empty_like(args[0])
        _soil_fwd[(triton.cdiv(N, BLOCK),)](*args, S, Fx, N, T, BLOCK=BLOCK)
        ctx.save_for_backward(*args, S)
        return S, Fx

    @staticmethod
    def backward(ctx, gS, gF):
        p, e, fc, lp, be, s0, S = ctx.saved_tensors
        gi, gp = _stored_backward(soil_step, S, s0, gS, [gF], [p, e], [fc, lp, be])
        return None, None, gp[0], gp[1], gp[2], None


class Upper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, pc, uz, k0, k1, u0):
        N, T = f.shape
        args = [v.detach().float().contiguous() for v in (f, pc, uz, k0, k1, u0)]
        U, Pe, Q = (torch.empty_like(args[0]) for _ in range(3))
        _upper_fwd[(triton.cdiv(N, BLOCK),)](*args, U, Pe, Q, N, T, BLOCK=BLOCK)
        ctx.save_for_backward(*args, U)
        return U, Pe, Q

    @staticmethod
    def backward(ctx, gU, gPe, gQ):
        f, pc, uz, k0, k1, u0, U = ctx.saved_tensors
        gi, gp = _stored_backward(upper_step, U, u0, gU, [gPe, gQ], [f], [pc, uz, k0, k1])
        return gi[0], gp[0], gp[1], gp[2], gp[3], None


def hbv(P, E, prm, state=None, carry=None):
    """P, E [N, T]; prm dict of [N] (hbv9.NAMES); state (sm, suz, slz) [N] or None (hbv9 initial state); carry [N, L_BUF - 1] = the
    last L_BUF - 1 days of qgw before this run (water still in the routing buffer). Returns Q [N, T], end state, qgw tail."""
    FC, LP, BETA, PERC, UZL, K0, K1, K2, MB = (prm[k] for k in hbv9.NAMES)
    if state is None:
        state = (0.5 * FC.detach(), torch.full_like(FC, 10.0), torch.full_like(FC, 10.0))
    sm0, suz0, slz0 = state
    S, F = Soil.apply(P, E, FC, LP, BETA, sm0)
    U, Pe, Q01 = Upper.apply(F, PERC, UZL, K0, K1, suz0)
    k2 = K2[:, None].expand_as(Pe)
    L = lin_rec((1.0 - k2).contiguous(), ((1.0 - k2) * Pe).contiguous(), slz0)
    Lprev = torch.cat([slz0[:, None], L[:, :-1]], 1)
    qgw = Q01 + K2[:, None] * (Lprev + Pe)
    c = hbv9.weights(MB, "paper", torch)                                                # [N, L_BUF], c[..., 0] = current day
    H = hbv9.L_BUF - 1
    pad = torch.zeros(qgw.shape[0], H, device=qgw.device, dtype=qgw.dtype) if carry is None else carry
    Qr = uh_conv(torch.cat([pad, qgw], 1), c)[:, H:]
    return Qr, (S[:, -1], U[:, -1], L[:, -1]), qgw[:, -H:]

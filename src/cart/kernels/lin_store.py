"""Fused Triton recursion (hand-written adjoint) of the CaRT+ quick-release store with a time-varying release coefficient.

MODE 0 (quick store of the derived release, used by CaRT+), per series and day, exchange applied once with the 0.9 / 0.1 split:
    j = Xj (max(R, 0)/30)^3.5;  Rm = max(R + x9 + 0.9 j, 0);  Qd = max(x1 + 0.1 j, 0)
    f = ln(max(Rm, 1e-3) / X3);  A = a_f exp(6 tanh((c0_t + c1_t f) / 6));  k = 1/4 ln(1 + 4 A Rm^4)
    R+ = Rm exp(-k);  Q = Rm - R+ + Qd;  realised exchange xr = (Rm - R - x9) + (Qd - x1)
MODE 1 (groundwater form, not used by the paper's configurations): Wm = W + g;  k = ln(1 + A Wm).
c0, c1 [N, T] are given series (CaRT+: c0 = ln A_theta + viscosity term - aF frost, c1 = k).
lin_store(u1, u2, p1, p2, sc, s0, c0, c1, mode) -> Q, xr, R_after (each [N, T]);  mode 0: u1 = x9, u2 = x1, p1 = a_f, p2 = Xj, sc = X3.
"""

import torch
import triton
import triton.language as tl

H = 32
BLOCK = 32
GC = 0.7978845608028654                                                                       # sqrt(2 / pi)


@triton.jit
def _tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


# ----------------------------------------------------------------------------- store law linear in its own state (analytic laws)
# same stores as above, but m = 6 tanh((c0_t + c1_t f) / 6) with c0, c1 [N, T] given series (from an analytic law of forcing / season)

@triton.jit
def _lin_fwd(u1_ptr, u2_ptr, p1_ptr, p2_ptr, sc_ptr, s0_ptr, c0_ptr, c1_ptr, q_ptr, xr_ptr, sv_ptr, ra_ptr, send_ptr, N, T,
             BLOCK: tl.constexpr, MODE: tl.constexpr, STORE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    p1 = tl.load(p1_ptr + offs, mask=m, other=0.0)
    sc = tl.load(sc_ptr + offs, mask=m, other=1.0)
    S = tl.load(s0_ptr + offs, mask=m, other=0.0)
    if MODE == 0:
        p2 = tl.load(p2_ptr + offs, mask=m, other=0.0)
    for t in range(T):
        u1 = tl.load(u1_ptr + offs * T + t, mask=m, other=0.0)
        c0 = tl.load(c0_ptr + offs * T + t, mask=m, other=0.0)
        c1 = tl.load(c1_ptr + offs * T + t, mask=m, other=0.0)
        if STORE:
            tl.store(sv_ptr + offs * T + t, S, mask=m)
        if MODE == 0:
            u2 = tl.load(u2_ptr + offs * T + t, mask=m, other=0.0)
            Sc = tl.maximum(S, 0.0) / 30.0
            j = p2 * Sc * Sc * Sc * tl.sqrt(Sc)
            Sm = tl.maximum(S + u1 + 0.9 * j, 0.0)
            Qd = tl.maximum(u2 + 0.1 * j, 0.0)
        else:
            Sm = S + u1
        f = tl.log(tl.maximum(Sm, 1e-3) / sc)
        A = p1 * tl.exp(6.0 * _tanh((c0 + c1 * f) / 6.0))
        if MODE == 0:
            S2 = Sm * Sm
            k = 0.25 * tl.log(1.0 + 4.0 * A * S2 * S2)
        else:
            k = tl.log(1.0 + A * Sm)
        Sp = Sm * tl.exp(-k)
        if MODE == 0:
            tl.store(q_ptr + offs * T + t, Sm - Sp + Qd, mask=m)
            tl.store(xr_ptr + offs * T + t, (Sm - S - u1) + (Qd - u2), mask=m)
        else:
            tl.store(q_ptr + offs * T + t, Sm - Sp, mask=m)
        tl.store(ra_ptr + offs * T + t, Sp, mask=m)
        S = Sp
    tl.store(send_ptr + offs, S, mask=m)


@triton.jit
def _lin_bwd(u1_ptr, u2_ptr, p1_ptr, p2_ptr, sc_ptr, sv_ptr, c0_ptr, c1_ptr, gq_ptr, gxr_ptr, gra_ptr,
             gu1_ptr, gu2_ptr, gp1_ptr, gp2_ptr, gsc_ptr, gs0_ptr, gc0_ptr, gc1_ptr, N, T, BLOCK: tl.constexpr, MODE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    p1 = tl.load(p1_ptr + offs, mask=m, other=0.0)
    sc = tl.load(sc_ptr + offs, mask=m, other=1.0)
    if MODE == 0:
        p2 = tl.load(p2_ptr + offs, mask=m, other=0.0)
    lam = tl.zeros([BLOCK], tl.float32)
    gp1 = tl.zeros([BLOCK], tl.float32)
    gp2 = tl.zeros([BLOCK], tl.float32)
    gsc = tl.zeros([BLOCK], tl.float32)
    for tt in range(T):
        t = T - 1 - tt
        S = tl.load(sv_ptr + offs * T + t, mask=m, other=0.0)
        u1 = tl.load(u1_ptr + offs * T + t, mask=m, other=0.0)
        c0 = tl.load(c0_ptr + offs * T + t, mask=m, other=0.0)
        c1 = tl.load(c1_ptr + offs * T + t, mask=m, other=0.0)
        gq = tl.load(gq_ptr + offs * T + t, mask=m, other=0.0)
        lam = lam + tl.load(gra_ptr + offs * T + t, mask=m, other=0.0)
        if MODE == 0:
            u2 = tl.load(u2_ptr + offs * T + t, mask=m, other=0.0)
            gxr = tl.load(gxr_ptr + offs * T + t, mask=m, other=0.0)
            Sc = tl.maximum(S, 0.0) / 30.0
            r35 = Sc * Sc * Sc * tl.sqrt(Sc)
            j = p2 * r35
            a = S + u1 + 0.9 * j
            Sm = tl.maximum(a, 0.0)
            qa = u2 + 0.1 * j
        else:
            Sm = S + u1
        fa = tl.maximum(Sm, 1e-3)
        f = tl.log(fa / sc)
        tho = _tanh((c0 + c1 * f) / 6.0)
        em = tl.exp(6.0 * tho)
        A = p1 * em
        if MODE == 0:
            S2 = Sm * Sm
            zz = 4.0 * A * S2 * S2
            den = 1.0 + zz
            k = 0.25 * tl.log(den)
            dk_dm = 0.25 * zz / den
            dk_dS = 4.0 * A * S2 * Sm / den
            dk_dp = em * S2 * S2 / den
        else:
            zz = A * Sm
            den = 1.0 + zz
            k = tl.log(den)
            dk_dm = zz / den
            dk_dS = A / den
            dk_dp = em * Sm / den
        E = tl.exp(-k)
        gSp = lam - gq
        gk = -gSp * Sm * E
        gx = gk * dk_dm * (1.0 - tho * tho)                                                   # d / d(c0 + c1 f)
        gp1 += gk * dk_dp
        tl.store(gc0_ptr + offs * T + t, gx, mask=m)
        tl.store(gc1_ptr + offs * T + t, gx * f, mask=m)
        gf = gx * c1
        gsc -= gf / sc
        gSm = gq + gSp * E + gk * dk_dS + tl.where(Sm > 1e-3, gf / fa, 0.0)
        if MODE == 0:
            gSm += gxr
            ga = tl.where(a > 0.0, gSm, 0.0)
            gqa = tl.where(qa > 0.0, gq + gxr, 0.0)
            tl.store(gu1_ptr + offs * T + t, ga - gxr, mask=m)
            tl.store(gu2_ptr + offs * T + t, gqa - gxr, mask=m)
            gj = 0.9 * ga + 0.1 * gqa
            gp2 += gj * r35
            lam = ga - gxr + tl.where(S > 0.0, gj * p2 * 3.5 * Sc * Sc * tl.sqrt(Sc) / 30.0, 0.0)
        else:
            tl.store(gu1_ptr + offs * T + t, gSm, mask=m)
            lam = gSm
    tl.store(gs0_ptr + offs, lam, mask=m)
    tl.store(gp1_ptr + offs, gp1, mask=m)
    tl.store(gsc_ptr + offs, gsc, mask=m)
    if MODE == 0:
        tl.store(gp2_ptr + offs, gp2, mask=m)


class LinStore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u1, u2, p1, p2, sc, s0, c0, c1, mode):
        N, T = u1.shape
        f = lambda x: x.detach().contiguous().float()
        u1c, p1c, scc, s0c, c0c, c1c = f(u1), f(p1), f(sc), f(s0), f(c0), f(c1)
        u2c = f(u2) if mode == 0 else u1c
        p2c = f(p2) if mode == 0 else p1c
        q = torch.empty_like(u1c); xr = torch.zeros_like(u1c) if mode == 1 else torch.empty_like(u1c)
        ra = torch.empty_like(u1c); send = torch.empty_like(s0c)
        store = any(ctx.needs_input_grad)
        sv = torch.empty_like(u1c) if store else q
        _lin_fwd[(triton.cdiv(N, BLOCK),)](u1c, u2c, p1c, p2c, scc, s0c, c0c, c1c, q, xr, sv, ra, send, N, T, BLOCK=BLOCK, MODE=mode, STORE=store)
        if store:
            ctx.save_for_backward(u1c, u2c, p1c, p2c, scc, sv, c0c, c1c)
        ctx.mode = mode
        return q, xr, ra

    @staticmethod
    def backward(ctx, gq, gxr, gra):
        u1, u2, p1, p2, sc, sv, c0, c1 = ctx.saved_tensors
        N, T = u1.shape
        mode = ctx.mode
        z = lambda g: torch.zeros_like(u1) if g is None else g.contiguous().float()
        gq, gxr, gra = z(gq), z(gxr), z(gra)
        gu1, gu2 = torch.empty_like(u1), torch.zeros_like(u1)
        gp1, gp2, gsc, gs0 = (torch.zeros_like(p1) for _ in range(4))
        gc0, gc1 = torch.empty_like(c0), torch.empty_like(c1)
        _lin_bwd[(triton.cdiv(N, BLOCK),)](u1, u2, p1, p2, sc, sv, c0, c1, gq, gxr, gra, gu1, gu2, gp1, gp2, gsc, gs0, gc0, gc1, N, T,
                                           BLOCK=BLOCK, MODE=mode)
        return (gu1, gu2 if mode == 0 else None, gp1, gp2 if mode == 0 else None, gsc, gs0, gc0, gc1, None)


def lin_store(u1, u2, p1, p2, sc, s0, c0, c1, mode):
    """store whose coefficient is p1 exp(6 tanh((c0_t + c1_t ln(S/sc)) / 6)) -> Q, xr, S_after [N, T]"""
    return LinStore.apply(u1, u2, p1, p2, sc, s0, c0, c1, mode)

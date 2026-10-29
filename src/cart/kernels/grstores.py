"""Fused GPU recursion (Triton) for the GR4J-type non-linear store, with an exact hand-written backward pass.

Per series (one thread lane each), daily:  S+ = S(t-1) + x(t);  Q(t) = S+ [1 - (1 + (S+/X3)^4)^(-1/4)];  S(t) = S+ - Q(t).
Forward stores S+ for the backward recursion; the backward runs the chain rule in reverse time:
  dQ/dS+ = f + r^4 u^(-5/4),  dQ/dX3 = -r^5 u^(-5/4)   (r = S+/X3, u = 1 + r^4, f = 1 - u^(-1/4))
  gS+ = gQ dQ/dS+ + gS (1 - dQ/dS+);  gX3 += (gQ - gS) dQ/dX3;  gx(t) = gS+;  gS(t-1) = gS+.
The initial storage S0 is treated as a constant (spin-up computed without gradient).
GProdK (production store): all three outputs D, G and AET are differentiable; tests/check_kernel_grads.py compares its gradients with
float64 autograd of the plain loop.
"""

import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------------- routing store (GR4J X2/X3), time-varying inputs
# Same equations as gr_route; X2, X3 are [N, T] (a static value may be passed expanded). Gradients are returned per day.

@triton.jit
def _tvfwd(x9_ptr, x1_ptr, x2_ptr, x3_ptr, r0_ptr, q_ptr, rp_ptr, xr_ptr, rend_ptr, lam_ptr, x5_ptr, N, T, BLOCK: tl.constexpr,
           CR: tl.constexpr = 1.0, CD: tl.constexpr = 1.0, LIN: tl.constexpr = 0, EXL: tl.constexpr = 0):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    R = tl.load(r0_ptr + offs, mask=m, other=0.0)
    for t in range(T):
        x9 = tl.load(x9_ptr + t * N + offs, mask=m, other=0.0)
        x1 = tl.load(x1_ptr + t * N + offs, mask=m, other=0.0)
        X2 = tl.load(x2_ptr + t * N + offs, mask=m, other=0.0)
        X3 = tl.load(x3_ptr + t * N + offs, mask=m, other=1.0)
        tl.store(rp_ptr + t * N + offs, R, mask=m)
        X5 = tl.load(x5_ptr + t * N + offs, mask=m, other=0.0)
        rr = R / X3
        if EXL:                                                                           # GR6J form X2 (R/X3 - X5)
            gp = rr
        else:
            gp = tl.where(rr > 0, tl.exp(3.5 * tl.log(tl.maximum(rr, 1e-30))), 0.0)
        F = X2 * (gp - X5)
        Rn = tl.maximum(R + x9 + CR * F, 0.0)
        if LIN:                                                                           # linear store: daily fraction lam
            lam = tl.load(lam_ptr + t * N + offs, mask=m, other=0.0)
            Qr = Rn * lam
        else:
            r = Rn / X3
            r4 = r * r * r * r
            Qr = Rn * (1.0 - tl.exp(-0.25 * tl.log(1.0 + r4)))
        Qd = tl.maximum(x1 + CD * F, 0.0)
        tl.store(q_ptr + t * N + offs, Qr + Qd, mask=m)
        tl.store(xr_ptr + t * N + offs, (Rn - R - x9) + (Qd - x1), mask=m)
        R = Rn - Qr
    tl.store(rend_ptr + offs, R, mask=m)


@triton.jit
def _tvbwd(gq_ptr, gxr_ptr, x9_ptr, x1_ptr, x2_ptr, x3_ptr, rp_ptr, g9_ptr, g1_ptr, gx2_ptr, gx3_ptr, lam_ptr, glam_ptr, x5_ptr, gx5_ptr,
           N, T, BLOCK: tl.constexpr, CR: tl.constexpr = 1.0, CD: tl.constexpr = 1.0, LIN: tl.constexpr = 0, EXL: tl.constexpr = 0):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    gR = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(T):
        t = T - 1 - i
        R = tl.load(rp_ptr + t * N + offs, mask=m, other=0.0)
        x9 = tl.load(x9_ptr + t * N + offs, mask=m, other=0.0)
        x1 = tl.load(x1_ptr + t * N + offs, mask=m, other=0.0)
        X2 = tl.load(x2_ptr + t * N + offs, mask=m, other=0.0)
        X3 = tl.load(x3_ptr + t * N + offs, mask=m, other=1.0)
        gq = tl.load(gq_ptr + t * N + offs, mask=m, other=0.0)
        gx = tl.load(gxr_ptr + t * N + offs, mask=m, other=0.0)                          # gradient w.r.t. the realised exchange
        X5 = tl.load(x5_ptr + t * N + offs, mask=m, other=0.0)
        rr = tl.maximum(R / X3, 1e-30)
        if EXL:
            gp = R / X3
            dg = 1.0 + 0.0 * R
        else:
            gp = tl.where(R > 0, tl.exp(3.5 * tl.log(rr)), 0.0)
            dg = tl.where(R > 0, 3.5 * tl.exp(2.5 * tl.log(rr)), 0.0)
        p35 = gp - X5                                                                     # dF/dX2
        F = X2 * p35
        dF_dR = X2 * dg / X3
        s = R + x9 + CR * F
        m1 = tl.where(s > 0, 1.0, 0.0)
        Rn = tl.maximum(s, 0.0)
        c2 = tl.where(x1 + CD * F > 0, 1.0, 0.0)                                         # dQd/dx1
        if CD == 0.0:
            c2 = 1.0 + 0.0 * F                                                            # Qd = x1 exactly
        m2 = c2 * CD                                                                      # dQd/dF
        if LIN:
            lam = tl.load(lam_ptr + t * N + offs, mask=m, other=0.0)
            dQ_dRn = lam
            dQ_dX3 = 0.0 * Rn
            tl.store(glam_ptr + t * N + offs, (gq - gR) * Rn, mask=m)
        else:
            r = Rn / X3
            r4 = r * r * r * r
            u = 1.0 + r4
            u54 = tl.exp(-1.25 * tl.log(u))
            f = 1.0 - tl.exp(-0.25 * tl.log(u))
            dQ_dRn = f + r4 * u54
            dQ_dX3 = -r4 * r * u54
        gRn = gq * dQ_dRn + gR * (1.0 - dQ_dRn) + gx
        gF = gRn * m1 * CR + gq * m2 + gx * m2
        tl.store(g9_ptr + t * N + offs, gRn * m1 - gx, mask=m)
        tl.store(g1_ptr + t * N + offs, gq * c2 + gx * (c2 - 1.0), mask=m)
        tl.store(gx2_ptr + t * N + offs, gF * p35, mask=m)
        tl.store(gx3_ptr + t * N + offs, (gq - gR) * dQ_dX3 + gF * (-X2 * dg * (R / X3) / X3), mask=m)
        tl.store(gx5_ptr + t * N + offs, gF * (-X2), mask=m)
        gR = gRn * m1 + gF * dF_dR - gx


class GRRouteTV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x9, x1, X2, X3, R0, cr=1.0, cd=1.0, lam=None, X5=None, exl=False):
        N, T = x9.shape
        a9, a1 = x9.t().contiguous().float(), x1.t().contiguous().float()
        b2, b3 = X2.t().contiguous().float(), X3.t().contiguous().float()
        lin = lam is not None
        bl = lam.t().contiguous().float() if lin else a9
        b5 = X5.t().contiguous().float() if X5 is not None else torch.zeros_like(a9)
        q = torch.empty_like(a9); rp = torch.empty_like(a9); xr = torch.empty_like(a9)
        rend = torch.empty(N, device=x9.device, dtype=torch.float32)
        _tvfwd[(triton.cdiv(N, 128),)](a9, a1, b2, b3, R0.contiguous().float(), q, rp, xr, rend, bl, b5, N, T, BLOCK=128, CR=float(cr), CD=float(cd),
                                       LIN=int(lin), EXL=int(exl))
        ctx.cr, ctx.cd, ctx.lin, ctx.exl, ctx.has5 = float(cr), float(cd), int(lin), int(exl), X5 is not None
        ctx.save_for_backward(a9, a1, b2, b3, rp, bl, b5)
        ctx.mark_non_differentiable(rend)
        return q.t(), xr.t(), rend

    @staticmethod
    def backward(ctx, gq, gxr, grend):
        a9, a1, b2, b3, rp, bl, b5 = ctx.saved_tensors
        T, N = a9.shape
        gqt = gq.t().contiguous().float() if gq is not None else torch.zeros_like(a9)
        gxt = gxr.t().contiguous().float() if gxr is not None else torch.zeros_like(a9)
        g9, g1, g2, g3, gl, g5 = (torch.empty_like(gqt) for _ in range(6))
        _tvbwd[(triton.cdiv(N, 128),)](gqt, gxt, a9, a1, b2, b3, rp, g9, g1, g2, g3, bl, gl, b5, g5, N, T, BLOCK=128, CR=ctx.cr, CD=ctx.cd,
                                       LIN=ctx.lin, EXL=ctx.exl)
        return (g9.t(), g1.t(), g2.t(), g3.t(), None, None, None, (gl.t() if ctx.lin else None), (g5.t() if ctx.has5 else None), None)


def gr_route_tv(x9, x1, X2, X3, R0, use_qd=True, split=None, lam=None, X5=None, exch_lin=False):
    """x9, x1, X2, X3 [..., T]; R0 [...] -> Q [..., T], realised exchange [..., T], R_end [...].
    lam [..., T] given -> LINEAR store, Qr = lam Rn (X3 then only scales the exchange). X5 [..., T]: exchange threshold,
    F = X2 (g(R/X3) - X5) with g(r) = r^3.5, or g(r) = r if exch_lin (GR6J form). The exchange F enters the routing store as CR F and
    the direct flow as CD F: GR4J (1, 1) (default); use_qd=False -> (1, 0)
    (store only); split=(cr, cd) explicit, e.g. (0.9, 0.1): F once, shared like the 0.9 / 0.1 split of the routed water."""
    shp = x9.shape
    f = lambda v: v.expand(shp).reshape(-1, shp[-1])
    cr, cd = split if split is not None else ((1.0, 1.0) if use_qd else (1.0, 0.0))
    q, xr, re = GRRouteTV.apply(f(x9), f(x1), f(X2), f(X3), R0.reshape(-1), cr, cd, f(lam) if lam is not None else None,
                                f(X5) if X5 is not None else None, exch_lin)
    return q.reshape(shp), xr.reshape(shp), re.reshape(shp[:-1])


# As gprod, but percolation G = S2 [1 - (1 + kq (S2/C)^4)^(-1/4)], kq = 4 Kp dt / C per series (exact integral of dS/dt = -Kp (S/C)^5 over one
# day); GR4J: kq = (4/9)^4, i.e. Kp = C (4/9)^4 / 4. Gradients w.r.t. p, e, C (kq held fixed) and kq.

@triton.jit
def _pkfwd(p_ptr, e_ptr, c_ptr, k_ptr, s0_ptr, d_ptr, g_ptr, a_ptr, sp_ptr, send_ptr, N, T, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    S = tl.load(s0_ptr + offs, mask=m, other=0.0)
    Cc = tl.load(c_ptr + offs, mask=m, other=1.0)
    kq = tl.load(k_ptr + offs, mask=m, other=0.0)
    for t in range(T):
        p = tl.load(p_ptr + t * N + offs, mask=m, other=0.0)
        e = tl.load(e_ptr + t * N + offs, mask=m, other=0.0)
        tl.store(sp_ptr + t * N + offs, S, mask=m)
        pn = tl.maximum(p - e, 0.0)
        en = tl.maximum(e - p, 0.0)
        s = S / Cc
        tp = 1.0 - 2.0 / (tl.exp(2.0 * pn / Cc) + 1.0)
        te = 1.0 - 2.0 / (tl.exp(2.0 * en / Cc) + 1.0)
        ps = Cc * (1.0 - s * s) * tp / (1.0 + s * tp)
        es = S * (2.0 - s) * te / (1.0 + (1.0 - s) * te)
        S2 = S - es + ps
        q = S2 / Cc
        r4 = kq * q * q * q * q
        g = S2 * (1.0 - tl.exp(-0.25 * tl.log(1.0 + r4)))
        tl.store(d_ptr + t * N + offs, pn - ps, mask=m)
        tl.store(g_ptr + t * N + offs, g, mask=m)
        tl.store(a_ptr + t * N + offs, es + tl.minimum(p, e), mask=m)
        S = S2 - g
    tl.store(send_ptr + offs, S, mask=m)


@triton.jit
def _pkbwd(gd_ptr, gg_ptr, ga_ptr, p_ptr, e_ptr, c_ptr, k_ptr, sp_ptr, gp_ptr, ge_ptr, gc_ptr, gk_ptr, N, T, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    Cc = tl.load(c_ptr + offs, mask=m, other=1.0)
    kq = tl.load(k_ptr + offs, mask=m, other=0.0)
    gS = tl.zeros([BLOCK], dtype=tl.float32)
    gC = tl.zeros([BLOCK], dtype=tl.float32)
    gK = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(T):
        t = T - 1 - i
        S = tl.load(sp_ptr + t * N + offs, mask=m, other=0.0)
        p = tl.load(p_ptr + t * N + offs, mask=m, other=0.0)
        e = tl.load(e_ptr + t * N + offs, mask=m, other=0.0)
        gd = tl.load(gd_ptr + t * N + offs, mask=m, other=0.0)
        gg = tl.load(gg_ptr + t * N + offs, mask=m, other=0.0)
        ga = tl.load(ga_ptr + t * N + offs, mask=m, other=0.0)          # adjoint of AET = es + min(p, e)
        w = tl.where(p >= e, 1.0, 0.0)
        pn = tl.maximum(p - e, 0.0)
        en = tl.maximum(e - p, 0.0)
        s = S / Cc
        tp = 1.0 - 2.0 / (tl.exp(2.0 * pn / Cc) + 1.0)
        te = 1.0 - 2.0 / (tl.exp(2.0 * en / Cc) + 1.0)
        Nn = (1.0 - s * s) * tp
        Dn = 1.0 + s * tp
        dps_dS = (-2.0 * s * tp * Dn - Nn * tp) / (Dn * Dn)
        dps_dpn = Cc * ((1.0 - s * s) * Dn - Nn * s) / (Dn * Dn) * (1.0 - tp * tp) / Cc
        Mm = S * (2.0 - s) * te
        Ld = 1.0 + (1.0 - s) * te
        es = Mm / Ld
        des_dS = ((2.0 - 2.0 * s) * te * Ld - Mm * (-te / Cc)) / (Ld * Ld)
        des_den = (S * (2.0 - s) * Ld - Mm * (1.0 - s)) / (Ld * Ld) * (1.0 - te * te) / Cc
        ps = Cc * Nn / Dn
        S2 = S - es + ps
        q = S2 / Cc
        q4 = q * q * q * q
        r4 = kq * q4
        u = 1.0 + r4
        u54 = tl.exp(-1.25 * tl.log(u))
        dg_dS2 = (1.0 - tl.exp(-0.25 * tl.log(u))) + r4 * u54
        gS2 = gS * (1.0 - dg_dS2) + gg * dg_dS2
        gps = gS2 - gd
        ges = -gS2 + ga
        gpn = gd + gps * dps_dpn
        gen = ges * des_den
        tl.store(gp_ptr + t * N + offs, gpn * w - gen * (1.0 - w) + ga * (1.0 - w), mask=m)
        tl.store(ge_ptr + t * N + offs, -gpn * w + gen * (1.0 - w) + ga * w, mask=m)
        ds_dC = -s / Cc
        dtp_dC = -(1.0 - tp * tp) * pn / (Cc * Cc)
        dte_dC = -(1.0 - te * te) * en / (Cc * Cc)
        dN_dC = -2.0 * s * ds_dC * tp + (1.0 - s * s) * dtp_dC
        dD_dC = ds_dC * tp + s * dtp_dC
        dps_dC = Nn / Dn + Cc * (dN_dC * Dn - Nn * dD_dC) / (Dn * Dn)
        dM_dC = -S * ds_dC * te + S * (2.0 - s) * dte_dC
        dL_dC = -ds_dC * te + (1.0 - s) * dte_dC
        des_dC = (dM_dC * Ld - Mm * dL_dC) / (Ld * Ld)
        dg_dC = -S2 * r4 * u54 / Cc
        dg_dk = 0.25 * S2 * q4 * u54
        gC += gps * dps_dC + ges * des_dC + (gg - gS) * dg_dC
        gK += (gg - gS) * dg_dk
        gS = gS2 + gps * dps_dS + ges * des_dS
    tl.store(gc_ptr + offs, gC, mask=m)
    tl.store(gk_ptr + offs, gK, mask=m)


class GProdK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, e, Cc, kq, S0):
        N, T = p.shape
        pt, et = p.t().contiguous().float(), e.t().contiguous().float()
        d = torch.empty_like(pt); g = torch.empty_like(pt); a = torch.empty_like(pt); sp = torch.empty_like(pt)
        send = torch.empty(N, device=p.device, dtype=torch.float32)
        cc, kk = Cc.contiguous().float(), kq.contiguous().float()
        _pkfwd[(triton.cdiv(N, 128),)](pt, et, cc, kk, S0.contiguous().float(), d, g, a, sp, send, N, T, BLOCK=128)
        ctx.save_for_backward(pt, et, cc, kk, sp)
        ctx.mark_non_differentiable(send)                       # AET is differentiable: it feeds the uptake demand and the soil state
        return d.t(), g.t(), a.t(), send

    @staticmethod
    def backward(ctx, gd, gg, ga, gsend):
        pt, et, cc, kk, sp = ctx.saved_tensors
        T, N = pt.shape
        gdt = gd.t().contiguous().float() if gd is not None else torch.zeros_like(pt)
        ggt = gg.t().contiguous().float() if gg is not None else torch.zeros_like(pt)
        gat = ga.t().contiguous().float() if ga is not None else torch.zeros_like(pt)
        gp, ge = torch.empty_like(gdt), torch.empty_like(gdt)
        gc, gk = torch.empty(N, device=pt.device, dtype=torch.float32), torch.empty(N, device=pt.device, dtype=torch.float32)
        _pkbwd[(triton.cdiv(N, 128),)](gdt, ggt, gat, pt, et, cc, kk, sp, gp, ge, gc, gk, N, T, BLOCK=128)
        return gp.t(), ge.t(), gc, gk, None


def gprod_k(p, e, Cc, kq, S0):
    """p, e [N, T]; Cc, kq, S0 [N] -> D, G, AET [N, T], S_end [N]; percolation G = S2 [1 - (1 + kq (S2/C)^4)^(-1/4)]"""
    return GProdK.apply(p, e, Cc, kq, S0)


# ----------------------------------------------------------------------------- quadratic store, exact daily drainage after the inflow
# W- = W + g;  W+ = W- / (1 + a W-)  (exact solution of dW/dt = -a W^2 over one day);  Q = W- - W+.

@triton.jit
def _qsfwd(g_ptr, a_ptr, w0_ptr, q_ptr, wm_ptr, wend_ptr, N, T, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    W = tl.load(w0_ptr + offs, mask=m, other=0.0)
    a = tl.load(a_ptr + offs, mask=m, other=0.0)
    for t in range(T):
        Wm = W + tl.load(g_ptr + t * N + offs, mask=m, other=0.0)
        tl.store(wm_ptr + t * N + offs, Wm, mask=m)
        Wp = Wm / (1.0 + a * Wm)
        tl.store(q_ptr + t * N + offs, Wm - Wp, mask=m)
        W = Wp
    tl.store(wend_ptr + offs, W, mask=m)


@triton.jit
def _qsbwd(gq_ptr, a_ptr, wm_ptr, gg_ptr, ga_ptr, N, T, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    a = tl.load(a_ptr + offs, mask=m, other=0.0)
    gW = tl.zeros([BLOCK], dtype=tl.float32)
    gA = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(T):
        t = T - 1 - i
        Wm = tl.load(wm_ptr + t * N + offs, mask=m, other=0.0)
        gq = tl.load(gq_ptr + t * N + offs, mask=m, other=0.0)
        den = 1.0 + a * Wm
        d = 1.0 / (den * den)
        gWm = gq * (1.0 - d) + gW * d
        gA += (gq - gW) * Wm * Wm * d
        tl.store(gg_ptr + t * N + offs, gWm, mask=m)
        gW = gWm
    tl.store(ga_ptr + offs, gA, mask=m)


class QuadStore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, a, W0):
        N, T = g.shape
        gt = g.t().contiguous().float()
        q = torch.empty_like(gt); wm = torch.empty_like(gt)
        wend = torch.empty(N, device=g.device, dtype=torch.float32)
        aa = a.contiguous().float()
        _qsfwd[(triton.cdiv(N, 128),)](gt, aa, W0.contiguous().float(), q, wm, wend, N, T, BLOCK=128)
        ctx.save_for_backward(aa, wm)
        ctx.mark_non_differentiable(wend)
        return q.t(), wend

    @staticmethod
    def backward(ctx, gq, gwend):
        aa, wm = ctx.saved_tensors
        T, N = wm.shape
        gqt = gq.t().contiguous().float()
        gg = torch.empty_like(gqt); ga = torch.empty(N, device=gqt.device, dtype=torch.float32)
        _qsbwd[(triton.cdiv(N, 128),)](gqt, aa, wm, gg, ga, N, T, BLOCK=128)
        return gg.t(), ga, None


def quad_store(g, a, W0):
    """g [N, T] inflow; a, W0 [N] -> Q [N, T], W_end [N] (dW/dt = -a W^2, exact per day after the inflow)"""
    return QuadStore.apply(g, a, W0)

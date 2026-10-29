"""Fused Triton kernel for the HBV daily recursion of baselines.dhbv (hydrodl2 HBV 1.0 equations), with a hand-written adjoint backward.

One program lane per (catchment x component) series; the forward stores the five states before each day, the backward recomputes the
day from them and propagates the adjoints of SP (snowpack), MW (meltwater), SM (soil), SUZ, SLZ backwards in time.
Static parameters (11, per lane): TT, CFMAX, CFR, CWH, FC, LP, PERC, UZL, K0, K1, K2; per-day parameters (per lane): BETA and, when
DYN_ET, BETAET (the ET-shape exponent exists only in the dynamic version, as hydrodl2). Inputs P, T, PET are not differentiated; the
initial states come from a gradient-free spin-up.
hbv_fused(P, T, E, stat, beta, betaet, states) -> Q [N, L] (component runoff, before averaging and routing), final states [5, N]
"""

import torch
import triton
import triton.language as tl

STAT = ["TT", "CFMAX", "CFR", "CWH", "FC", "LP", "PERC", "UZL", "K0", "K1", "K2"]


@triton.jit
def _hbv_fwd(p_ptr, t_ptr, e_ptr, st_ptr, b_ptr, g_ptr, s0_ptr, q_ptr, sv_ptr, send_ptr, N, L,
             BLOCK: tl.constexpr, DYN_ET: tl.constexpr, STORE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    TT = tl.load(st_ptr + 0 * N + offs, mask=m, other=0.0)
    CFMAX = tl.load(st_ptr + 1 * N + offs, mask=m, other=1.0)
    CFR = tl.load(st_ptr + 2 * N + offs, mask=m, other=0.0)
    CWH = tl.load(st_ptr + 3 * N + offs, mask=m, other=0.0)
    FC = tl.load(st_ptr + 4 * N + offs, mask=m, other=100.0)
    LP = tl.load(st_ptr + 5 * N + offs, mask=m, other=0.5)
    PERC = tl.load(st_ptr + 6 * N + offs, mask=m, other=0.0)
    UZL = tl.load(st_ptr + 7 * N + offs, mask=m, other=0.0)
    K0 = tl.load(st_ptr + 8 * N + offs, mask=m, other=0.1)
    K1 = tl.load(st_ptr + 9 * N + offs, mask=m, other=0.1)
    K2 = tl.load(st_ptr + 10 * N + offs, mask=m, other=0.01)
    SP = tl.load(s0_ptr + 0 * N + offs, mask=m, other=0.0)
    MW = tl.load(s0_ptr + 1 * N + offs, mask=m, other=0.0)
    SM = tl.load(s0_ptr + 2 * N + offs, mask=m, other=1.0)
    SUZ = tl.load(s0_ptr + 3 * N + offs, mask=m, other=0.0)
    SLZ = tl.load(s0_ptr + 4 * N + offs, mask=m, other=0.0)
    for t in range(L):
        if STORE:
            tl.store(sv_ptr + (0 * L + t) * N + offs, SP, mask=m)
            tl.store(sv_ptr + (1 * L + t) * N + offs, MW, mask=m)
            tl.store(sv_ptr + (2 * L + t) * N + offs, SM, mask=m)
            tl.store(sv_ptr + (3 * L + t) * N + offs, SUZ, mask=m)
            tl.store(sv_ptr + (4 * L + t) * N + offs, SLZ, mask=m)
        p = tl.load(p_ptr + t * N + offs, mask=m, other=0.0)
        tc = tl.load(t_ptr + t * N + offs, mask=m, other=10.0)
        e = tl.load(e_ptr + t * N + offs, mask=m, other=0.0)
        beta = tl.load(b_ptr + t * N + offs, mask=m, other=1.0)
        warm = tc >= TT
        r = tl.where(warm, p, 0.0)
        s = tl.where(warm, 0.0, p)
        SP = SP + s
        melt = tl.minimum(tl.maximum(CFMAX * (tc - TT), 0.0), SP)
        MW = MW + melt
        SP = SP - melt
        refr = tl.minimum(tl.maximum(CFR * CFMAX * (TT - tc), 0.0), MW)
        SP = SP + refr
        MW = MW - refr
        tos = tl.maximum(MW - CWH * SP, 0.0)
        MW = MW - tos
        x = SM / FC
        w = tl.minimum(tl.maximum(tl.exp(beta * tl.log(tl.maximum(x, 1e-30))), 0.0), 1.0)
        inp = r + tos
        rech = inp * w
        SM = SM + inp - rech
        exc = tl.maximum(SM - FC, 0.0)
        SM = SM - exc
        y = SM / (LP * FC)
        if DYN_ET:
            bet = tl.load(g_ptr + t * N + offs, mask=m, other=1.0)
            efr = tl.exp(bet * tl.log(tl.maximum(y, 1e-30)))
        else:
            efr = y
        ef = tl.minimum(tl.maximum(efr, 0.0), 1.0)
        et = tl.minimum(SM, e * ef)
        SM = tl.maximum(SM - et, 1e-5)
        SUZ = SUZ + rech + exc
        perc = tl.minimum(SUZ, PERC)
        SUZ = SUZ - perc
        q0 = K0 * tl.maximum(SUZ - UZL, 0.0)
        SUZ = SUZ - q0
        q1 = K1 * SUZ
        SUZ = SUZ - q1
        SLZ = SLZ + perc
        q2 = K2 * SLZ
        SLZ = SLZ - q2
        tl.store(q_ptr + t * N + offs, q0 + q1 + q2, mask=m)
    tl.store(send_ptr + 0 * N + offs, SP, mask=m)
    tl.store(send_ptr + 1 * N + offs, MW, mask=m)
    tl.store(send_ptr + 2 * N + offs, SM, mask=m)
    tl.store(send_ptr + 3 * N + offs, SUZ, mask=m)
    tl.store(send_ptr + 4 * N + offs, SLZ, mask=m)


@triton.jit
def _hbv_bwd(gq_ptr, p_ptr, t_ptr, e_ptr, st_ptr, b_ptr, g_ptr, sv_ptr, gst_ptr, gb_ptr, gg_ptr, N, L,
             BLOCK: tl.constexpr, DYN_ET: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    TT = tl.load(st_ptr + 0 * N + offs, mask=m, other=0.0)
    CFMAX = tl.load(st_ptr + 1 * N + offs, mask=m, other=1.0)
    CFR = tl.load(st_ptr + 2 * N + offs, mask=m, other=0.0)
    CWH = tl.load(st_ptr + 3 * N + offs, mask=m, other=0.0)
    FC = tl.load(st_ptr + 4 * N + offs, mask=m, other=100.0)
    LP = tl.load(st_ptr + 5 * N + offs, mask=m, other=0.5)
    PERC = tl.load(st_ptr + 6 * N + offs, mask=m, other=0.0)
    UZL = tl.load(st_ptr + 7 * N + offs, mask=m, other=0.0)
    K0 = tl.load(st_ptr + 8 * N + offs, mask=m, other=0.1)
    K1 = tl.load(st_ptr + 9 * N + offs, mask=m, other=0.1)
    K2 = tl.load(st_ptr + 10 * N + offs, mask=m, other=0.01)
    zero = tl.zeros([BLOCK], dtype=tl.float32)
    gSP = zero; gMW = zero; gSM = zero; gSUZ = zero; gSLZ = zero
    gTT = zero; gCFMAX = zero; gCFR = zero; gCWH = zero; gFC = zero; gLP = zero
    gPERC = zero; gUZL = zero; gK0 = zero; gK1 = zero; gK2 = zero
    for i in range(L):
        t = L - 1 - i
        SP0 = tl.load(sv_ptr + (0 * L + t) * N + offs, mask=m, other=0.0)
        MW0 = tl.load(sv_ptr + (1 * L + t) * N + offs, mask=m, other=0.0)
        SM0 = tl.load(sv_ptr + (2 * L + t) * N + offs, mask=m, other=1.0)
        SUZ0 = tl.load(sv_ptr + (3 * L + t) * N + offs, mask=m, other=0.0)
        SLZ0 = tl.load(sv_ptr + (4 * L + t) * N + offs, mask=m, other=0.0)
        p = tl.load(p_ptr + t * N + offs, mask=m, other=0.0)
        tc = tl.load(t_ptr + t * N + offs, mask=m, other=10.0)
        e = tl.load(e_ptr + t * N + offs, mask=m, other=0.0)
        beta = tl.load(b_ptr + t * N + offs, mask=m, other=1.0)
        gQ = tl.load(gq_ptr + t * N + offs, mask=m, other=0.0)
        # ---- recompute the day
        warm = tc >= TT
        r = tl.where(warm, p, 0.0)
        s = tl.where(warm, 0.0, p)
        SP1 = SP0 + s
        mraw = CFMAX * (tc - TT)
        mpos = tl.maximum(mraw, 0.0)
        ind_m = SP1 < mpos                                                                   # melt = SP1
        melt = tl.minimum(mpos, SP1)
        MW1 = MW0 + melt
        SP2 = SP1 - melt
        fraw = CFR * CFMAX * (TT - tc)
        fpos = tl.maximum(fraw, 0.0)
        ind_r = MW1 < fpos                                                                   # refr = MW1
        refr = tl.minimum(fpos, MW1)
        SP3 = SP2 + refr
        MW2 = MW1 - refr
        traw = MW2 - CWH * SP3
        ind_t = traw > 0.0
        tos = tl.maximum(traw, 0.0)
        x = SM0 / FC
        lx = tl.log(tl.maximum(x, 1e-30))
        wraw = tl.exp(beta * lx)
        ind_w = (wraw > 0.0) & (wraw < 1.0)
        w = tl.minimum(tl.maximum(wraw, 0.0), 1.0)
        inp = r + tos
        SM1 = SM0 + inp - inp * w
        ind_x = SM1 > FC
        exc = tl.maximum(SM1 - FC, 0.0)
        SM2 = SM1 - exc
        y = SM2 / (LP * FC)
        ly = tl.log(tl.maximum(y, 1e-30))
        if DYN_ET:
            bet = tl.load(g_ptr + t * N + offs, mask=m, other=1.0)
            efr = tl.exp(bet * ly)
            defr_dy = bet * efr / tl.maximum(y, 1e-30)
        else:
            bet = 1.0 + 0.0 * y
            efr = y
            defr_dy = 1.0 + 0.0 * y
        ind_c = (efr > 0.0) & (efr < 1.0)
        ef = tl.minimum(tl.maximum(efr, 0.0), 1.0)
        etp = e * ef
        ind_e = SM2 < etp                                                                    # et = SM2
        et = tl.minimum(SM2, etp)
        ind_s = (SM2 - et) > 1e-5
        rech = inp * w
        SUZ1 = SUZ0 + rech + exc
        ind_p = SUZ1 < PERC                                                                  # perc = SUZ1
        perc = tl.minimum(SUZ1, PERC)
        SUZ2 = SUZ1 - perc
        ind_q = (SUZ2 - UZL) > 0.0
        q0 = K0 * tl.maximum(SUZ2 - UZL, 0.0)
        SUZ3 = SUZ2 - q0
        SLZ1 = SLZ0 + perc
        # ---- adjoints (g* of the states after the day are gSP, gMW, gSM, gSUZ, gSLZ)
        gq0 = gQ; gq1 = gQ; gq2 = gQ
        gSLZ1 = gSLZ * (1.0 - K2) + gq2 * K2
        gK2 += SLZ1 * (gq2 - gSLZ)
        gperc = gSLZ1
        gSUZ3 = gSUZ * (1.0 - K1) + gq1 * K1
        gK1 += SUZ3 * (gq1 - gSUZ)
        gq0t = gq0 - gSUZ3
        iq = tl.where(ind_q, 1.0, 0.0)
        gSUZ2 = gSUZ3 + gq0t * K0 * iq
        gUZL += -gq0t * K0 * iq
        gK0 += gq0t * tl.maximum(SUZ2 - UZL, 0.0)
        gpt = gperc - gSUZ2
        ip = tl.where(ind_p, 1.0, 0.0)
        gSUZ1 = gSUZ2 + gpt * ip
        gPERC += gpt * (1.0 - ip)
        grech = gSUZ1
        gexc = gSUZ1
        gSLZ_prev = gSLZ1
        gSUZ_prev = gSUZ1
        # soil: SM3 = max(SM2 - et, 1e-5)
        gSMm = tl.where(ind_s, gSM, 0.0)
        gSM2 = gSMm
        get = -gSMm
        ie = tl.where(ind_e, 1.0, 0.0)
        gSM2 += get * ie
        getp = get * (1.0 - ie)
        gef = getp * e
        gefr = tl.where(ind_c, gef, 0.0)
        gy = gefr * defr_dy
        if DYN_ET:
            tl.store(gg_ptr + t * N + offs, gefr * efr * ly, mask=m)
        gSM2 += gy / (LP * FC)
        gLP += -gy * SM2 / (LP * LP * FC)
        gFC += -gy * SM2 / (LP * FC * FC)
        # SM2 = SM1 - exc, exc = max(SM1 - FC, 0)
        gexct = gexc - gSM2
        ix = tl.where(ind_x, 1.0, 0.0)
        gSM1 = gSM2 + gexct * ix
        gFC += -gexct * ix
        # SM1 = SM0 + inp - rech, rech = inp w
        grt = grech - gSM1
        ginp = gSM1 + grt * w
        gw = grt * inp
        gwr = tl.where(ind_w, gw, 0.0)
        tl.store(gb_ptr + t * N + offs, gwr * wraw * lx, mask=m)
        gx = gwr * beta * wraw / tl.maximum(x, 1e-30)
        gSM_prev = gSM1 + gx / FC
        gFC += -gx * SM0 / (FC * FC)
        # inp = r + tos; MW3 = MW2 - tos, tos = max(MW2 - CWH SP3, 0)
        gtos = ginp
        gtt = gtos - gMW
        it = tl.where(ind_t, 1.0, 0.0)
        gMW2 = gMW + gtt * it
        gSP3 = gSP - gtt * it * CWH
        gCWH += -gtt * it * SP3
        # SP3 = SP2 + refr, MW2 = MW1 - refr, refr = min(fpos, MW1)
        grf = gSP3 - gMW2
        ir = tl.where(ind_r, 1.0, 0.0)
        gMW1 = gMW2 + grf * ir
        gfp = grf * (1.0 - ir)
        gf = tl.where(fraw > 0.0, gfp, 0.0)
        gCFR += gf * CFMAX * (TT - tc)
        gCFMAX += gf * CFR * (TT - tc)
        gTT += gf * CFR * CFMAX
        gSP2 = gSP3
        # MW1 = MW0 + melt, SP2 = SP1 - melt, melt = min(mpos, SP1)
        gml = gMW1 - gSP2
        im = tl.where(ind_m, 1.0, 0.0)
        gSP1 = gSP2 + gml * im
        gmp = gml * (1.0 - im)
        gmr = tl.where(mraw > 0.0, gmp, 0.0)
        gCFMAX += gmr * (tc - TT)
        gTT += -gmr * CFMAX
        gSP = gSP1
        gMW = gMW1
        gSM = gSM_prev
        gSUZ = gSUZ_prev
        gSLZ = gSLZ_prev
    tl.store(gst_ptr + 0 * N + offs, gTT, mask=m)
    tl.store(gst_ptr + 1 * N + offs, gCFMAX, mask=m)
    tl.store(gst_ptr + 2 * N + offs, gCFR, mask=m)
    tl.store(gst_ptr + 3 * N + offs, gCWH, mask=m)
    tl.store(gst_ptr + 4 * N + offs, gFC, mask=m)
    tl.store(gst_ptr + 5 * N + offs, gLP, mask=m)
    tl.store(gst_ptr + 6 * N + offs, gPERC, mask=m)
    tl.store(gst_ptr + 7 * N + offs, gUZL, mask=m)
    tl.store(gst_ptr + 8 * N + offs, gK0, mask=m)
    tl.store(gst_ptr + 9 * N + offs, gK1, mask=m)
    tl.store(gst_ptr + 10 * N + offs, gK2, mask=m)


class HBVFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P, T, E, stat, beta, betaet, s0):
        """P, T, E, beta, betaet: [L, N] (time-major, contiguous); stat [11, N]; s0 [5, N]"""
        L, N = P.shape
        dyn_et = betaet is not None
        q = torch.empty_like(P)
        need = any(ctx.needs_input_grad[3:6])                                                  # (grad mode is off inside forward)
        sv = torch.empty((5, L, N), device=P.device, dtype=torch.float32) if need else torch.empty(1, device=P.device)
        send = torch.empty((5, N), device=P.device, dtype=torch.float32)
        g_ = betaet if dyn_et else beta
        _hbv_fwd[(triton.cdiv(N, 128),)](P, T, E, stat, beta, g_, s0, q, sv, send, N, L, BLOCK=128, DYN_ET=int(dyn_et), STORE=int(need))
        ctx.dyn_et = dyn_et
        ctx.save_for_backward(P, T, E, stat, beta, g_, sv)
        ctx.mark_non_differentiable(send)
        return q, send

    @staticmethod
    def backward(ctx, gq, gsend):
        P, T, E, stat, beta, g_, sv = ctx.saved_tensors
        L, N = P.shape
        gst = torch.empty_like(stat); gb = torch.empty_like(beta)
        gg = torch.empty_like(beta) if ctx.dyn_et else torch.empty(1, device=P.device)
        _hbv_bwd[(triton.cdiv(N, 128),)](gq.contiguous(), P, T, E, stat, beta, g_, sv, gst, gb, gg, N, L, BLOCK=128, DYN_ET=int(ctx.dyn_et))
        return None, None, None, gst, gb, (gg if ctx.dyn_et else None), None


def hbv_fused(P, T, E, stat, beta, betaet, s0):
    """time-major float32 inputs; see HBVFused.forward"""
    f = lambda v: v.contiguous().float() if v is not None else None
    return HBVFused.apply(f(P), f(T), f(E), f(stat), f(beta), f(betaet), f(s0))

"""Regional HBV with the HBV-light snow routine on elevation bands, on CAMELS-GB, CAMELS-US and CAMELS-AUS under the unified held-out
protocol.

Model = HBV-9 (the nine HBV-light parameters of kernels/hbv9_core.py, fused kernels of kernels/hbvstore.py) preceded by the HBV-light
snow routine (kernels/hbvsnow.py: TT, CFMAX, SFCF, CFR, CWH) applied on the same K = 5 equal-area elevation bands with the same
per-catchment lapse rate Gamma (3-9 C/km) as CaRT: T_k = T - Gamma (h_k - h_mean), liquid input = sum_k a_k liquid_k.
  GB bands: hypsometry from the CAMELS-GB elevation quantiles (elev_min, 10, 50, 90, max), equal-area centres;
  US / AUS bands: equal-area groups of the 100 m elevation bands with bf > 0 (empty groups a_k = 0),
  the construction of cart.data, so HBV receives the information CaRT receives. --no_snow: HBV-9 alone.
Parameter ranges (--ranges): feng2022 (default, the paper's runs) = FC 50-1000 mm, LP 0.2-1, K2 0.001-0.2 /d as Feng et al. (2022),
BETA 1-6, PERC 0-10 mm/d, UZL 0-100 mm, K0 0.05-0.5, K1 0.01-0.3 /d (K0 + K1 < 1 for the simultaneous upper-zone update), MAXBAS 1-7 d;
hbv9 = FC 50-700 mm, LP 0.3-1, K2 0.0001-0.1 /d, the others the same. Snow: TT -2.5-2.5 C, CFMAX 0.5-10 mm/C/d, SFCF 0.5-1.5,
CFR 0-0.1, CWH 0-0.2, Gamma 3-9 C/km.
Snow parameters per band share the catchment values; the snow kernel runs on C x K lanes (hbvsnow._hbvsnow_fwd, stored-state adjoint
with the band temperature offset as a parameter, so that Gamma is trained). The soil kernel is hbvstore.Soil with the gradient to its
input (SoilP), so the snow parameters are trained through HBV.
Training as CaRT: all gauged catchments jointly, loss mean(1 - NSE) over the training period, selection on the median val NSE of the
gauged catchments, AdamW lr 2e-3 one-cycle (pct_start 0.05), weight decay 1e-4, gradient clip 1, attribute noise 0.1, 3-layer 128-unit
GELU encoder, sigmoid-bounded head initialised at the medians of a per-catchment HBV-9 calibration on CAMELS-GB
(configs/hbv9_init_medians.json; as a fraction of the range, clipped to 0.02-0.98) for the HBV-9 part and at the CaRT initial values
for the snow part (TT 0 C, CFMAX 3, SFCF 1, CFR 0.05, CWH 0.1, Gamma 6.5 C/km). Spin-up: 730 days from SM = 0.5 FC, SUZ = SLZ = 10 mm,
snow stores 0; the end state (snow, stores, routing buffer) starts the full run. Final simulation and water balance in float64 numpy
(hbv9_core.run_numpy + a numpy transcription of the band snow routine).
Periods (unified protocol): training 1989-01-01..2008-12-31, selection 2009-01-01..2010-12-31; nothing after 2010 enters the loss or the
selection. --dataset gb: RESULT prints val, 2011-15, 2015-19. --dataset us / aus (CAMELS-AUS v2 on the US code path, bands from
elev_bands_aus.npz): RESULT prints the val period only. Held-out catchments are never in the loss or the selection, only simulated.
The full-record simulation is saved.
Output: --out (default runs/<dataset>_hbv_s<seed>)/{sim.npy [T, C] mm/d, net.pt, run.json, params.csv}.

Usage (from src/): python -m baselines.hbv_regional --dataset gb|us|aus --seed 0 [--ranges feng2022|hbv9] [--iters 5000] [--gpu 0] [--out DIR]
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import triton

from cart import data as D
from . import data_files
from cart.kernels.adjoint import backward_k
from .kernels import hbv9_core as hbv9
from .kernels import hbvstore
from .kernels.hbvsnow import BLOCK, _hbvsnow_fwd, hbvsnow_step

PKG = Path(__file__).resolve().parents[2]
SNOW = {"TT": (-2.5, 2.5, 0.0), "lCF": (np.log(0.5), np.log(10.0), np.log(3.0)), "SFCF": (0.5, 1.5, 1.0), "CFR": (0.0, 0.1, 0.05),
        "CWH": (0.0, 0.2, 0.1), "lGam": (np.log(3.0), np.log(9.0), np.log(6.5))}                       # (lo, hi, init) as the CaRT snow parameters
US_PERIODS = {"train": ("1989-01-01", "2008-12-31"), "val": ("2009-01-01", "2010-12-31")}


# ------------------------------------------------------------------ kernels
class SoilP(hbvstore.Soil):
    """hbvstore.Soil with the gradient w.r.t. its water input (needed to train the snow routine through HBV)"""
    @staticmethod
    def backward(ctx, gS, gF):
        p, e, fc, lp, be, s0, S = ctx.saved_tensors
        gi, gp = hbvstore._stored_backward(hbvstore.soil_step, S, s0, gS, [gF], [p, e], [fc, lp, be])
        return gi[0], None, gp[0], gp[1], gp[2], None


def hbv_in(P, E, prm, state=None, carry=None):
    """hbvstore.hbv with SoilP (same forward, extra input gradient)"""
    FC, LP, BETA, PERC, UZL, K0, K1, K2, MB = (prm[k] for k in hbv9.NAMES)
    if state is None:
        state = (0.5 * FC.detach(), torch.full_like(FC, 10.0), torch.full_like(FC, 10.0))
    sm0, suz0, slz0 = state
    S, F = SoilP.apply(P, E, FC, LP, BETA, sm0)
    U, Pe, Q01 = hbvstore.Upper.apply(F, PERC, UZL, K0, K1, suz0)
    k2 = K2[:, None].expand_as(Pe)
    L = hbvstore.lin_rec((1.0 - k2).contiguous(), ((1.0 - k2) * Pe).contiguous(), slz0)
    Lprev = torch.cat([slz0[:, None], L[:, :-1]], 1)
    qgw = Q01 + K2[:, None] * (Lprev + Pe)
    c = hbv9.weights(MB, "paper", torch)
    H = hbv9.L_BUF - 1
    pad = torch.zeros(qgw.shape[0], H, device=qgw.device, dtype=qgw.dtype) if carry is None else carry
    Qr = hbvstore.uh_conv(torch.cat([pad, qgw], 1), c)[:, H:]
    return Qr, (S[:, -1], U[:, -1], L[:, -1]), qgw[:, -H:]


def _snow_step(xp, p, ta, tt, cf, sfc, cr, cw, dT):
    return hbvsnow_step(xp, p, ta - dT, tt, cf, sfc, cr, cw)


class SnowLanes(torch.autograd.Function):
    """HBV-light snow routine on lanes (catchment x band) with a trainable temperature offset dT per lane: forward = hbvsnow Triton
    kernel on ta - dT, backward = stored-state adjoint (cart.kernels.adjoint.backward_k) with dT as a parameter."""
    @staticmethod
    def forward(ctx, p, ta, dT, x0, tt, cf, sfc, cr, cw):
        N, T = p.shape
        a = [v.detach().float().contiguous() for v in (p, ta - dT[:, None], tt, cf, sfc, cr, cw, x0)]
        X = torch.empty(N, T, 2, device=p.device); q = torch.empty(N, T, device=p.device)
        _hbvsnow_fwd[(triton.cdiv(N, BLOCK),)](*a, X, q, N, T, BLOCK=BLOCK)
        ctx.save_for_backward(a[0], ta.detach().float().contiguous(), dT.detach().float(), a[-1], X, *a[2:7])
        return q, X

    @staticmethod
    def backward(ctx, gq, gX):
        p, ta, dT, x0, X, *prm = ctx.saved_tensors
        gi, gp = backward_k(_snow_step, X, x0, None, [gq], [p, ta], prm + [dT])
        return (None, None, gp[-1], None, *gp[:-1])


def snow_bands_np(P, Tc, Hdev, A, ps, x0=None):
    """float64 numpy: P, Tc [T, C]; Hdev, A [C, K]; ps dict of [C] (TT, CFMAX, SFCF, CFR, CWH, Gam) -> catchment liquid input [T, C],
    corrected input sum_k a_k (rain_k + SFCF snow_k) [T, C], final (SP, MW) [C, K]"""
    tt, cf, sfc, cr, cw, gam = (np.asarray(ps[k], np.float64)[:, None] for k in ("TT", "CFMAX", "SFCF", "CFR", "CWH", "Gam"))
    dT = gam * Hdev
    sp, mw = (np.zeros(A.shape), np.zeros(A.shape)) if x0 is None else (x0[0].copy(), x0[1].copy())
    out = np.zeros(P.shape); pin = np.zeros(P.shape)
    for t in range(P.shape[0]):
        p = P[t][:, None]; ta = Tc[t][:, None] - dT
        cold = ta < tt
        rain = np.where(cold, 0.0, p)
        sp = sp + np.where(cold, sfc * p, 0.0)
        melt = np.minimum(np.maximum(cf * (ta - tt), 0.0), sp); mw = mw + melt; sp = sp - melt
        refr = np.minimum(np.maximum(cr * cf * (tt - ta), 0.0), mw); sp = sp + refr; mw = mw - refr
        tos = np.maximum(mw - cw * sp, 0.0); mw = mw - tos
        out[t] = (A * (rain + tos)).sum(1); pin[t] = (A * np.where(cold, sfc * p, p)).sum(1)
    return out, pin, (sp, mw)


def build_bands(ds, K):
    """equal-area elevation bands as CaRT (cart.data)"""
    assert ds.band_H.shape[1] == K
    return ds.band_H, ds.band_A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gb", choices=["gb", "us", "aus"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--snoise", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--bands", type=int, default=5, help="number of equal-area elevation bands of the snow routine")
    ap.add_argument("--no_snow", action="store_true", help="HBV-9 without the snow routine")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ranges", default="feng2022", choices=sorted(hbv9.RANGES), help="parameter ranges of the HBV-9 part: feng2022 (paper; "
                    "FC 50-1000, LP 0.2-1, K2 0.001-0.2 as Feng et al. 2022) or hbv9 (FC 50-700, LP 0.3-1, K2 1e-4-0.1); K0 0.05-0.5, K1 0.01-0.3 in both")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    args.tag = f"{args.dataset}_hbv"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    out = Path(args.out or f"runs/{args.dataset}_hbv_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    ds = D.load(args.dataset, args.bands)
    data = {"ids": ds.ids, "basin_mask": ds.gauged, "ug_mask": ds.ungauged, "static": ds.static}
    days, P, E, q, Tm = ds.days, ds.P, ds.E, ds.q, ds.Tm
    PERIODS_ = US_PERIODS if args.dataset in D.US_LAYOUT else {k: ds.periods[k] for k in ("train", "val", "2011-15", "2015-19")}
    T, C = P.shape
    K = args.bands
    ids = [int(i) for i in data["ids"]]
    Tc = Tm - 273.15
    gauged, ug_mask = data["basin_mask"], data["ug_mask"]
    idx = {k: np.array([date.fromisoformat(a) <= d <= date.fromisoformat(b) for d in days]) for k, (a, b) in PERIODS_.items()}
    tr = idx["train"]
    Ttr = int(np.where(idx["val"])[0].max()) + 1
    xs = torch.from_numpy(data["static"].astype(np.float32)).to(dev)
    Pg = torch.from_numpy(P.T.astype(np.float32)).to(dev); Eg = torch.from_numpy(E.T.astype(np.float32)).to(dev)
    Tg = torch.from_numpy(np.ascontiguousarray(Tc.T).astype(np.float32)).to(dev)
    qg = torch.from_numpy(q.T.astype(np.float32)).to(dev)
    okq = torch.isfinite(qg); qz = torch.where(okq, qg, torch.zeros_like(qg))

    def masks(mask):
        ok = okq & mask[None]; okf = ok.float()
        mean = (qz * okf).sum(1) / ok.sum(1).clamp(min=1)
        return okf, ((qz - mean[:, None]) * okf).pow(2).sum(1).clamp(min=1e-6), ok.sum(1)
    okf_tr, den_tr, n_tr = masks(torch.from_numpy(tr).to(dev))
    okf_va, den_va, n_va = masks(torch.from_numpy(idx["val"]).to(dev))
    train_c = torch.from_numpy(gauged).to(dev) & (n_tr > 730)
    val_c = torch.from_numpy(gauged).to(dev) & (n_va > 365)
    snow = not args.no_snow
    if snow:
        H, A_ = build_bands(ds, K)
        Hdev_np = (H - (A_ * H).sum(1, keepdims=True)) / 1000.0                                # km above the area-weighted mean
        Hdev = torch.from_numpy(Hdev_np.astype(np.float32)).to(dev); Aband = torch.from_numpy(A_.astype(np.float32)).to(dev)
        lane = lambda v: v[:, None].expand(C, K).reshape(C * K)                                # catchment parameter -> lanes

    # linear search coordinates; head initialised at the per-catchment calibrated medians (GB) + CaRT snow init
    med = json.load(open(PKG / "configs" / "hbv9_init_medians.json"))
    lo, hi = (np.asarray(x, float) for x in hbv9.RANGES[args.ranges])
    u0 = np.clip((np.array([med[k] for k in hbv9.NAMES]) - lo) / (hi - lo), 0.02, 0.98)
    bias0 = list(np.log(u0 / (1 - u0)))
    if snow:
        lo = np.concatenate([lo, [v[0] for v in SNOW.values()]]); hi = np.concatenate([hi, [v[1] for v in SNOW.values()]])
        bias0 += [np.log((v[2] - v[0]) / (v[1] - v[2])) for v in SNOW.values()]
    PN = list(hbv9.NAMES) + (list(SNOW) if snow else [])
    lo_t, hi_t = torch.tensor(lo, dtype=torch.float32, device=dev), torch.tensor(hi, dtype=torch.float32, device=dev)
    senc = nn.Sequential(nn.Linear(xs.shape[1], 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU()).to(dev)
    shead = nn.Linear(128, len(PN)).to(dev)
    nn.init.zeros_(shead.weight)
    with torch.no_grad():
        shead.bias.copy_(torch.tensor(bias0, dtype=torch.float32))
    net = nn.ModuleList([senc, shead])

    def params(noise=0.0):
        x = xs + noise * torch.randn_like(xs) if noise > 0 else xs
        v = lo_t + (hi_t - lo_t) * torch.sigmoid(shead(senc(x)))
        return {k: v[:, i] for i, k in enumerate(PN)}

    def snow_in(p, TT, x0=None):
        """band snow routine -> catchment liquid input [C, TT], end state [C K, 2]"""
        dT = (torch.exp(p["lGam"])[:, None] * Hdev).reshape(C * K)                              # T_k = T - Gamma (h_k - h_mean)
        x0 = torch.zeros(C * K, 2, device=dev) if x0 is None else x0
        liq, X = SnowLanes.apply(Pg[:, None, :TT].expand(C, K, TT).reshape(C * K, TT), Tg[:, None, :TT].expand(C, K, TT).reshape(C * K, TT),
                                 dT, x0, lane(p["TT"]), lane(torch.exp(p["lCF"])), lane(p["SFCF"]), lane(p["CFR"]), lane(p["CWH"]))
        return (liq.view(C, K, TT) * Aband[:, :, None]).sum(1), X[:, -1]

    def simulate(p, TT):
        pk, ek = Pg[:, :TT].contiguous(), Eg[:, :TT].contiguous()
        with torch.no_grad():                                                                  # spin-up: 730 days, restart
            pd_ = {k: v.detach() for k, v in p.items()}
            pin, xs0 = snow_in(pd_, 730) if snow else (pk[:, :730], None)
            _, st, carry = hbv_in(pin.contiguous(), ek[:, :730], pd_)
        if snow:
            pin, _ = snow_in(p, TT, xs0)
            Q, _, _ = hbv_in(pin.contiguous(), ek, p, state=st, carry=carry)
        else:
            Q, _, _ = hbvstore.hbv(pk, ek, p, state=st, carry=carry)
        return Q

    def score(Q, okf, den, sel):
        TT = Q.shape[1]
        return (1 - (((Q - qz[:, :TT]) * okf[:, :TT]) ** 2).sum(1) / den)[sel]

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05)
    best, best_state, hist = -9.0, None, []
    nskip = 0
    t0 = time.time()
    print(f"{args.tag} seed {args.seed}: regional HBV ({args.dataset}), {len(PN)} parameters from {xs.shape[1]} attributes, snow "
          f"{'on (' + str(K) + ' bands)' if snow else 'off'}; {C} catchments, {int(train_c.sum())} training (gauged), {int(ug_mask.sum())} held-out; "
          f"train {PERIODS_['train']}, selection {PERIODS_['val']}", flush=True)
    for it in range(args.iters):
        ntr = score(simulate(params(args.snoise), Ttr), okf_tr, den_tr, train_c)
        loss = (1 - ntr).mean()
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if torch.isfinite(loss) and torch.isfinite(gnorm):
            opt.step()
        else:
            nskip += 1
            if nskip <= 5 or nskip % 50 == 0:
                print(f"  iter {it}: non-finite loss / gradient, step skipped ({nskip})", flush=True)
        sched.step()
        if it % args.eval_every == 0 or it == args.iters - 1:
            with torch.no_grad():
                v = score(simulate(params(), Ttr), okf_va, den_va, val_c).median().item()
            hist.append({"iter": it, "train_median": ntr.median().item(), "val_median": v})
            if v > best:
                best, best_state = v, {k: x.detach().clone() for k, x in net.state_dict().items()}
            if it % (4 * args.eval_every) == 0 or it == args.iters - 1:
                print(f"  iter {it:5d}  train median NSE {ntr.median().item():.4f}  val {v:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    net.load_state_dict(best_state)
    with torch.no_grad():
        pb = {k: v.double().cpu().numpy() for k, v in params().items()}
        Qgpu = simulate(params(), T).cpu().numpy().T
    pb_out = {k: pb[k] for k in hbv9.NAMES}
    if snow:
        pb_out.update({"TT": pb["TT"], "CFMAX": np.exp(pb["lCF"]), "SFCF": pb["SFCF"], "CFR": pb["CFR"], "CWH": pb["CWH"], "Gam": np.exp(pb["lGam"])})
    # float64 reference run (numpy band snow + hbv9.run_numpy, same spin-up incl. the routing buffer) and water balance
    if snow:
        Pin_s, _, xs0 = snow_bands_np(P[:730], Tc[:730], Hdev_np, A_, pb_out)
        Pin, Pcorr, (sp1, mw1) = snow_bands_np(P, Tc, Hdev_np, A_, pb_out, x0=xs0)
        dsnow = (A_ * (sp1 + mw1 - xs0[0] - xs0[1])).sum(1)
    else:
        Pin_s, Pin, Pcorr, dsnow = P[:730], P, P, 0.0
    _, st, _ = hbv9.run_numpy(Pin_s, E[:730], pb_out)
    sm0, suz0, slz0, buf0 = (np.array(x, copy=True) for x in st)
    Qn, (sm1, suz1, slz1, buf1), (AET, QGW) = hbv9.run_numpy(Pin, E, pb_out, state=st)
    resid = Pcorr.sum(0) - AET - QGW - (sm1 - sm0) - (suz1 - suz0) - (slz1 - slz0) - dsnow
    np.save(out / "sim.npy", Qn.astype(np.float32))                                             # [T, C], full record
    torch.save(best_state, out / "net.pt")
    pd.DataFrame(pb_out, index=ids).to_csv(out / "params.csv")
    json.dump({"args": vars(args), "dataset": args.dataset, "periods": PERIODS_, "n_train_basins": int(train_c.sum()),
               "n_ungauged": int(ug_mask.sum()), "npz": data_files(args.dataset)[0], "ungauged_list": data_files(args.dataset)[1],
               "best_val_median": best, "history": hist}, open(out / "run.json", "w"), indent=1)
    print(f"checks: float64 water balance (input incl. SFCF correction) max |residual|/P {np.abs(resid / Pcorr.sum(0)).max():.2e}; "
          f"GPU vs float64 max |dQ| {np.nanmax(np.abs(Qgpu - Qn)):.1e} mm/d; median SFCF-corrected input / P {np.median(Pcorr.sum(0) / P.sum(0)):.4f}", flush=True)
    print("   parameter medians (gauged): " + ", ".join(f"{k} {np.median(pb_out[k][gauged]):.3g}" for k in pb_out), flush=True)

    def nse_(sim, ob):
        k = np.isfinite(ob) & np.isfinite(sim)
        return 1 - np.sum((sim[k] - ob[k]) ** 2) / np.sum((ob[k] - ob[k].mean()) ** 2) if k.sum() > 365 else np.nan
    line = f"RESULT {args.tag}_s{args.seed} |"
    for grp, msk in (("gauged", gauged), ("ungauged", ug_mask)):
        if not msk.any():
            continue
        for pn in (("val",) if args.dataset in D.US_LAYOUT else ("val", "2011-15", "2015-19")):   # US, AUS: nothing after 2010 printed
            m = idx[pn]
            line += f" {grp} {pn} {np.nanmedian([nse_(Qn[m, c], q[m, c]) for c in np.where(msk)[0]]):.4f} |"
    print(line, flush=True)


if __name__ == "__main__":
    main()

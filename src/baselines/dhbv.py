"""dHBV (Feng et al. 2022, WRR 58, e2022WR032404): differentiable HBV with 16 parallel components whose parameters come from an LSTM,
re-implemented on our data and protocol for comparison with CaRT.

HBV (per component, daily; equations and order as hydroDL HBVMulTDET):
  snow     rain = P [T >= TT], snow = P [T < TT];  SP += snow;  melt = min(CFMAX max(T - TT, 0), SP);  MW += melt, SP -= melt;
           refr = min(CFR CFMAX max(TT - T, 0), MW);  SP += refr, MW -= refr;  tosoil = max(MW - CWH SP, 0);  MW -= tosoil
  soil     recharge = (rain + tosoil) min((SM/FC)^BETA, 1);  SM += rain + tosoil - recharge;  excess = max(SM - FC, 0);  SM -= excess
           ET = min(SM, PET min((SM / (LP FC))^BETAET, 1));  SM = max(SM - ET, 1e-5)
  response SUZ += recharge + excess;  perc = min(SUZ, PERC);  SUZ -= perc;  Q0 = K0 max(SUZ - UZL, 0);  SUZ -= Q0;  Q1 = K1 SUZ;  SUZ -= Q1
           SLZ += perc;  Q2 = K2 SLZ;  SLZ -= Q2;   Q = Q0 + Q1 + Q2
  mean over the 16 components, then a gamma unit hydrograph (shape a + 0.1, scale b + 0.5, 15 days), as hydroDL UH_gamma.
Parameters (13 per component x 16, + 2 routing): BETA [1, 6], FC [50, 1000], K0 [0.05, 0.9], K1 [0.01, 0.5], K2 [0.001, 0.2],
  LP [0.2, 1], PERC [0, 10], UZL [0, 100], TT [-2.5, 2.5], CFMAX [0.5, 10], CFR [0, 0.1], CWH [0, 0.2], BETAET [0.3, 5];
  routing a [0, 2.9], b [0, 6.5].  --dyn (default): BETA and BETAET vary daily (Feng et al.'s dynamic version); --static: none.
LSTM: input = standardised P, T, PET + the static attributes (as CaRT), hidden 256, sigmoid outputs scaled to the ranges.
  Static parameters: LSTM output on the last day of the training window (training) / on the last training day (evaluation: no
  test-period information; 2008-12-31); dynamic parameters: the output of each day.
Training: 100 random gauged catchments x 730-day windows inside 1989-2008; the first 365 days spin up the states without gradient and
  with the static parameters only (hydrodl2 HBV 1.0, warmup_states); the static parameters of the window are the LSTM output on its last
  day. Loss: --loss nse (our protocol: mean(1 - NSE) of the last 365 days with the catchment's training-period variance, Adam 1e-3) or
  --loss feng (Feng et al. 2022 eq. 1-2: 0.75 RMSE(Q) + 0.25 RMSE(log10(sqrt(Q) + 0.1)) over the batch, Adadelta lr 1.0, as hydroDL).
  Model selection on the 2009-10 median NSE of the gauged catchments (full-period simulation). Ungauged catchments never enter the loss.
Daily equations, bounds and gamma UH as hydrodl2 1.4.0 (hydrodl2/models/hbv/hbv.py, core/calc/uh_routing.py).
  Differences: in evaluation the static parameters are taken on 2008-12-31 (no information from the test period); BETAET exists only
  in the dynamic version (as hydrodl2).
Datasets: gb (CAMELS-GB), us (CAMELS-US) and aus (CAMELS-AUS v2, the US configuration), read with cart.data.
Usage (from src/): python -m baselines.dhbv --dataset gb|us|aus --seed 0 [--iters 6000] [--gpu 0] [--out DIR]
  -> sim.npy [T, N] (mm/d, full record), net.pt, run.json; paper: --loss nse (default), dynamic BETA/BETAET (default), 6000 iterations
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cart import data as D
from . import data_files
NAMES = ["BETA", "FC", "K0", "K1", "K2", "LP", "PERC", "UZL", "TT", "CFMAX", "CFR", "CWH", "BETAET"]
LO = torch.tensor([1.0, 50.0, 0.05, 0.01, 0.001, 0.2, 0.0, 0.0, -2.5, 0.5, 0.0, 0.0, 0.3])
HI = torch.tensor([6.0, 1000.0, 0.9, 0.5, 0.2, 1.0, 10.0, 100.0, 2.5, 10.0, 0.1, 0.2, 5.0])
DYN = ["BETA", "BETAET"]
NMUL, LENF = 16, 15


def uh_gamma(a, b):
    """a, b [N] -> unit hydrograph weights [N, LENF] (hydroDL UH_gamma)"""
    aa = torch.relu(a)[:, None] + 0.1
    th = torch.relu(b)[:, None] + 0.5
    t = torch.arange(0.5, LENF, device=a.device)[None]
    w = torch.exp(-torch.lgamma(aa) - aa * torch.log(th) + (aa - 1) * torch.log(t) - t / th)
    return w / w.sum(1, keepdim=True)


def hbv(P, T, E, ps, pdyn, states=None):
    """P, T, E [N, L]; ps: dict of static parameters [N, NMUL]; pdyn: dict of dynamic parameters [N, L, NMUL] (or None).
    -> mean component runoff [N, L], final states"""
    N, L = P.shape
    dev = P.device
    if states is None:
        z = torch.full((N, NMUL), 0.001, device=dev)
        SP, MW, SM, SUZ, SLZ = z.clone(), z.clone(), z.clone(), z.clone(), z.clone()
    else:
        SP, MW, SM, SUZ, SLZ = states
    out = []
    for t in range(L):
        p, tt, e = P[:, t:t + 1], T[:, t:t + 1], E[:, t:t + 1]
        beta = pdyn["BETA"][:, t] if (pdyn is not None and "BETA" in pdyn) else ps["BETA"]
        betaet = pdyn["BETAET"][:, t] if (pdyn is not None and "BETAET" in pdyn) else None
        rain = p * (tt >= ps["TT"]).float()
        snow = p * (tt < ps["TT"]).float()
        SP = SP + snow
        melt = torch.minimum(torch.relu(ps["CFMAX"] * (tt - ps["TT"])), SP)
        MW = MW + melt; SP = SP - melt
        refr = torch.minimum(torch.relu(ps["CFR"] * ps["CFMAX"] * (ps["TT"] - tt)), MW)
        SP = SP + refr; MW = MW - refr
        tosoil = torch.relu(MW - ps["CWH"] * SP); MW = MW - tosoil
        wet = torch.clamp((SM / ps["FC"]).clamp(min=1e-6) ** beta, 0.0, 1.0)
        rech = (rain + tosoil) * wet
        SM = SM + rain + tosoil - rech
        excess = torch.relu(SM - ps["FC"]); SM = SM - excess
        ef = SM / (ps["LP"] * ps["FC"])
        if betaet is not None:                                                              # BETAET acts only when dynamic (hydrodl2)
            ef = ef.clamp(min=1e-6) ** betaet
        ef = torch.clamp(ef, 0.0, 1.0)
        et = torch.minimum(SM, e * ef); SM = torch.clamp(SM - et, min=1e-5)
        SUZ = SUZ + rech + excess
        perc = torch.minimum(SUZ, ps["PERC"]); SUZ = SUZ - perc
        q0 = ps["K0"] * torch.relu(SUZ - ps["UZL"]); SUZ = SUZ - q0
        q1 = ps["K1"] * SUZ; SUZ = SUZ - q1
        SLZ = SLZ + perc; q2 = ps["K2"] * SLZ; SLZ = SLZ - q2
        out.append((q0 + q1 + q2).mean(1))
    return torch.stack(out, 1), (SP, MW, SM, SUZ, SLZ)


def hbv_fast(P, T, E, ps, pdyn, states=None):
    """same as hbv() through the fused Triton kernel (hbv_kernel.py, adjoint backward): lanes = catchment x component"""
    from .kernels.hbv_kernel import STAT, hbv_fused
    N, L = P.shape
    K = NMUL
    lanes = lambda x: x[:, None, :].expand(L, N, K).reshape(L, N * K) if x.dim() == 2 and x.shape == (L, N) else x
    tm = lambda x: x.t().contiguous()[:, :, None].expand(L, N, K).reshape(L, N * K)                 # forcing [N, L] -> [L, N K]
    stat = torch.stack([ps[k].reshape(N * K) for k in STAT])
    beta = (pdyn["BETA"].permute(1, 0, 2).reshape(L, N * K) if (pdyn is not None and "BETA" in pdyn)
            else ps["BETA"].reshape(1, N * K).expand(L, N * K))
    betaet = pdyn["BETAET"].permute(1, 0, 2).reshape(L, N * K) if (pdyn is not None and "BETAET" in pdyn) else None
    if states is None:
        s0 = torch.full((5, N * K), 0.001, device=P.device)
    else:
        s0 = torch.stack([x.reshape(N * K) for x in states])
    q, send = hbv_fused(tm(P), tm(T), tm(E), stat, beta, betaet, s0.detach())
    return q.view(L, N, K).mean(-1).t(), tuple(send[i].view(N, K) for i in range(5))


def route(Q, a, b):
    """Q [N, L] -> routed [N, L] with the gamma UH (causal convolution)"""
    w = uh_gamma(a, b)                                                                     # [N, LENF]
    Qp = torch.nn.functional.pad(Q, (LENF - 1, 0))
    return torch.nn.functional.conv1d(Qp[None], w.flip(1)[:, None], groups=Q.shape[0])[0]


class Net(nn.Module):
    def __init__(self, n_in, dyn, hid=256):
        super().__init__()
        self.lin_in = nn.Linear(n_in, hid)                                                  # hydroDL CudnnLstmModel: linear + ReLU + LSTM
        self.lstm = nn.LSTM(hid, hid, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.dyn = dyn
        self.head_s = nn.Linear(hid, len(NAMES) * NMUL + 2)
        self.head_d = nn.Linear(hid, len(DYN) * NMUL) if dyn else None

    def forward(self, x):
        h, _ = self.lstm(torch.relu(self.lin_in(x)))                                        # [N, L, hid]
        return self.drop(h)


def params_from(net, h_static, h_seq, dev):
    """static parameters from one hidden state [N, hid], dynamic from the sequence [N, L, hid]"""
    lo, hi = LO.to(dev), HI.to(dev)
    u = torch.sigmoid(net.head_s(h_static))
    us = u[:, :len(NAMES) * NMUL].view(-1, len(NAMES), NMUL)
    ps = {k: lo[i] + (hi[i] - lo[i]) * us[:, i] for i, k in enumerate(NAMES)}
    ra, rb = u[:, -2] * 2.9, u[:, -1] * 6.5
    pdyn = None
    if net.dyn:
        ud = torch.sigmoid(net.head_d(h_seq)).view(h_seq.shape[0], h_seq.shape[1], len(DYN), NMUL)
        pdyn = {k: lo[NAMES.index(k)] + (hi[NAMES.index(k)] - lo[NAMES.index(k)]) * ud[:, :, j] for j, k in enumerate(DYN)}
    return ps, pdyn, ra, rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", action="store_true", help="no dynamic parameters")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--loss", default="nse", choices=["nse", "feng"])
    ap.add_argument("--dataset", default="gb", choices=["gb", "us", "aus"], help="gb: CAMELS-GB; us: CAMELS-US 671 basins, unified held-out "
                    "protocol: train 1989-2008, selection 2009-2010, static LSTM parameters from 2008-12-31; "
                    "aus: CAMELS-AUS v2 561 catchments, as us")
    ap.add_argument("--out", default="")
    ap.add_argument("--slow", action="store_true", help="PyTorch daily loop instead of the fused kernel (reference)")
    args = ap.parse_args()
    args.tag = f"{args.dataset}_dhbv"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    run_hbv = hbv if args.slow else hbv_fast
    dev = torch.device(f"cuda:{args.gpu}")
    out = Path(args.out or f"runs/{args.dataset}_dhbv_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    ds = D.load(args.dataset)
    data = {"ids": ds.ids, "basin_mask": ds.gauged, "ug_mask": ds.ungauged, "static": ds.static}
    days, P, E, q = ds.days, ds.P, ds.E, ds.q
    PERIODS_ = {k: ds.periods[k] for k in (("train", "val", "test") if args.dataset in D.US_LAYOUT else ("train", "val", "2011-15", "2015-19"))}
    T_, C = P.shape
    G, U = data["basin_mask"], data["ug_mask"]
    Tm = ds.Tm - 273.15
    idx = {k: np.array([date.fromisoformat(a) <= d <= date.fromisoformat(b) for d in days]) for k, (a, b) in PERIODS_.items()}
    tr = idx["train"]
    t_tr = np.where(tr)[0]
    t_end_tr = int(t_tr[-1])
    z = lambda x: ((x - x[tr][:, G].mean()) / (x[tr][:, G].std() + 1e-6)).astype(np.float32)
    X = np.concatenate([np.stack([z(P), z(Tm), z(E)], -1).transpose(1, 0, 2),
                        np.repeat(data["static"].astype(np.float32)[:, None], T_, 1)], -1)                      # [C, T, 24]
    Xg = torch.from_numpy(X).to(dev)
    Pg, Tg, Eg = (torch.from_numpy(np.ascontiguousarray(a.T).astype(np.float32)).to(dev) for a in (P, Tm, E))
    qg = torch.from_numpy(np.ascontiguousarray(q.T).astype(np.float32)).to(dev)
    okq = torch.isfinite(qg)
    var_tr = torch.from_numpy(np.nanvar(np.where(tr[:, None], q, np.nan), 0).astype(np.float32)).to(dev)
    train_c = np.where(G & (np.isfinite(np.where(tr[:, None], q, np.nan)).sum(0) > 730))[0]
    net = Net(X.shape[-1], not args.static).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr) if args.loss == "nse" else torch.optim.Adadelta(net.parameters(), lr=1.0)
    rng = np.random.default_rng(args.seed)
    W, WU = 730, 365

    def simulate_full(sel):
        """full-record simulation for catchments sel (no gradient): static parameters from 2008-12-31"""
        with torch.no_grad():
            outq = []
            for i0 in range(0, len(sel), 48):                                                  # chunks: LSTM over 11 000 days
                cs = torch.as_tensor(sel[i0:i0 + 48], device=dev)
                h = net(Xg[cs])
                ps, pdyn, ra, rb = params_from(net, h[:, t_end_tr], h, dev)
                Qm, _ = run_hbv(Pg[cs], Tg[cs], Eg[cs], ps, pdyn)
                outq.append(route(Qm, ra, rb))
            return torch.cat(outq, 0)

    def med_nse(Q, sel, mask):
        o = qg[sel][:, mask]; s_ = Q[:, mask]; k = torch.isfinite(o)
        oz = torch.where(k, o, torch.zeros_like(o)); n = k.sum(1).clamp(min=1)
        mu = oz.sum(1) / n
        num = (((s_ - oz) * k) ** 2).sum(1); den = (((oz - mu[:, None]) * k) ** 2).sum(1).clamp(min=1e-6)
        v = 1 - num / den
        return v[n > 365].median().item()

    val_m = torch.from_numpy(idx["val"]).to(dev)
    gsel = np.where(G)[0]
    best, best_state, hist = -9.0, None, []
    t0 = time.time()
    print(f"{args.tag} seed {args.seed}: dHBV, {NMUL} components, dynamic {not args.static} ({DYN if not args.static else []}); "
          f"{len(train_c)} training catchments", flush=True)
    for it in range(args.iters):
        net.train()
        cs = torch.as_tensor(rng.choice(train_c, args.batch, replace=False), device=dev)
        st = int(rng.integers(t_tr[0], t_end_tr - W + 2))
        with torch.no_grad():                                                               # spin-up: static parameters, no gradient
            hw = net(Xg[cs, st:st + WU])
            psw, _, _, _ = params_from(net, hw[:, -1], hw, dev)
            _, states = run_hbv(Pg[cs, st:st + WU], Tg[cs, st:st + WU], Eg[cs, st:st + WU], psw, None)
        h = net(Xg[cs, st:st + W])[:, WU:]
        ps, pdyn, ra, rb = params_from(net, h[:, -1], h, dev)
        Qm, _ = run_hbv(Pg[cs, st + WU:st + W], Tg[cs, st + WU:st + W], Eg[cs, st + WU:st + W], ps, pdyn, states=states)
        Qr = route(Qm, ra, rb)
        o = qg[cs, st + WU:st + W]; k = okq[cs, st + WU:st + W]
        err = torch.where(k, Qr - o, torch.zeros_like(Qr))
        if args.loss == "nse":
            nse_w = 1 - (err ** 2).sum(1) / (k.sum(1).clamp(min=1) * var_tr[cs]).clamp(min=1e-6)
            loss = (1 - nse_w)[k.sum(1) > 100].mean()
        else:                                                                               # Feng et al. 2022, eq. 1-2 (alpha 0.25)
            tq = lambda x: torch.log10(torch.sqrt(x.clamp(min=0)) + 0.1)
            el = torch.where(k, tq(Qr) - tq(torch.where(k, o, torch.zeros_like(o))), torch.zeros_like(Qr))
            nk = k.sum().clamp(min=1)
            loss = 0.75 * torch.sqrt((err ** 2).sum() / nk) + 0.25 * torch.sqrt((el ** 2).sum() / nk)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if (it + 1) % args.eval_every == 0 or it == args.iters - 1:
            net.eval()
            Qv = simulate_full(gsel)
            v = med_nse(Qv, torch.as_tensor(gsel, device=dev), val_m)
            hist.append({"iter": it, "loss": loss.item(), "val_median": v})
            if v > best:
                best, best_state = v, {k_: x.detach().clone() for k_, x in net.state_dict().items()}
            print(f"  iter {it + 1:5d}  window loss {loss.item():.4f}  val median NSE {v:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    net.load_state_dict(best_state); net.eval()
    Q = simulate_full(np.arange(C)).cpu().numpy().T
    np.save(out / "sim.npy", Q.astype(np.float32))
    torch.save(best_state, out / "net.pt")
    json.dump({"args": vars(args), "periods": PERIODS_, "n_train_basins": int(len(train_c)), "n_ungauged": int(U.sum()),
               "npz": data_files(args.dataset)[0], "ungauged_list": data_files(args.dataset)[1],
               "static_param_date": str(days[t_end_tr]), "best_val_median": best, "history": hist}, open(out / "run.json", "w"), indent=1)

    def nse(sim, o):
        k = np.isfinite(o) & np.isfinite(sim)
        return 1 - np.sum((sim[k] - o[k]) ** 2) / np.sum((o[k] - o[k].mean()) ** 2) if k.sum() > 365 else np.nan
    line = f"RESULT {args.tag}_s{args.seed} |"
    near = args.dataset in D.US_LAYOUT
    for grp, msk in (("gauged", G), ("ungauged", U)):
        if not msk.any():
            continue
        for pn in (("val",) if near else ("val", "2011-15", "2015-19")):   # near: nothing after 2010
            m = idx[pn]
            line += f" {grp} {pn} {np.nanmedian([nse(Q[m, c], q[m, c]) for c in np.where(msk)[0]]):.4f} |"
    print(line, flush=True)


if __name__ == "__main__":
    main()

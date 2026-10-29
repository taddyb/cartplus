"""Full-system water balance of CaRT+ (paper, Appendix E): accumulated over the whole simulation, for every catchment,

    sum P + sum (SFCF - 1) P_snow + sum J  =  sum Q + sum (sublimation + e_u + e_s + e_z + e_d + e_W) + exchange loss + channel loss
                                              + change of storage (snow bands, surface layer, root zone, near-stream zone, depressions,
                                                water in transit in the unit hydrographs, quick store, groundwater)

(Supporting Information S1.8). The model is run once over the whole record for all catchments at once (float32, Triton kernels, as in
training); every term is taken from the forward pass (CaRT.simulate(keep=...)) and booked in float64 on the CPU. Also reported: the
residual of each stage, the residual of the stores-only balance that training prints and, for a trained run, the simulated Q against
the run's sim.npy.

    python tests/check_balance.py [--run runs/us_cartplus_s0] [--dataset us] [--seed 0] [--K N] [--gpu 0] [--json OUT]

--run: a run directory written by cart.train (net.pt, run.json; the dataset is taken from run.json). Without --run the network is
freshly initialised with --seed (the water balance holds for any parameter values). --K: only the first K catchments.
Pass: max |residual| / accumulated P < 1e-5 (float32).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))
from cart import data as D  # noqa: E402
from cart.config import config_for  # noqa: E402
from cart.model import CaRT  # noqa: E402
from cart.predict import load_model, run_config  # noqa: E402


def book(L, ps, model, T):
    g = lambda x_: (L[x_] if isinstance(x_, str) else x_).detach().cpu().double()
    P = g("Pg")[:, :T]
    wz = g(ps["wz"])
    # 1 snow: snowfall correction booked from the kernel's own rain/snow switch (same dtype and operations as the kernel)
    dT_ = (-torch.exp(ps["lGam"])[:, None] * model.Hdev).detach().cpu()
    cold = (L["Tcl"][:, :T].detach().cpu()[:, :, None] + dT_[:, None, :]) < ps["TTs"].detach().cpu()[:, None, None]
    Ab = model.Aband.detach().cpu().double()
    snow_raw = (Ab * (cold.double() * P[:, :, None]).sum(1)).sum(1)                         # area-weighted raw snowfall
    ADJ = (g(ps["SFCF"]) - 1.0) * snow_raw
    SUB, PIN = g("subl").sum(1), g("Pin").sum(1)
    dSN = (Ab * g("Xsn")[:, -1, :]).sum(1)                                                  # bands start empty
    # 2 surface layer U (overflow -> soil: du_ already includes it)
    EU, DU = g("eu_").sum(1), g("du_").sum(1)
    dU = g("User")[:, -1] - g("Usp")[:, -1]
    # 3 infiltration excess
    IE, PK = g("ie").sum(1), g("pk").sum(1)
    # 4 production store (per unit area of its share 1 - w)
    Dp, Gp, Ap = (g(k_).sum(1) for k_ in ("Dp", "Gp", "Ap"))
    dSk = g("Sk")[:, -1] - g("s0")
    # 5 near-stream zone (per unit area of its share w)
    DZ, QZz, AZ = g("dzz").sum(1), g("qzz").sum(1), g("Az").sum(1)
    dZ = g("Zser")[:, -1] - g("z0_")
    # 6 depressions
    Din, Dd, Gd, Ad = g("Din_dep").sum(1), g("Dd").sum(1), g("Gd").sum(1), g("Ad").sum(1)
    dV = g("Sdser")[:, -1] - g("s0d")
    # 7 unit hydrographs: water still in transit at the end, from the ordinates
    Duh, Dby, x9, x1, o1, o2 = g("Duh"), g("Dby"), g("x9"), g("x1"), g("o1"), g("o2")

    def tail(x_, o_):
        Lk = o_.shape[1]; cum = torch.cumsum(o_, 1)
        return (x_[:, T - Lk:].flip(1) * (1.0 - cum)).sum(1)
    Hend = tail(Duh - Dby, o1) + tail(Dby, o2)
    # 8 quick store with exterior inflow J (0.9 / 0.1 into x9e / x1e) and dissipative exchange xr
    cX = g("cX"); J = cX * T
    XIN = g("x9e").sum(1) + g("x1e").sum(1)
    XR, QG = g("xr").sum(1), g("Qg").sum(1)
    dR = g("Rser")[:, -1] - g("r0")
    # 9 groundwater with uptake
    GW, QS, EW = g("G").sum(1), g("Qs").sum(1), g("ewg").sum(1)
    dW = g("Wser")[:, -1] - g("W0")
    # 10 outlet and channel loss
    QPRE, CL, QF, QZw = g("Qpre_cl").sum(1), g("closs").sum(1), g("Q").sum(1), g("QZ").sum(1)
    Pr = P.sum(1)
    inp = Pr + ADJ + J
    outs = SUB + EU + (1 - wz) * Ap + wz * AZ + Ad + EW - XR + CL + QF
    dst = dSN + dU + (1 - wz) * dSk + wz * dZ + dV + Hend + dR + dW
    res = inp - outs - dst
    stage = dict(snow=Pr + ADJ - PIN - SUB - dSN, surface=PIN - EU - DU - dU, ie_split=DU - IE - PK,
                 rootzone=PK - Dp - Gp - Ap - dSk, nearstream=PK - DZ - QZz - AZ - dZ,
                 d_into_depr=Din - ((1 - wz) * Dp + wz * DZ + IE), depressions=Din - Dd - Gd - Ad - dV,
                 uh=Duh.sum(1) - x9.sum(1) - x1.sum(1) - Hend, quick_inflow=XIN - x9.sum(1) - x1.sum(1) - J,
                 quick=XIN + XR - QG - dR, g_into_gw=GW - ((1 - wz) * Gp + Gd), gw=GW - QS - EW - dW,
                 outlet=QPRE - QG - QS - QZw, channel=QPRE - CL - QF)
    shares = dict(SFCF_adj=ADJ, J_inflow=J, sublimation=SUB, e_u_surface=EU, e_s_rootzone=(1 - wz) * Ap, e_z_nearstream=wz * AZ,
                  e_d_depressions=Ad, e_W_gw_uptake=EW, exch_loss=-XR, channel_loss=CL, Q=QF)
    return res, Pr, stage, shares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="us", choices=["gb", "us", "aus"])
    ap.add_argument("--model", default="cartplus", choices=["cartplus"])
    ap.add_argument("--run", default="", help="run directory written by cart.train (net.pt, run.json); default: untrained network")
    ap.add_argument("--seed", type=int, default=0, help="initialisation seed without --run")
    ap.add_argument("--K", type=int, default=0, help="only the first K catchments (0 = all)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(dev)
    if a.run:
        cfg, dsn = run_config(a.run)
        assert cfg.name == "cartplus", f"the balance bookkeeping covers CaRT+ (run has {cfg.name})"
    else:
        cfg, dsn = config_for(a.model, a.dataset), a.dataset
    ds = D.load(dsn)
    if a.K:
        ds = ds.subset(a.K)
    if a.run:
        model = load_model(a.run, ds, cfg, dev)
    else:
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        model = CaRT(ds, cfg, dev)
    keep = {}
    torch.cuda.reset_peak_memory_stats(dev)
    with torch.no_grad():
        Q, ps, bal = model.simulate(TT=model.T, keep=keep)
    peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    T = model.T
    Qn = Q.cpu().numpy().T
    res, Pr, stage, shares = book(keep, ps, model, T)
    rel = (res / Pr).numpy()
    b = {k: v.detach().cpu().double() for k, v in bal.items()}
    res_st = (b["P"] + b["X"] - b["A"] - b["Q"] - b["dS"] - b["dR"] - b["dW"] - b["transit"]) / b["P"]
    out = {"what": "balance", "run": str(Path(a.run).name) if a.run else f"untrained seed {a.seed}", "dataset": dsn,
           "precision": "float32 (Triton kernels), booked in float64", "n_catchments": int(len(rel)), "days": int(T),
           "max_abs_resid_over_P": float(np.abs(rel).max()), "median_abs_resid_over_P": float(np.median(np.abs(rel))),
           "stores_only_max_abs_resid_over_P": float(res_st.abs().max()),
           "stage_max_abs_over_P": {k: float((v / Pr).abs().max()) for k, v in stage.items()},
           "median_share_of_P": {k: float(np.median((v / Pr).numpy())) for k, v in shares.items()},
           "peak_gpu_GB": round(peak, 2)}
    sim = Path(a.run) / "sim.npy" if a.run else None
    if sim is not None and sim.exists() and not a.K:
        saved = np.load(sim)
        out["Q_vs_sim_npy_max_abs"] = float(np.nanmax(np.abs(Qn - saved)))
    print(f"{out['run']} ({dsn}): full-system residual / accumulated P: max {out['max_abs_resid_over_P']:.2e}, median "
          f"{out['median_abs_resid_over_P']:.2e} over {out['n_catchments']} catchments ({T} days); stores-only balance max "
          f"{out['stores_only_max_abs_resid_over_P']:.2e}; peak GPU {peak:.2f} GB"
          + (f"; Q vs sim.npy max |dQ| {out['Q_vs_sim_npy_max_abs']:.1e}" if "Q_vs_sim_npy_max_abs" in out else ""))
    print("  largest stage residual / P:", {k: f"{v:.1e}" for k, v in sorted(out["stage_max_abs_over_P"].items(), key=lambda kv: -kv[1])[:4]})
    ok = out["max_abs_resid_over_P"] < 1e-5
    print(f"BALANCE CHECK {'PASSED' if ok else 'FAILED'} (tolerance 1e-5, float32)")
    print("SUMMARY", json.dumps(out))
    if a.json:
        open(a.json, "a").write(json.dumps(out) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

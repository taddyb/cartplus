"""Unified-protocol figures for Section 5 (held-out 2011-2019): cdf_unified.pdf (Figure fig:cdf: CDFs of NSE, NSE_log, KGE of the single
models, per-catchment mean over the three seeds, GB top / US bottom) and hydro_gb.pdf, hydro_us.pdf (Figures fig:hydro_gb, fig:hydro_us:
one water year, observed and seed-0 CaRT+, CaRT, LSTM in six gauged catchments selected by rule: class medians of CaRT+ and the largest
advantages of CaRT+ over the LSTM and of the LSTM over CaRT+). Also imported by fig_maps.py (load_ds, metrics; load_ds also reads
CAMELS-AUS for Figure fig:map_aus).
usage: python scripts/eval/fig_unified.py [cdf|hydro|all] [--runs DIR] [--out DIR]
(CAMELS-US gauge names for the titles from $CAMELS_US_NAMES = CAMELS-US camels_name.txt)"""
import os, sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalcommon as EC
_ap = EC.parser(__doc__); _ap.add_argument("what", nargs="?", default="all"); A_ = _ap.parse_known_args()[0]    # (fig_maps.py passes its own positional)
PP = str(EC.outdirs(A_)) + "/"
KEY = {"HBV": "HBV", "dHBV": "dHBV", "CaRT": "CaRT", "CaRT+": "CaRTp", "LSTM": "LSTM", "EA-LSTM": "EALSTM"}        # figure label -> evalcommon model key
COL = {"HBV": "#eda100", "dHBV": "#e87ba4", "CaRT": "#eb6834", "CaRT+": "#2a78d6", "LSTM": "#1baf7a", "EA-LSTM": "#7a5cc4", "obs": "#1f1f1e"}
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5})


def load_ds(ds):
    OB = EC.load_obs(ds)
    dd, q, P, ids, mask, ug, names = OB["dd"], OB["q"], OB["P"], OB["ids"], OB["mask"], OB["ung"], OB["names"]
    if ds == "aus":                                                                                       # CAMELS-AUS v2, ids = running npz ids "9xxxxxxx"
        ai = OB["aridity"]
        cls = {"humid (PET/P $\\leq1$)": ai <= 1, "intermediate ($1<$ PET/P $\\leq2$)": (ai > 1) & (ai <= 2), "arid (PET/P $>2$)": ai > 2}
        wy = ("2013-07-01", "2014-06-30")
    elif ds == "us":
        sn, ai = OB["snow"], OB["aridity"]
        cls = {"snow-dominated ($f_{\\rm snow}\\geq0.3$)": sn >= 0.3, "arid (PET/P $>1.1$)": (ai > 1.1) & (sn < 0.3), "humid (PET/P $\\leq0.7$)": (ai <= 0.7) & (sn < 0.3),
               "intermediate": (ai > 0.7) & (ai <= 1.1) & (sn < 0.3)}
        wy = ("2013-10-01", "2014-09-30")
    else:
        bfi, sn = OB["bfi"], OB["snow"]
        cls = {"flashy (BFI $<0.5$)": (bfi < 0.5) & (sn < 0.04), "intermediate ($0.5\\leq$ BFI $<0.65$)": (bfi >= 0.5) & (bfi < 0.65) & (sn < 0.04),
               "groundwater-fed (BFI $\\geq0.65$)": bfi >= 0.65, "snow-affected ($f_{\\rm snow}\\geq0.04$)": sn >= 0.04}
        wy = ("2013-10-01", "2014-09-30")
    per = np.asarray((dd >= "2011-01-01") & (dd <= "2019-12-31"))
    return dict(dd=dd, q=q, P=P, ids=ids, mask=mask, ug=ug, names=names, cls=cls, wy=wy, per=per)


def metrics(sim, q, per):
    out = np.full((q.shape[1], 3), np.nan)
    for c in range(q.shape[1]):
        o = q[per, c]; s = np.maximum(sim[per, c], 0); k = np.isfinite(o) & np.isfinite(s)
        if k.sum() < 365: continue
        s, o = s[k], o[k]; e = 0.01 * o.mean(); r = np.corrcoef(s, o)[0, 1]
        out[c] = [1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2), 1 - np.sum((np.log(s + e) - np.log(o + e)) ** 2) / np.sum((np.log(o + e) - np.log(o + e).mean()) ** 2),
                  1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)]
    return out


def sims(ds, m, seeds=(0, 1, 2)):
    L = [EC.sim_path(A_, ds, KEY[m], k) for k in seeds]; L = [f for f in L if os.path.exists(f)]
    return [np.load(f).astype(float) for f in L]


def cdf():
    f, ax = plt.subplots(2, 3, figsize=(11, 6.4), constrained_layout=True)
    for row, ds in enumerate(("gb", "us")):
        try: D = load_ds(ds)
        except FileNotFoundError as e: print(ds, "skipped:", e); continue
        for m in ("HBV", "dHBV", "CaRT", "CaRT+", "LSTM", "EA-LSTM"):
            S = sims(ds, m)
            if not S: continue
            Mt = np.mean([metrics(x, D["q"], D["per"]) for x in S], 0)[D["mask"]]          # single model: per-catchment mean of the seed metrics
            for j, (nm, lo) in enumerate((("NSE", 0.0), ("NSE of log flow", 0.0), ("KGE", 0.0))):
                v = np.sort(Mt[np.isfinite(Mt[:, j]), j]); ax[row, j].plot(v, np.linspace(0, 1, len(v)), color=COL[m], lw=1.4 if m in ("CaRT+", "LSTM") else 1.0, label=f"{m} ({np.median(v):.3f})")
                ax[row, j].set_xlim(lo, 1); ax[row, j].set_ylim(0, 1); ax[row, j].set_xlabel(nm); ax[row, j].set_ylabel("fraction of catchments" if j == 0 else "")
                ax[row, j].legend(frameon=False, fontsize=7.5, loc="upper left")
        ax[row, 0].set_title("CAMELS-GB, gauged" if ds == "gb" else "CAMELS-US, gauged", loc="left", fontsize=9.5)
    f.savefig(PP + "figs/cdf_unified.pdf"); f.savefig(PP + "figs/cdf_unified.png", dpi=130); plt.close(f)


def hydro(ds):
    try: D = load_ds(ds)
    except FileNotFoundError as e: print(ds, "skipped:", e); return
    q, dd, per, mask = D["q"], D["dd"], D["per"], D["mask"]
    S = {m: sims(ds, m, (0,)) for m in ("CaRT+", "CaRT", "LSTM")}
    if any(len(v) == 0 for v in S.values()): print(ds, "hydro: missing seed-0 sims", {m: len(v) for m, v in S.items()}); return
    S = {m: np.maximum(v[0], 0) for m, v in S.items()}; N = {m: metrics(s, q, per)[:, 0] for m, s in S.items()}
    pick = []
    for lab, cm in D["cls"].items():
        c = np.where(cm & mask & np.isfinite(N["CaRT+"]))[0]; pick.append((lab + ", CaRT$^+$ at class median", int(c[np.argmin(np.abs(N["CaRT+"][c] - np.nanmedian(N["CaRT+"][c])))])))
    d = N["CaRT+"] - N["LSTM"]; ok = mask & np.isfinite(d) & (np.maximum(N["CaRT+"], N["LSTM"]) > 0.5)
    pick.append(("largest advantage of CaRT$^+$ over the LSTM", int(np.where(ok)[0][np.argmax(d[ok])]))); pick.append(("largest advantage of the LSTM over CaRT$^+$", int(np.where(ok)[0][np.argmin(d[ok])])))
    w = np.asarray((dd >= D["wy"][0]) & (dd <= D["wy"][1]))
    f, ax = plt.subplots(3, 2, figsize=(11, 9.2), constrained_layout=True)
    for a, (lab, c) in zip(ax.ravel(), pick):
        x = dd[w]; a.plot(x, q[w, c], color=COL["obs"], lw=1.5, label="observed")
        for m, ls in (("CaRT", "--"), ("LSTM", ":"), ("CaRT+", "-")): a.plot(x, S[m][w, c], color=COL[m], lw=1.0 if m != "CaRT+" else 1.1, ls=ls, label=m)
        a.set_title(f"{D['ids'][c]} {D['names'][c]}\n{lab}", fontsize=8.5, loc="left"); a.set_ylabel("Q (mm d$^{-1}$)")
        a.text(0.99, 0.97, "NSE  CaRT$^+$ %.2f · CaRT %.2f · LSTM %.2f" % (N["CaRT+"][c], N["CaRT"][c], N["LSTM"][c]), transform=a.transAxes, ha="right", va="top", fontsize=7.5, color="#4a4a47", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    h, l = ax[0, 0].get_legend_handles_labels(); f.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    f.savefig(PP + f"figs/hydro_{ds}.pdf", bbox_inches="tight"); f.savefig(PP + f"figs/hydro_{ds}.png", dpi=130, bbox_inches="tight"); plt.close(f)
    print(ds, [(D["ids"][c], lab, round(N["CaRT+"][c], 3), round(N["LSTM"][c], 3)) for lab, c in pick])


if __name__ == "__main__":
    what = A_.what
    assert what in ("cdf", "hydro", "all"), what
    if what in ("cdf", "all"): cdf()
    if what in ("hydro", "all"):
        for ds in ("gb", "us"): hydro(ds)

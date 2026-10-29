"""Held-out (ungauged) basins of CAMELS-US, test period 2011-2019 (Section 5.5, Figure fig:ung_us, Table tab:ung_us): per-basin NSE of
CaRT+ against the stateful LSTM and the EA-LSTM (mean over three seeds, bars = seed range), by class; a class table with paired
differences; and the diagnosis of the failing basins (runoff ratio against the gauged basins nearest in attribute space; NSE
decomposition into correlation, amplitude and bias). Writes <out>/figs/ung_us.pdf/png, <out>/tables/u_ung_us.tex, <out>/data/u_ung_us.json.
Gauge names from $CAMELS_US_NAMES (CAMELS-US camels_name.txt; required).
usage: python scripts/eval/fig_ungauged_us.py [--runs DIR] [--out DIR]
"""
import os, sys, json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalcommon as EC
A_ = EC.parser(__doc__).parse_args(); PP = str(EC.outdirs(A_)) + "/"
OB = EC.load_obs("us"); dd, q, P = OB["dd"], OB["q"], OB["P"]
ids = OB["ids"]; Sn = np.asarray(OB["static_norm"], float); St = pd.DataFrame({"aridity": OB["aridity"], "frac_snow": OB["snow"]})
ug = OB["ung"]; G = np.where(~ug)[0]; U = np.where(ug)[0]
per = np.asarray((dd >= "2011-01-01") & (dd <= "2019-12-31")); tr = np.asarray((dd >= "1989-01-01") & (dd <= "2008-12-31"))
nm = pd.read_csv(os.environ["CAMELS_US_NAMES"], sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
T = {"CaRT+": "CaRTp", "LSTM": "LSTM", "EA-LSTM": "EALSTM"}                                     # evalcommon model keys


def dec(s, o):
    k = np.isfinite(o)
    if k.sum() < 365: return [np.nan] * 4
    s, o = np.maximum(s[k], 0), o[k]; r = np.corrcoef(s, o)[0, 1]; a = s.std() / o.std(); b = (s.mean() - o.mean()) / o.std()
    return [1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2), r, a, b]                # NSE = 2 a r - a^2 - b^2


M = {m: np.array([[dec(np.load(EC.sim_path(A_, "us", t, k))[per, c].astype(float), q[per, c]) for c in range(q.shape[1])] for k in range(3)]) for m, t in T.items()}
N = {m: M[m][:, :, 0] for m in M}                                                          # [seed, basin]
ai, sn = St.aridity.values, St.frac_snow.values
cls = np.where(ai > 1.1, "arid", np.where(sn >= 0.3, "snow", "other"))
LAB = {"other": "other", "snow": r"snow-dominated ($f_{\rm snow}\geq0.3$)", "arid": r"arid (PET/P $>1.1$)"}
COL = {"other": "#2a78d6", "snow": "#1baf7a", "arid": "#eb6834"}; INK, MUTED = "#1f1f1e", "#6b6a66"


def boot(d):
    d = d[np.isfinite(d)]; b = [np.median(np.random.default_rng(i).choice(d, len(d))) for i in range(2000)]
    return [float(np.median(d)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), float(np.mean(d > 0))]


# ---- figure
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
LO = -1.0; f, axs = plt.subplots(1, 2, figsize=(10, 5.0)); f.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.12, wspace=0.2)
out = {"clipped": {}}
for ax, ref in zip(axs, ("LSTM", "EA-LSTM")):
    x, y = N[ref].mean(0)[U], N["CaRT+"].mean(0)[U]; xs, ys = np.clip(x, LO, 1), np.clip(y, LO, 1)
    ax.plot([LO, 1], [LO, 1], color="#b9b7b0", lw=0.8, zorder=0)
    for c in ("other", "snow", "arid"):
        k = cls[U] == c
        xe = [np.clip(x[k] - N[ref].min(0)[U][k], 0, None), np.clip(N[ref].max(0)[U][k] - x[k], 0, None)]
        ye = [np.clip(y[k] - N["CaRT+"].min(0)[U][k], 0, None), np.clip(N["CaRT+"].max(0)[U][k] - y[k], 0, None)]
        ax.errorbar(xs[k], ys[k], xerr=xe, yerr=ye, fmt="none", ecolor=COL[c], alpha=0.35, lw=0.8, zorder=1)
        ax.scatter(xs[k], ys[k], s=30, color=COL[c], edgecolor="white", linewidth=0.7, zorder=2, label=f"{LAB[c]} ($n={k.sum()}$)")
    clipped = (x < LO) | (y < LO)
    ax.scatter(xs[clipped], ys[clipped], s=58, facecolor="none", edgecolor=INK, lw=0.8, zorder=3, label=f"NSE $<{LO:.0f}$, drawn at the edge")
    d = y - x; ok = np.where(~clipped & np.isfinite(d))[0]
    for i in list(ok[np.argsort(d[ok])[:1]]) + list(ok[np.argsort(d[ok])[-1:]]):
        ax.annotate(str(nm.loc[ids[U[i]], "gauge_name"]).split(",")[0][:22].title(), (xs[i], ys[i]), xytext=(5, -3), textcoords="offset points", fontsize=6.5, color=MUTED)
    out["clipped"][ref] = [(ids[U[i]], str(nm.loc[ids[U[i]], "gauge_name"])[:30].title(), round(float(y[i]), 1), round(float(x[i]), 1)) for i in np.where(clipped)[0]]
    bb = boot(d); ax.set_title(f"({'ab'[ref == 'EA-LSTM']}) CaRT$^+$ against the {ref}: median $\\Delta$NSE {bb[0]:+.3f} [{bb[1]:+.3f}, {bb[2]:+.3f}]", loc="left", fontsize=8.5, color=INK)
    ax.set_xlim(LO - 0.03, 1); ax.set_ylim(LO - 0.03, 1); ax.set_aspect("equal"); ax.grid(color="#e4e3df", lw=0.5)
    ax.set_xlabel(f"NSE of the {ref} (mean of three seeds; horizontal bars: seed range)"); ax.set_ylabel("NSE of CaRT$^+$ (mean of three seeds; vertical bars: seed range)")
axs[0].legend(loc="lower right", bbox_to_anchor=(1, 0.06), fontsize=7.5, frameon=False)
f.savefig(PP + "figs/ung_us.pdf"); f.savefig(PP + "figs/ung_us.png", dpi=150); plt.close(f)

# ---- class table (ungauged US) with paired differences
rows = []; out["classes"] = {}
for name, k in (("all held-out basins", np.ones(len(U), bool)), ("non-arid", cls[U] != "arid"), ("arid (PET/P $>1.1$)", cls[U] == "arid"), ("snow-dominated, not arid", cls[U] == "snow")):
    sel = U[k]; med = {m: float(np.nanmedian(N[m].mean(0)[sel])) for m in N}
    pl = boot(N["CaRT+"].mean(0)[sel] - N["LSTM"].mean(0)[sel]); pe = boot(N["CaRT+"].mean(0)[sel] - N["EA-LSTM"].mean(0)[sel])
    out["classes"][name] = {"n": int(k.sum()), "median": med, "vs_LSTM": pl, "vs_EALSTM": pe}
    rows.append(f"{name} & {k.sum()} & {med['CaRT+']:.3f} & {med['LSTM']:.3f} & {med['EA-LSTM']:.3f} & ${pl[0]:+.3f}$ [{pl[1]:+.3f}, {pl[2]:+.3f}] & ${pe[0]:+.3f}$ [{pe[1]:+.3f}, {pe[2]:+.3f}] \\\\")
open(PP + "tables/u_ung_us.tex", "w").write("\\begin{tabular}{lcccccc}\n\\toprule\n & & \\multicolumn{3}{c}{median NSE} & \\multicolumn{2}{c}{paired $\\Delta$NSE, \\CaRTp{} minus} \\\\\n"
    "Class & $n$ & \\CaRTp & LSTM & EA-LSTM & LSTM & EA-LSTM \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- diagnosis of the failing basins
rr_tr = np.nanmean(np.where(tr[:, None], q, np.nan), 0) / np.mean(P[tr], 0)
dist = np.array([np.sort(np.linalg.norm(Sn[G] - Sn[c], axis=1))[0 if ug[c] else 1] for c in range(len(ids))])
nn5 = lambda c: G[np.argsort(np.linalg.norm(Sn[G] - Sn[c], axis=1))[(0 if ug[c] else 1):(5 if ug[c] else 6)]]
yc = N["CaRT+"].mean(0); bad = U[yc[U] < 0]; good = U[yc[U] >= 0]
out["failing"] = {"n": int(len(bad)), "rr_median": float(np.nanmedian(rr_tr[bad])), "rr_others": float(np.nanmedian(rr_tr[good])),
                  "rr_neighbours_median": float(np.nanmedian([np.median(rr_tr[nn5(c)]) for c in bad])), "dist_median": float(np.median(dist[bad])), "dist_others": float(np.median(dist[good])),
                  "basins": [{"id": ids[c], "name": str(nm.loc[ids[c], "gauge_name"])[:30].title(), "rr": round(float(rr_tr[c]), 3), "rr_nn": [round(float(v), 2) for v in rr_tr[nn5(c)]],
                              **{m: [round(float(v), 2) for v in M[m][:, c].mean(0)] for m in ("CaRT+", "LSTM")}} for c in bad]}
low = (~ug) & (rr_tr < 0.12); knn = np.array([np.median(rr_tr[nn5(c)]) for c in range(len(ids))])
beta = {m: np.exp(np.mean(np.log(np.maximum([[np.nanmean(np.maximum(np.load(EC.sim_path(A_, "us", T[m], k))[per, c], 0)) / np.nanmean(q[per, c]) for c in range(q.shape[1])] for k in range(3)], 1e-6)), 0)) for m in ("CaRT+",)}
out["gauged_low_rr"] = {"n": int(low.sum()), "knn_rr": float(np.nanmedian(knn[low])), "obs_rr": float(np.nanmedian(rr_tr[low])), "beta_low": float(np.nanmedian(beta["CaRT+"][low])), "beta_other": float(np.nanmedian(beta["CaRT+"][(~ug) & (rr_tr >= 0.12)]))}
json.dump(out, open(PP + "data/u_ung_us.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "failing"}, indent=0)[:1500]); print([(b["name"], b["rr"], b["CaRT+"], b["LSTM"]) for b in out["failing"]["basins"]])

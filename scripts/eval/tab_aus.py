"""CAMELS-AUS (paper Section 5.6, Table tab:aus): the models of the paper with the CAMELS-US configuration, unified protocol (train
1989-2008, select 2009-10, test 2011-2019; 56 held-out catchments). Single model = per-catchment metric averaged over seeds 0-2, then the
median over gauged (or held-out) catchments; ensemble = mean of the three simulations. Dry years = test years whose precipitation is below
the catchment's lower quartile of annual precipitation 1980-2019 (forcing only; 502 of 505 gauged catchments have at least one, mostly
2018-19); arid = PET/P > 2; no flow = share of test days with flow below 0.01 mm/d, simulated minus observed (percentage points), over the
gauged catchments with at least 5 % such days in the observations. Two extra rows: CaRT+ with the flows of 32 attribute draws averaged
(sigma 0.3 and 0.5, Section 5.5; sim_noise0.3.npy / sim_noise0.5.npy in the CaRT+ run folders, see tab_hedge.py). Writes
<out>/tables/u_aus.tex (medians), <out>/tables/u_aus_paired.tex (paired differences with 95 % bootstrap intervals) and <out>/data/u_aus.json
and prints paired differences quoted in the text.
Data: CAMELS-AUS v2 through cart.data (CART_AUS_NPZ, CART_AUS_UNGAUGED, CART_AUS_BANDS; DATA.md).
usage: python scripts/eval/tab_aus.py [--runs DIR] [--out DIR]
"""
import json, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalcommon as EC
A_ = EC.parser(__doc__).parse_args()
O = EC.outdirs(A_); PP = str(O) + "/"
OB = EC.load_obs("aus")
dd = OB["dd"]; q = OB["q"].astype(float); P = OB["P"].astype(float); T = len(dd)
ai = OB["aridity"].astype(float)
ug = OB["ung"].copy(); C = q.shape[1]
yr = dd.year.values; test = (yr >= 2011) & (yr <= 2019)
Py = pd.DataFrame(P, index=dd).groupby(yr).sum(); q25 = Py.quantile(0.25)
dry = np.zeros((T, C), bool)
for c in range(C): dry[:, c] = np.isin(yr, [y for y in range(2011, 2020) if Py.loc[y, c] < q25[c]])
ZF = 0.01                                                                                     # mm/d, cessation of flow
zf_obs = np.array([np.mean(q[test, c][np.isfinite(q[test, c])] < ZF) if np.isfinite(q[test, c]).sum() >= 365 else np.nan for c in range(C)])
ROWS = [("HBV", "HBV", "sim.npy"), ("dHBV", "dHBV", "sim.npy"), (r"\CaRT", "CaRT", "sim.npy"), (r"\CaRTp", "CaRTp", "sim.npy"),       # (label, evalcommon key, file)
        ("LSTM", "LSTM", "sim.npy"), ("EA-LSTM", "EALSTM", "sim.npy"),
        (r"\CaRTp, $\sigma=0.3$", "CaRTp", "sim_noise0.3.npy"), (r"\CaRTp, $\sigma=0.5$", "CaRTp", "sim_noise0.5.npy")]


def met(s, o, nmin=365):                                                                         # 365 valid days as unified_eval.py; dry years 180
    k = np.isfinite(o) & np.isfinite(s)
    if k.sum() < nmin or np.nanstd(o[k]) <= 0: return [np.nan] * 4
    s, o = np.maximum(s[k], 0), o[k]; e = 0.01 * o.mean(); r = np.corrcoef(s, o)[0, 1] if s.std() > 0 else np.nan
    a, b = s.std() / o.std(), s.mean() / o.mean()
    return [1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2),
            1 - np.sum((np.log(s + e) - np.log(o + e)) ** 2) / np.sum((np.log(o + e) - np.log(o + e).mean()) ** 2),
            1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2), b]


def evals(S):
    full = np.nanmean([[met(s[test, c], q[test, c]) for c in range(C)] for s in S], 0)
    dr = np.nanmean([[met(s[dry[:, c], c], q[dry[:, c], c], 180) if dry[:, c].any() else [np.nan] * 4 for c in range(C)] for s in S], 0)
    E = np.mean(S, 0); ens = np.array([met(E[test, c], q[test, c]) for c in range(C)])
    zf = np.mean([[np.mean(np.maximum(s[test, c][np.isfinite(q[test, c])], 0) < ZF) - zf_obs[c] for c in range(C)] for s in S], 0)
    return full, dr, ens, zf


def boot(d):
    d = d[np.isfinite(d)]; b = [np.median(np.random.default_rng(i).choice(d, len(d))) for i in range(2000)]
    return [float(np.median(d)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


g, u, ar = ~ug, ug, (~ug) & (ai > 2)
im = g & (zf_obs >= 0.05)                                                                      # intermittent gauged catchments
res, rows, per = {}, [], {}
for lab, tag, fn in ROWS:
    S = [np.load(EC.sim_path(A_, "aus", tag, k, fn)).astype(float)[:T] for k in range(3)]
    full, dr, ens, zf = evals(S); per[lab] = (full, dr, ens, zf)
    v = {"NSE": np.nanmedian(full[g, 0]), "logNSE": np.nanmedian(full[g, 1]), "KGE": np.nanmedian(full[g, 2]), "beta": np.nanmedian(full[g, 3]),
         "dry_NSE": np.nanmedian(dr[g, 0]), "dry_KGE": np.nanmedian(dr[g, 2]), "dry_beta": np.nanmedian(dr[g, 3]), "noflow_pp": 100 * np.nanmedian(zf[im]), "arid_NSE": np.nanmedian(full[ar, 0]),
         "arid_KGE": np.nanmedian(full[ar, 2]), "held_NSE": np.nanmedian(full[u, 0]), "ens_NSE": np.nanmedian(ens[g, 0]), "ens_held_NSE": np.nanmedian(ens[u, 0])}
    res[lab] = {k: float(x) for k, x in v.items()}
    if lab == r"\CaRTp, $\sigma=0.3$": rows.append("\\midrule")
    rows.append(f"{lab} & {v['NSE']:.3f} & {v['logNSE']:.3f} & {v['KGE']:.3f} & {v['beta']:.2f} & {v['dry_NSE']:.3f} & {v['dry_KGE']:.3f} & {v['dry_beta']:.2f} & ${v['noflow_pp']:+.1f}$ & "
                f"{v['arid_KGE']:.3f} & {v['held_NSE']:.3f} & {v['ens_NSE']:.3f} & {v['ens_held_NSE']:.3f} \\\\")
hdr = ("\\begin{tabular}{lcccccccccccc}\n\\toprule\n & \\multicolumn{4}{c}{gauged, 2011--2019} & \\multicolumn{3}{c}{gauged, dry years} & no flow & arid & held-out & \\multicolumn{2}{c}{3-seed ensemble} \\\\\n"
       "Model & NSE & NSE$_{\\log}$ & KGE & $\\beta$ & NSE & KGE & $\\beta$ & (pp) & KGE & NSE & NSE & NSE held-out \\\\\n\\midrule\n")
open(PP + "tables/u_aus.tex", "w").write(hdr + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
pair = {}
for x in ("LSTM", "EA-LSTM", "dHBV", r"\CaRT"):
    a, b = per[r"\CaRTp"], per[x]
    pair[x] = {"NSE": boot(a[0][g, 0] - b[0][g, 0]), "logNSE": boot(a[0][g, 1] - b[0][g, 1]), "dry_NSE": boot(a[1][g, 0] - b[1][g, 0]),
               "dry_absbeta": boot(np.abs(a[1][g, 3] - 1) - np.abs(b[1][g, 3] - 1)), "arid_KGE": boot(a[0][ar, 2] - b[0][ar, 2]),
               "held_NSE": boot(a[0][u, 0] - b[0][u, 0]), "ens_NSE": boot(a[2][g, 0] - b[2][g, 0]),
               "noflow_abs_pp": boot(100 * (np.abs(a[3][im]) - np.abs(b[3][im])))}
for s in ("0.3", "0.5"):
    h = per[rf"\CaRTp, $\sigma={s}$"]
    pair[f"sigma{s}"] = {"held_vs_plain": boot(h[0][u, 0] - per[r"\CaRTp"][0][u, 0]), "held_vs_LSTM": boot(h[0][u, 0] - per["LSTM"][0][u, 0]),
                         "held_vs_EALSTM": boot(h[0][u, 0] - per["EA-LSTM"][0][u, 0])}
info = {"n_intermittent": int(im.sum()), "obs_noflow_share_median_intermittent": float(np.nanmedian(zf_obs[im])), "n_gauged": int(g.sum()), "n_held": int(u.sum()), "n_arid": int(ar.sum()), "n_dry_catch": int((dry[:, g].any(0)).sum()),
        "dry_share_by_year": {int(y): float(np.mean([y in set(yr[dry[:, c]]) for c in np.where(g)[0]])) for y in range(2011, 2020)},
        "gauged_aridity_median": float(np.median(ai[g])), "share_arid": float(np.mean(ai[g] > 2))}
# ---- panel (b): paired per-catchment differences, median and 95 % bootstrap interval (catchments resampled)
COLS = [("NSE", lambda X: X[0][g, 0], 3), ("NSE$_{\\log}$", lambda X: X[0][g, 1], 3), ("NSE", lambda X: X[1][g, 0], 3),
        ("KGE", lambda X: X[0][ar, 2], 3), ("NSE", lambda X: X[0][u, 0], 3), ("NSE", lambda X: X[2][g, 0], 3)]           # the columns the text quotes
PAIRS = [(r"\CaRTp", "LSTM"), (r"\CaRTp", "EA-LSTM"), (r"\CaRTp", "dHBV"), (r"\CaRTp", r"\CaRT"), (r"\CaRTp", "HBV"),
         (r"\CaRTp, $\sigma=0.3$", "LSTM"), (r"\CaRTp, $\sigma=0.5$", "LSTM")]
prow, pjs = [], {}
for x, y in PAIRS:
    cells, cis, js = [], [], {}
    for (nm, f, nd), k in zip(COLS, range(len(COLS))):
        m, lo, hi = boot(f(per[x]) - f(per[y])); js[k] = [m, lo, hi]
        cells.append(f"${m:+.{nd}f}$"); cis.append(f"{{\\footnotesize [{lo:+.{nd}f}, {hi:+.{nd}f}]}}")
    if x == r"\CaRTp, $\sigma=0.3$": prow.append("\\midrule")
    prow.append(f"{x} $-$ {y} & " + " & ".join(cells) + " \\\\")                         # median, then the 95 % interval below it
    prow.append(" & " + " & ".join(cis) + " \\\\[2pt]"); pjs[f"{x} - {y}"] = js
phdr = ("\\begin{tabular}{lcccccc}\n\\toprule\n & \\multicolumn{2}{c}{gauged, 2011--2019} & gauged, dry years & arid & held-out & 3-seed ensemble \\\\\n"
        "Pair & " + " & ".join(c[0] for c in COLS) + " \\\\\n\\midrule\n")
open(PP + "tables/u_aus_paired.tex", "w").write(phdr + "\n".join(prow) + "\n\\bottomrule\n\\end{tabular}\n")
json.dump({"info": info, "medians": res, "paired_CaRTp_minus": pair, "paired_table": pjs}, open(PP + "data/u_aus.json", "w"), indent=1)
print(json.dumps(info)); [print(k, {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in res.items()]
for k, v in pair.items(): print(k, {kk: [round(x, 3) for x in vv] for kk, vv in v.items()})

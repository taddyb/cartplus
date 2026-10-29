"""Contribution of each physical process group of refinements to CaRT+ (Section 5.3, Figure fig:lofo, Table tab:lofo): leave-one-group-out,
CaRT+ minus CaRT+ without the group, held-out period 2011-2019, three seeds (per-catchment NSE averaged over the seeds; median of the paired
differences and 95 % bootstrap interval over catchments, 2000 resamples, numpy default_rng(i)), overall and by class. Groups (cart.config
GROUPS): P1 runoff generation, P2 quick release, P3 evaporation, P4 exchange with the exterior, P5 snow. Also the paired gain of CaRT+ over
CaRT, the sum of the five contributions, the low-flow volume bias (lowest 30 % of the flow-duration curve) of the arid US basins and the
NSE of Costa Beck (27038). Classes: CAMELS-GB groundwater-fed BFI >= 0.65, flashy BFI < 0.5, snow-affected f_snow >= 0.04; CAMELS-US
snow-dominated f_snow >= 0.3, arid PET/P > 1.1, humid PET/P <= 0.7 with f_snow < 0.3.
Runs (configs/paper_runs.csv): cartplus, cart, cartplus_no_p1 .. cartplus_no_p5.
Writes <out>/tables/u_lofo_phys.tex, <out>/figs/refine_lofo.pdf/.png and <out>/data/u_lofo_phys.json.
usage: python scripts/eval/lofo_phys.py [--runs DIR] [--out DIR]
"""
import os, sys, json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalcommon as EC
A_ = EC.parser(__doc__).parse_args(); PP = str(EC.outdirs(A_)) + "/"
GROUPS = [("P1", "Runoff generation", "surface layer, Pareto partition, infiltration excess, depression storage, near-stream zone"),
          ("P2", "Quick release", "connected release, connected bypass, groundwater-controlled connectivity"),
          ("P3", "Evaporation", "vegetation demand, groundwater uptake"),
          ("P4", "Exchange with the exterior", "exterior head, exterior inflow, channel loss"),
          ("P5", "Snow", "melt energy term")]
def load(ds):
    OB = EC.load_obs(ds)
    if ds == "us":
        sn, ai = OB["snow"], OB["aridity"]
        return OB["dd"], OB["q"], OB["ids"], OB["mask"], {"snow-dominated": sn >= 0.3, "arid": ai > 1.1, "humid": (ai <= 0.7) & (sn < 0.3)}
    bfi, sn = OB["bfi"], OB["snow"]
    return OB["dd"], OB["q"], OB["ids"], OB["mask"], {"groundwater-fed": bfi >= 0.65, "flashy": bfi < 0.5, "snow-affected": sn >= 0.04}
def boot(d):
    d = d[np.isfinite(d)]; b = [np.median(np.random.default_rng(k).choice(d, len(d))) for k in range(2000)]
    return [float(np.median(d)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(len(d))]
out = {}
for ds in ("gb", "us"):
    dd, q, ids, mask, cls = load(ds); per = np.asarray((dd >= "2011-01-01") & (dd <= "2019-12-31")); cache = {}
    sim = lambda m, k: EC.sim_path(A_, ds, m, k) if m in EC.RUNS[ds] else Path(A_.runs) / f"{m}_s{k}" / "sim.npy"
    lab = lambda m: EC.run_label(A_, ds, m) if m in EC.RUNS[ds] else m
    def N(m):                                                                                   # seed-mean NSE and low-flow volume bias per catchment
        if m not in cache:
            v, w = [], []
            for k in range(3):
                S = np.load(sim(m, k))[per].astype(float); n, lv = [], []
                for c in range(q.shape[1]):
                    o = q[per, c]; mk = np.isfinite(o)
                    if mk.sum() < 365: n.append(np.nan); lv.append(np.nan); continue
                    s_, o_ = np.maximum(S[mk, c], 0), o[mk]
                    n.append(1 - np.sum((s_ - o_) ** 2) / np.sum((o_ - o_.mean()) ** 2))
                    so, oo = np.sort(s_)[::-1], np.sort(o_)[::-1]; i70 = int(0.7 * len(oo))
                    lv.append(100 * (so[i70:].sum() - oo[i70:].sum()) / max(oo[i70:].sum(), 1e-9))
                v.append(n); w.append(lv)
            cache[m] = (np.mean(v, 0), np.mean(w, 0))
        return cache[m]
    full = N("CaRTp")[0]; o = {"groups": {}}
    o["total"] = boot((full - N("CaRT")[0])[mask])
    o["total_classes"] = {c: boot((full - N("CaRT")[0])[mask & k]) for c, k in cls.items()}
    o["median_nse_full"] = float(np.nanmedian(full[mask]))
    for g, _, _ in GROUPS:
        d = full - N(g)[0]
        o["groups"][g] = {"lofo": boot(d[mask]), "classes": {c: boot(d[mask & k]) for c, k in cls.items()}, "tag": lab(g)}
    o["sum_lofo"] = float(sum(o["groups"][g]["lofo"][0] for g, _, _ in GROUPS))
    if ds == "us":
        k = mask & cls["arid"]
        o["arid_lowvol"] = {lab(m): float(np.nanmedian(N(m)[1][k])) for m in ["CaRTp", "P4", "P5"]}
    else:
        c = ids.index(27038); o["costa_beck"] = {lab(m): float(N(m)[0][c]) for m in ["CaRTp", "P4"]}
    out[ds] = o
    print(ds, "total", [round(x, 4) for x in o["total"][:3]], "sum of lofo", round(o["sum_lofo"], 4), "median NSE", round(o["median_nse_full"], 4))
    for g, lb, _ in GROUPS:
        r = o["groups"][g]; print(f"  {g} {lb:18s} {r['lofo'][0]:+.4f} [{r['lofo'][1]:+.4f}, {r['lofo'][2]:+.4f}]  " +
                                  "  ".join(f"{c} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for c, v in r["classes"].items()))
    print("  total by class", {c: [round(x, 4) for x in v[:3]] for c, v in o["total_classes"].items()})
    print(" ", {k_: v for k_, v in o.items() if k_ in ("arid_lowvol", "costa_beck")})
json.dump(out, open(PP + "data/u_lofo_phys.json", "w"), indent=1)
# ---- table
f = lambda r: f"${r[0]:+.4f}$ [{r[1]:+.4f}, {r[2]:+.4f}]"
rows = [f"{lb} ({desc}) & {f(out['gb']['groups'][g]['lofo'])} & {f(out['us']['groups'][g]['lofo'])} \\\\" for g, lb, desc in GROUPS]
rows += ["\\midrule", f"Sum of the five & ${out['gb']['sum_lofo']:+.4f}$ & ${out['us']['sum_lofo']:+.4f}$ \\\\",
         f"\\CaRTp{{}} minus \\CaRT{{}} & {f(out['gb']['total'])} & {f(out['us']['total'])} \\\\"]
open(PP + "tables/u_lofo_phys.tex", "w").write("\\begin{tabular}{p{0.5\\textwidth}cc}\n\\toprule\nGroup removed from \\CaRTp & CAMELS-GB & CAMELS-US \\\\\n\\midrule\n"
                                              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
# ---- figure: per continent, the leave-one-out contribution of each group overall and by class (median and 95 % interval)
COL = {"all": "#1f1f1e", "groundwater-fed": "#2a78d6", "flashy": "#d9622b", "snow-affected": "#7a5cc2",
       "snow-dominated": "#7a5cc2", "arid": "#d9622b", "humid": "#2a78d6"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True, sharey=True)
for ax, ds, ttl in zip(axes, ("gb", "us"), ("CAMELS-GB", "CAMELS-US")):
    o = out[ds]; keys = ["all"] + list(o["groups"]["P1"]["classes"]); off = np.linspace(0.27, -0.27, len(keys))
    for j, kk in enumerate(keys):
        for i, (g, lb, _) in enumerate(GROUPS):
            r = o["groups"][g]["lofo"] if kk == "all" else o["groups"][g]["classes"][kk]; y = len(GROUPS) - 1 - i + off[j]
            ax.errorbar(r[0], y, xerr=[[r[0] - r[1]], [r[2] - r[0]]], fmt="o", ms=4.5 if kk == "all" else 3.5, color=COL[kk], elinewidth=1, capsize=0,
                        label=(f"{kk} (n={r[3]})" if i == 0 else None))
    ax.axvline(0, color="#4a4a47", lw=0.8); ax.grid(axis="x", color="#e4e3df", lw=0.5); ax.spines[["top", "right"]].set_visible(False)
    ax.set_yticks(range(len(GROUPS))[::-1]); ax.set_yticklabels([lb for _, lb, _ in GROUPS], fontsize=9)
    ax.set_xlabel("$\\Delta$NSE, CaRT$^+$ minus CaRT$^+$ without the group"); ax.set_title(ttl, fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
fig.savefig(PP + "figs/refine_lofo.pdf"); fig.savefig(PP + "figs/refine_lofo.png", dpi=130)

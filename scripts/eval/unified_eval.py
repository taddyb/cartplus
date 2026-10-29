"""Unified-protocol evaluation (paper Section 5, Tables tab:u_gb / tab:u_us, tab:u_paired, tab:u_groups): both continents, held-out
test 2011-01-01 .. 2019-12-31, observations only. Models HBV, dHBV, CaRT, CaRT+, LSTM, EA-LSTM; seeds 0-2 (a model is left out until
all three seeds exist). Writes <out>/tables/u_main_<ds>.tex, u_groups_<ds>.tex, u_paired_<ds>.tex and <out>/data/unified_<ds>.json.
Metrics per catchment (simulation clipped at 0, >= 365 valid days): NSE; NSE of log(Q + 0.01 mean(Q_obs)); KGE (Gupta et al. 2009);
FHV (top 2 % of the flow-duration curve), FLV (bottom 30 % of the log flow-duration curve; Yilmaz et al. 2008) and the volume bias of
the low-flow segment (lowest 30 % of the flow-duration curve). Single model first (per-catchment metric averaged over the seeds, then
the median; and the median NSE of each seed), the 3-seed ensemble (mean of the simulations) alongside; paired per-catchment differences
of the seed-mean metric with a 95 % bootstrap interval of the median (2000 resamples, numpy default_rng(i), i = 0..1999).
usage: python scripts/eval/unified_eval.py gb|us [--runs DIR] [--out DIR]
"""
import json, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalcommon as EC
ap = EC.parser(__doc__); ap.add_argument("ds", choices=["gb", "us"]); A_ = ap.parse_args()
O = EC.outdirs(A_); T = str(O / "tables") + "/"; D = str(O / "data") + "/"
NAME = {"HBV": "HBV", "dHBV": "dHBV", "CaRT": r"\CaRT", "CaRTp": r"\CaRTp", "LSTM": "LSTM", "EALSTM": "EA-LSTM"}
ds = A_.ds; MODELS = [(m, EC.run_label(A_, ds, m)) for m in ("HBV", "dHBV", "CaRT", "CaRTp", "LSTM", "EALSTM")]
OB = EC.load_obs(ds); dd, q, mask, ung, ids = OB["dd"], OB["q"], OB["mask"], OB["ung"], OB["ids"]
if ds == "us":
    snow = OB["snow"]; ai = OB["aridity"]
    groups = {"snow<0.1": snow < 0.1, "snow0.1-0.3": (snow >= 0.1) & (snow < 0.3), "snow>=0.3": snow >= 0.3, "AI<=0.7": ai <= 0.7, "AI0.7-1.1": (ai > 0.7) & (ai <= 1.1), "AI>1.1": ai > 1.1}
    GLAB = {"snow<0.1": r"$f_{\rm snow}<0.1$", "snow0.1-0.3": r"$0.1\le f_{\rm snow}<0.3$", "snow>=0.3": r"$f_{\rm snow}\ge0.3$", "AI<=0.7": r"PET/P $\le0.7$", "AI0.7-1.1": r"$0.7<$ PET/P $\le1.1$", "AI>1.1": r"PET/P $>1.1$"}
    sb = snow >= 0.3
else:
    bfi, snow = OB["bfi"], OB["snow"]
    groups = {"BFI<0.5": bfi < 0.5, "BFI0.5-0.65": (bfi >= 0.5) & (bfi < 0.65), "BFI>=0.65": bfi >= 0.65, "snow>=0.04": snow >= 0.04}
    GLAB = {"BFI<0.5": r"BFI $<0.5$", "BFI0.5-0.65": r"$0.5\le$ BFI $<0.65$", "BFI>=0.65": r"BFI $\ge0.65$", "snow>=0.04": r"$f_{\rm snow}\ge0.04$"}
    sb = (snow >= 0.04) & mask
per = np.asarray((dd >= "2011-01-01") & (dd <= "2019-12-31")); mon = dd.month.values
nse = lambda s, o: 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def met(s, o):
    k = np.isfinite(o) & np.isfinite(s)
    if k.sum() < 365: return [np.nan] * 6
    s, o = np.maximum(s[k], 0), o[k]; e = 0.01 * o.mean(); r = np.corrcoef(s, o)[0, 1]
    kge = 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)
    so, oo = np.sort(s)[::-1], np.sort(o)[::-1]; n = len(oo); h = max(int(0.02 * n), 1)
    fhv = 100 * (so[:h].sum() - oo[:h].sum()) / oo[:h].sum()
    ls_, lo = np.log(np.maximum(so, 1e-4)), np.log(np.maximum(oo, 1e-4)); i70 = int(0.7 * n)
    aa, bb = np.sum(ls_[i70:] - ls_[i70:].min()), np.sum(lo[i70:] - lo[i70:].min()); flv = -100 * (aa - bb) / (bb + 1e-9)
    lv = 100 * (so[i70:].sum() - oo[i70:].sum()) / max(oo[i70:].sum(), 1e-9)                    # volume bias of the low-flow segment (lowest 30 % of the FDC)
    return [nse(s, o), nse(np.log(s + e), np.log(o + e)), kge, fhv, flv, lv]


def M(sim): return np.array([met(sim[per, c], q[per, c]) for c in range(q.shape[1])])


def boot(d, n=2000):
    d = d[np.isfinite(d)]; b = [np.median(np.random.default_rng(i).choice(d, len(d))) for i in range(n)]
    return np.median(d), np.percentile(b, 2.5), np.percentile(b, 97.5)


def seasonal(sim):
    out = []
    for ms in ([2, 3, 4], [5, 6, 7]):
        w = per & np.isin(mon, ms); v = []
        for c in np.where(sb)[0]:
            o = q[w, c]; k = np.isfinite(o)
            if k.sum() > 30: v.append(sim[w, c][k].sum() / o[k].sum() - 1)
        out.append(100 * np.median(v))
    return out


sims, note = {}, []
for m, t in MODELS:
    L = [EC.sim_path(A_, ds, m, k) for k in range(3) if EC.sim_path(A_, ds, m, k).exists()]
    if len(L) < 3: note.append(f"{m}: {len(L)}/3 seeds ({t}), left out until all seeds exist"); continue
    if L:
        S = [np.load(f).astype(float) for f in L]
        if any(s.shape != q.shape for s in S): note.append(f"{m}: shape mismatch, skipped"); continue
        sims[m] = S
print("models:", list(sims), "| notes:", note)
res = {m: {"single": [M(s) for s in S], "ens": M(np.mean(S, 0)), "seas": seasonal(np.mean(S, 0))} for m, S in sims.items()}
for m in res: res[m]["seedmean"] = np.mean(res[m]["single"], 0)
names = ["NSE", "logNSE", "KGE", "FHV", "FLV", "LowVol"]
J = {"notes": note, "n_gauged": int(mask.sum()), "n_ungauged": int(ung.sum()), "models": {}}
# ---- main table: single model (per-catchment mean of the seed metrics, then median) first, 3-seed ensemble alongside
rows = []
for m, r in res.items():
    sing = " / ".join(f"{np.nanmedian(x[mask, 0]):.3f}" for x in r["single"]); E, Sm = r["ens"], r["seedmean"]
    sm = [np.nanmedian(Sm[mask, i]) for i in range(6)]; e = [np.nanmedian(E[mask, i]) for i in range(6)]
    nneg_s = int(np.nansum(Sm[mask, 0] < 0)); nneg = int(np.nansum(E[mask, 0] < 0))
    ug_s = np.nanmedian(Sm[ung, 0]) if ung.any() else np.nan; ug = np.nanmedian(E[ung, 0]) if ung.any() else np.nan
    rows.append(f"{NAME[m]} & {sing} & {sm[0]:.3f} & {sm[1]:.3f} & {sm[2]:.3f} & {sm[3]:.1f} & {sm[4]:.1f} & {sm[5]:.1f} & {nneg_s} & {ug_s:.3f} & {e[0]:.3f} & {e[1]:.3f} & {e[2]:.3f} & {ug:.3f} \\\\")
    J["models"][m] = {"single": [float(np.nanmedian(x[mask, 0])) for x in r["single"]], "seedmean": dict(zip(names, map(float, sm))), "nneg_seedmean": nneg_s,
                      "ens": dict(zip(names, map(float, e))), "nneg": nneg, "ungauged_ens": float(ug), "ungauged_seedmean": float(ug_s),
                      "seas_FebApr_MayJul": list(map(float, r["seas"])),
                      "groups_ens": {g: float(np.nanmedian(E[gm & mask, 0])) for g, gm in groups.items()},
                      "groups_seedmean": {g: float(np.nanmedian(Sm[gm & mask, 0])) for g, gm in groups.items()},
                      "groups_seedmean_log": {g: float(np.nanmedian(Sm[gm & mask, 1])) for g, gm in groups.items()}}
open(T + f"u_main_{ds}.tex", "w").write("\\begin{tabular}{lccccccccc|cccc}\n\\toprule\n & \\multicolumn{9}{c|}{single model} & \\multicolumn{4}{c}{3-seed ensemble} \\\\\n"
    "Model & NSE (seeds 0/1/2) & NSE & NSE$_{\\log}$ & KGE & FHV (\\%) & FLV (\\%) & low-flow volume (\\%) & $n$(NSE$<$0) & NSE ungauged & NSE & NSE$_{\\log}$ & KGE & NSE ungauged \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
# ---- groups table: single-model NSE / logNSE, ensemble NSE in parentheses
hdr = " & ".join(GLAB[g] for g in groups); rows = []
for m, r in res.items():
    E, Sm = r["ens"], r["seedmean"]
    rows.append(f"{NAME[m]} & " + " & ".join(f"{np.nanmedian(Sm[gm & mask, 0]):.3f} / {np.nanmedian(Sm[gm & mask, 1]):.3f} ({np.nanmedian(E[gm & mask, 0]):.3f})" for gm in groups.values()) + " \\\\")
valid = np.isfinite(res["CaRTp"]["seedmean"][:, 0])                                          # catchments with >= 365 observed test days
ns = " & ".join(f"$n={int((gm & mask & valid).sum())}$" for gm in groups.values())
open(T + f"u_groups_{ds}.tex", "w").write("\\begin{tabular}{l" + "c" * len(groups) + "}\n\\toprule\nModel & " + hdr + " \\\\\n & " + ns + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
# ---- paired differences: single model (seed-mean per catchment) on gauged and ungauged, and the ensembles
rows = []; J["paired"] = {}
for a, b in (("CaRTp", "LSTM"), ("CaRTp", "EALSTM"), ("CaRTp", "CaRT"), ("CaRTp", "dHBV"), ("CaRTp", "HBV"), ("CaRT", "HBV"), ("CaRT", "dHBV"), ("dHBV", "LSTM"), ("LSTM", "CaRT"), ("EALSTM", "LSTM")):
    if a not in res or b not in res: continue
    cells = []; J["paired"][f"{a}-{b}"] = {}
    for i, nm in enumerate(["NSE", "logNSE", "KGE"]):
        md, lo, hi = boot((res[a]["seedmean"][:, i] - res[b]["seedmean"][:, i])[mask]); cells.append(f"${md:+.3f}$ [{lo:+.3f}, {hi:+.3f}]"); J["paired"][f"{a}-{b}"][nm] = [md, lo, hi]
    if ung.any():
        md, lo, hi = boot((res[a]["seedmean"][:, 0] - res[b]["seedmean"][:, 0])[ung]); cells.append(f"${md:+.3f}$ [{lo:+.3f}, {hi:+.3f}]"); J["paired"][f"{a}-{b}"]["NSE_ung"] = [md, lo, hi]
    md, lo, hi = boot((res[a]["ens"][:, 0] - res[b]["ens"][:, 0])[mask]); cells.append(f"${md:+.3f}$ [{lo:+.3f}, {hi:+.3f}]"); J["paired"][f"{a}-{b}"]["NSE_ens"] = [md, lo, hi]
    if ung.any():
        md, lo, hi = boot((res[a]["ens"][:, 0] - res[b]["ens"][:, 0])[ung]); cells.append(f"${md:+.3f}$ [{lo:+.3f}, {hi:+.3f}]"); J["paired"][f"{a}-{b}"]["NSE_ens_ung"] = [md, lo, hi]
    rows.append(f"{NAME[a]} $-$ {NAME[b]} & " + " & ".join(cells) + " \\\\")
open(T + f"u_paired_{ds}.tex", "w").write("\\begin{tabular}{lcccc|cc}\n\\toprule\n & \\multicolumn{4}{c|}{single model} & \\multicolumn{2}{c}{3-seed ensemble} \\\\\n"
    "Pair & $\\Delta$NSE & $\\Delta$NSE$_{\\log}$ & $\\Delta$KGE & $\\Delta$NSE ungauged & $\\Delta$NSE & $\\Delta$NSE ungauged \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
os.makedirs(D, exist_ok=True); json.dump(J, open(D + f"unified_{ds}.json", "w"), indent=1, default=float)
for m in J["models"]: print(m, J["models"][m]["single"], "single", {k: round(v, 3) for k, v in J["models"][m]["seedmean"].items()}, "nneg", J["models"][m]["nneg_seedmean"], "ung", round(J["models"][m]["ungauged_seedmean"], 3), "| ens", {k: round(v, 3) for k, v in J["models"][m]["ens"].items()}, "ung", round(J["models"][m]["ungauged_ens"], 3))
for k, v in J["paired"].items(): print(k, {kk: [round(x, 4) for x in vv] for kk, vv in v.items()})

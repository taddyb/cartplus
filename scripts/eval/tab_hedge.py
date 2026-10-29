"""Parameter-space ensemble of CaRT+ on the held-out catchments (paper Section 5.5, Table tab:hedge): the standardised attributes of every
catchment are perturbed with Gaussian noise of s.d. sigma, CaRT+ is run once per draw with the trained network of each seed and the 32
flows are averaged; the averaged flows are stored as sim_noise<sigma>.npy next to sim.npy in the CaRT+ run folders (sigma = 0.1, 0.2, 0.3,
0.5, 0.7, 1.0; written by python -m cart.predict --run <CaRT+ run> --noise SIGMA --draws 32, configs/paper_forward.csv).
This script evaluates the averaged flows against the plain simulation on CAMELS-GB and CAMELS-US, test period 2011-2019, per-catchment
NSE averaged over the three seeds, then the median over held-out and over gauged catchments and over the classes of Table tab:ung_us (US
arid, PET/P > 1.1) and the groundwater-fed class (GB, BFI >= 0.65); paired differences with 95 % catchment-bootstrap intervals; NSE
decomposition (r, alpha, bias) of the US arid class. Writes <out>/tables/u_hedge.tex and <out>/data/u_hedge.json and prints the numbers
quoted in the text.
usage: python scripts/eval/tab_hedge.py [--runs DIR] [--out DIR]
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalcommon as EC
A_ = EC.parser(__doc__).parse_args()
O = EC.outdirs(A_); PP = str(O) + "/"
SIG = ["0.1", "0.2", "0.3", "0.5", "0.7", "1.0"]; F = ["sim.npy"] + [f"sim_noise{s}.npy" for s in SIG]; SL = ["0"] + SIG


def dec(s, o):
    k = np.isfinite(o) & np.isfinite(s)
    if k.sum() < 365: return [np.nan] * 4
    s, o = np.maximum(s[k], 0), o[k]; r = np.corrcoef(s, o)[0, 1]; a = s.std() / o.std(); b = (s.mean() - o.mean()) / o.std()
    return [1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2), r, a, b]


def boot(d):
    d = d[np.isfinite(d)]; b = [np.median(np.random.default_rng(i).choice(d, len(d))) for i in range(2000)]
    return [float(np.median(d)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), float(np.mean(d > 0))]


out = {}; rows = []
for ds in ("gb", "us"):
    OB = EC.load_obs(ds); dd, q, ug = OB["dd"], OB["q"], OB["ung"]
    ref = {"LSTM": "LSTM", "EA-LSTM": "EALSTM"}                                                  # label -> evalcommon model key
    if ds == "us":
        cls = np.where(OB["aridity"] > 1.1, "arid", "non-arid"); CL = [("arid (PET/P $>1.1$)", "arid"), ("non-arid", "non-arid")]
    else:
        cls = np.where(OB["bfi"] >= 0.65, "chalk", "other"); CL = [("groundwater-fed (BFI $\\ge0.65$)", "chalk"), ("other", "other")]
    per = np.asarray((dd >= "2011-01-01") & (dd <= "2019-12-31")); C = q.shape[1]

    def seedmean(model, f, j=None):
        S = [np.load(EC.sim_path(A_, ds, model, k, f)) for k in range(3)]
        X = np.nanmean([[dec(s[per, c].astype(float), q[per, c]) for c in range(C)] for s in S], 0) if j is None else \
            np.nanmean([[dec(s[per, c].astype(float), q[per, c])[j] for c in range(C)] for s in S], 0)
        return X
    M = {f: seedmean("CaRTp", f) for f in F}
    RF = {m: seedmean(k, "sim.npy", 0) for m, k in ref.items()}
    U = np.where(ug & np.isfinite(M["sim.npy"][:, 0]))[0]; G = np.where(~ug & np.isfinite(M["sim.npy"][:, 0]))[0]
    o = {"n_held": int(len(U)), "n_gauged": int(len(G)), "sigma": {}, "ref_held": {m: float(np.nanmedian(RF[m][U])) for m in RF},
         "ref_class": {m: {c: float(np.nanmedian(RF[m][U[cls[U] == c]])) for _, c in CL} for m in RF}}
    for s, f in zip(SL, F):
        N = M[f][:, 0]; e = {"held": float(np.nanmedian(N[U])), "gauged": float(np.nanmedian(N[G])),
                             **{c: float(np.nanmedian(N[U[cls[U] == c]])) for _, c in CL}, "n_class": {c: int((cls[U] == c).sum()) for _, c in CL}}
        if s != "0":
            e["paired_vs_plain"] = boot(N[U] - M["sim.npy"][U, 0]); e["gain_class"] = {c: float(np.nanmedian((N - M["sim.npy"][:, 0])[U[cls[U] == c]])) for _, c in CL}
        e["paired_vs"] = {m: boot(N[U] - RF[m][U]) for m in RF}
        if ds == "us":
            a = U[cls[U] == "arid"]; e["arid_decomp"] = {"r": float(np.nanmedian(M[f][a, 1])), "alpha": float(np.nanmedian(M[f][a, 2])), "abs_bias": float(np.nanmedian(np.abs(M[f][a, 3])))}
            neg = U[M["sim.npy"][U, 0] < 0]; e["failing"] = {"n_still_negative": int((N[neg] < 0).sum()), "median": float(np.nanmedian(N[neg]))}
        o["sigma"][s] = e
    out[ds] = o
    lab = {"gb": "CAMELS-GB", "us": "CAMELS-US"}[ds]
    for s in SL:
        e = o["sigma"][s]; sig = "0 (plain)" if s == "0" else s
        rows.append(f"{lab if s == '0' else ''} & {sig} & {e['held']:.3f} & {e[CL[0][1]]:.3f} & {e[CL[1][1]]:.3f} & {e['gauged']:.3f} \\\\")
    rows.append("\\midrule" if ds == "gb" else "")
hdr = ("\\begin{tabular}{llcccc}\n\\toprule\n & & \\multicolumn{3}{c}{held-out catchments} & gauged \\\\\n"
       "Dataset & $\\sigma$ & all & arid (US) / groundwater-fed (GB) & other & all \\\\\n\\midrule\n")
open(PP + "tables/u_hedge.tex", "w").write(hdr + "\n".join(r for r in rows if r) + "\n\\bottomrule\n\\end{tabular}\n")
json.dump(out, open(PP + "data/u_hedge.json", "w"), indent=1)
for ds in out:
    o = out[ds]; print(f"\n{ds}: held-out n {o['n_held']} (classes {o['sigma']['0']['n_class']}), gauged n {o['n_gauged']}; reference held-out medians {o['ref_held']} classes {o['ref_class']}")
    for s in SL:
        e = o["sigma"][s]; x = f"  sigma {s:4s} held {e['held']:.3f} gauged {e['gauged']:.3f} classes " + " ".join(f"{c} {e[c]:.3f}" for c in e["n_class"])
        if s != "0": x += f" | vs plain {e['paired_vs_plain'][0]:+.3f} [{e['paired_vs_plain'][1]:+.3f}, {e['paired_vs_plain'][2]:+.3f}] improved {e['paired_vs_plain'][3]:.2f} gain {e['gain_class']}"
        x += " | vs " + " ".join(f"{m} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for m, v in e["paired_vs"].items())
        if ds == "us": x += f" | arid r {e['arid_decomp']['r']:.3f} a {e['arid_decomp']['alpha']:.2f} |b| {e['arid_decomp']['abs_bias']:.2f} | failing still <0: {e['failing']['n_still_negative']} median {e['failing']['median']:.2f}"
        print(x)

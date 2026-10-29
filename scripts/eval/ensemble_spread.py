"""Seed averaging (Section 5.2, Table tab:u_spread). Ambiguity decomposition (Krogh & Vedelsby 1995) of the 3-seed ensemble gain, test
2011-2019, gauged catchments: 1 - NSE_ens = mean_k(1 - NSE_k) - A, A = mean_t mean_k (s_k - s_bar)^2 / var(o)  => NSE_ens - mean_k NSE_k
= A exactly (per catchment). Also: where in the flow range the spread sits (share of A on days below the observed 30 % quantile and above
the 95 % quantile). Models CaRT+, stateful LSTM, EA-LSTM. Writes <out>/tables/u_spread.tex.
usage: python scripts/eval/ensemble_spread.py [--runs DIR] [--out DIR]
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalcommon as EC
A_ = EC.parser(__doc__).parse_args(); PP = str(EC.outdirs(A_)) + "/"
ROWS = []
for ds in ("gb", "us"):
    OB = EC.load_obs(ds); dd, q, mask = OB["dd"], OB["q"], OB["mask"]
    T = {"CaRT+": "CaRTp", "LSTM": "LSTM", "EA-LSTM": "EALSTM"}                                 # evalcommon model keys
    per = np.asarray((dd >= "2011-01-01") & (dd <= "2019-12-31"))
    for m, t in T.items():
        S = np.stack([np.maximum(np.load(EC.sim_path(A_, ds, t, k)).astype(float), 0) for k in range(3)])[:, per]
        O = q[per]; A, lo, hi, gain = [], [], [], []
        for c in np.where(mask)[0]:
            o = O[:, c]; k = np.isfinite(o) & np.isfinite(S[:, :, c]).all(0)
            if k.sum() < 365: continue
            s = S[:, k, c]; o = o[k]; v = np.var(o); sb = s.mean(0); sp = ((s - sb) ** 2).mean(0)
            A.append(sp.mean() / v); p30, p95 = np.percentile(o, [30, 95])
            lo.append(sp[o <= p30].sum() / max(sp.sum(), 1e-12)); hi.append(sp[o >= p95].sum() / max(sp.sum(), 1e-12))
            nse_k = [1 - np.mean((s[j] - o) ** 2) / v for j in range(3)]; gain.append((1 - np.mean((sb - o) ** 2) / v) - np.mean(nse_k))
        A, gain = np.array(A), np.array(gain)
        print(f"{ds} {m:8s} median ensemble gain {np.median(gain):+.4f} (= ambiguity A {np.median(A):.4f}); share of spread on days <= obs Q30 {np.median(lo):.2f}, >= obs Q95 {np.median(hi):.2f}")
        ROWS.append((ds, m, np.median(gain), np.median(hi), np.median(lo)))
NAME = {"CaRT+": "\\CaRTp", "LSTM": "stateful LSTM", "EA-LSTM": "EA-LSTM"}
lines = []
for ds, nm in (("gb", "CAMELS-GB"), ("us", "CAMELS-US")):
    for i, (d, m, g, hi, lo) in enumerate([r for r in ROWS if r[0] == ds]):
        lines.append(f"{nm if i == 0 else ''} & {NAME[m]} & ${g:+.4f}$ & {100 * hi:.0f}\\,\\% & {100 * lo:.0f}\\,\\% \\\\")
    lines.append(r"\midrule")
tex = "\\begin{tabular}{llccc}\n\\toprule\nDataset & model & ensemble gain & spread on days $\\ge Q_{95}$ & spread on days $\\le Q_{30}$ \\\\\n\\midrule\n" + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n"
open(PP + "tables/u_spread.tex", "w").write(tex); print(tex)

"""Full Daymet-v4 rebuild of the CAMELS-US 671 record, 1980-01-01..2019-12-31 -> processed/camels_us_daymet671_v4full.npz
(same keys/layout as camels_us_daymet671.npz). All inputs from Daymet v4 basin means (daymet_basin_means.py) on v4's own dates, NO shift
(v4 assigns morning-read station totals to the previous day, Thornton et al. 2021). Leap-year Dec 31 (absent in Daymet) = mean(Dec 29, Dec 30),
the CAMELS v1.2 rule. PET: the Priestley-Taylor formula of CAMELS. q: CAMELS v1.2 for 1980-2014 (copied unchanged), NWIS for 2015-2019 (build_ext2019.py).
Also: CAMELS climate indices recomputed from v4 over the CAMELS
window 1989-10-01..2009-09-30 -> processed/camels_us_clim_v4.npz; and the v4 vs v1.2 comparison over 1980-2014 (v1.2 aligned to v4 timing).
Usage: python -I build_v4full.py <work_dir> <dir with build_v4ovl.py, also receives the log and tables>"""
import os  # data root: env CAMELS_US_ROOT (default ./camels_us)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[2])))  # folder with build_v4ovl.py (helpers); -I keeps everything else out
import numpy as np, pandas as pd
from build_v4ovl import load_v4, pet_all, write_like_old, O, old, ids, N, days as days_old, PROC
WORK, NOTES = Path(sys.argv[1]), Path(sys.argv[2])
log = []
def P_(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.append(s)
cal, V = load_v4(1980, 2019)
nan = {k: int(np.isnan(V[k]).sum()) for k in V}; P_("[v4] NaN basin-days 1980-2019 after leap fill:", nan); assert sum(nan.values()) == 0
V["PET"] = pet_all(cal, V)
Q = np.load(PROC / "camels_us_ext2019_q_qualifiers.npz"); assert (Q["ids"] == ids).all(); dq = pd.DatetimeIndex(Q["days"])
q = np.concatenate([O["q"], Q["q_mm"][dq.get_indexer(cal[cal >= "2015-01-01"])]], 0)
assert len(cal) == 14610 and (cal[:12784] == days_old).all()
series = {k: V[k] for k in ("P", "Tmax", "Tmin", "srad", "vp", "dayl", "PET")}; series["q"] = q
TMP = PROC / "camels_us_daymet671_v4full.tmp.npz"  # written under a temporary name, renamed only after all checks and the clim file
chk = write_like_old(TMP, cal, series)
assert chk["q"][:12784].tobytes() == old["q"].tobytes()
for k in ("static_raw", "static_norm", "static_names", "ids", "lat", "elev", "area_m2"): assert chk[k].tobytes() == old[k].tobytes(), k
P_(f"[write] camels_us_daymet671_v4full (tmp): {chk['days'][0]}..{chk['days'][-1]} ({len(chk['days'])} days); NaN in inputs "
   f"{ {k: int(np.isnan(chk[k]).sum()) for k in ('P','Tmax','Tmin','srad','vp','dayl','PET')} }"
   f" q valid 2015-19 {np.isfinite(chk['q'][12784:]).mean():.3f}")

# ---------------- climate indices (Addor et al. 2017 definitions), window 1989-10-01..2009-09-30
def clim(P, Tx, Tn, PET, dts):
    Tm = 0.5 * (Tx + Tn); t = (dts - pd.Timestamp("1989-10-01")).days.values.astype(float); tau = 365.25
    out = {"p_mean": P.mean(0), "pet_mean": PET.mean(0)}; out["aridity"] = out["pet_mean"] / out["p_mean"]
    out["frac_snow"] = (P * (Tm < 0)).sum(0) / P.sum(0)
    nyr = len(dts) / 365.25; hi = P >= 5 * P.mean(0); lo = P < 1
    def runs(b):  # mean length of consecutive True runs, per basin
        res = np.zeros(b.shape[1])
        for j in range(b.shape[1]):
            x = np.r_[0, b[:, j].astype(int), 0]; d = np.diff(x); s, e = np.where(d == 1)[0], np.where(d == -1)[0]; res[j] = (e - s).mean() if len(s) else 0
        return res
    out["high_prec_freq"] = hi.sum(0) / nyr; out["high_prec_dur"] = runs(hi); out["low_prec_freq"] = lo.sum(0) / nyr; out["low_prec_dur"] = runs(lo)
    # Woods (2009) seasonality: P = Pbar (1 + dp sin(2 pi (t - sp)/tau)), T = Tbar + dT sin(2 pi (t - sT)/tau); index = dp sign(dT) cos(2 pi (sp - sT)/tau)
    X = np.c_[np.ones_like(t), np.sin(2 * np.pi * t / tau), np.cos(2 * np.pi * t / tau)]
    bp = np.linalg.lstsq(X, P, rcond=None)[0]; bt = np.linalg.lstsq(X, Tm, rcond=None)[0]
    ampP = np.hypot(bp[1], bp[2]); dp = ampP / bp[0]; phP = np.arctan2(-bp[2], bp[1]); dT = np.hypot(bt[1], bt[2]); phT = np.arctan2(-bt[2], bt[1])
    # a sin(w t) + b cos(w t) = A sin(w (t - s)) with A = hypot(a, b), -w s = atan2(b, a)  -> phase difference via angles
    out["p_seasonality"] = dp * np.sign(dT) * np.cos(phP - phT)
    return out
NAMES = ["p_mean", "pet_mean", "aridity", "p_seasonality", "frac_snow", "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur"]
w = (cal >= "1989-10-01") & (cal <= "2009-09-30"); wo = days_old.get_indexer(cal[w])
c4 = clim(V["P"][w], V["Tmax"][w], V["Tmin"][w], V["PET"][w], cal[w])
c1 = clim(O["P"][wo].astype(float), O["Tmax"][wo].astype(float), O["Tmin"][wo].astype(float), O["PET"][wo].astype(float), cal[w])
off = pd.read_csv(os.path.join(os.environ.get("CAMELS_US_ROOT", "camels_us"), "raw", "camels_clim.txt"), sep=";", dtype={"gauge_id": str}); off.index = off.gauge_id.str.zfill(8); off = off.reindex(ids)
P_("[clim] code check: same code on v1.2 inputs vs official camels_clim.txt (corr / median ratio); pet_mean differs by construction (CAMELS used its own PET):")
for k in NAMES:
    o = pd.to_numeric(off[k], errors="coerce").values; m = np.isfinite(o)
    P_(f"    {k:15s} v1.2-recomputed vs official: r {np.corrcoef(c1[k][m], o[m])[0,1]:.3f}, median ratio {np.median(c1[k][m] / np.where(o[m] == 0, np.nan, o[m])):.3f} |"
       f" v4 vs v1.2-recomputed: r {np.corrcoef(c4[k], c1[k])[0,1]:.3f}, median diff {np.median(c4[k] - c1[k]):.4f}")
np.savez(PROC / "camels_us_clim_v4.npz", ids=ids, names=np.array(NAMES), clim_v4=np.stack([c4[k] for k in NAMES], 1), clim_v12_recomputed=np.stack([c1[k] for k in NAMES], 1),
         clim_official=np.stack([pd.to_numeric(off[k], errors="coerce").values for k in NAMES], 1), window=np.array(["1989-10-01", "2009-09-30"]),
         note=np.array("Addor et al. 2017 definitions; pet_mean/aridity from the npz Priestley-Taylor PET (CAMELS official uses its own PET); "
                       "Tmean=(Tmax+Tmin)/2; frac_snow = share of P on days with Tmean<0; high: P>=5*mean P; low: P<1 mm; freq in days/yr, dur in days"))

import os; os.replace(TMP, PROC / "camels_us_daymet671_v4full.npz"); P_("[write] renamed ->", PROC / "camels_us_daymet671_v4full.npz")
# ---------------- v4 vs v1.2 over 1980-2014, v1.2 aligned to v4 timing
ov = cal <= "2014-12-31"; leap31 = (cal.month == 12) & (cal.day == 31) & (cal.year % 4 == 0)
okd = ov & ~leap31 & ~np.r_[False, leap31[:-1]] & ~np.r_[leap31[1:], False]; okd[0] = False; okd[12783] = False
sn = list(O["static_names"]); fsnow = O["static_raw"][:, sn.index("frac_snow")]; zmean = O["static_raw"][:, sn.index("elev_mean")]; snow = fsnow >= 0.3
def corr(a, b):
    a = a - a.mean(0); b = b - b.mean(0); return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))
def lead1(a): o = np.full_like(a, np.nan); o[:-1] = a[1:]; return o  # v1.2 value of day t+1 -> compared with v4 day t
R = {k: np.concatenate([O[k].astype(np.float64), np.full((len(cal) - 12784, N), np.nan)], 0) for k in ("P", "Tmax", "Tmin", "srad", "vp", "dayl", "PET")}  # padded to 2019
L = {}
for k in ("P", "Tmax", "srad"):
    L[k] = (corr(V[k][okd], lead1(R[k])[okd]) > corr(V[k][okd], R[k][okd])).astype(int)
Lext = np.load(WORK / "v4_to_v12_correction.npz")
P_("[full overlap] per-basin lag (v1.2 one day late) chosen on 1980-2014:", {k: int(L[k].sum()) for k in L}, "| agreement with 2010-2014 choice:",
   {k: round(float((L[k] == Lext[f'lag_{k}']).mean()), 3) for k in L})
RA = {k: (np.where(L[k][None, :] == 1, lead1(R[k]), R[k]) if k in L else R[k]) for k in R}
rows = []
eras = {"1980-10..1995-09": (cal >= "1980-10-01") & (cal <= "1995-09-30"), "1995-10..2010-09": (cal >= "1995-10-01") & (cal <= "2010-09-30"), "1980-2014": ov}
for k in ("P", "Tmax", "Tmin", "srad", "vp", "dayl", "PET"):
    for en, em in eras.items():
        m = okd & em; a4, a1 = V[k][m], RA[k][m]
        bias = (a4 - a1).mean(0); ratio = a4.sum(0) / a1.sum(0); r = corr(a4, a1); r0 = corr(a4, R[k][m])
        for g, gm in (("all", np.ones(N, bool)), ("frac_snow>=0.3", snow), ("snow & elev>=1500m", snow & (zmean >= 1500))):
            rows.append(dict(var=k, period=en, group=g, n=int(gm.sum()), bias_median=np.median(bias[gm]), bias_p10=np.percentile(bias[gm], 10), bias_p90=np.percentile(bias[gm], 90),
                             ratio_median=np.median(ratio[gm]), ratio_p10=np.percentile(ratio[gm], 10), ratio_p90=np.percentile(ratio[gm], 90),
                             r_median_aligned=np.median(r[gm]), r_p10_aligned=np.percentile(r[gm], 10), r_median_unaligned=np.median(r0[gm])))
ST = pd.DataFrame(rows); ST.to_csv(NOTES / "v4_vs_v12_1980_2014.csv", index=False, float_format="%.4f")
P_(ST[ST.period == "1980-2014"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
P_(ST[(ST.period != "1980-2014") & ST.var.isin(["P", "Tmax", "Tmin", "PET"])].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
# temperature drift between the Feng train (1980-10..1995-09) and test (1995-10..2010-09) windows
a, b = eras["1980-10..1995-09"], eras["1995-10..2010-09"]
for k in ("Tmax", "Tmin", "Tmean", "P", "PET"):
    get = (lambda S: 0.5 * (S["Tmax"] + S["Tmin"])) if k == "Tmean" else (lambda S: S[k])
    d4 = get(V)[b].mean(0) - get(V)[a].mean(0); d1 = get(R)[b].mean(0) - get(R)[a].mean(0)
    P_(f"[drift {k}] mean(1995-10..2010-09) - mean(1980-10..1995-09), median over basins: snow (n={snow.sum()}) v1.2 {np.median(d1[snow]):+.3f}  v4 {np.median(d4[snow]):+.3f} |"
       f" snow & >=1500 m v1.2 {np.median(d1[snow & (zmean >= 1500)]):+.3f} v4 {np.median(d4[snow & (zmean >= 1500)]):+.3f} | all v1.2 {np.median(d1):+.3f} v4 {np.median(d4):+.3f}")
yb = pd.DataFrame(V["Tmin"][okd] - RA["Tmin"][okd], index=cal[okd]).groupby(cal[okd].year).mean()
yx = pd.DataFrame(V["Tmax"][okd] - RA["Tmax"][okd], index=cal[okd]).groupby(cal[okd].year).mean()
P_("[yearly v4 - v1.2, median over snow basins] Tmax:", {int(y): round(float(np.median(yx.loc[y][snow])), 2) for y in yx.index})
P_("[yearly v4 - v1.2, median over snow basins] Tmin:", {int(y): round(float(np.median(yb.loc[y][snow])), 2) for y in yb.index})
(NOTES / "build_v4full.log").write_text("\n".join(log) + "\n")
print("done")

"""Extend camels_us_daymet671.npz to 2019-12-31 with Daymet v4 basin means (daymet_basin_means.py) and USGS NWIS daily discharge.
Stages (all in this file, run with python -I from outside the download folder):
  1. calendar: determine how CAMELS v1.2 filled Dec 31 of leap years (Daymet 365-day calendar) and apply the same rule to v4;
  2. flow: parse NWIS RDB (00060 daily mean, cfs), rule = numeric value >= 0 kept (incl. estimated 'e' and provisional 'P'),
     non-numeric (Ice, Eqp, Ssn, Bkw, Dis, Rat, Mnt, ***, ...) or negative -> NaN; cfs -> mm/d exactly as camels_us_prep671.py (forcing-header area);
  3. overlap 2010-2014 (inputs) and 2010-10..2014-12 (flow): v4 vs CAMELS v1.2 basin means, PET; flow NWIS vs CAMELS;
  4. per-basin, per-variable, per-calendar-month correction from the 2010-2014 overlap (inputs only) + split-sample check (fit 2010-12, test 2013-14);
  5. write camels_us_daymet671_ext2019.npz (adjusted) and camels_us_daymet671_ext2019_rawv4.npz; first 12784 days copied unchanged (asserted).
No model and no skill score on 2015-2019 observed flow is computed here (flow QC = coverage / units / consistency only).
Usage: python -I build_ext2019.py <nwis_dir> <work_dir> <log_dir>"""
import os  # data root: env CAMELS_US_ROOT (default ./camels_us)
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
NWIS, WORK, NOTES = map(Path, sys.argv[1:4])
PROC = (Path(os.environ.get("CAMELS_US_ROOT", "camels_us")) / "processed")
old = np.load(PROC / "camels_us_daymet671.npz"); O = {k: old[k] for k in old.files}
ids, area, lat, elev = O["ids"], O["area_m2"], O["lat"], O["elev"]; N = len(ids)
days_old = pd.DatetimeIndex(O["days"]); assert days_old[0] == pd.Timestamp("1980-01-01") and days_old[-1] == pd.Timestamp("2014-12-31") and len(days_old) == 12784
sn = list(O["static_names"]); fsnow = O["static_raw"][:, sn.index("frac_snow")]; zmean = O["static_raw"][:, sn.index("elev_mean")]
log = []
def P_(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.append(s)
VMAP = {"prcp": "P", "tmax": "Tmax", "tmin": "Tmin", "srad": "srad", "vp": "vp", "dayl": "dayl"}

# ---------------- PET: identical formula to camels_us_prep671.py
SIGMA, LAM, ALPHA, ALB = 4.903e-9, 2.45, 1.26, 0.23
def pet(days, tx, tn, vp, srad, dayl, lat_j, elev_j):
    tx, tn = tx.astype(np.float64), tn.astype(np.float64)  # prep script computed from float64 arrays read from text
    ea = vp.astype(np.float64) / 1000.0; tm = 0.5 * (tx + tn); doy = days.dayofyear.values
    phi = np.deg2rad(lat_j); dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365); dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    Ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    Rs = srad.astype(np.float64) * dayl.astype(np.float64) / 1e6; Rso = (0.75 + 2e-5 * elev_j) * Ra
    Rnl = SIGMA * ((tx + 273.16) ** 4 + (tn + 273.16) ** 4) / 2 * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) * (1.35 * np.clip(Rs / np.maximum(Rso, 1e-6), 0, 1) - 0.35)
    Rn = (1 - ALB) * Rs - Rnl
    delta = 4098 * 0.6108 * np.exp(17.27 * tm / (tm + 237.3)) / (tm + 237.3) ** 2; gamma = 0.665e-3 * 101.3 * ((293 - 0.0065 * elev_j) / 293) ** 5.26
    return np.maximum(ALPHA * delta / (delta + gamma) * Rn / LAM, 0.0)
def pet_all(days, F):
    return np.stack([pet(days, F["Tmax"][:, j], F["Tmin"][:, j], F["vp"][:, j], F["srad"][:, j], F["dayl"][:, j], lat[j], elev[j]) for j in range(N)], 1)
# sanity: the formula reproduces the stored PET on 1980-2014 (float32 inputs -> tiny differences only)
pchk = pet_all(days_old, O)
P_(f"[PET] recomputed from stored float32 inputs vs stored PET 1980-2014: max|diff| {np.nanmax(np.abs(pchk - O['PET'])):.2e} mm/d")

# ---------------- 1. leap-year Dec 31 rule in CAMELS v1.2
leapY = [y for y in range(1980, 2015) if y % 4 == 0]
res = {}
for v in ["prcp", "tmax", "tmin", "srad", "vp", "dayl"]:
    a = O[VMAP[v]]; e1, e2, e3 = [], [], []
    for y in leapY:
        i31 = days_old.get_loc(pd.Timestamp(f"{y}-12-31")); d29, d30, d31 = a[i31 - 2], a[i31 - 1], a[i31]
        e1.append(np.abs(d31 - 0.5 * (d29 + d30))); e2.append(np.abs(d31 - d30))
        if i31 + 1 < len(a): e3.append(np.abs(d31 - 0.5 * (d30 + a[i31 + 1])))
    res[v] = (np.nanmax(e1), np.nanmax(e2), np.nanmax(e3))
P_("[calendar] CAMELS v1.2 Dec 31 of leap years, max|diff| over 671 basins x", len(leapY), "leap years vs (mean(Dec29,Dec30), Dec30, mean(Dec30,Jan1)):",
   {k: tuple(round(float(x), 3) for x in t) for k, t in res.items()})

# ---------------- Daymet v4 basin means -> calendar days 2010-01-01..2019-12-31
D = np.load(WORK / "daymet_v4_basinmean_2010_2019.npz"); assert (D["ids"] == ids).all()
ddates = pd.DatetimeIndex(D["dates"]); cal = pd.date_range("2010-01-01", "2019-12-31", freq="D")
V4 = {}
for v, k in VMAP.items():
    s = pd.DataFrame(D[v], index=ddates).reindex(cal).values.astype(np.float64)
    for y in [2012, 2016]:
        i = cal.get_loc(pd.Timestamp(f"{y}-12-31")); assert np.isnan(s[i]).all(); s[i] = 0.5 * (s[i - 2] + s[i - 1])
    V4[k] = s
    P_(f"[v4] {v}: NaN basin-days after leap fill {int(np.isnan(s).sum())}")
nanf = {v: np.load(WORK / f"nanfrac_{v}_2010_2019.npy") for v in VMAP}
P_("[v4] basins with >1% of weight on NaN (masked) Daymet cells, max over years:", {v: int((nanf[v].max(0) > 0.01).sum()) for v in VMAP},
   "max frac", {v: round(float(nanf[v].max()), 3) for v in VMAP})
wi = np.load(WORK / "daymet_weights.npz"); inf = wi["info"]
P_(f"[weights] cells/basin min {inf[:,0].min():.0f} (basin {ids[inf[:,0].argmin()]}, {inf[inf[:,0].argmin(),4]:.1f} km2) median {np.median(inf[:,0]):.0f} max {inf[:,0].max():.0f};"
   f" basins with <10 cells {(inf[:,0] < 10).sum()}; with 0 fully-covered cells {(inf[:,5] == 0).sum()};"
   f" shapefile AREA vs geodesic polygon area max rel diff {np.max(np.abs(inf[:,4]/inf[:,3]-1)):.4f};"
   f" header area vs shapefile AREA median ratio {np.median(area/1e6/inf[:,4]):.3f}")

# ---------------- 2. NWIS flow
def parse_rdb(path):
    txt = path.read_text(errors="replace"); blocks = re.split(r"\n(?=agency_cd\t)", txt)
    best = None
    for b in blocks[1:]:
        lines = [l for l in b.splitlines() if l and not l.startswith("#")]
        hdr = lines[0].split("\t"); rows = [l.split("\t") for l in lines[2:]]
        vcols = [c for c in hdr if re.fullmatch(r"\d+_00060_00003", c)]
        for c in vcols:
            iv, ic = hdr.index(c), hdr.index(c + "_cd")
            df = pd.DataFrame({"date": [r[2] for r in rows], "val": [r[iv] if len(r) > iv else "" for r in rows], "cd": [r[ic] if len(r) > ic else "" for r in rows]})
            num = pd.to_numeric(df.val, errors="coerce"); n = int(num.notna().sum())
            if best is None or n > best[0]: best = (n, c, df, len(vcols) + (0 if best is None else best[3]))
    return best
cal_q = pd.date_range("2010-10-01", "2019-12-31", freq="D")
Qcfs = np.full((len(cal_q), N), np.nan); QCD = np.full((len(cal_q), N), "", dtype="<U12"); QRAW = np.full((len(cal_q), N), "", dtype="<U12")
nser, nofile = {}, []
for j, b in enumerate(ids):
    f = NWIS / f"{b}.rdb"
    if not f.exists(): nofile.append(b); continue
    r = parse_rdb(f)
    if r is None: nofile.append(b); continue
    n, c, df, ns = r; nser[b] = ns
    df = df.drop_duplicates("date").set_index(pd.DatetimeIndex(df.drop_duplicates("date").date)).reindex(cal_q)
    num = pd.to_numeric(df.val, errors="coerce")
    Qcfs[:, j] = num.where(num >= 0).values; QCD[:, j] = df.cd.fillna("").values; QRAW[:, j] = df.val.fillna("").values
P_(f"[flow] sites without data file/series: {len(nofile)} {nofile[:20]}; sites with >1 daily-mean 00060 series (kept the one with most values): {[k for k, v in nser.items() if v > 1]}")
nonnum = pd.Series(QRAW[(QRAW != "") & ~np.char.isnumeric(np.char.replace(np.char.replace(QRAW, ".", ""), "-", ""))]).value_counts()
P_("[flow] non-numeric value codes set to NaN:", dict(nonnum)); P_("[flow] negative values set to NaN:", int((pd.to_numeric(pd.Series(QRAW.ravel()), errors="coerce") < 0).sum()))
P_("[flow] qualifier counts (2010-10..2019-12):", dict(pd.Series(QCD[QCD != ""].ravel()).value_counts().head(15)))
Qmm = 28316846.592 * Qcfs * 86400 / (area[None, :] * 1e6)
# flow consistency 2010-10-01..2014-12-31 vs CAMELS v1.2 (stored q)
ov = cal_q <= "2014-12-31"; io_ = days_old.get_indexer(cal_q[ov]); qo = O["q"][io_].astype(np.float64); qn = Qmm[ov]
both = np.isfinite(qo) & np.isfinite(qn); rel = np.abs(qn - qo) / np.maximum(np.abs(qo), 1e-3)
same = both & ((rel <= 0.005) | (np.abs(qn - qo) <= 2e-3))
P_(f"[flow overlap 2010-10..2014-12] basin-days both valid {both.sum()}, identical within 0.5% {same.sum() / both.sum():.4f};"
   f" valid only in CAMELS {int((np.isfinite(qo) & ~np.isfinite(qn)).sum())}, only in NWIS {int((~np.isfinite(qo) & np.isfinite(qn)).sum())};"
   f" basins with >1% differing days {(((both & ~same).sum(0) / np.maximum(both.sum(0), 1)) > 0.01).sum()};"
   f" volume ratio NWIS/CAMELS on common days median {np.nanmedian(np.nansum(np.where(both, qn, 0), 0) / np.nansum(np.where(both, qo, 0), 0)):.4f}")
vr = np.nansum(np.where(both, qn, 0), 0) / np.maximum(np.nansum(np.where(both, qo, 0), 0), 1e-9)
bad = np.where((np.abs(vr - 1) > 0.02) & (both.sum(0) > 100))[0]
P_("[flow overlap] basins with |volume ratio - 1| > 2%:", [(ids[j], round(float(vr[j]), 3), int(both[:, j].sum())) for j in bad])
cov = pd.DataFrame(np.isfinite(Qmm), index=cal_q).groupby(cal_q.year).mean().mean(1)
P_("[flow] NWIS coverage (fraction of basin-days valid) by year:", {int(k): round(float(v), 3) for k, v in cov.items()})
ev = cal_q >= "2015-01-01"; nval = np.isfinite(Qmm[ev]).sum(0)
P_(f"[flow] 2015-2019: basins with >=365 valid days {(nval >= 365).sum()}, with full record (1826 d) {(nval == ev.sum()).sum()}, with 0 valid days {(nval == 0).sum()} {list(ids[nval == 0])}")
prov = np.char.startswith(QCD[ev], "P"); P_(f"[flow] 2015-2019 valid values flagged provisional (P*): {(prov & np.isfinite(Qmm[ev])).sum() / np.isfinite(Qmm[ev]).sum():.4f}")

# ---------------- 3. overlap check of inputs 2010-01-02..2014-12-31 (excl. leap Dec 31 filled in v1.2 and the following Jan 1)
def corr(a, b):
    a = a - a.mean(0); b = b - b.mean(0); return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))
leap31 = (cal.month == 12) & (cal.day == 31) & (cal.year % 4 == 0); afterleap = np.r_[False, leap31[:-1]]
OV = (cal <= "2014-12-31") & (cal > "2010-01-01") & ~leap31 & ~afterleap
def ref(mask, k): return O[k][days_old.get_indexer(cal[mask])].astype(np.float64)
def shift1(a): out = np.full_like(a, np.nan); out[1:] = a[:-1]; return out
SHIFTABLE = ["P", "Tmax", "srad"]   # Daymet v4 re-assigned prcp/tmax to the previous day for morning-observing stations (time-of-observation)
def choose_lag(mask):
    L = {}
    for k in SHIFTABLE:
        a1 = ref(mask, k); L[k] = (corr(shift1(V4[k])[mask], a1) > corr(V4[k][mask], a1)).astype(int)
    return L
def align(L):
    return {k: (np.where(L[k][None, :] == 1, shift1(V4[k]), V4[k]) if k in L else V4[k].copy()) for k in V4}
for k in ["P", "Tmax", "Tmin", "srad", "vp", "dayl"]:
    a1 = ref(OV, k); a4 = V4[k]; lagr = [np.median(corr(a4[np.r_[False, OV[:-1]]], a1)), np.median(corr(a4[OV], a1)), np.median(corr(shift1(a4)[OV], a1))]
    P_(f"[timing {k}] median daily corr v1.2(t) vs v4(t+1) / v4(t) / v4(t-1): {np.round(lagr, 4)}")
y_a = OV & (cal.year <= 2012); y_b = OV & (cal.year >= 2013)
La, Lb, L = choose_lag(y_a), choose_lag(y_b), choose_lag(OV)
for k in SHIFTABLE:
    P_(f"[timing {k}] basins best aligned with v4(t-1): {L[k].sum()}/{N}; lag choice 2010-12 vs 2013-14 agrees for {(La[k] == Lb[k]).mean():.3f} of basins;"
       f" snow basins with lag 1: {L[k][fsnow >= 0.3].sum()}/{(fsnow >= 0.3).sum()}")
VA = align(L)
grp = {"all": np.ones(N, bool), "frac_snow>=0.3": fsnow >= 0.3, "elev_mean>=1500m": zmean >= 1500, "snow & >=1500m": (fsnow >= 0.3) & (zmean >= 1500)}
rows = []
for ver, VV in [("raw_v4", V4), ("time_aligned_v4", VA)]:
    for k in ["P", "Tmax", "Tmin", "srad", "vp", "dayl"]:
        a1 = ref(OV, k); a4 = VV[k][OV]
        bias = (a4 - a1).mean(0); r = corr(a4, a1); ratio = a4.sum(0) / a1.sum(0); rmse = np.sqrt(((a4 - a1) ** 2).mean(0))
        for g, m in grp.items():
            rows.append(dict(version=ver, var=k, group=g, n=int(m.sum()), bias_median=np.median(bias[m]), bias_p10=np.percentile(bias[m], 10), bias_p90=np.percentile(bias[m], 90),
                             absbias_max=np.abs(bias[m]).max(), ratio_median=np.median(ratio[m]), ratio_p10=np.percentile(ratio[m], 10), ratio_p90=np.percentile(ratio[m], 90),
                             r_median=np.median(r[m]), r_p10=np.percentile(r[m], 10), r_min=r[m].min(), rmse_median=np.median(rmse[m])))
        if ver == "raw_v4":
            by = pd.DataFrame(a4 - a1, index=cal[OV]).groupby(cal[OV].year).mean()
            P_(f"[overlap {k}] median-over-basins bias v4-v1.2 by year (all / snow):", {int(y): (round(float(np.median(by.loc[y])), 3), round(float(np.median(by.loc[y][grp['frac_snow>=0.3']])), 3)) for y in by.index})
            if k in ("Tmax", "Tmin"):
                bm = pd.DataFrame(a4 - a1, index=cal[OV]).groupby(cal[OV].month).mean(); sg = grp["snow & >=1500m"]
                P_(f"[overlap {k}] median bias by month, snow & >=1500 m basins:", {int(mm): round(float(np.median(bm.loc[mm][sg])), 2) for mm in bm.index})
                worst = np.argsort(-np.abs(bias))[:5]; P_(f"[overlap {k}] largest |bias| basins:", [(ids[j], round(float(bias[j]), 2), round(float(zmean[j])), round(float(fsnow[j]), 2)) for j in worst])
ST = pd.DataFrame(rows); ST.to_csv(NOTES / "overlap_inputs_2010_2014.csv", index=False, float_format="%.4f")
P_(ST.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
PET4 = pet_all(cal, V4); PETA = pet_all(cal, VA); p1 = ref(OV, "PET")
for nm, PP in [("raw v4", PET4), ("time-aligned v4", PETA)]:
    p4 = PP[OV]; rt = p4.sum(0) / p1.sum(0)
    P_(f"[overlap PET {nm} vs npz] bias median {np.median((p4 - p1).mean(0)):.3f} mm/d, ratio median {np.median(rt):.3f} (p10 {np.percentile(rt,10):.3f}, p90 {np.percentile(rt,90):.3f}), corr median {np.median(corr(p4, p1)):.4f}")

# ---------------- 4. per-basin x variable x calendar-month correction on the time-aligned series (inputs only, 2010-2014)
ADD = ["Tmax", "Tmin"]; MUL = ["P", "srad", "vp"]
def fit(mask, src):
    mon = cal[mask].month.values; C = {}
    for k in ADD + MUL:
        a1 = ref(mask, k); a4 = src[k][mask]; c = np.zeros((12, N))
        for mm in range(1, 13):
            if k in ADD:
                s_ = mon == mm; c[mm - 1] = (a1[s_] - a4[s_]).mean(0)
            else:  # ratio of sums pooled over months m-1, m, m+1 (stabilises dry months); guarded
                s_ = np.isin(mon, [(mm - 2) % 12 + 1, mm, mm % 12 + 1]); num, den = a1[s_].sum(0), a4[s_].sum(0)
                c[mm - 1] = np.where(den > (5.0 if k == "P" else 1e-6), num / np.maximum(den, 1e-12), 1.0)
        if k in MUL: c = np.clip(c, 0.5, 2.0)
        C[k] = c
    return C
def apply(C, src, dates, use):
    out = {k: src[k].copy() for k in src}; mi = dates.month.values - 1
    for k in use: out[k] = src[k] + C[k][mi] if k in ADD else src[k] * C[k][mi]
    return out
# split-sample: lags + factors fitted on 2010-2012, evaluated on 2013-2014
VAa = align(La); Ca = fit(y_a, VAa); Fb = apply(Ca, {k: VAa[k] for k in VAa}, cal, ADD + MUL)
USE = []
for k in ADD + MUL:
    a1 = ref(y_b, k); raw, ali, adj = V4[k][y_b], VAa[k][y_b], Fb[k][y_b]; sg = grp["frac_snow>=0.3"]
    if k in ADD: f = lambda x, m=slice(None): np.median(np.abs((x - a1).mean(0))[m])
    else: f = lambda x, m=slice(None): np.median(np.abs(x.sum(0) / a1.sum(0) - 1)[m])
    rm = lambda x: np.median(np.sqrt(((x - a1) ** 2).mean(0)))
    if f(adj) < 0.8 * f(ali) and rm(adj) <= 1.01 * rm(ali): USE.append(k)
    P_(f"[split-sample 2013-14 | fit 2010-12] {k}: median |{'bias' if k in ADD else 'ratio-1'}| raw {f(raw):.3f} / aligned {f(ali):.3f} / aligned+monthly {f(adj):.3f};"
       f" snow {f(raw, sg):.3f} / {f(ali, sg):.3f} / {f(adj, sg):.3f}; daily RMSE median {rm(raw):.3f} / {rm(ali):.3f} / {rm(adj):.3f}")
P_("[correction] monthly factors kept (split-sample |bias| reduced >20% without RMSE increase):", USE)
pt1 = ref(y_b, "PET"); Fb2 = apply(Ca, {k: VAa[k] for k in VAa}, cal, USE)
for nm, PP in [("raw", PET4[y_b]), ("aligned", pet_all(cal, VAa)[y_b]), ("aligned+kept monthly", pet_all(cal, Fb2)[y_b])]:
    P_(f"[split-sample PET 2013-14] {nm}: median |ratio-1| {np.median(np.abs(PP.sum(0) / pt1.sum(0) - 1)):.4f}, median corr {np.median(corr(PP, pt1)):.4f}")
for nm, PP in [("raw", V4["P"][y_b]), ("aligned", VAa["P"][y_b]), ("aligned+kept", Fb2["P"][y_b])]:
    P_(f"[split-sample P 2013-14] {nm}: median daily corr {np.median(corr(PP, ref(y_b, 'P'))):.4f}")
C = fit(OV, VA)
np.savez(WORK / "v4_to_v12_correction.npz", ids=ids, **{f"lag_{k}": L[k] for k in L}, **{f"{k}_{'add' if k in ADD else 'mul'}": C[k] for k in C}, applied=np.array(USE),
         note=np.array("lag_*: 1 = v1.2-consistent value at date t is v4 at t-1; factors per calendar month (rows Jan..Dec) x basin on the time-aligned series; "
                       "Tmax/Tmin additive (v1.2-v4), P/srad/vp multiplicative (pooled m-1..m+1 sums, clipped 0.5..2); fit 2010-2014; only 'applied' are used"))
for k in C: P_(f"[correction {k}] {'additive' if k in ADD else 'multiplicative'} (applied={k in USE}): median {np.median(C[k]):.3f}, p5 {np.percentile(C[k],5):.3f}, p95 {np.percentile(C[k],95):.3f}")
pd.DataFrame({"gauge_id": ids, **{f"lag_{k}": L[k] for k in L}, **{f"{k}_m{mm+1:02d}": C[k][mm] for k in C for mm in range(12)}}).to_csv(NOTES / "v4_to_v12_correction.csv", index=False, float_format="%.4f")
VADJ = apply(C, VA, cal, USE)

# ---------------- 5. write extended npz files
days_new = pd.date_range("1980-01-01", "2019-12-31", freq="D"); T = len(days_new); ext = cal >= "2015-01-01"; dx = cal[ext]
assert (days_new[:12784] == days_old).all() and (days_new[12784:] == dx).all()
qext = Qmm[cal_q >= "2015-01-01"]
for tag, fname in [("adj", "camels_us_daymet671_ext2019.npz"), ("raw", "camels_us_daymet671_ext2019_rawv4.npz")]:
    src = {k: (VADJ if tag == "adj" else V4)[k][ext] for k in VMAP.values()}
    src["PET"] = pet_all(dx, src); src["q"] = qext
    out = {}
    for k in O:
        if k == "days": out[k] = np.array([d.strftime("%Y-%m-%d") for d in days_new])
        elif k in ("P", "Tmax", "Tmin", "srad", "vp", "dayl", "PET", "q"):
            out[k] = np.concatenate([O[k], src[k].astype(np.float32)], 0)
        else: out[k] = O[k]
    np.savez_compressed(PROC / fname, **out)
    chk = np.load(PROC / fname); assert chk.files == old.files, (chk.files, old.files)
    for k in old.files:
        a, b = chk[k], old[k]
        if a.ndim and a.shape[0] == T and k != "days": a = a[:12784]
        if k == "days": a = a[:12784]
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), k
    nanin = {k: int(np.isnan(chk[k][12784:]).sum()) for k in ("P", "Tmax", "Tmin", "srad", "vp", "dayl", "PET")}
    P_(f"[write] {fname}: days {chk['days'][0]}..{chk['days'][-1]} ({T}); NaN in 2015-2019 inputs {nanin};"
       f" q valid fraction 2015-2019 {np.isfinite(chk['q'][12784:]).mean():.3f}; 2015-19 means P {np.nanmean(chk['P'][12784:]):.3f} PET {np.nanmean(chk['PET'][12784:]):.3f} Tmax {np.nanmean(chk['Tmax'][12784:]):.2f}")
np.savez_compressed(PROC / "camels_us_ext2019_q_qualifiers.npz", ids=ids, days=np.array([d.strftime("%Y-%m-%d") for d in cal_q]), q_mm=Qmm.astype(np.float32),
                    q_cfs=Qcfs.astype(np.float32), qualifier=QCD, raw_value=QRAW)
(NOTES / "build_ext2019.log").write_text("\n".join(log) + "\n")
print("done")

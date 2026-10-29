"""Version-swap test files: camels_us_daymet671.npz layout (1980-01-01..2014-12-31, same keys), identical except that the inputs
P, Tmax, Tmin, srad, vp, dayl (and PET, recomputed with the identical Priestley-Taylor formula) for 2010-01-01..2014-12-31 are Daymet v4:
  camels_us_daymet671_v4ovl_raw.npz         raw v4 on its own dates (no shift)
  camels_us_daymet671_v4ovl_aligned.npz     v4 time-aligned to v1.2 with the per-basin lags of the ext2019 file (lag 1: value at t = v4 at t-1)
  camels_us_daymet671_v4ovl_aligned_adj.npz aligned + the monthly factors that were applied in the ext2019 file (srad only)
q and statics unchanged. Daymet leap-year Dec 31 filled as mean(Dec 29, Dec 30) (CAMELS v1.2 rule), before any shift.
Usage: python -I build_v4ovl.py <work_dir>"""
import os  # data root: env CAMELS_US_ROOT (default ./camels_us)
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
WORK = Path(sys.argv[1]); PROC = (Path(os.environ.get("CAMELS_US_ROOT", "camels_us")) / "processed")
old = np.load(PROC / "camels_us_daymet671.npz"); O = {k: old[k] for k in old.files}; ids = O["ids"]; N = len(ids)
days = pd.DatetimeIndex(O["days"]); lat, elev = O["lat"], O["elev"]
VMAP = {"prcp": "P", "tmax": "Tmax", "tmin": "Tmin", "srad": "srad", "vp": "vp", "dayl": "dayl"}
def load_v4(y0, y1):
    """calendar-day v4 basin means for y0..y1 from all daymet_v4_basinmean_*.npz in WORK (+ per-year caches); leap Dec 31 = mean(Dec 29, Dec 30)."""
    cal = pd.date_range(f"{y0}-01-01", f"{y1}-12-31"); out = {k: np.full((len(cal), N), np.nan) for k in VMAP.values()}
    srcs = [np.load(f) for f in sorted(glob.glob(str(WORK / "daymet_v4_basinmean_*.npz")))]
    for s in srcs:
        assert (s["ids"] == ids).all(); d = pd.DatetimeIndex(s["dates"]); m = (d >= cal[0]) & (d <= cal[-1]); i = cal.get_indexer(d[m])
        for v, k in VMAP.items(): out[k][i] = s[v][m]
    for f in sorted(glob.glob(str(WORK / "years" / "bm_*_*.npz"))):
        v, Y = Path(f).stem.split("_")[1], int(Path(f).stem.split("_")[2])
        if y0 <= Y <= y1:
            d = pd.date_range(f"{Y}-01-01", periods=365); out[VMAP[v]][cal.get_indexer(d)] = np.load(f)["bm"]
    for Y in range(y0, y1 + 1):
        if Y % 4 == 0:
            i = cal.get_loc(pd.Timestamp(f"{Y}-12-31"))
            for k in out: out[k][i] = 0.5 * (out[k][i - 2] + out[k][i - 1])
    return cal, out
SIGMA, LAM, ALPHA, ALB = 4.903e-9, 2.45, 1.26, 0.23
def pet(days, tx, tn, vp, srad, dayl, lat_j, elev_j):  # identical to camels_us_prep671.py
    tx, tn = tx.astype(np.float64), tn.astype(np.float64)
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
def write_like_old(path, days_new, series, keep_q=None):
    """series: dict of full-length arrays for P..PET (and optionally q); everything else copied from the old npz."""
    out = {}
    for k in old.files:
        if k == "days": out[k] = np.array([d.strftime("%Y-%m-%d") for d in days_new])
        elif k in series: out[k] = series[k].astype(np.float32)
        else: out[k] = O[k]
    np.savez_compressed(path, **out)
    chk = np.load(path); assert chk.files == old.files
    return chk
if __name__ == "__main__":
    C = np.load(WORK / "v4_to_v12_correction.npz"); applied = list(C["applied"]); lags = {k: C[f"lag_{k}"] for k in ("P", "Tmax", "srad")}
    print("lags (n basins with lag 1):", {k: int(v.sum()) for k, v in lags.items()}, "| monthly factors applied:", applied)
    cal, V = load_v4(2009, 2014)
    assert not any(np.isnan(V[k]).any() for k in V), "missing v4 days in 2009-2014"
    def shift1(a): o = np.full_like(a, np.nan); o[1:] = a[:-1]; return o
    VA = {k: (np.where(lags[k][None, :] == 1, shift1(V[k]), V[k]) if k in lags else V[k].copy()) for k in V}
    VJ = {k: VA[k].copy() for k in VA}
    for k in applied:
        f = C[f"{k}_add"] if k in ("Tmax", "Tmin") else C[f"{k}_mul"]; mi = cal.month.values - 1
        VJ[k] = VA[k] + f[mi] if k in ("Tmax", "Tmin") else VA[k] * f[mi]
    sel = cal >= "2010-01-01"; dsel = cal[sel]; i0 = days.get_loc(pd.Timestamp("2010-01-01")); assert (days[i0:] == dsel).all()
    for tag, src in [("raw", V), ("aligned", VA), ("aligned_adj", VJ)]:
        F = {k: src[k][sel] for k in src}; F["PET"] = pet_all(dsel, F)
        series = {k: np.concatenate([O[k][:i0], F[k].astype(np.float32)], 0) for k in ("P", "Tmax", "Tmin", "srad", "vp", "dayl", "PET")}
        chk = write_like_old(PROC / f"camels_us_daymet671_v4ovl_{tag}.npz", days, series)
        for k in old.files:
            a, b = chk[k], old[k]
            if k in series: a, b = a[:i0], b[:i0]
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), k
        nan = {k: int(np.isnan(chk[k][i0:]).sum()) for k in series}
        dif = {k: round(float(np.mean(chk[k][i0:] - old[k][i0:])), 3) for k in series}
        print(f"wrote camels_us_daymet671_v4ovl_{tag}.npz: days {chk['days'][0]}..{chk['days'][-1]} ({len(chk['days'])}); <2010 + q + statics bit-identical: OK;"
              f" NaN 2010-14 {nan}; mean(new-old) 2010-14 {dif}")

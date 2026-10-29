"""Convert CAMELS-AUS v2 (Fowler, Zhang & Hou 2025, ESSD 17, 4079; Zenodo 10.5281/zenodo.12575680, record 14289037 = v2.03, CC-BY 4.0)
into the layout of camels_us_daymet671_v4full.npz, read by cart.data.load_aus.

    python -I scripts/data/aus/build_camels_aus_v2.py <raw/ex dir> <master table csv> <out dir>

<raw/ex dir> is the folder into which 03_streamflow.zip and 05_hydrometeorology.zip of the Zenodo record are unzipped
(streamflow_mmd.csv at its top level, 05_hydrometeorology/... below it); <master table csv> is CAMELS_AUS_Attributes&Indices_MasterTable.csv
of the same record (scripts/data/README.md).

Outputs in <out dir>:
  camels_aus_v2.npz       ids (N,) <U8 = "9" + 7-digit running index (sorted by station id), station_ids (N,) original ids,
                          days 1980-01-01..2019-12-31, P (AGCD), PET (SILO Morton wet-environment areal PET), Tmax/Tmin (AGCD),
                          q (streamflow_mmd, NaN missing), static_raw/static_norm/static_names (27), lat (centroid), elev (elev_mean),
                          area_m2; extras: P_silo (SILO precipitation, the P of Zhang et al. 2026), lon, state, station_names
  elev_bands_aus.npz      5 equal-area bands from a truncated-normal hypsometry on [elev_min, elev_max] with mean elev_mean
  aus_ungauged_10pct.txt  10 % stratified random holdout (terciles aridity x elev_mean, round(10 %) per stratum, default_rng(0)),
                          excluding the three study catchments 406213, 406214, 407230
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import truncnorm

RAW, MASTER, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
HM = os.path.join(RAW, "05_hydrometeorology")
D0, D1 = "1980-01-01", "2019-12-31"
W0, W1 = "1989-01-01", "2008-12-31"                     # streamflow-availability window
MIN_VALID_DAYS = 365                                      # catchments need >= 1 year of q in W0..W1
STUDY = ["406213", "406214", "407230"]
SIG_RATIO = 0.17                                          # hypsometric sd / elevation range (median of CAMELS-US elev_bands671)
os.makedirs(OUT, exist_ok=True)


def read_daily(path, stations, missing_below=None):
    d = pd.read_csv(path)
    t = pd.to_datetime(d[["year", "month", "day"]])
    d.index = t
    d = d.drop(columns=["year", "month", "day"])
    d.columns = [str(c) for c in d.columns]
    full = pd.date_range(D0, D1, freq="D")
    x = d.reindex(full)[stations].to_numpy(np.float64)
    if missing_below is not None:
        x[x < missing_below] = np.nan
    return x


# ---- metadata and attributes
m = pd.read_csv(MASTER, dtype={"station_id": str}).set_index("station_id")
m = m.replace(-99.0, np.nan)                             # -99 sentinels (claya / clayb / sanda)
stations_all = sorted(m.index)
days = pd.date_range(D0, D1, freq="D")
T = len(days)

# ---- streamflow (mm/d, CAMELS-AUS v2 adopted areas), select catchments with q in 1989-2008
q_all = read_daily(os.path.join(RAW, "streamflow_mmd.csv"), stations_all, missing_below=-1e-9)
win = (days >= W0) & (days <= W1)
nvalid = np.isfinite(q_all[win]).sum(0)
keep = nvalid >= MIN_VALID_DAYS
dropped = [(s, int(n)) for s, n, k in zip(stations_all, nvalid, keep) if not k]
stations = [s for s, k in zip(stations_all, keep) if k]
q = q_all[:, keep]
N = len(stations)
print(f"{len(stations_all)} catchments in CAMELS-AUS v2; {N} with >= {MIN_VALID_DAYS} valid q days in {W0}..{W1}; dropped: {dropped}")
print(f"  catchments with any q in window: {(nvalid > 0).sum()}")

# ---- forcing
P = read_daily(os.path.join(HM, "01_precipitation_timeseries", "precipitation_AGCD.csv"), stations)
P_silo = read_daily(os.path.join(HM, "01_precipitation_timeseries", "precipitation_SILO.csv"), stations)
PET = read_daily(os.path.join(HM, "02_EvaporativeDemand_timeseries", "et_morton_wet_SILO.csv"), stations)
Tmax = read_daily(os.path.join(HM, "03_Other", "AGCD", "tmax_AGCD.csv"), stations)
Tmin = read_daily(os.path.join(HM, "03_Other", "AGCD", "tmin_AGCD.csv"), stations)
for k, a in {"P": P, "P_silo": P_silo, "PET": PET, "Tmax": Tmax, "Tmin": Tmin}.items():
    assert np.isfinite(a).all(), f"{k}: {np.isnan(a).sum()} missing values"
    assert (a > -90).all(), k

# ---- static attributes: (output name, source column, scale); US names kept where the quantity is the same concept
mm = m.loc[stations]
npp = mm[[f"npp_{i}" for i in range(1, 13)]].to_numpy(np.float64)
SPEC = [("p_mean", "p_mean", 1.0), ("pet_mean", "pet_mean", 1.0), ("aridity", "aridity", 1.0), ("p_seasonality", "p_seasonality", 1.0),
        ("frac_snow", "frac_snow", 1.0), ("high_prec_freq", "high_prec_freq", 1.0), ("high_prec_dur", "high_prec_dur", 1.0),
        ("low_prec_freq", "low_prec_freq", 1.0), ("low_prec_dur", "low_prec_dur", 1.0),
        ("elev_mean", "elev_mean", 1.0), ("slope_mean", "mean_slope_pct", 10.0),          # % -> m/km (US unit)
        ("area_gages2", "catchment_area", 1.0), ("elev_range", "elev_range", 1.0), ("strdensity", "strdensity", 1.0),
        ("frac_forest", "prop_forested", 1.0), ("npp_ann", "npp_ann", 1.0), ("npp_diff", None, 1.0),
        ("solum_thickness", "solum_thickness", 1.0), ("ksat", "ksat", 1.0), ("solpawhc", "solpawhc", 1.0),
        ("claya", "claya", 1.0), ("clayb", "clayb", 1.0), ("sanda", "sanda", 1.0),
        ("unconsoldted", "unconsoldted", 1.0), ("carbnatesed", "carbnatesed", 1.0), ("igneous", "igneous", 1.0), ("silicsed", "silicsed", 1.0)]
cols = []
for name, src, sc in SPEC:
    if name == "npp_diff":
        cols.append(npp.max(1) - npp.min(1))             # seasonal amplitude of pre-European NPP (analogue of lai_diff / gvf_diff)
    else:
        cols.append(mm[src].to_numpy(np.float64) * sc)
static_raw = np.column_stack(cols)
static_names = np.array([s[0] for s in SPEC])
mu, sd = np.nanmean(static_raw, 0), np.nanstd(static_raw, 0)
static_norm = (static_raw - mu) / np.where(sd > 0, sd, 1.0)
static_norm = np.nan_to_num(static_norm, nan=0.0)
print("static NaN per attribute:", {n: int(c) for n, c in zip(static_names, np.isnan(static_raw).sum(0)) if c})

ids = np.array([f"9{i + 1:07d}" for i in range(N)])
lat = mm["lat_centroid"].to_numpy(np.float64)
lon = mm["long_centroid"].to_numpy(np.float64)
elev = mm["elev_mean"].to_numpy(np.float64)
area_m2 = mm["catchment_area"].to_numpy(np.float64) * 1e6

np.savez(os.path.join(OUT, "camels_aus_v2.npz"), ids=ids, station_ids=np.array(stations), days=np.array([str(d.date()) for d in days]),
         P=P.astype(np.float32), Tmax=Tmax.astype(np.float32), Tmin=Tmin.astype(np.float32), PET=PET.astype(np.float32), q=q.astype(np.float32),
         static_raw=static_raw.astype(np.float32), static_norm=static_norm.astype(np.float32), static_names=static_names,
         lat=lat, elev=elev, area_m2=area_m2, P_silo=P_silo.astype(np.float32), lon=lon,
         state=mm["state_outlet"].to_numpy(str), station_names=mm["station_name"].to_numpy(str))

# ---- elevation bands: truncated normal on [elev_min, elev_max], sd = SIG_RATIO * range, location solved so its mean = elev_mean
K = 5
qs = (np.arange(K) + 0.5) / K
band_elev = np.zeros((N, K)); elev_std = np.zeros(N)
emin, emax = mm["elev_min"].to_numpy(np.float64), mm["elev_max"].to_numpy(np.float64)
for i in range(N):
    lo, hi, mean = emin[i], emax[i], elev[i]
    if mean <= lo:                                        # elev_min (gauge, 1" DEM) above elev_mean (9" DEM): inconsistent sources;
        lo = mean - 0.5 * (hi - mean)                     # use the median CAMELS-AUS (mean - min) / range = 1/3
        print(f"  elev bands: {stations[i]} elev_mean {mean:.1f} <= elev_min {emin[i]:.1f}; lower bound set to {lo:.1f}")
    s = SIG_RATIO * (hi - lo)
    mean_c = min(max(mean, lo + 0.02 * (hi - lo)), hi - 0.02 * (hi - lo))

    def tmean(loc):
        return truncnorm.mean((lo - loc) / s, (hi - loc) / s, loc=loc, scale=s) - mean_c
    loc = brentq(tmean, lo - 20 * (hi - lo), hi + 20 * (hi - lo))
    dist = truncnorm((lo - loc) / s, (hi - loc) / s, loc=loc, scale=s)
    band_elev[i] = dist.ppf(qs); elev_std[i] = dist.std()
np.savez(os.path.join(OUT, "elev_bands_aus.npz"), ids=ids, band_elev=band_elev, band_frac=np.full((N, K), 1.0 / K),
         n_bands=np.full(N, K, dtype=np.int64), elev_std=elev_std, elev_mean=elev.copy())

# ---- holdout: terciles of aridity x elev_mean, round(10 %) per stratum, default_rng(0); study catchments never drawn
St = pd.DataFrame(static_raw, columns=static_names)
ai = pd.qcut(St.aridity.rank(method="first"), 3, labels=False)
el = pd.qcut(St.elev_mean.rank(method="first"), 3, labels=False)
strata = (ai * 3 + el).to_numpy()
study_idx = {stations.index(s) for s in STUDY}
rng = np.random.default_rng(0)
pick = []
for s_ in range(9):
    idx = np.where(strata == s_)[0]
    n = int(round(0.1 * len(idx)))
    pool = np.array([i for i in idx if i not in study_idx])
    pick += list(rng.choice(pool, n, replace=False))
held = sorted(ids[i] for i in pick)
with open(os.path.join(OUT, "aus_ungauged_10pct.txt"), "w") as f:
    f.write(f"# CAMELS-AUS v2 ungauged validation set: 10 % of the {N} catchments, stratified random (terciles of aridity x elev_mean, 10 % per "
            "stratum, numpy default_rng(0); study catchments 406213, 406214, 407230 excluded from the draw); ids = 8-digit npz ids "
            "(station_ids in camels_aus_v2.npz); never used in training or model selection\n" + "\n".join(held) + "\n")
print(f"holdout: {len(held)} catchments")
print("study catchments:", {s: ids[stations.index(s)] for s in STUDY})

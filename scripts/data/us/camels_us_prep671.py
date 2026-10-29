"""CAMELS-US, all 671 basins, Daymet forcing (CAMELS v1.2 basin_mean_forcing/daymet, 1980-01-01 .. 2014-12-31) -> one npz, for the protocol of
Feng et al. (2022, WRR, doi:10.1029/2022WR032404): train 1980-10-01 .. 1995-09-30, test 1995-10-01 .. 2010-09-30.
Discharge (cfs -> mm/d with the forcing-header area), PET (Priestley-Taylor alpha 1.26, FAO-56 net radiation) and the 27 static attributes as
camels_us_prep.py; attributes missing for a basin are filled with the 671-basin median (count printed). Also writes elev_bands671.npz.
Run with python -I from a directory outside the download folders."""
import os  # data root: env CAMELS_US_ROOT (default ./camels_us)
import io, zipfile
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(os.environ.get("CAMELS_US_ROOT", "camels_us")); OUT = ROOT / "processed"
zq = zipfile.ZipFile(ROOT / "raw/basin_timeseries_v1p2_metForcing_obsFlow.zip")
names_all = zq.namelist()
fmap = {Path(x).name.split("_")[0]: x for x in names_all if "basin_mean_forcing/daymet/" in x and x.endswith("_lump_cida_forcing_leap.txt") and "__MACOSX" not in x}
qmap = {Path(x).name.split("_")[0]: x for x in names_all if "usgs_streamflow" in x and x.endswith(".txt") and "__MACOSX" not in x}
_att = pd.read_csv(ROOT / "raw/camels_topo.txt", sep=";", dtype={"gauge_id": str}); _att_ids = set(_att.gauge_id.str.zfill(8))
basins = sorted(set(fmap) & set(qmap) & _att_ids); print("basins with Daymet forcing, streamflow and CAMELS attributes:", len(basins),
      "| excluded (no attributes):", sorted((set(fmap) & set(qmap)) - _att_ids))
days = pd.date_range("1980-01-01", "2014-12-31", freq="D"); T, N = len(days), len(basins)
V = {k: np.full((T, N), np.nan) for k in ("prcp", "tmax", "tmin", "srad", "vp", "dayl", "q", "pet")}
lat, elev, area = np.zeros(N), np.zeros(N), np.zeros(N)
SIGMA, LAM, ALPHA, ALB = 4.903e-9, 2.45, 1.26, 0.23
for j, b in enumerate(basins):
    raw = zq.read(fmap[b]).decode(); head = raw.splitlines()[:3]
    lat[j], elev[j], area[j] = float(head[0]), float(head[1]), float(head[2])
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+", skiprows=3)
    df.index = pd.to_datetime(dict(year=df.Year, month=df.Mnth, day=df.Day)); df = df[~df.index.duplicated()].reindex(days)
    V["prcp"][:, j] = df["prcp(mm/day)"]; V["tmax"][:, j] = df["tmax(C)"]; V["tmin"][:, j] = df["tmin(C)"]
    V["srad"][:, j] = df["srad(W/m2)"]; V["vp"][:, j] = df["vp(Pa)"]; V["dayl"][:, j] = df["dayl(s)"]
    qd = pd.read_csv(io.BytesIO(zq.read(qmap[b])), sep=r"\s+", header=None, names=["id", "y", "m", "d", "Q", "flag"])
    qd.index = pd.to_datetime(dict(year=qd.y, month=qd.m, day=qd.d)); qd = qd[~qd.index.duplicated()]
    Qc = qd.Q.astype(float).where(qd.Q >= 0).reindex(days)
    V["q"][:, j] = 28316846.592 * Qc.values * 86400 / (area[j] * 1e6)
    tx, tn, ea = V["tmax"][:, j], V["tmin"][:, j], V["vp"][:, j] / 1000.0; tm = 0.5 * (tx + tn); doy = days.dayofyear.values
    phi = np.deg2rad(lat[j]); dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365); dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    Ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    Rs = V["srad"][:, j] * V["dayl"][:, j] / 1e6; Rso = (0.75 + 2e-5 * elev[j]) * Ra
    Rnl = SIGMA * ((tx + 273.16) ** 4 + (tn + 273.16) ** 4) / 2 * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) * (1.35 * np.clip(Rs / np.maximum(Rso, 1e-6), 0, 1) - 0.35)
    Rn = (1 - ALB) * Rs - Rnl
    delta = 4098 * 0.6108 * np.exp(17.27 * tm / (tm + 237.3)) / (tm + 237.3) ** 2; gamma = 0.665e-3 * 101.3 * ((293 - 0.0065 * elev[j]) / 293) ** 5.26
    V["pet"][:, j] = np.maximum(ALPHA * delta / (delta + gamma) * Rn / LAM, 0.0)
attr = {}
for f in ("camels_clim", "camels_topo", "camels_vege", "camels_soil", "camels_geol"):
    d = pd.read_csv(ROOT / f"raw/{f}.txt", sep=";", dtype={"gauge_id": str}).set_index("gauge_id"); d.index = d.index.str.zfill(8); attr[f] = d
A = pd.concat(attr.values(), axis=1)
names = ["p_mean", "pet_mean", "aridity", "p_seasonality", "frac_snow", "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur", "elev_mean",
         "slope_mean", "area_gages2", "frac_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff", "soil_depth_pelletier", "soil_depth_statsgo",
         "soil_porosity", "soil_conductivity", "max_water_content", "sand_frac", "silt_frac", "clay_frac", "carbonate_rocks_frac", "geol_permeability"]
S = A.reindex(basins)[names].apply(pd.to_numeric, errors="coerce")
miss = S.isna().sum(); print("missing attribute values filled with the median:", {k: int(v) for k, v in miss.items() if v > 0})
S = S.fillna(S.median()).values.astype(float); Sn = (S - S.mean(0)) / (S.std(0) + 1e-6)
print(f"days {T}, basins {N}; missing forcing {int(np.isnan(V['prcp']).sum())}; q coverage 1980-2010 {np.isfinite(V['q'][(days <= '2010-09-30')]).mean():.3f}")
np.savez_compressed(OUT / "camels_us_daymet671.npz", ids=np.array(basins), days=np.array([d.strftime("%Y-%m-%d") for d in days]),
                    P=V["prcp"].astype(np.float32), Tmax=V["tmax"].astype(np.float32), Tmin=V["tmin"].astype(np.float32), srad=V["srad"].astype(np.float32),
                    vp=V["vp"].astype(np.float32), dayl=V["dayl"].astype(np.float32), PET=V["pet"].astype(np.float32), q=V["q"].astype(np.float32),
                    static_raw=S.astype(np.float32), static_norm=Sn.astype(np.float32), static_names=np.array(names), lat=lat, elev=elev, area_m2=area)
# elevation bands
lists = [x for x in names_all if "elev_bands_forcing/daymet" in x and x.endswith(".list") and "__MACOSX" not in x]
bands = {}
for x in lists:
    b = x.split("/")[-1].split(".")[0]
    if b not in basins: continue
    for ln in zq.read(x).decode().splitlines()[1:]:
        f, ar = ln.split()[:2]; band = int(f.split("_elev_band_")[1].split("_")[0]); bands.setdefault(b, []).append((band * 100.0 + 50.0, float(ar)))
nb = [len(bands.get(b, [])) for b in basins]; K = max(nb); print("basins with elevation bands", sum(n > 0 for n in nb), "of", N, "; max bands", K)
E = np.zeros((N, K)); Af = np.zeros((N, K)); sig = np.zeros(N); zm = np.zeros(N)
for i, b in enumerate(basins):
    if not bands.get(b):
        E[i, 0], Af[i, 0] = elev[i], 1.0; zm[i] = elev[i]; continue
    arr = np.array(sorted(bands[b])); a = arr[:, 1] / arr[:, 1].sum(); e = arr[:, 0]
    E[i, :len(e)] = e; Af[i, :len(e)] = a; zm[i] = (a * e).sum(); sig[i] = np.sqrt((a * (e - zm[i]) ** 2).sum())
np.savez(OUT / "elev_bands671.npz", ids=np.array(basins), band_elev=E, band_frac=Af, n_bands=np.array(nb), elev_std=sig, elev_mean=zm)
print("wrote camels_us_daymet671.npz and elev_bands671.npz")

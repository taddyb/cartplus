"""Daymet v4 (1980-2019 files identical in v4 R1, ORNL DAAC ds 2129; v4 = ds 1840) basin-mean daily forcing for the 671 CAMELS basins, 2010-2019.
Source: Microsoft Planetary Computer Zarr copy of Daymet v4 daily NA (abfs daymet-zarr/daily/na.zarr, account daymeteuwest), anonymous SAS token
from https://planetarycomputer.microsoft.com/api/sas/v1/token/daymeteuwest/daymet-zarr . Chunks (365 d, 284 y, 584 x) = one Daymet year x tile;
only the tiles touching a CAMELS basin are fetched, decoded in memory, and reduced to basin means (nothing but the zarr metadata is stored).
Weights: CAMELS polygons (basin_set_full_res.zip, HCDN_nhru_final_671, NAD83 lon/lat) projected to the Daymet LCC grid; weight of each 1-km cell =
exact area of cell-box n polygon (shapely). Cells with NaN (Daymet land mask) are dropped and weights renormalised per day.
Daymet 365-day calendar: day k of Daymet year Y = Jan 1 + k (leap years: Feb 29 kept, Dec 31 dropped). Output keeps Daymet days only
(Dec 31 of leap years is filled later, in build_ext2019.py, as CAMELS v1.2 did).
Run: python -I daymet_basin_means.py <raw_dir (zmetadata.json)> <work_dir> [year0 year1]   (from a directory outside the download folder)"""
import os  # data root: env CAMELS_US_ROOT (default ./camels_us)
import sys, io, json, time, zipfile, urllib.request, concurrent.futures as cf
from pathlib import Path
import numpy as np, shapefile, shapely, pyproj, numcodecs, scipy.sparse as sp
from shapely.geometry import shape as shp_shape
RAW, WORK = Path(sys.argv[1]), Path(sys.argv[2]); WORK.mkdir(parents=True, exist_ok=True)
ROOT = Path(os.environ.get("CAMELS_US_ROOT", "camels_us"))
Y0, Y1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (2010, 2019)  # optional year range (e.g. 1990 1991 for an extra version check)
YEARS = list(range(Y0, Y1 + 1)); VARS = ["prcp", "tmax", "tmin", "srad", "vp", "dayl"]; TAG = f"{Y0}_{Y1}"
CY, CX = 284, 584; NY, NX = 8075, 7814
BASE = "https://daymeteuwest.blob.core.windows.net/daymet-zarr/daily/na.zarr"
TOKURL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/daymeteuwest/daymet-zarr"
UA = {"User-Agent": "python-data-client"}
_tok = {"t": None, "at": 0}
def token():
    if _tok["t"] is None or time.time() - _tok["at"] > 1500:
        with urllib.request.urlopen(urllib.request.Request(TOKURL, headers=UA), timeout=60) as r: _tok["t"] = json.load(r)["token"]
        _tok["at"] = time.time()
    return _tok["t"]
def fetch(key):
    for k in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{BASE}/{key}?{token()}", headers=UA), timeout=300) as r: return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            if e.code == 403: _tok["t"] = None
            err = e
        except Exception as e: err = e
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"failed {key}: {err}")
blosc = numcodecs.Blosc()
def chunk(var, t, iy, ix, shape):
    b = fetch(f"{var}/{t}.{iy}.{ix}")
    if b is None: return np.full(shape, np.nan, np.float32)
    return np.frombuffer(blosc.decode(b), dtype="<f4").reshape(shape)
# ---- grid coordinates and time check (from the store itself)
x = np.concatenate([np.frombuffer(blosc.decode(fetch(f"x/{i}")), "<f4") for i in range(int(np.ceil(NX / CX)))])[:NX].astype(np.float64)
y = np.concatenate([np.frombuffer(blosc.decode(fetch(f"y/{i}")), "<f4") for i in range(int(np.ceil(NY / CY)))])[:NY].astype(np.float64)
assert np.allclose(np.diff(x), 1000) and np.allclose(np.diff(y), -1000), "grid not regular"
print("grid x0", x[0], "y0", y[0], flush=True)
for Y in YEARS:
    tt = np.frombuffer(blosc.decode(fetch(f"time/{Y - 1980}")), "<i8")
    d = np.datetime64("1980-01-01") + tt.astype("timedelta64[D]")
    exp = np.datetime64(f"{Y}-01-01") + np.arange(365).astype("timedelta64[D]")
    assert (d == exp).all(), (Y, d[:3], d[-3:])
print("time chunks = Jan 1 + 0..364 for", YEARS, flush=True)
# ---- weights
ids = np.load(ROOT / "processed/camels_us_daymet671.npz")["ids"]
z = zipfile.ZipFile(ROOT / "raw/basin_set_full_res.zip")
r = shapefile.Reader(shp=io.BytesIO(z.read("HCDN_nhru_final_671.shp")), shx=io.BytesIO(z.read("HCDN_nhru_final_671.shx")), dbf=io.BytesIO(z.read("HCDN_nhru_final_671.dbf")))
tr = pyproj.Transformer.from_crs("EPSG:4269", "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +a=6378137 +rf=298.257223563 +units=m +no_defs", always_xy=True)
geod = pyproj.Geod(ellps="GRS80")
polys = {}
for i in range(len(r)):
    rec = r.record(i); g = shapely.make_valid(shp_shape(r.shape(i).__geo_interface__))
    polys[f"{int(rec['hru_id']):08d}"] = (g, rec["AREA"])
missing = [b for b in ids if b not in polys]; assert not missing, missing
WFILE = WORK / "daymet_weights.npz"
if WFILE.exists():  # weights are deterministic; reuse them so that parallel runs never rewrite the file
    _w = np.load(WFILE); assert (_w["ids"] == ids).all()
    cells = _w["cells"]; W = sp.csr_matrix((_w["W_data"], _w["W_indices"], _w["W_indptr"]), shape=tuple(_w["W_shape"])); print("reusing", WFILE, flush=True)
else:
    rows, cols, vals, info = [], [], [], []
    cell_index = {}
    for j, b in enumerate(ids):
        g, a_attr = polys[b]
        area_geod = abs(geod.geometry_area_perimeter(g)[0])
        gp = shapely.transform(g, lambda c: np.column_stack(tr.transform(c[:, 0], c[:, 1])))
        x0, y0, x1, y1 = gp.bounds
        ix = np.arange(max(0, int(np.floor((x0 - x[0]) / 1000 + 0.5)) - 1), min(NX, int(np.ceil((x1 - x[0]) / 1000 + 0.5)) + 1))
        iy = np.arange(max(0, int(np.floor((y[0] - y1) / 1000 + 0.5)) - 1), min(NY, int(np.ceil((y[0] - y0) / 1000 + 0.5)) + 1))
        IX, IY = np.meshgrid(ix, iy); IX, IY = IX.ravel(), IY.ravel()
        boxes = shapely.box(x[IX] - 500, y[IY] - 500, x[IX] + 500, y[IY] + 500)
        shapely.prepare(gp); a = shapely.area(shapely.intersection(boxes, gp))
        k = a > 0; IX, IY, a = IX[k], IY[k], a[k]
        for cx, cy, w in zip(IX, IY, a):
            c = cell_index.setdefault((int(cy), int(cx)), len(cell_index)); rows.append(j); cols.append(c); vals.append(w)
        info.append((b, len(a), a.sum() / 1e6, gp.area / 1e6, area_geod / 1e6, a_attr / 1e6, (a >= 0.999e6).sum()))
        if j % 100 == 0: print(j, info[-1], flush=True)
    W = sp.csr_matrix((vals, (rows, cols)), shape=(len(ids), len(cell_index)))
    cells = np.array(sorted(cell_index, key=cell_index.get))  # (ncell, 2) = (iy, ix)
    inf = np.array([t[1:] for t in info], float)
    print(f"unique cells {len(cells)}; cells per basin min {inf[:,0].min():.0f} median {np.median(inf[:,0]):.0f} max {inf[:,0].max():.0f}; "
          f"fully-covered cells min {inf[:,5].min():.0f}; LCC poly area / geodesic area range {np.min(inf[:,2]/inf[:,3]):.4f}..{np.max(inf[:,2]/inf[:,3]):.4f}", flush=True)
    np.savez(WORK / "daymet_weights.npz", ids=ids, cells=cells, W_data=W.data, W_indices=W.indices, W_indptr=W.indptr, W_shape=W.shape,
             info=inf, info_cols=np.array(["ncell", "sum_w_km2", "poly_lcc_km2", "poly_geod_km2", "shp_AREA_km2", "n_full_cells"]))
# ---- extraction
tiles = {}
for c, (cy, cx) in enumerate(cells): tiles.setdefault((cy // CY, cx // CX), []).append(c)
tiles = {k: np.array(v) for k, v in tiles.items()}
print("tiles to fetch per var-year:", len(tiles), flush=True)
Wd = W.tocsr()
def task(var, Y, key):
    iy, ix = key; cidx = tiles[key]
    arr = chunk(var, Y - 1980, iy, ix, (365, CY, CX))
    return cidx, arr[:, cells[cidx, 0] - iy * CY, cells[cidx, 1] - ix * CX].copy()
out = {v: np.full((len(YEARS) * 365, len(ids)), np.nan, np.float32) for v in VARS}
nanfrac = {v: np.zeros((len(YEARS), len(ids)), np.float32) for v in VARS}
t0 = time.time()
with cf.ThreadPoolExecutor(12) as ex:
    for v in VARS:
        f = WORK / f"bm_{v}_{TAG}.npy"
        if f.exists(): out[v] = np.load(f); print("cached", v, flush=True); continue
        for yi, Y in enumerate(YEARS):
            fy = WORK / "years" / f"bm_{v}_{Y}.npz"
            if fy.exists():
                _c = np.load(fy); out[v][yi * 365:(yi + 1) * 365] = _c["bm"]; nanfrac[v][yi] = _c["nanfrac"]; continue
            V = np.full((365, len(cells)), np.nan, np.float32)
            for cidx, vals_ in ex.map(lambda k: task(v, Y, k), list(tiles)): V[:, cidx] = vals_
            ok = np.isfinite(V); V0 = np.where(ok, V, 0.0).astype(np.float64)
            num = (Wd @ V0.T).T; den = (Wd @ ok.T.astype(np.float64)).T; tot = np.asarray(Wd.sum(1)).ravel()
            out[v][yi * 365:(yi + 1) * 365] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
            nanfrac[v][yi] = 1 - den.mean(0) / tot
            fy.parent.mkdir(exist_ok=True); np.savez(str(fy)[:-4] + ".tmp.npz", bm=out[v][yi * 365:(yi + 1) * 365], nanfrac=nanfrac[v][yi]); Path(str(fy)[:-4] + ".tmp.npz").rename(fy)
            print(f"{v} {Y} done {time.time() - t0:.0f}s; basins with NaN-cell weight>1%: {(nanfrac[v][yi] > 0.01).sum()}", flush=True)
        np.save(f, out[v]); np.save(WORK / f"nanfrac_{v}_{TAG}.npy", nanfrac[v])
dates = np.concatenate([np.datetime64(f"{Y}-01-01") + np.arange(365).astype("timedelta64[D]") for Y in YEARS])
np.savez(WORK / f"daymet_v4_basinmean_{TAG}.npz", ids=ids, dates=dates.astype(str), **out)
print("wrote", WORK / f"daymet_v4_basinmean_{TAG}.npz")

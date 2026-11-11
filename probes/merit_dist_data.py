"""MERIT loader for distributed CaRT+: the unit-catchment inputs of the distributed dHBV run, and the chunk plan.

Author: tbindas

What it does: reads /mnt/ssd1/data/water_loss/training_inputs.zarr and merit_idx.json (both read-only) and returns
  - per-divide forcing P, Temp, PET [divide, day] float32 (store layout; 73,959 divides, 1980-10-01 .. 2010-09-30),
  - per-divide scaled attributes [divide, 14] float32, the raw meanP attribute, the elevation, the unit-catchment area,
  - the gauge discharge [gauge, day] float32 (2717 gauges, order of `gage_staid`),
  - the gauge-to-divide map and the area fraction of each divide in its gauge,
  - the counts of the missing values before the fill.
"Divide" = one MERIT unit catchment = one row of the store. divide_dataset() gives a cart.data.Dataset for a set of
divides in the layout of probes/merit_data.py (arrays [day, divide] float64); the tests and the water-balance check use
it. chunk_plan() splits the gauges into chunks for gradient accumulation.

Experimental purpose: the equifinality paper compares lumped CaRT+ with distributed CaRT+ (CaRT+ on each divide, area
weighted sum to the gauge, gamma unit hydrograph at the gauge) on the data of the distributed dHBV run
CONUS2717_AORC2F_v3_gradaccum. The loader must give CaRT+ the same divides, forcing, attributes and periods.

Settings (decisions 8 to 12 of the brief of 2026-11-11):
  area weights    fraction_i = catchsize_i / sum of catchsize over the divides of the gauge (float64, then float32).
                  `gage_area_km2` is not used.
  attributes      (x - mean) / std per column, mean and std over all 73,959 divides, std < 0.001 -> 1, NaN -> 0 after
                  the scaling (scale.cal_stat and scale.trans_norm of the distributed dHBV script, lines 106-115). The
                  code is merit_data.scale_attrs; the statistics of the lumped store are not used.
  NaN forcing     P = 0, Temp = 0 deg C, PET = 0 (the distributed dHBV script sets xTrain2 NaN to 0, line 112).
  NaN elevation   -> 0 m. Each divide has one elevation band with area fraction 1, so the band offset from the mean
                  (cart.model.CaRT.Hdev) is 0 for any finite elevation and the value has no effect; load() asserts this.
  p_mean          the raw `meanP` attribute of each divide.
  periods         those of probes/merit_data.py (loss 1981-10-01 .. 1995-09-30, test 1995-10-01 .. 2010-09-30).
"""
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import zarr

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cart.data import Dataset  # noqa: E402
import merit_data  # noqa: E402
from merit_data import LOSS, N_ATTR, N_DAYS, PERIODS, RECORD, TEST  # noqa: E402,F401

STORE = "/mnt/ssd1/data/water_loss/training_inputs.zarr"
MAP = "/mnt/ssd1/data/water_loss/merit_idx.json"
N_GAUGES, N_DIVIDES = 2717, 73959
FORCING = ("P", "Temp", "PET")


@dataclass
class DistData:
    ids: np.ndarray             # [G] gauge ids (str), order of `gage_staid`
    q: np.ndarray               # [G, T] float32 observed discharge, mm/d (NaN = missing)
    P: np.ndarray               # [D, T] float32 precipitation, mm/d (NaN -> 0)
    Temp: np.ndarray            # [D, T] float32 air temperature, deg C (NaN -> 0)
    PET: np.ndarray             # [D, T] float32 potential evapotranspiration, mm/d (NaN -> 0)
    static: np.ndarray          # [D, 14] float32 scaled attributes (NaN -> 0)
    attrs_raw: np.ndarray       # [D, 14] float32 attributes of the store (with NaN)
    attr_mean: np.ndarray       # [14] float
    attr_std: np.ndarray        # [14] float
    static_names: list
    p_mean: np.ndarray          # [D] raw meanP attribute
    elev: np.ndarray            # [D] float32 mean elevation, m (NaN -> 0)
    catchsize: np.ndarray       # [D] float32 unit-catchment area, km2
    comid: np.ndarray           # [D] MERIT COMID of the divide
    days: list                  # [T] datetime.date
    gmap: list                  # [G] int64 arrays: store rows of the divides of each gauge
    frac: list                  # [G] float64 arrays: area fraction of each divide in the gauge (sum 1)
    nan: dict                   # counts of the missing values before the fill

    def mask(self, period):
        a, b = (date.fromisoformat(s) for s in PERIODS[period])
        return np.array([a <= d <= b for d in self.days])


def read_map(ids, path=MAP):
    """-> list over the gauges (order of ids) of int64 arrays with the store rows of the gauge's divides"""
    m = json.load(open(path))
    assert list(m) == [str(s) for s in ids], "merit_idx.json and gage_staid differ in gauges or order"
    return [np.asarray(m[str(s)], dtype=np.int64) for s in ids]


def fractions(catchsize, gmap):
    """area fraction of each divide in its gauge: catchsize_i / sum(catchsize of the gauge's divides), float64"""
    out = []
    for rows in gmap:
        a = catchsize[rows].astype(np.float64)
        out.append(a / a.sum())
    return out


def nan_counts(raw, gmap):
    """counts of missing values in the raw store arrays (before any fill)"""
    D = raw["P"].shape[0]
    all_nan = {k: np.isnan(raw[k]).all(1) for k in FORCING}
    any_nan = {k: np.isnan(raw[k]).any(1) for k in FORCING}
    forc_all = all_nan["P"] | all_nan["Temp"] | all_nan["PET"]
    forc_any = any_nan["P"] | any_nan["Temp"] | any_nan["PET"]
    an = np.isnan(raw["attrs"])
    bad_div = forc_any | an.any(1) | np.isnan(raw["meanelevation"])
    g_forc = [i for i, rows in enumerate(gmap) if forc_any[rows].any()]
    g_any = [i for i, rows in enumerate(gmap) if bad_div[rows].any()]
    return {"n_divides": int(D),
            "forcing_all_nan_divides": int(forc_all.sum()), "forcing_any_nan_divides": int(forc_any.sum()),
            "forcing_all_nan_rows": np.where(forc_all)[0].tolist(),
            "forcing_nan_values": {k: int(np.isnan(raw[k]).sum()) for k in FORCING},
            "attr_nan_per_column": {n: int(an[:, j].sum()) for j, n in enumerate(raw["attr_list"]) if an[:, j].any()},
            "attr_nan_divides": int(an.any(1).sum()), "elev_nan_divides": int(np.isnan(raw["meanelevation"]).sum()),
            "catchsize_nan": int(np.isnan(raw["catchsize"]).sum()),
            "gauges_with_nan_forcing": g_forc, "n_gauges_with_nan_forcing": len(g_forc),
            "n_gauges_with_any_nan_divide": len(g_any),
            "flow_nan_values": int(np.isnan(raw["flow_mmday"]).sum())}


def read_store(path=STORE):
    """-> dict of the raw store arrays in the store layout and the store attrs. Nothing is converted or filled."""
    z = zarr.open_group(path, mode="r")
    raw = {k: z[k][:] for k in ("P", "Temp", "PET", "flow_mmday", "attrs", "meanelevation", "catchsize", "gage_staid", "divide_comid")}
    raw["attr_list"] = list(z.attrs["attr_list"])
    raw["time_start"], raw["time_end"] = z.attrs["time_start"], z.attrs["time_end"]
    return raw


def load(path=STORE, map_path=MAP, raw=None):
    """-> DistData (all 73,959 divides and 2717 gauges). raw: the result of read_store (the tests pass it to keep one copy)."""
    raw = read_store(path) if raw is None else raw
    assert (raw["time_start"], raw["time_end"]) == RECORD, (raw["time_start"], raw["time_end"])
    d0 = date.fromisoformat(RECORD[0])
    days = [d0 + timedelta(days=i) for i in range(raw["P"].shape[1])]
    assert days[-1] == date.fromisoformat(RECORD[1]), days[-1]
    ids = raw["gage_staid"]
    gmap = read_map(ids, map_path)
    nan = nan_counts(raw, gmap)
    static, mean, std = merit_data.scale_attrs(raw["attrs"])                                  # decision 9 (NaN -> 0 after scaling)
    filled = {}
    for k in FORCING:                                                                         # decision 10: NaN -> 0 (P, deg C, PET)
        a = raw[k].copy()
        a[a != a] = 0
        filled[k] = a
    elev = raw["meanelevation"].copy()
    elev[elev != elev] = 0                                                                    # decision 11
    hdev = (elev.astype(np.float64)[:, None] - (np.ones((len(elev), 1)) * elev.astype(np.float64)[:, None]).sum(1, keepdims=True)).astype(np.float32)
    assert (hdev == 0).all(), "one band per divide: the band offset from the mean must be 0"
    names = raw["attr_list"]
    return DistData(ids=ids, q=raw["flow_mmday"], P=filled["P"], Temp=filled["Temp"], PET=filled["PET"], static=static,
                    attrs_raw=raw["attrs"], attr_mean=mean, attr_std=std, static_names=names,
                    p_mean=raw["attrs"][:, names.index("meanP")].copy(), elev=elev, catchsize=raw["catchsize"],
                    comid=raw["divide_comid"], days=days, gmap=gmap, frac=fractions(raw["catchsize"], gmap), nan=nan)


def divide_dataset(dd, rows):
    """-> cart.data.Dataset "merit" with one column per divide of `rows` (layout of merit_data.load: [day, divide] float64,
    Tm = Temp + 273.15, one band at the divide's elevation). q is NaN: a divide has no observed discharge."""
    rows = np.asarray(rows)
    n = len(rows)
    f64T = lambda a: np.ascontiguousarray(a[rows].T).astype(np.float64)                       # [divide, day] -> [day, divide]
    return Dataset(name="merit", ids=dd.comid[rows], gauged=np.zeros(n, bool), ungauged=np.ones(n, bool), static=dd.static[rows],
                   days=dd.days, P=f64T(dd.P), E=f64T(dd.PET), q=np.full((len(dd.days), n), np.nan), Tm=f64T(dd.Temp) + 273.15,
                   periods=dict(PERIODS), band_H=dd.elev[rows].astype(np.float64)[:, None], band_A=np.ones((n, 1)),
                   static_names=dd.static_names, p_mean=dd.p_mean[rows].copy())


def model_inputs(dd, rows):
    """model inputs of the divides `rows` with the values cart.model.CaRT.__init__ gives for a Dataset of these divides:
    Tcl [n, T] float32 = ((Temp + 273.15) - 273.15) in float64, then float32;  Pbar [n] float32 = max(mean of P over the
    loss days, 0.1), mean in float64 over the day axis of a [day, divide] array (model.py:67, 72)."""
    rows = np.asarray(rows)
    tr = dd.mask("train")
    Tm = np.ascontiguousarray(dd.Temp[rows].T).astype(np.float64) + 273.15
    Tcl = np.ascontiguousarray((Tm - 273.15).T).astype(np.float32)
    P = np.ascontiguousarray(dd.P[rows].T).astype(np.float64)
    Pbar = np.maximum(P[tr].mean(0), 0.1).astype(np.float32)
    return Tcl, Pbar


# ---------------------------------------------------------------------------------------------------- chunk plan
def chunk_plan(gmap, cap, n_gauges=None):
    """Split the first n_gauges gauges into chunks for gradient accumulation. -> list of chunks, each
    {"gauges": gauge indices (ascending), "rows": store rows of the union of their divides (ascending)}.

    Rules: each gauge is in exactly one chunk; a chunk holds all divides of its gauges, each divide once; no chunk has
    more than `cap` divides. The plan depends on gmap, cap and n_gauges only (no random numbers), so it is the same for
    each run with the same arguments.

    Method (first fit, largest gauge first): take the gauges in the order of falling divide count (ties: gauge index).
    Put the gauge in the first chunk where its divides that the chunk does not hold yet still fit. A gauge nested in a
    gauge of a chunk adds 0 divides to that chunk, so nested gauges land in the chunk of the gauge that holds them."""
    G = len(gmap) if n_gauges is None else n_gauges
    size = np.array([len(gmap[g]) for g in range(G)])
    assert size.max() <= cap, f"the largest gauge has {size.max()} divides; cap {cap} is too small"
    D = max(int(gmap[g].max()) for g in range(G)) + 1
    order = sorted(range(G), key=lambda g: (-size[g], g))
    held, count, members = [], [], []                                                          # per chunk: bool [D], divides, gauges
    for g in order:
        rows = gmap[g]
        for c in range(len(held)):
            need = int(size[g] - held[c][rows].sum())
            if count[c] + need <= cap:
                held[c][rows] = True; count[c] += need; members[c].append(g)
                break
        else:
            h = np.zeros(D, bool); h[rows] = True
            held.append(h); count.append(int(size[g])); members.append([g])
    return [{"gauges": sorted(m), "rows": np.where(h)[0].tolist()} for m, h in zip(members, held)]


def plan_summary(plan, gmap):
    """numbers of a chunk plan for the report"""
    cols = [len(c["rows"]) for c in plan]
    ng = [len(c["gauges"]) for c in plan]
    G = sum(ng)
    gs = [g for c in plan for g in c["gauges"]]
    return {"n_chunks": len(plan), "n_gauges": G, "columns_per_pass": int(sum(cols)), "largest_chunk": int(max(cols)),
            "smallest_chunk": int(min(cols)), "gauges_per_chunk_min_max": [int(min(ng)), int(max(ng))],
            "unique_divides": int(len(set(r for c in plan for r in c["rows"]))),
            "sum_of_divides_over_gauges": int(sum(len(gmap[g]) for g in gs))}

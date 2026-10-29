"""Datasets and the unified evaluation protocol of the paper.

All datasets are read from derived .npz files in data/ (or $CART_DATA; see DATA.md for their sources):
  gb  CAMELS-GB, 671 catchments (602 gauged training catchments, 67 held-out "ungauged" catchments), 1988-12-03 .. 2019-12-31
      file: $CART_GB_NPZ  (default $CART_DATA/camels_gb_cart.npz)
  us  CAMELS-US, 671 basins, Daymet v4 forcing 1980-01-01 .. 2019-12-31 (603 training basins, 68 held out = 10 %)
      files: $CAMELS_US_NPZ671 (default $CART_DATA/camels_us_daymet671_v4full.npz; scripts/data/us/),
             $CART_US_BANDS    (default $CART_DATA/elev_bands671.npz; CAMELS elevation bands),
             $CAMELS_US_UNGAUGED (default data_lists/us_ungauged_10pct.txt)
  aus CAMELS-AUS version 2, 561 catchments, AGCD / SILO forcing 1980-01-01 .. 2019-12-31 (505 training catchments, 56 held out = 10 %),
      in the layout of the US file and read by the US loader
      files: $CART_AUS_NPZ (default $CART_DATA/camels_aus_v2.npz; scripts/data/aus/), $CART_AUS_BANDS (default $CART_DATA/elev_bands_aus.npz),
             $CART_AUS_UNGAUGED (default data_lists/aus_ungauged_10pct.txt)
Protocol (all three continents): train 1989-01-01 .. 2008-12-31, checkpoint selection 2009-01-01 .. 2010-12-31 (median NSE of the gauged
training catchments), test 2011-01-01 .. 2019-12-31 (evaluated by scripts/eval/unified_eval.py, never used in training or selection).
All arrays are [T days, N catchments], mm/d; temperature Tm in K.
"""
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[2]                                                     # package root (contains data_lists/)
DATA = Path(os.environ.get("CART_DATA", PKG / "data"))

# training / selection periods; the GB run also prints the two halves of the test period (diagnostics only)
PERIODS_GB = {"train": ("1989-01-01", "2008-12-31"), "val": ("2009-01-01", "2010-12-31"),
              "2011-15": ("2011-01-01", "2015-09-30"), "2015-19": ("2015-10-01", "2019-12-31")}
# US and AUS: "test" = the selection period here (training diagnostics); the test period 2011-2019 is evaluated by scripts/eval
PERIODS_US = {"train": ("1989-01-01", "2008-12-31"), "val": ("2009-01-01", "2010-12-31"), "test": ("2009-01-01", "2010-12-31")}


@dataclass
class Dataset:
    name: str
    ids: np.ndarray
    gauged: np.ndarray          # [N] bool, catchments used for training and checkpoint selection
    ungauged: np.ndarray        # [N] bool, held-out catchments (never in the loss or in selection)
    static: np.ndarray          # [N, F] float32, standardised static attributes (input of the mapping network)
    days: list                  # [T] datetime.date
    P: np.ndarray               # [T, N] float64 precipitation
    E: np.ndarray               # [T, N] float64 potential evapotranspiration
    q: np.ndarray               # [T, N] float64 observed specific discharge (NaN = missing)
    Tm: np.ndarray              # [T, N] float64 mean air temperature (K)
    periods: dict
    band_H: np.ndarray          # [N, K] band elevation (m), K equal-area elevation bands
    band_A: np.ndarray          # [N, K] band area fraction
    static_names: list = None   # names of the static attributes
    p_mean: np.ndarray = None   # [N] raw mean precipitation attribute (LSTM target scaling)
    spin_days: list = None      # GB only: days before days[0] with forcing (LSTM spin-up)
    spin_forcing: np.ndarray = None  # GB only: [D0, N, 3] P, Tm, PET on spin_days (float32)
    band_elev_raw: np.ndarray = None  # US only: CAMELS elevation bands (100 m) of elev_bands671.npz, [N, B]
    band_frac_raw: np.ndarray = None  # US only: their area fractions, [N, B]

    def subset(self, K):
        """the first K catchments"""
        c = slice(0, K)
        return Dataset(self.name, self.ids[c], self.gauged[c], self.ungauged[c], self.static[c], self.days, self.P[:, c], self.E[:, c],
                       self.q[:, c], self.Tm[:, c], self.periods, self.band_H[c], self.band_A[c], self.static_names,
                       None if self.p_mean is None else self.p_mean[c], self.spin_days,
                       None if self.spin_forcing is None else self.spin_forcing[:, c],
                       None if self.band_elev_raw is None else self.band_elev_raw[c], None if self.band_frac_raw is None else self.band_frac_raw[c])

    def mask(self, period):
        a, b = self.periods[period]
        a, b = date.fromisoformat(a), date.fromisoformat(b)
        return np.array([a <= d <= b for d in self.days])


def _gb_bands(qv, K):
    """CAMELS-GB: K equal-area bands from the elevation quantiles (min, 10, 50, 90, max %) by linear interpolation."""
    qk = np.array([0.0, 10.0, 50.0, 90.0, 100.0])
    qm = 100.0 * (np.arange(K) + 0.5) / K                                                      # equal-area band centres
    H = np.stack([np.interp(qm, qk, np.maximum.accumulate(qv[i])) for i in range(len(qv))])
    return H, np.full((len(qv), K), 1.0 / K)


def _us_bands(be, bf, K):
    """CAMELS-US: K equal-area groups of the CAMELS elevation-band hypsometry (only positive band fractions are used)."""
    C = len(be)
    H = np.zeros((C, K)); A_ = np.zeros((C, K))
    for i in range(C):
        e, a = be[i][bf[i] > 0], bf[i][bf[i] > 0]; o_ = np.argsort(e); e, a = e[o_], a[o_]
        g = np.minimum((np.cumsum(a) - 1e-9) // (1.0 / K), K - 1).astype(int)
        for j in range(K):
            if (g == j).any():
                H[i, j] = (a[g == j] * e[g == j]).sum() / a[g == j].sum(); A_[i, j] = a[g == j].sum()
        A_[i] /= A_[i].sum()
    return H, A_


def load_gb(K=5):
    z = np.load(os.environ.get("CART_GB_NPZ", DATA / "camels_gb_cart.npz"))
    f64 = lambda k: z[k].astype(np.float64)
    H, A_ = _gb_bands(z["elev_quantiles"].astype(float), K)
    return Dataset("gb", z["ids"], z["basin_mask"], z["ug_mask"], z["static"].astype(np.float32), [date.fromisoformat(s) for s in z["days"]],
                   f64("P"), f64("PET"), f64("q"), f64("Tm"), dict(PERIODS_GB), H, A_, [str(s) for s in z["static_names"]], z["p_mean"],
                   [date.fromisoformat(s) for s in z["spin_days"]], z["spin_forcing"])


def us_bands_file():
    return os.environ.get("CART_US_BANDS", DATA / "elev_bands671.npz")


def load_us(K=5):
    z = np.load(os.environ.get("CAMELS_US_NPZ671", DATA / "camels_us_daymet671_v4full.npz"))
    ids = z["ids"]
    ugf = os.environ.get("CAMELS_US_UNGAUGED", PKG / "data_lists" / "us_ungauged_10pct.txt")
    held = {l.strip().zfill(8) for l in open(ugf) if l.strip() and not l.startswith("#")}
    ug = np.array([str(i).zfill(8) in held for i in ids])
    assert ug.sum() == len(held), (ug.sum(), len(held))
    eb = np.load(us_bands_file())
    assert list(eb["ids"]) == [str(i).zfill(8) for i in ids]
    H, A_ = _us_bands(eb["band_elev"], eb["band_frac"], K)
    Tm = 0.5 * (z["Tmax"].astype(np.float64) + z["Tmin"].astype(np.float64)) + 273.15
    return Dataset("us", ids, ~ug, ug, z["static_norm"].astype(np.float32), [date.fromisoformat(s) for s in z["days"]],
                   z["P"].astype(np.float64), z["PET"].astype(np.float64), z["q"].astype(np.float64), Tm, dict(PERIODS_US), H, A_,
                   [str(s) for s in z["static_names"]], z["static_raw"][:, [str(s) for s in z["static_names"]].index("p_mean")].astype(np.float32),
                   None, None, eb["band_elev"], eb["band_frac"])


US_LAYOUT = ("us", "aus")                                                                      # datasets that follow the US code paths


def aus_files():
    """the CAMELS-AUS input files (npz, held-out list, elevation bands) under the names of the US loader's environment variables"""
    return {"CAMELS_US_NPZ671": os.environ.get("CART_AUS_NPZ", str(DATA / "camels_aus_v2.npz")),
            "CAMELS_US_UNGAUGED": os.environ.get("CART_AUS_UNGAUGED", str(PKG / "data_lists" / "aus_ungauged_10pct.txt")),
            "CART_US_BANDS": os.environ.get("CART_AUS_BANDS", str(DATA / "elev_bands_aus.npz"))}


def load_aus(K=5):
    """CAMELS-AUS v2 through the US loader: same file layout, same periods and code paths; only the files and the held-out list differ."""
    env = aus_files()
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        ds = load_us(K)
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    ds.name = "aus"
    return ds


def load(name, K=5):
    return {"gb": load_gb, "us": load_us, "aus": load_aus}[name](K)

"""MERIT loader for CaRT+: the lumped dHBV inputs of water_loss as a cart.data.Dataset.

Author: tbindas

What it does: reads /mnt/ssd1/data/water_loss/lumped_training_inputs.zarr (read-only) and returns a cart.data.Dataset with
one column per USGS gauge (2717 gauges, 1980-10-01 .. 2010-09-30, daily). The gauge order is the order of `gage_staid`.

Experimental purpose: the equifinality paper needs a CaRT+ run that is comparable with the lumped dHBV run of water_loss
(CONUS2717_AORC2F_LUMPED_v2_gradaccum). CaRT+ must see the same gauges, forcing, static attributes and periods.

Settings (user decisions of 2026-11-10):
  loss period     1981-10-01 .. 1995-09-30. The simulation starts on 1980-10-01; the first 365 days are warm-up
                  (BUFFTIME of the lumped dHBV script). Dataset.periods["train"] is this loss period.
  test period     1995-10-01 .. 2010-09-30 (forward_conus_divides.py, EVAL_START / EVAL_END).
  elevation bands one band per gauge at `meanelevation`, area fraction 1. The band offset from the mean is then 0, so the
                  lapse-rate parameter lGam has no effect on the simulation.
  attributes      (x - mean) / std per column, mean and std over all 2717 gauges, std < 0.001 -> 1, NaN -> 0. This is the
                  transform of CONUS_water_loss_lumped_AORC_gradaccum.py:105-115 (scale.cal_stat, scale.trans_norm).
                  That script has log_norm_cols = [], so no column is log-normalised.
  temperature     the store holds Temp in deg C (prepare_lumped_aorc_inputs.py:137). Tm = Temp + 273.15 (K).

Dataset.periods also holds a key "val". cart.model.CaRT.__init__ reads
ds.mask("val") to find the last day of the training simulation. It is set to the loss period, so the training simulation
ends on 1995-09-30. It is not a validation period: probes/train_merit.py does no checkpoint selection.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import zarr

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))
from cart.data import Dataset  # noqa: E402

STORE = "/mnt/ssd1/data/water_loss/lumped_training_inputs.zarr"
RECORD = ("1980-10-01", "2010-09-30")
DHBV_TRAIN = ("1980-10-01", "1995-09-30")          # train window of the lumped dHBV script (5478 days, incl. warm-up; the script comment says 5479)
WARMUP_DAYS = 365                                  # BUFFTIME
LOSS = ("1981-10-01", "1995-09-30")                # DHBV_TRAIN less the warm-up
TEST = ("1995-10-01", "2010-09-30")
PERIODS = {"train": LOSS, "test": TEST, "val": LOSS}   # "val": end marker of the training simulation only (see docstring)
N_GAUGES, N_DAYS, N_ATTR = 2717, 10957, 14
STD_FLOOR = 0.001


def read_store(path=STORE):
    """-> dict of the raw store arrays in the store layout [gauge, day] and the store attrs. Nothing is converted."""
    z = zarr.open_group(path, mode="r")
    raw = {k: z[k][:] for k in ("P", "Temp", "PET", "flow_mmday", "attrs", "meanelevation", "gage_staid")}
    raw["attr_list"] = list(z.attrs["attr_list"])
    raw["time_start"], raw["time_end"] = z.attrs["time_start"], z.attrs["time_end"]
    return raw


def attr_stats(attrs):
    """mean and std per column as scale.cal_stat gives them (float32 reduction, NaN left out, std < 0.001 -> 1)"""
    mean, std = [], []
    for k in range(attrs.shape[1]):
        b = attrs[:, k].flatten()
        b = b[~np.isnan(b)]
        if b.size == 0:
            b = np.array([0])
        m, s = float(np.mean(b)), float(np.std(b))
        if s < STD_FLOOR:
            s = 1
        mean.append(m); std.append(s)
    return np.array(mean), np.array(std)


def scale_attrs(attrs):
    """-> standardised attributes [N, F] float32 (scale.trans_norm with to_norm=True, then NaN -> 0), mean [F], std [F]"""
    mean, std = attr_stats(attrs)
    out = np.zeros(attrs.shape)
    for k in range(attrs.shape[1]):
        out[:, k] = (attrs[:, k] - mean[k]) / std[k]
    out[out != out] = 0
    return out.astype(np.float32), mean, std


def load(K=None, path=STORE):
    """-> cart.data.Dataset "merit". K: only the first K gauges (the attribute statistics always come from all gauges)."""
    raw = read_store(path)
    assert (raw["time_start"], raw["time_end"]) == RECORD, (raw["time_start"], raw["time_end"])
    d0 = date.fromisoformat(RECORD[0])
    days = [d0 + timedelta(days=i) for i in range(raw["P"].shape[1])]
    assert days[-1] == date.fromisoformat(RECORD[1]), days[-1]
    ids = raw["gage_staid"]
    n = len(ids)
    static, _, _ = scale_attrs(raw["attrs"])
    f64T = lambda a: np.ascontiguousarray(a.T).astype(np.float64)                             # [gauge, day] -> [day, gauge]
    names = raw["attr_list"]
    ds = Dataset(name="merit", ids=ids, gauged=np.ones(n, bool), ungauged=np.zeros(n, bool), static=static, days=days,
                 P=f64T(raw["P"]), E=f64T(raw["PET"]), q=f64T(raw["flow_mmday"]), Tm=f64T(raw["Temp"]) + 273.15,
                 periods=dict(PERIODS), band_H=raw["meanelevation"].astype(np.float64)[:, None], band_A=np.ones((n, 1)),
                 static_names=names, p_mean=raw["attrs"][:, names.index("meanP")].copy())
    return ds if K is None else ds.subset(K)

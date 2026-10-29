"""Shared settings of the evaluation scripts: where the simulations are, the observations, catchment classes, output folders.

Simulations: <runs>/<run name>/sim.npy  [T, N] float32 mm/d on the dataset days (full record, all catchments incl. held-out).
  run names <dataset>_<model>_s<seed>, as written by the training commands of README.md and configs/train_all.sh;
  other outputs of a run (e.g. the attribute-draw averages sim_noise<sigma>.npy of CaRT+) are read from the same folder (sim_path(..., fname)).
Observations and catchment sets: cart.data (CART_DATA, CAMELS_US_NPZ671, CAMELS_US_UNGAUGED, CART_US_BANDS; CAMELS-AUS: CART_AUS_NPZ,
CART_AUS_UNGAUGED, CART_AUS_BANDS).
Catchment classes: GB baseflow index and snow fraction from the CAMELS-GB attribute tables ($CAMELS_GB_ATTR = folder with
CAMELS_GB_{hydrologic,climatic,topographic}_attributes.csv); US and AUS snow fraction and aridity from static_raw of the npz.
Outputs: <out>/tables/*.tex, <out>/figs/*.pdf|png, <out>/data/*.json (default <repository>/results).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "src"))
from cart import data as D  # noqa: E402

# model key -> run name stem per dataset (configs/paper_runs.csv). CAMELS-AUS (Section 5.6) has the six models of the main comparison only.
_CART = {"CaRT": "cart", "CaRTp": "cartplus"}
_CART.update({f"P{_i}": f"cartplus_no_p{_i}" for _i in range(1, 6)})                          # CaRT+ without process group P<i> (leave one out)
RUNS = {
    _ds: {"HBV": f"{_ds}_hbv", "dHBV": f"{_ds}_dhbv", "LSTM": f"{_ds}_lstm", "EALSTM": f"{_ds}_ealstm"} for _ds in ("gb", "us", "aus")
}
_AUS_CART = ("CaRT", "CaRTp")                                                                  # CAMELS-AUS: CaRT and CaRT+ only
_ORDER = ["HBV", "dHBV", "CaRT", "CaRTp", "LSTM", "EALSTM"] + [k for k in _CART if k not in ("CaRT", "CaRTp")]
for _ds in ("gb", "us", "aus"):
    RUNS[_ds].update({_k: f"{_ds}_{_m}" for _k, _m in _CART.items() if _ds != "aus" or _k in _AUS_CART})
    RUNS[_ds] = {_k: RUNS[_ds][_k] for _k in _ORDER if _k in RUNS[_ds]}
TEST = ("2011-01-01", "2019-12-31")


def parser(doc):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(PKG / "runs"), help="folder with the run directories")
    ap.add_argument("--out", default=str(PKG / "results"))
    return ap


def sim_path(a, ds, model, seed, fname="sim.npy"):
    return Path(a.runs) / f"{RUNS[ds][model]}_s{seed}" / fname


def run_label(a, ds, model):
    return RUNS[ds][model]


def outdirs(a):
    o = Path(a.out)
    for s in ("tables", "figs", "data"):
        (o / s).mkdir(parents=True, exist_ok=True)
    return o


def load_obs(ds):
    """-> dict(dd, q [T, N] float64, P, ids, mask (gauged), ung (held out), static attributes for the classes, names)"""
    d = D.load(ds)
    dd = pd.to_datetime([x.isoformat() for x in d.days])
    out = dict(dd=dd, q=d.q, P=d.P, ids=None, mask=np.asarray(d.gauged, bool), ung=np.asarray(d.ungauged, bool))
    if ds == "aus":                                                                           # CAMELS-AUS v2: ids = running npz ids "9xxxxxxx"
        z = np.load(os.environ.get("CART_AUS_NPZ", D.DATA / "camels_aus_v2.npz"))
        St = pd.DataFrame(z["static_raw"], columns=[str(s) for s in z["static_names"]])
        out.update(ids=[str(i) for i in z["ids"]], snow=St.frac_snow.values, aridity=St.aridity.values, static_norm=d.static,
                   station_ids=[str(s) for s in z["station_ids"]], names=[f"{s} {n}"[:38] for s, n in zip(z["station_ids"], z["station_names"])])
    elif ds == "us":
        z = np.load(os.environ.get("CAMELS_US_NPZ671", D.DATA / "camels_us_daymet671_v4full.npz"))
        St = pd.DataFrame(z["static_raw"], columns=[str(s) for s in z["static_names"]])
        out.update(ids=[str(i).zfill(8) for i in z["ids"]], snow=St.frac_snow.values, aridity=St.aridity.values, static_norm=d.static)
        nmf = os.environ.get("CAMELS_US_NAMES", "")                                           # camels_name.txt of CAMELS-US (optional)
        if nmf and os.path.exists(nmf):
            nm = pd.read_csv(nmf, sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
            out["names"] = [nm.loc[i, "gauge_name"][:38] if i in nm.index else "" for i in out["ids"]]
        else:
            out["names"] = ["" for _ in out["ids"]]
    else:
        ids = [int(i) for i in d.ids]
        A = Path(os.environ.get("CAMELS_GB_ATTR", D.DATA / "CAMELS_GB_attributes"))
        out.update(ids=ids,
                   bfi=pd.read_csv(A / "CAMELS_GB_hydrologic_attributes.csv").set_index("gauge_id").loc[ids].baseflow_index.values,
                   snow=pd.read_csv(A / "CAMELS_GB_climatic_attributes.csv").set_index("gauge_id").loc[ids].frac_snow.values,
                   names=[str(s)[:40] for s in pd.read_csv(A / "CAMELS_GB_topographic_attributes.csv").set_index("gauge_id").loc[ids].gauge_name.values])
    out["per"] = np.asarray((dd >= TEST[0]) & (dd <= TEST[1]))
    return out

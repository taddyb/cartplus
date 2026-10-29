"""Checks of the model input files (CPU only): checksums, layout, catchment order and roles, held-out sets, and what the loaders return.

    python tests/check_data.py us|aus|gb

us / aus  the released files in data/ (or the files named by the environment variables of cart.data):
            SHA256 of the released files (scripts/data/fetch_data.py; skipped for files given through environment variables),
            keys, shapes and dtypes, days 1980-01-01 .. 2019-12-31, catchment order of data_lists/<dataset>_*.csv, the held-out list
            redrawn from static_raw (US: terciles of frac_snow x aridity; AUS: terciles of aridity x elev_mean, catchments 406213,
            406214 and 407230 never drawn; round(10 %) per stratum, numpy default_rng(0)), complete forcing, and the dataset returned by
            cart.data.load (68 / 56 held-out catchments, no overlap with the gauged set, 5 elevation bands with area fractions summing to 1).
gb        SHA256 of the released camels_gb_cart.npz (skipped for a file given through CART_GB_NPZ), layout, catchment order of data_lists/gb_catchments_671.csv, the 602 gauged and 67
            held-out catchments, and cart.data.load("gb").
"""
import hashlib
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "scripts" / "data"))
from cart import data as D  # noqa: E402
from fetch_data import SHA256  # noqa: E402

N_OK, N_BAD = 0, 0


def check(name, ok, info=""):
    global N_OK, N_BAD
    N_OK += bool(ok); N_BAD += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({info})" if info else ""), flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_list(path):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]


def draw_heldout(static_raw, names, ids, strat, exclude=()):
    """stratified random draw: terciles of strat[0] x terciles of strat[1], round(10 %) per stratum, numpy default_rng(0)"""
    St = pd.DataFrame(static_raw, columns=names)
    a = pd.qcut(St[strat[0]].rank(method="first"), 3, labels=False)
    b = pd.qcut(St[strat[1]].rank(method="first"), 3, labels=False)
    strata = (a * 3 + b).to_numpy()
    rng = np.random.default_rng(0)
    pick = []
    for s in range(9):
        idx = np.where(strata == s)[0]
        n = int(round(0.1 * len(idx)))
        pool = np.array([i for i in idx if i not in set(exclude)])
        pick += list(rng.choice(pool, n, replace=False))
    return sorted(str(ids[i]) for i in pick)


def check_us_layout(name):
    if name == "us":
        files = {"npz": ("CAMELS_US_NPZ671", "camels_us_daymet671_v4full.npz"), "bands": ("CART_US_BANDS", "elev_bands671.npz")}
        ug_list = os.environ.get("CAMELS_US_UNGAUGED", PKG / "data_lists" / "us_ungauged_10pct.txt")
        order = pd.read_csv(PKG / "data_lists" / "us_basins_671.csv", comment="#", dtype={"gauge_id": str})
        n, n_ug, n_static, strat, excl_st = 671, 68, 27, ("frac_snow", "aridity"), ()
    else:
        files = {"npz": ("CART_AUS_NPZ", "camels_aus_v2.npz"), "bands": ("CART_AUS_BANDS", "elev_bands_aus.npz")}
        ug_list = os.environ.get("CART_AUS_UNGAUGED", PKG / "data_lists" / "aus_ungauged_10pct.txt")
        order = pd.read_csv(PKG / "data_lists" / "aus_catchments_561.csv", dtype={"id": str, "station_id": str})
        n, n_ug, n_static, strat, excl_st = 561, 56, 27, ("aridity", "elev_mean"), ("406213", "406214", "407230")
    paths = {}
    for k, (env, fname) in files.items():
        paths[k] = Path(os.environ.get(env, D.DATA / fname))
        check(f"{fname} present", paths[k].exists(), str(paths[k]))
        if not paths[k].exists():
            print("  (python scripts/data/fetch_data.py downloads the released files)")
            return
        if env not in os.environ and fname in SHA256:
            check(f"{fname} SHA256", sha256(paths[k]) == SHA256[fname])
    z = np.load(paths["npz"])
    T = 14610
    shapes = {"ids": (n,), "days": (T,), "P": (T, n), "Tmax": (T, n), "Tmin": (T, n), "PET": (T, n), "q": (T, n),
              "static_raw": (n, n_static), "static_norm": (n, n_static), "static_names": (n_static,), "lat": (n,), "elev": (n,),
              "area_m2": (n,)}
    check("keys and shapes", all(k in z.files and z[k].shape == s for k, s in shapes.items()),
          ", ".join(f"{k} {z[k].shape}" for k in shapes if k in z.files and z[k].shape != shapes[k]))
    days = list(z["days"])
    check("days 1980-01-01 .. 2019-12-31, consecutive", days[0] == "1980-01-01" and days[-1] == "2019-12-31"
          and all((date.fromisoformat(b) - date.fromisoformat(a)).days == 1 for a, b in zip(days[:-1], days[1:])))
    ids = [str(i) for i in z["ids"]]
    ref_ids = list(order["gauge_id"] if name == "us" else order["id"])
    check("catchment order = data_lists", ids == ref_ids)
    if name == "aus":
        check("station ids = data_lists", list(z["station_ids"]) == list(order["station_id"]))
    fin = all(np.isfinite(z[k]).all() for k in ("P", "Tmax", "Tmin", "PET"))
    check("forcing complete (P, Tmax, Tmin, PET finite)", fin)
    q = z["q"]
    check("discharge >= 0 where observed", bool((q[np.isfinite(q)] >= 0).all()), f"{np.isfinite(q).mean():.3f} of the basin-days observed")
    held = read_list(ug_list)
    names = [str(s) for s in z["static_names"]]
    stations = [str(s) for s in z["station_ids"]] if name == "aus" else ids
    excl = [stations.index(s) for s in excl_st]
    redraw = draw_heldout(z["static_raw"], names, ids, strat, excl)
    check(f"held-out list ({len(held)}) = stratified draw from static_raw", sorted(held) == redraw)
    roles = dict(zip(ref_ids, order["role"]))
    check("held-out list = role held_out in data_lists", sorted(held) == sorted(i for i, r in roles.items() if r == "held_out"))
    eb = np.load(paths["bands"])
    check("bands file: same ids", [str(i) for i in eb["ids"]] == ids)
    ds = D.load(name)
    check("loader: held-out / gauged counts", int(ds.ungauged.sum()) == n_ug and int(ds.gauged.sum()) == n - n_ug,
          f"{int(ds.gauged.sum())} gauged, {int(ds.ungauged.sum())} held out")
    check("loader: no catchment both gauged and held out", not (ds.gauged & ds.ungauged).any())
    check("loader: 5 bands, area fractions sum to 1", ds.band_H.shape == (n, 5) and np.allclose(ds.band_A.sum(1), 1.0))
    check("loader: Tm in K", 250 < float(np.nanmean(ds.Tm)) < 310, f"mean {np.nanmean(ds.Tm):.1f} K")
    check("loader: standardised attributes finite", bool(np.isfinite(ds.static).all()))


def check_gb():
    path = Path(os.environ.get("CART_GB_NPZ", D.DATA / "camels_gb_cart.npz"))
    check("camels_gb_cart.npz present", path.exists(), str(path))
    if not path.exists():
        print("  (python scripts/data/fetch_data.py downloads the released files)")
        return
    if "CART_GB_NPZ" not in os.environ:
        check("camels_gb_cart.npz SHA256", sha256(path) == SHA256["camels_gb_cart.npz"])
    z = np.load(path)
    need = ("ids", "days", "P", "PET", "Tm", "q", "static", "static_names", "basin_mask", "ug_mask", "elev_quantiles", "p_mean",
            "spin_days", "spin_forcing")
    check("keys", all(k in z.files for k in need), ", ".join(k for k in need if k not in z.files))
    n, T = 671, len(z["days"])
    check("shapes", z["P"].shape == (T, n) and z["static"].shape[0] == n and z["elev_quantiles"].shape == (n, 5)
          and z["spin_forcing"].shape == (len(z["spin_days"]), n, 3))
    order = pd.read_csv(PKG / "data_lists" / "gb_catchments_671.csv", comment="#")
    check("catchment order = data_lists", [int(i) for i in z["ids"]] == list(order["gauge_id"]))
    gauged = [int(i) for i in read_list(PKG / "data_lists" / "gb_train602_basins.txt")]
    ids = [int(i) for i in z["ids"]]
    check("gauged = gb_train602_basins.txt", sorted(np.array(ids)[z["basin_mask"]].tolist()) == sorted(gauged))
    check("held out = role held_out (67)", sorted(np.array(ids)[z["ug_mask"]].tolist())
          == sorted(order.gauge_id[order.role == "held_out"].tolist()))
    ds = D.load("gb")
    check("loader: 602 gauged, 67 held out, no overlap", int(ds.gauged.sum()) == 602 and int(ds.ungauged.sum()) == 67
          and not (ds.gauged & ds.ungauged).any())
    check("loader: days 1988-12-03 .. 2019-12-31", ds.days[0] == date(1988, 12, 3) and ds.days[-1] == date(2019, 12, 31))


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "us"
    print(f"DATA CHECK {name}")
    {"us": lambda: check_us_layout("us"), "aus": lambda: check_us_layout("aus"), "gb": check_gb}[name]()
    print(f"DATA CHECK {name}: {N_OK} passed, {N_BAD} failed -> {'PASSED' if N_BAD == 0 else 'FAILED'}")
    sys.exit(0 if N_BAD == 0 else 1)

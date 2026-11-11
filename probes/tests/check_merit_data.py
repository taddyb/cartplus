"""Checks of the MERIT loader (probes/merit_data.py) and of the metric code (probes/eval_merit.py). CPU only.

Author: tbindas

What it does: compares the cart.data.Dataset of merit_data.load() with the zarr store of water_loss and with the code of
water_loss. Prints PASSED / FAILED with numbers for each check: gauge order, shapes, values (bit for bit), units, missing
data, periods, attribute scaling, metric code.

    python probes/tests/check_merit_data.py [--run <run dir of train_merit.py>]

--run: the metric check also uses the test-period simulation of that run (without it: a surrogate simulation only).

Experimental purpose: evidence that CaRT+ and the lumped dHBV run of water_loss see the same data, and that the per-gauge
scores of the two models come from the same formulas. It reads water_loss files; it does not change them. The two
water_loss modules (hydroDL/data/scale.py, forward_lumped_eval.py) are loaded by file path, not edited or patched.
"""
import argparse
import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import eval_merit  # noqa: E402
import merit_data  # noqa: E402

WL = Path.home() / "projects/water_loss/dPLHBVrelease-master/hydroDL-dev"
SCALE_PY = WL / "hydroDL/data/scale.py"
EVAL_PY = WL / "example/water_loss_aorc/forward_lumped_eval.py"
REPORT_NAN_PCT, REPORT_RR_MEDIAN = 2.0682, 0.354            # /mnt/ssd1/data/water_loss/lumped_prep_report.txt

fails = []


def check(name, ok, msg):
    print(f"{'PASSED' if ok else 'FAILED'}  {name}: {msg}", flush=True)
    if not ok:
        fails.append(name)


def by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    a = ap.parse_args()
    z = zarr.open_group(merit_data.STORE, mode="r")
    ds = merit_data.load()
    N, T, F = merit_data.N_GAUGES, merit_data.N_DAYS, merit_data.N_ATTR

    # gauge order
    staid = z["gage_staid"][:]
    check("gauge order", len(ds.ids) == N and np.array_equal(ds.ids, staid) and len(set(ds.ids.tolist())) == N,
          f"{len(ds.ids)} ids, equal to gage_staid in order: {np.array_equal(ds.ids, staid)}, unique: {len(set(ds.ids.tolist()))}")

    # shapes
    shp = {k: getattr(ds, k).shape for k in ("P", "E", "Tm", "q", "static", "band_H", "band_A", "p_mean", "gauged", "ungauged")}
    ok = all(shp[k] == (T, N) for k in ("P", "E", "Tm", "q")) and shp["static"] == (N, F) and shp["band_H"] == (N, 1) \
        and shp["band_A"] == (N, 1) and len(ds.days) == T and ds.gauged.all() and not ds.ungauged.any()
    check("shapes", ok, f"{shp}; days {len(ds.days)} ({ds.days[0]} .. {ds.days[-1]}); gauged {int(ds.gauged.sum())}, ungauged {int(ds.ungauged.sum())}")

    # values, bit for bit, before conversion and scaling
    raw = merit_data.read_store()
    res = {}
    for dk, zk in (("P", "P"), ("E", "PET"), ("q", "flow_mmday")):
        back = np.ascontiguousarray(getattr(ds, dk).T).astype(np.float32)        # float32 -> float64 is exact, so the way back is too
        res[zk] = same_bits(back, z[zk][:])
    res["Temp (raw array of the loader)"] = same_bits(raw["Temp"], z["Temp"][:])
    res["attrs (raw array of the loader)"] = same_bits(raw["attrs"], z["attrs"][:])
    res["p_mean vs attrs[:, meanP]"] = same_bits(ds.p_mean, z["attrs"][:][:, list(z.attrs["attr_list"]).index("meanP")])
    res["band_H vs meanelevation"] = np.array_equal(ds.band_H[:, 0], z["meanelevation"][:].astype(np.float64))
    check("values bit for bit", all(res.values()), str(res))

    # units
    temp64 = z["Temp"][:].T.astype(np.float64)
    tm_exact = np.array_equal(ds.Tm, temp64 + 273.15)
    rr = np.nanmean(z["flow_mmday"][:], axis=1) * 365.25 / ds.p_mean              # formula of prepare_lumped_aorc_inputs.py
    rr_med = float(np.nanmedian(rr))
    p_year = ds.P.mean(0) * 365.25
    ratio_p = float(np.median(p_year / ds.p_mean))
    ok = tm_exact and ds.Tm.min() > 200 and ds.Tm.max() < 330 and abs(rr_med - REPORT_RR_MEDIAN) < 5e-4 and ds.P.min() >= 0 and ds.E.min() >= 0
    check("units", ok, f"Tm == Temp + 273.15 exactly: {tm_exact}; Tm range {ds.Tm.min():.2f} .. {ds.Tm.max():.2f} K; "
          f"median runoff ratio {rr_med:.4f} (report {REPORT_RR_MEDIAN}); min P {ds.P.min():.3g}, min PET {ds.E.min():.3g} mm/d; "
          f"median of (mean P x 365.25) / meanP {ratio_p:.3f} (information only: P of the store against the meanP attribute)")

    # missing data
    nan = {k: int(np.isnan(getattr(ds, k)).sum()) for k in ("P", "E", "Tm", "static", "band_H", "p_mean")}
    q_pct = 100.0 * np.isnan(ds.q).mean()
    ok = all(v == 0 for v in nan.values()) and round(q_pct, 4) == REPORT_NAN_PCT
    check("missing data", ok, f"NaN counts {nan}; discharge NaN {q_pct:.4f} % (report {REPORT_NAN_PCT} %)")

    # periods
    tr, te = ds.mask("train"), ds.mask("test")
    d = lambda s: date.fromisoformat(s)
    n_dhbv = (d(merit_data.DHBV_TRAIN[1]) - d(merit_data.DHBV_TRAIN[0])).days + 1
    te_days = [x for x, m in zip(ds.days, te) if m]
    tr_days = [x for x, m in zip(ds.days, tr) if m]
    ok = n_dhbv == 5478 and int(te.sum()) == 5479 and int(tr.sum()) == n_dhbv - merit_data.WARMUP_DAYS and tr_days[0] == d("1981-10-01") and tr_days[-1] == d("1995-09-30") \
        and te_days[0] == d("1995-10-01") and te_days[-1] == d("2010-09-30") and not (tr & te).any() and ds.days[0] == d("1980-10-01") \
        and int(np.argmax(tr)) == merit_data.WARMUP_DAYS and np.array_equal(ds.mask("val"), tr)
    # The brief and the comment at CONUS_water_loss_lumped_AORC_gradaccum.py:76 give 5479 days for 1980-10-01 .. 1995-09-30.
    # The calendar (and pd.date_range of that script) gives 5478; the test period has 5479. The dates decide here.
    check("periods", ok, f"dHBV train window {n_dhbv} d (the brief says 5479; the calendar gives 5478); loss mask {int(tr.sum())} d ({tr_days[0]} .. {tr_days[-1]}) = {n_dhbv} - {merit_data.WARMUP_DAYS}; "
          f"first loss day index {int(np.argmax(tr))}; test mask {int(te.sum())} d ({te_days[0]} .. {te_days[-1]}); overlap {int((tr & te).sum())} d; "
          f"'val' (end marker) equal to the loss mask: {np.array_equal(ds.mask('val'), tr)}")

    # attribute scaling against the water_loss functions (the lines of CONUS_water_loss_lumped_AORC_gradaccum.py:105-115)
    scale = by_path("wl_scale", SCALE_PY)
    attr2, names = z["attrs"][:], list(z.attrs["attr_list"])
    stat_dict = {n_: scale.cal_stat(attr2[:, i]) for i, n_ in enumerate(names)}
    ref = scale.trans_norm(attr2, names, stat_dict, to_norm=True)
    ref[ref != ref] = 0
    mine, mean, std = merit_data.scale_attrs(attr2)
    same = same_bits(mine, ref.astype(np.float32))
    m, s = mine.astype(np.float64).mean(0), mine.astype(np.float64).std(0)
    ok = same and np.isfinite(ds.static).all() and same_bits(ds.static, mine) and np.abs(m).max() < 1e-4 and np.abs(s - 1).max() < 1e-4 \
        and ds.static_names == names
    check("attribute scaling", ok, f"equal to scale.cal_stat + scale.trans_norm bit for bit (float32): {same}; all finite: {bool(np.isfinite(ds.static).all())}; "
          f"column means max |.| {np.abs(m).max():.2e}; column stds max |. - 1| {np.abs(s - 1).max():.2e}; range {mine.min():.2f} .. {mine.max():.2f}; "
          f"stds below the 0.001 floor: {int((np.array([stat_dict[n_][3] for n_ in names]) == 1).sum())}")
    print("          attribute      mean          std")
    for n_, mu, sd in zip(names, mean, std):
        print(f"          {n_:<14s} {mu:<13.6g} {sd:.6g}")

    # metric code against forward_lumped_eval.py on the same arrays
    wl = by_path("wl_eval", EVAL_PY)
    obs = z["flow_mmday"][:][:, te]                                               # [N, test days] float32
    sims = {"surrogate (0.35 x P)": (0.35 * z["P"][:][:, te]).astype(np.float32)}
    ids = staid
    if a.run:
        import json
        K = json.load(open(Path(a.run) / "run.json"))["args"]["K"] or N
        sims[f"run {Path(a.run).name}"] = np.load(Path(a.run) / "sim.npy")[te].T
        assert sims[f"run {Path(a.run).name}"].shape[0] == K
    for label, sim in sims.items():
        n = sim.shape[0]
        df = eval_merit.metrics_table(ids[:n], sim, obs[:n])
        ref_nse = np.array([wl.nse(sim[i], obs[i]) for i in range(n)])
        ref_kge = np.array([wl.kge(sim[i], obs[i]) for i in range(n)])
        ref_pb = np.array([100.0 * (np.nansum(sim[i]) - np.nansum(obs[i])) / np.nansum(obs[i]) for i in range(n)])
        dn = np.nanmax(np.abs(df.NSE.values - ref_nse)); dk = np.nanmax(np.abs(df.KGE.values - ref_kge)); dp = np.nanmax(np.abs(df.PBIAS_pct.values - ref_pb))
        same_nan = np.array_equal(np.isnan(df.NSE.values), np.isnan(ref_nse)) and np.array_equal(np.isnan(df.KGE.values), np.isnan(ref_kge))
        ok = dn < 1e-6 and dk < 1e-6 and dp < 1e-6 and same_nan and list(df.columns) == ["STAID", "NSE", "KGE", "PBIAS_pct"]
        check(f"metric code, {label}", ok, f"{n} gauges; max |dNSE| {dn:.1e}, max |dKGE| {dk:.1e}, max |dPBIAS| {dp:.1e}; same NaN gauges: {same_nan}; "
              f"NaN NSE {int(np.isnan(ref_nse).sum())}; columns {list(df.columns)}")

    print(f"## check_merit_data: {'all checks passed' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

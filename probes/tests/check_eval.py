"""Checks of the NSE evaluation of a trained CaRT+ MERIT run (probes/eval_merit.py). CPU only.

Author: tbindas

What it does:
  formulas      nse, kge and pbias_pct of eval_merit.py on small arrays with known answers (exact simulation, mean
                simulation, a hand-computed case, NaN in the observations, fewer than 2 valid days, constant observations).
  eval CSV      with --run: the CSV that eval_merit.py wrote for the run against a second computation that does not use
                eval_merit.py or merit_data.py (vectorised float64, straight from the zarr store and sim.npy):
                gauge ids and order, row count, column names, NSE, KGE and PBIAS per gauge.
  training NSE  with --run: the NSE of the CaRT+ training code (cart.train.nse_score, the function inside the loss) on the
                test period against the NSE of eval_merit.py. Both must give the same number per gauge (float32 limits).
  dHBV CSV      with --run: the run CSV and the dHBV CSV hold the same gauges in the same order, so rows compare one to one.
                The medians of both are printed. This is information, not a pass / fail on model skill.

    python probes/tests/check_eval.py [--run <run dir>] [--eval-csv <csv of eval_merit.py>] [--dhbv-csv <csv>]

Experimental purpose: the equifinality paper compares the test-period NSE of CaRT+ with the NSE of lumped dHBV per gauge.
This check is the evidence that the CaRT+ number is computed correctly and on the same gauges, days and formula.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(PKG / "src"))
import eval_merit  # noqa: E402

STORE = "/mnt/ssd1/data/water_loss/lumped_training_inputs.zarr"
DHBV_CSV = "/mnt/ssd1/data/water_loss/models/CONUS2717_AORC2F_LUMPED_v2_gradaccum/gage_eval_metrics_ep100.csv"
TEST = ("1995-10-01", "2010-09-30")
COLS = ["STAID", "NSE", "KGE", "PBIAS_pct"]

fails = []


def check(name, ok, msg):
    print(f"{'PASSED' if ok else 'FAILED'}  {name}: {msg}", flush=True)
    if not ok:
        fails.append(name)


def formulas():
    f = np.float32
    obs = np.array([1, 2, 3, 4, 5], f)
    r = {}
    r["exact sim: NSE 1, KGE 1, PBIAS 0"] = eval_merit.nse(obs, obs) == 1 and abs(eval_merit.kge(obs, obs) - 1) < 1e-7 and eval_merit.pbias_pct(obs, obs) == 0
    mean_sim = np.full(5, 3, f)
    r["sim = mean of obs: NSE 0, KGE NaN (sim has no variance)"] = eval_merit.nse(mean_sim, obs) == 0 and np.isnan(eval_merit.kge(mean_sim, obs))
    sim = np.array([2, 2, 2, 5, 5], f)                               # errors 1, 0, -1, 1, 0 -> SSE 3; obs variance sum 10
    r["hand case: NSE 1 - 3/10"] = abs(eval_merit.nse(sim, obs) - 0.7) < 1e-7
    r["hand case: PBIAS 100 (16 - 15) / 15"] = abs(eval_merit.pbias_pct(sim, obs) - 100.0 / 15.0) < 1e-5
    rr = np.corrcoef(sim.astype(float), obs.astype(float))[0, 1]
    kg = 1 - np.sqrt((rr - 1) ** 2 + (sim.astype(float).std() / obs.astype(float).std() - 1) ** 2 + (16 / 15 - 1) ** 2)
    r["hand case: KGE (2009) from r, std ratio, mean ratio"] = abs(eval_merit.kge(sim, obs) - kg) < 1e-6
    r["2 x sim: KGE 1 - sqrt(2), NSE 1 - 55/10"] = abs(eval_merit.kge(2 * obs, obs) - (1 - np.sqrt(2))) < 1e-6 and abs(eval_merit.nse(2 * obs, obs) + 4.5) < 1e-6
    obs_nan = np.array([1, np.nan, 3, 4, 5], f)                      # day 2 is left out: obs 1, 3, 4, 5 (mean 3.25), sim 2, 2, 5, 5
    o, s = np.array([1, 3, 4, 5.0]), np.array([2, 2, 5, 5.0])
    r["NaN obs day left out of NSE"] = abs(eval_merit.nse(sim, obs_nan) - (1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())) < 1e-6
    r["NaN obs day: PBIAS sums sim over all days (dHBV line)"] = abs(eval_merit.pbias_pct(sim, obs_nan) - 100.0 * (16 - 13) / 13) < 1e-5
    one = np.array([np.nan, np.nan, 3, np.nan, np.nan], f)
    r["fewer than 2 valid days: NaN"] = np.isnan(eval_merit.nse(sim, one)) and np.isnan(eval_merit.kge(sim, one))
    const = np.full(5, 2, f)
    r["constant obs: NaN"] = np.isnan(eval_merit.nse(sim, const)) and np.isnan(eval_merit.kge(sim, const))
    r["NSE can be below 0"] = eval_merit.nse(np.full(5, 10, f), obs) < 0
    bad = [k for k, v in r.items() if not v]
    check("formulas", not bad, f"{len(r) - len(bad)} of {len(r)} cases correct" + (f"; wrong: {bad}" if bad else ""))


def second_computation(sim, obs):
    """NSE, KGE, PBIAS per gauge, vectorised in float64. sim, obs [N, days]; obs has NaN. No code of eval_merit.py."""
    sim, obs = sim.astype(np.float64), obs.astype(np.float64)
    ok = np.isfinite(obs)
    n = ok.sum(1)
    om = np.where(ok, obs, 0).sum(1) / n
    sm = np.where(ok, sim, 0).sum(1) / n
    do, dsim = np.where(ok, obs - om[:, None], 0), np.where(ok, sim - sm[:, None], 0)
    sse = np.where(ok, (sim - obs) ** 2, 0).sum(1)
    nse = 1 - sse / (do ** 2).sum(1)
    r = (do * dsim).sum(1) / np.sqrt((do ** 2).sum(1) * (dsim ** 2).sum(1))
    alpha = np.sqrt((dsim ** 2).sum(1) / (do ** 2).sum(1))
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (sm / om - 1) ** 2)
    pb = 100 * (sim.sum(1) - np.where(ok, obs, 0).sum(1)) / np.where(ok, obs, 0).sum(1)
    return nse, kge, pb, n


def run_checks(run, eval_csv, dhbv_csv):
    import torch
    from cart.train import nse_score
    rj = json.load(open(run / "run.json"))
    z = zarr.open_group(STORE, mode="r")
    K = rj["args"]["K"] or z["gage_staid"].shape[0]
    days = pd.date_range(z.attrs["time_start"], z.attrs["time_end"], freq="D")
    te = np.asarray((days >= TEST[0]) & (days <= TEST[1]))
    staid = z["gage_staid"][:][:K]
    obs = z["flow_mmday"][:][:K][:, te]                              # [K, test days] float32
    sim = np.load(run / "sim.npy")[te].T                             # [K, test days] float32
    df = pd.read_csv(eval_csv, dtype={"STAID": str})

    nse2, kge2, pb2, nval = second_computation(sim, obs)
    dn, dk, dp = (float(np.nanmax(np.abs(df[c].values - v))) for c, v in (("NSE", nse2), ("KGE", kge2), ("PBIAS_pct", pb2)))
    ids_ok = list(df.STAID) == [str(s) for s in staid] and all(len(s) == 8 for s in df.STAID)
    # float32 sums in eval_merit.py against float64 here: 1e-4 on NSE and KGE, 1e-2 % on PBIAS (relative for large values)
    tol = lambda c, v, t: bool(np.all(np.abs(df[c].values - v) <= t * np.maximum(1, np.abs(v))))
    ok = ids_ok and list(df.columns) == COLS and len(df) == K and int(te.sum()) == 5479 and sim.shape == obs.shape \
        and tol("NSE", nse2, 1e-4) and tol("KGE", kge2, 1e-4) and tol("PBIAS_pct", pb2, 1e-4) and not df[["NSE", "KGE", "PBIAS_pct"]].isna().any().any()
    check("eval CSV", ok, f"{eval_csv.name}: {len(df)} rows for {K} gauges; ids equal to gage_staid in order, 8 characters: {ids_ok}; columns {list(df.columns)}; "
          f"test days {int(te.sum())} ({days[te][0].date()} .. {days[te][-1].date()}); valid obs days per gauge min {int(nval.min())}, median {int(np.median(nval))}; "
          f"max |difference| to the second computation: NSE {dn:.1e}, KGE {dk:.1e}, PBIAS {dp:.1e} %; NaN scores {int(df[['NSE', 'KGE', 'PBIAS_pct']].isna().sum().sum())}")

    # the NSE inside the CaRT+ loss, on the test period
    Q, qg = torch.from_numpy(sim), torch.from_numpy(obs)
    okq = torch.isfinite(qg); qz = torch.where(okq, qg, torch.zeros_like(qg)); okf = okq.float()
    mean = (qz * okf).sum(1) / okq.sum(1).clamp(min=1)
    den = ((qz - mean[:, None]) * okf).pow(2).sum(1).clamp(min=1e-6)
    nse_tr = nse_score(Q, qz, okf, den, torch.ones(K, dtype=torch.bool)).numpy().astype(np.float64)
    d = np.abs(nse_tr - df.NSE.values)
    ok = bool(np.all(d <= 1e-3 * np.maximum(1, np.abs(df.NSE.values))))
    check("training NSE", ok, f"cart.train.nse_score against the NSE column on {K} gauges: max |difference| {d.max():.1e} (limit 1e-3, relative for |NSE| > 1); "
          f"median NSE {np.median(nse_tr):.4f} against {df.NSE.median():.4f}")

    # same gauges as the dHBV CSV
    dh = pd.read_csv(dhbv_csv, dtype={"STAID": str})
    same = list(dh.STAID[:K]) == list(df.STAID) and list(dh.columns) == COLS
    check("dHBV CSV", same and (K < len(dh) or len(dh) == K), f"{Path(dhbv_csv).parent.name}: {len(dh)} rows; the first {K} ids equal the run ids in order: {same}; columns {list(dh.columns)}")
    a, b = df, dh.iloc[:K]
    print(f"          information ({K} gauges, {rj['args']['iters']} iterations; smoke runs are not results):")
    print(f"          {'':<22s}{'CaRT+ run':>12s}{'lumped dHBV':>14s}")
    for c in ("NSE", "KGE"):
        print(f"          median {c:<15s}{a[c].median():>12.4f}{b[c].median():>14.4f}")
    print(f"          {'NSE > 0.55, share':<22s}{(a.NSE > 0.55).mean():>12.1%}{(b.NSE > 0.55).mean():>14.1%}")
    print(f"          {'NSE < 0, gauges':<22s}{int((a.NSE < 0).sum()):>12d}{int((b.NSE < 0).sum()):>14d}")
    print(f"          {'median |PBIAS| %':<22s}{a.PBIAS_pct.abs().median():>12.2f}{b.PBIAS_pct.abs().median():>14.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="run directory of train_merit.py that eval_merit.py has scored")
    ap.add_argument("--eval-csv", default="", help="default <run>/gage_eval_metrics_test.csv")
    ap.add_argument("--dhbv-csv", default=DHBV_CSV)
    a = ap.parse_args()
    formulas()
    if a.run:
        run = Path(a.run)
        run_checks(run, Path(a.eval_csv or run / "gage_eval_metrics_test.csv"), a.dhbv_csv)
    print(f"## check_eval: {'all checks passed' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

"""Evaluate a CaRT+ MERIT run on the test period: one CSV row per gauge.

Author: tbindas

What it does: reads sim.npy and run.json of a run written by probes/train_merit.py, takes the test period
1995-10-01 .. 2010-09-30 and writes STAID, NSE, KGE, PBIAS_pct per gauge.

    python probes/eval_merit.py --run /mnt/ssd1/data/cartplus/runs/<run> [--eval-csv <new file>]

Experimental purpose: a per-gauge score file that compares directly with the dHBV eval CSV of water_loss
(gage_eval_metrics_ep100.csv). The column names and the three formulas are those of
water_loss/.../water_loss_aorc/forward_lumped_eval.py (nse, kge, and the PBIAS line of run_eval). The functions are
copied, not imported, so this script does not need the water_loss repository.

Difference from the dHBV eval that remains: dHBV restarts on 1994-10-01 with one warm-up year before the test period.
CaRT+ simulates the whole record from 1980-10-01 in one pass, so its test-period states continue from the train period.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merit_data  # noqa: E402


def nse(sim, obs):
    """Nash-Sutcliffe efficiency, NaN-masked on obs. (forward_lumped_eval.py:51-60)"""
    m = ~np.isnan(obs) & ~np.isnan(sim)
    if m.sum() < 2:
        return np.nan
    o, s = obs[m], sim[m]
    denom = ((o - o.mean()) ** 2).sum()
    if denom == 0:
        return np.nan
    return 1.0 - ((s - o) ** 2).sum() / denom


def kge(sim, obs):
    """Kling-Gupta efficiency (2009), NaN-masked on obs. (forward_lumped_eval.py:63-74)"""
    m = ~np.isnan(obs) & ~np.isnan(sim)
    if m.sum() < 2:
        return np.nan
    o, s = obs[m], sim[m]
    if o.std() == 0 or o.mean() == 0 or s.std() == 0:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    alpha = s.std() / o.std()
    beta = s.mean() / o.mean()
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def pbias_pct(sim, obs):
    """the PBIAS line of forward_lumped_eval.run_eval: the sim sum runs over all days, the obs sum over days with obs"""
    return 100.0 * (np.nansum(sim) - np.nansum(obs)) / np.nansum(obs)


def metrics_table(ids, sim, obs):
    """ids [N]; sim, obs [N, days] float32 -> DataFrame STAID, NSE, KGE, PBIAS_pct"""
    return pd.DataFrame([{"STAID": str(s), "NSE": nse(sim[i], obs[i]), "KGE": kge(sim[i], obs[i]), "PBIAS_pct": pbias_pct(sim[i], obs[i])}
                         for i, s in enumerate(ids)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory written by probes/train_merit.py")
    ap.add_argument("--eval-csv", default="", help="output CSV; it must not exist (default <run>/gage_eval_metrics_test.csv)")
    a = ap.parse_args()
    run = Path(a.run)
    out = Path(a.eval_csv or run / "gage_eval_metrics_test.csv")
    if out.exists():
        sys.exit(f"{out} exists; pass a new path (nothing is overwritten)")
    rj = json.load(open(run / "run.json"))
    ds = merit_data.load(rj["args"]["K"] or None)
    sim = np.load(run / "sim.npy")                                                             # [days, gauges] float32
    assert sim.shape == ds.q.shape, (sim.shape, ds.q.shape)
    m = ds.mask("test")
    df = metrics_table(ds.ids, sim[m].T, ds.q[m].T.astype(np.float32))                         # float32 obs, as the store holds it
    df.to_csv(out, index=False)
    print(df[["NSE", "KGE"]].median().rename("median"))
    print(f"NSE>0.55: {(df.NSE > 0.55).mean():.1%} of gages; test days {int(m.sum())}; wrote {out}")


if __name__ == "__main__":
    main()

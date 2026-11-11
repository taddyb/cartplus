"""Checks of smoke runs written by probes/train_merit.py. CPU only (reads run.json and sim.npy).

Author: tbindas

What it does:
  smoke run   the loss is finite at each iteration, no step was skipped for a non-finite loss or gradient, sim.npy is
              finite and has the shape [10957 days, gauges], the training simulation covers 5478 days (1980-10-01 .. 1995-09-30) and the loss 5113 days.
  same seed   with --same <second run>: both runs have the same seed, gauges and iterations, and the same loss at each
              iteration (exact float equality), and the same sim.npy.

    python probes/tests/check_smoke.py --run <run dir> [--same <run dir of a second run with the same arguments>]

Experimental purpose: evidence that the training scaffolding for CaRT+ on the dHBV MERIT setup is stable and repeatable
before the full run starts.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

fails = []


def check(name, ok, msg):
    print(f"{'PASSED' if ok else 'FAILED'}  {name}: {msg}", flush=True)
    if not ok:
        fails.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--same", default="")
    a = ap.parse_args()
    rj = json.load(open(Path(a.run) / "run.json"))
    h = rj["history"]
    loss = np.array([x["loss"] for x in h]); gn = np.array([x["gnorm"] for x in h])
    sim = np.load(Path(a.run) / "sim.npy")
    n = rj["n_gauges"]
    ok = len(h) == rj["args"]["iters"] and np.isfinite(loss).all() and np.isfinite(gn).all() and rj["nskip"] == 0 \
        and not any(x["skipped"] for x in h) and sim.shape == (10957, n) and np.isfinite(sim).all() \
        and rj["sim_days_train"] == 5478 and rj["loss_days"] == 5113 and rj["sim_days_all"] == 10957
    check(f"smoke run {Path(a.run).name}", ok,
          f"{n} gauges ({rj['n_gauges_in_loss']} in the loss), {len(h)} iterations; non-finite losses {int((~np.isfinite(loss)).sum())}, "
          f"non-finite gradient norms {int((~np.isfinite(gn)).sum())}, skipped steps {rj['nskip']}; loss first {loss[0]:.6f}, last {loss[-1]:.6f}; "
          f"sim.npy {sim.shape}, non-finite {int((~np.isfinite(sim)).sum())}, min {np.nanmin(sim):.3g}; training simulation {rj['sim_days_train']} d, "
          f"loss {rj['loss_days']} d; {rj['timing']['median_s_per_iter']:.3f} s/iter (median), peak GPU {rj['peak_gpu_training']['allocated_GiB']:.2f} GiB "
          f"allocated / {rj['peak_gpu_training']['reserved_GiB']:.2f} GiB reserved")
    if a.same:
        r2 = json.load(open(Path(a.same) / "run.json"))
        l2 = np.array([x["loss"] for x in r2["history"]])
        keys = ("K", "seed", "iters", "lr", "snoise")
        same_args = all(rj["args"][k] == r2["args"][k] for k in keys)
        same_loss = loss.shape == l2.shape and bool((loss == l2).all())
        s2 = np.load(Path(a.same) / "sim.npy")
        dmax = float(np.abs(loss - l2).max()) if loss.shape == l2.shape else float("nan")
        same_sim = sim.shape == s2.shape and sim.tobytes() == s2.tobytes()
        check("same seed", same_args and same_loss,
              f"{Path(a.run).name} against {Path(a.same).name}: seed {rj['args']['seed']}; same arguments {same_args}; loss equal at each of "
              f"{len(loss)} iterations: {same_loss} (max |difference| {dmax:.1e}); sim.npy equal bit for bit: {same_sim}"
              + ("" if same_sim or s2.shape != sim.shape else f" (max |dQ| {float(np.abs(sim - s2).max()):.1e} mm/d)"))
    print(f"## check_smoke: {'all checks passed' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

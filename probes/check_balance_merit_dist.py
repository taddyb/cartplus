"""Run the repository water-balance check (tests/check_balance.py) on the divides of a distributed CaRT+ run.

Author: tbindas

What it does: tests/check_balance.py loads its dataset with cart.data.load and builds cart.model.CaRT on it. This
script replaces cart.data.load IN THIS PROCESS with a function that returns, for the name "merit", a Dataset with one
column per divide (merit_dist_data.divide_dataset). Then it calls the unchanged main() of tests/check_balance.py with
the mapping network of the run (net.pt). No file of src/cart/ or tests/ is edited. The bookkeeping, the tolerance
(1e-5 of the accumulated precipitation) and the output are those of the repository.

Which divides: the first --n divides (store order) of the gauges of the run that have forcing. The 71 divides whose
forcing is NaN in the store get P = 0; a residual relative to P is not defined for them, so they are left out here.
probes/train_merit_dist.py prints their absolute store residual. The check covers the divide step only. The gauge
step (area-weighted sum, unit hydrograph) is checked by probes/tests/check_dist.py --run.

    python probes/check_balance_merit_dist.py --run /mnt/ssd1/data/cartplus/runs/<run> [--n 500] [--gpu 0]

Experimental purpose: the brief asks for a closed water balance per divide on a smoke run, with the method of
probes/check_balance_merit.py. The divides have NaN-filled attributes and forcing that the lumped run never saw.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "tests"))
sys.path.insert(0, str(HERE))
import cart.data as D  # noqa: E402
import check_balance  # noqa: E402
import merit_dist_data as mdd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    plan = json.load(open(Path(a.run) / "chunk_plan.json"))
    dd = mdd.load()
    rows = np.unique(np.concatenate([np.asarray(c["rows"]) for c in plan["chunks"]]))
    dry = np.isin(rows, dd.nan["forcing_all_nan_rows"])
    rows = rows[~dry][:a.n]
    print(f"divides of the run {len(dry)}, without forcing {int(dry.sum())}; checked: the first {len(rows)} with forcing", flush=True)
    ds = mdd.divide_dataset(dd, rows)
    repo_load = D.load

    def load(name, *args, **kw):
        return ds if name == "merit" else repo_load(name, *args, **kw)
    D.load = load                                                                # check_balance calls D.load(dsn)
    sys.argv = [sys.argv[0], "--run", a.run, "--K", str(len(rows)), "--gpu", str(a.gpu)]   # --K = all columns: no compare with sim.npy (gauges)
    check_balance.main()


if __name__ == "__main__":
    main()

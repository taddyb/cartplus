"""Print the per-catchment parameters of the CaRT+ preset that probes/train_merit.py trains. CPU only.

Author: tbindas

What it does: builds config_for("cartplus", "merit") and prints each parameter of cart.parameters.static_table and
law_table with its bounds and start value, then the count. Names that start with "l" are logarithms (not the lam* coefficients); for those the
bounds are also printed as exp(.).

    python probes/tests/check_params.py

Experimental purpose: the parameter learnability census of the equifinality paper needs the exact list of CaRT+
parameters and their bounds on the MERIT setup. This is the record of that list.
"""
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "src"))
from cart.config import config_for  # noqa: E402
from cart.parameters import law_table, static_table  # noqa: E402


def main():
    cfg = config_for("cartplus", "merit")
    S = static_table(cfg)
    L = law_table(cfg) if cfg.fast_law_der else {}
    print(f"preset {cfg.name}, dataset merit; fc_min {getattr(cfg, 'fc_min', 'n/a')}")
    print(f"{'table':<7s}{'name':<8s}{'lo':>12s}{'hi':>12s}{'init':>12s}   {'exp(lo)':>10s}{'exp(hi)':>10s}{'exp(init)':>10s}")
    for tab, T in (("static", S), ("law", L)):
        for k, v in T.items():
            lo, hi, p0 = (float(x) for x in v[:3])
            ex = f"   {np.exp(lo):>10.4g}{np.exp(hi):>10.4g}{np.exp(p0):>10.4g}" if k.startswith("l") and not k.startswith("lam") else ""
            print(f"{tab:<7s}{k:<8s}{lo:>12.5g}{hi:>12.5g}{p0:>12.5g}{ex}")
            assert lo < hi and lo <= p0 <= hi, (k, v)
    print(f"count: {len(S)} static + {len(L)} law = {len(S) + len(L)} parameters per catchment")
    print("PASSED  parameters: each parameter has lo < hi and lo <= init <= hi")


if __name__ == "__main__":
    main()

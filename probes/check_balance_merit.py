"""Run the repository water-balance check (tests/check_balance.py) on a CaRT+ MERIT run.

Author: tbindas

What it does: tests/check_balance.py loads its dataset with cart.data.load, which knows gb, us and aus only. This script
replaces cart.data.load IN THIS PROCESS with a function that returns the MERIT dataset of probes/merit_data.py for the
name "merit" (with the gauge subset of the run), then calls the unchanged main() of tests/check_balance.py. No file of
src/cart/ or tests/ is edited. The balance bookkeeping, the tolerance (1e-5) and the output are those of the repository.

    python probes/check_balance_merit.py --run /mnt/ssd1/data/cartplus/runs/<run> [--gpu 0] [--json <new file>]

Experimental purpose: the brief asks that `tests/check_balance.py --run <smoke run>` passes on the MERIT setup. One
elevation band and the MERIT forcing are new inputs for the CaRT+ kernels; a closed water balance shows they are handled.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "tests"))
sys.path.insert(0, str(HERE))
import cart.data as D  # noqa: E402
import check_balance  # noqa: E402
import merit_data  # noqa: E402


def main():
    if "--run" not in sys.argv or "--K" in sys.argv:
        sys.exit("pass --run <run dir>; --K is taken from run.json")
    run = Path(sys.argv[sys.argv.index("--run") + 1])
    K = json.load(open(run / "run.json"))["args"]["K"] or None
    repo_load = D.load

    def load(name, *args, **kw):
        return merit_data.load(K) if name == "merit" else repo_load(name, *args, **kw)
    D.load = load                                                                # check_balance calls D.load(dsn)
    check_balance.main()


if __name__ == "__main__":
    main()

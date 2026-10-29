"""Draw the CAMELS-US held-out ("ungauged") validation set: 10 % of the 671 basins, stratified random by terciles of snow fraction x
aridity (9 strata, round(10 %) of each), numpy default_rng(0).

    python scripts/data/us/draw_ungauged_10pct.py <camels_us_daymet671.npz> <out.txt>
The draw uses the static_raw table of camels_us_daymet671.npz (CAMELS v1.2 attributes) and gives data_lists/us_ungauged_10pct.txt;
with --check <list> it compares the draw with an existing list instead of writing one.
"""
import sys

import numpy as np
import pandas as pd


def draw(npz):
    z = np.load(npz)
    St = pd.DataFrame(z["static_raw"], columns=[str(s) for s in z["static_names"]])
    sn = pd.qcut(St.frac_snow.rank(method="first"), 3, labels=False)
    ai = pd.qcut(St.aridity.rank(method="first"), 3, labels=False)
    strata = (sn * 3 + ai).to_numpy()
    rng = np.random.default_rng(0)
    pick = []
    for s in range(9):
        idx = np.where(strata == s)[0]
        n = int(round(0.1 * len(idx)))
        pick += list(rng.choice(idx, n, replace=False))
    return [str(z["ids"][i]).zfill(8) for i in pick]


if __name__ == "__main__":
    ids = draw(sys.argv[1])
    if "--check" in sys.argv:
        ref = [l.strip() for l in open(sys.argv[sys.argv.index("--check") + 1]) if l.strip() and not l.startswith("#")]
        print(f"{len(ids)} drawn; same set as {sys.argv[-1]}: {set(ids) == set(ref)}; same order after sorting: {sorted(ids) == ref}")
        sys.exit(0 if set(ids) == set(ref) else 1)
    with open(sys.argv[2], "w") as f:
        f.write("# CAMELS-US ungauged validation set: 10 % of the 671 basins, stratified random (terciles of frac_snow x aridity, 10 % per "
                "stratum, numpy default_rng(0)); never used in training or model selection\n" + "\n".join(sorted(ids)) + "\n")

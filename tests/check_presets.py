"""Checks of the model presets and of the run tables (CPU only, nothing is trained).

  * every preset builds (ModelConfig checks) on every dataset, and its mapping network builds with the parameter tables of the preset;
  * cartplus_no_p<i> differs from cartplus exactly in the switches of group P<i> (config.GROUPS), reset to the ModelConfig defaults
    except for the replacements of config.REPLACEMENT (P3: uptake kernel kept with lambda_E = 0; P4: loss-only GR4J exchange);
  * on CAMELS-AUS the presets with infiltration excess have fc_min = 1e-4 mm/d, the others none;
  * configs/paper_runs.csv and configs/paper_forward.csv name only existing presets, datasets and modules, and configs/train_all.sh runs
    exactly their commands.

    python tests/check_presets.py
"""
import csv
import importlib.util
import shlex
import sys
from dataclasses import fields
from pathlib import Path

import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))
from cart.config import GROUPS, PRESETS, REPLACEMENT, ModelConfig, config_for  # noqa: E402
from cart.parameters import MappingNetwork, head_rows, law_table, static_table  # noqa: E402

N_OK, N_BAD = 0, 0
MODULES = {"cart.train", "cart.predict", "baselines.hbv_regional", "baselines.dhbv", "baselines.lstm_stateful", "baselines.ealstm"}


def check(name, ok, info=""):
    global N_OK, N_BAD
    N_OK += bool(ok); N_BAD += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({info})" if info else ""), flush=True)


def diff(a, b):
    return {f.name for f in fields(ModelConfig) if f.name != "name" and getattr(a, f.name) != getattr(b, f.name)}


def main():
    print("PRESET CHECK")
    plus = PRESETS["cartplus"]
    for name, cfg in PRESETS.items():
        for ds in ("gb", "us", "aus"):
            c = config_for(name, ds)
            check(f"{name} on {ds}: fc_min {c.fc_min}", c.fc_min == (1e-4 if ds == "aus" and c.ie else 0.0))
        torch.manual_seed(0)
        net = MappingNetwork(27, cfg, "cpu")
        h, ps = net(torch.zeros(3, 27))
        n_static, n_law = len(static_table(cfg)), len(law_table(cfg)) if cfg.fast_law_der else 0
        rows = head_rows(cfg)
        check(f"{name}: mapping network, {n_static} static + {n_law} law parameters",
              list(ps) == list(static_table(cfg)) and net.shead.out_features == n_static
              and (rows is None) == (len(static_table(cfg, full=True)) == n_static),
              "inactive X5e row dropped" if rows is not None else "")
    for i in range(1, 6):
        g = f"P{i}"
        cfg = PRESETS[f"cartplus_no_p{i}"]
        want = set(GROUPS[g]) | set(REPLACEMENT.get(g, {}))
        want = {k for k in want if getattr(plus, k) != getattr(cfg, k)}
        ok = diff(plus, cfg) == want and all(getattr(cfg, k) == getattr(ModelConfig(), k) for k in GROUPS[g] if k not in REPLACEMENT.get(g, {}))
        ok &= all(getattr(cfg, k) == v for k, v in REPLACEMENT.get(g, {}).items())
        check(f"cartplus_no_p{i} = cartplus without {g}", ok, ", ".join(sorted(diff(plus, cfg))))
    check("cart: CaRT core with loss-only exchange", diff(ModelConfig(), PRESETS["cart"]) == {"exch_loss"})

    cmds = []
    for f in ("paper_runs.csv", "paper_forward.csv"):
        rows = list(csv.DictReader(open(PKG / "configs" / f)))
        bad = []
        for r in rows:
            argv = shlex.split(r["command"])
            mod = argv[2]
            opts = dict(zip(argv[3::2], argv[4::2]))
            if mod not in MODULES or importlib.util.find_spec(mod) is None:
                bad.append(f"module {mod}")
            if mod == "cart.train" and (opts.get("--model") != r["model"] or r["model"] not in PRESETS or opts.get("--dataset") != r["dataset"]):
                bad.append(r["command"])
            if mod.startswith("baselines") and opts.get("--dataset") != r["dataset"]:
                bad.append(r["command"])
            if mod == "cart.predict" and opts.get("--run") != f"runs/{r['dataset']}_{r['model']}_s{r['seed']}":
                bad.append(r["command"])
            cmds.append(r["command"] + " --gpu $G")
        check(f"configs/{f}: {len(rows)} commands consistent", not bad, "; ".join(bad[:3]))
    sh = [l.strip() for l in open(PKG / "configs" / "train_all.sh") if l.startswith("python ")]
    check("configs/train_all.sh runs exactly the commands of the two tables", sh == cmds, f"{len(sh)} lines, {len(cmds)} commands")
    print(f"PRESET CHECK: {N_OK} passed, {N_BAD} failed -> {'PASSED' if N_BAD == 0 else 'FAILED'}")
    sys.exit(0 if N_BAD == 0 else 1)


if __name__ == "__main__":
    main()

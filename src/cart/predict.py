"""Forward runs of a trained CaRT / CaRT+ network (no training): the simulation of the whole record, and the parameter-space ensemble of
the paper (Section 5.5).

    python -m cart.predict --run runs/gb_cartplus_s0 [--gpu 0]                          -> runs/gb_cartplus_s0/sim_forward.npy
    python -m cart.predict --run runs/gb_cartplus_s0 --noise 0.3 [--draws 32]           -> runs/gb_cartplus_s0/sim_noise0.3.npy
    python -m cart.predict --run runs/aus_cartplus_s0 --donor donors.npy --name donor5  -> runs/aus_cartplus_s0/sim_donor5.npy

Parameter-space ensemble (--noise SIGMA --draws K): for k = 0 .. K-1 the random generator is seeded with 1000 + k, Gaussian noise of
s.d. SIGMA is added to the standardised attributes of all catchments (as the training noise, one draw per catchment and attribute), the
model is run over the whole record without gradients, and the K flows are averaged in float64; the mean is written as float32 [T, C]
(mm/d). The same seeds give the same draws for every SIGMA.
Donor ensemble (--donor FILE): FILE is an int array [K, C]; in run k catchment c takes the attributes of catchment FILE[k, c]; the K
flows are averaged (with --noise, the noise is added to the donor attributes).
Without --noise and --donor the plain forward run is written (the sim.npy of a training run).

--run is a run directory written by cart.train (run.json, net.pt); the preset and dataset are read from run.json (--model / --dataset
override them).
"""
import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from . import data as data_mod
from .config import PRESETS, ModelConfig, config_for
from .model import CaRT

SEED0 = 1000                                                                                    # draw k uses torch.manual_seed(SEED0 + k)


def run_config(run, model=None, dataset=None):
    """-> (ModelConfig, dataset name) of a run directory written by cart.train"""
    rj = json.load(open(Path(run) / "run.json"))
    cfg = ModelConfig(**{k: v for k, v in rj["config"].items() if k in {f.name for f in fields(ModelConfig)}})
    d = dataset or rj["args"]["dataset"]
    return (config_for(model, d) if model else cfg), d


def load_model(run, ds, cfg, dev):
    """CaRT on dataset ds with the trained weights of the run"""
    model = CaRT(ds, cfg, dev)
    model.net.load_state_dict(torch.load(Path(run) / "net.pt", map_location=dev))
    return model


def forward(model):
    """the whole record, all catchments, no noise: [T, C] float32 (= sim.npy of the training run)"""
    with torch.no_grad():
        return model.simulate(TT=model.T)[0].cpu().numpy().T.astype(np.float32)


def ensemble(model, sigma=0.0, draws=32, donor=None):
    """mean flow [T, C] float32 of the parameter-space ensemble: draws runs with attribute noise of s.d. sigma (seeds SEED0 + k), or
    donor.shape[0] runs with the donor attributes donor[k] (int [K, C]), accumulated in float64"""
    K = donor.shape[0] if donor is not None else draws
    acc = None
    with torch.no_grad():
        for k in range(K):
            torch.manual_seed(SEED0 + k)
            xin = None if donor is None else model.xs[torch.from_numpy(donor[k].astype(np.int64)).to(model.dev)]
            Qk = model.simulate(noise=sigma, TT=model.T, xs_in=xin)[0]
            acc = Qk.double() if acc is None else acc + Qk.double()
    return (acc / K).cpu().numpy().T.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory written by cart.train (run.json, net.pt)")
    ap.add_argument("--model", default=None, choices=sorted(PRESETS), help="preset (default: from run.json)")
    ap.add_argument("--dataset", default=None, choices=["gb", "us", "aus"], help="dataset (default: from run.json)")
    ap.add_argument("--noise", type=float, default=0.0, help="s.d. of the attribute noise of the parameter-space ensemble (0 = none)")
    ap.add_argument("--draws", type=int, default=32, help="--noise: number of draws")
    ap.add_argument("--donor", default="", help="npy int [K, C]: donor ensemble, run k takes the attributes of catchment donor[k, c]")
    ap.add_argument("--name", default="", help="output sim_<name>.npy (default forward, noise<SIGMA>, or donor)")
    ap.add_argument("--out", default="", help="output file (default <run>/sim_<name>.npy)")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    cfg, dsn = run_config(a.run, a.model, a.dataset)
    name = a.name or (f"noise{a.noise}" if a.noise > 0 else "donor" if a.donor else "forward")
    out = Path(a.out or Path(a.run) / f"sim_{name}.npy")
    dev = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(dev)                                                                 # Triton kernels launch on the current device
    ds = data_mod.load(dsn)
    model = load_model(a.run, ds, cfg, dev)
    if a.noise > 0 or a.donor:
        donor = np.load(a.donor) if a.donor else None
        Q = ensemble(model, a.noise, a.draws, donor)
        what = f"{donor.shape[0] if donor is not None else a.draws} runs averaged"
    else:
        Q, what = forward(model), "forward run"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, Q)
    print(f"{cfg.name} on {dsn} (run {a.run}): {what} -> {out} {Q.shape}", flush=True)


if __name__ == "__main__":
    main()

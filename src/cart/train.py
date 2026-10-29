"""Train CaRT / CaRT+ (one regional model per dataset and seed) and write its daily simulation.

    python -m cart.train --model cartplus --dataset gb --seed 0 [--gpu 0] [--out runs/gb_cartplus_s0]

Datasets: gb (CAMELS-GB), us (CAMELS-US) and aus (CAMELS-AUS v2, read and run like us; see data.py). The configuration is the preset
as run on the dataset (config.config_for: on aus the presets with infiltration excess bound F_c below at 1e-4 mm/d); --fc_min
overrides that bound. --iters sets the number of iterations (at least 41 for the one-cycle schedule; e.g. --iters 50 --eval_every 25 for a quick test).

Training: all gauged training catchments at once, loss = mean over catchments of (1 - NSE) on 1989-2008 (the simulation starts at the
first day of the record, the years before 1989 are spin-up, and it is truncated at the end of 2010 while training), AdamW (lr 2e-3,
weight decay 1e-4), one-cycle schedule (5 % warm-up), gradient-norm clipping 1, Gaussian noise of SD 0.1 on the standardised attributes,
5000 iterations; the checkpoint with the best median NSE of the gauged catchments over 2009-2010 (evaluated every 50 iterations) is kept.
Outputs in --out: sim.npy [T, N] float32 (mm/d, whole record, all catchments incl. held-out), net.pt (mapping network state dict),
run.json (arguments, selection history), and a RESULT line on stdout (median NSE of gauged / held-out catchments, diagnostics only).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data as data_mod
from .config import PRESETS, config_for
from .model import CaRT


def nse_score(Q, qz, okf, den, sel):
    TT = Q.shape[1]
    nse = 1 - (((Q - qz[:, :TT]) * okf[:, :TT]) ** 2).sum(1) / den
    return nse[sel]


def clip_grad_norm(params, max_norm, net):
    """torch.nn.utils.clip_grad_norm_ with the norm taken at the full head size: where the full static head has the inactive X5e row
    (see parameters.py), the per-tensor norms of the head's gradients are taken with a zero row in its place, because the reduction
    rounds differently for a tensor of another size (1 ulp of the norm otherwise)."""
    if net.rows is None:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    n_r = net.shead.out_features + 1
    with torch.no_grad():
        gl = []
        for p in params:
            if p.grad is None:
                continue
            g = p.grad
            if p is net.shead.weight or p is net.shead.bias:
                g_ = torch.zeros((n_r,) + tuple(g.shape[1:]), device=g.device, dtype=g.dtype)
                g_[net.rows] = g
                g = g_
            gl.append(g)
        total = torch.nn.utils.get_total_norm(gl, 2.0)
        torch.nn.utils.clip_grads_with_norm_(params, max_norm, total)
    return total


def train(ds, cfg, seed, iters=5000, lr=2e-3, snoise=0.1, eval_every=50, dev="cuda:0", on_step=None, verbose=True):
    """Returns (model with the selected weights, best selection score, history). on_step(it, loss, gnorm) is called every iteration."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = CaRT(ds, cfg, dev)
    net = model.net
    qg = torch.from_numpy(ds.q.T.astype(np.float32)).to(dev)
    okq = torch.isfinite(qg); qz = torch.where(okq, qg, torch.zeros_like(qg))

    def masks(mask):
        ok = okq & mask[None]; okf = ok.float()
        mean = (qz * okf).sum(1) / ok.sum(1).clamp(min=1)
        return okf, ((qz - mean[:, None]) * okf).pow(2).sum(1).clamp(min=1e-6), ok.sum(1)
    okf_tr, den_tr, n_tr = masks(model.trm)
    okf_va, den_va, n_va = masks(torch.from_numpy(ds.mask("val")).to(dev))
    train_c = torch.from_numpy(ds.gauged).to(dev) & (n_tr > 730)                                # >= 2 years of training data
    val_c = torch.from_numpy(ds.gauged).to(dev) & (n_va > 365)

    params = net.train_parameters()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.05)
    best, best_state, hist, nskip = -9.0, None, [], 0
    t0 = time.time()
    if verbose:
        print(f"{cfg.name} seed {seed}: per-catchment parameters {len(net.static_par) + len(net.law_par)}: static {list(net.static_par)}"
              + (f", law {list(net.law_par)}" if net.law_par else ""), flush=True)
    for it in range(iters):
        Q, _, _ = model.simulate(snoise)
        ntr = nse_score(Q, qz, okf_tr, den_tr, train_c)
        loss = (1 - ntr).mean()
        opt.zero_grad(); loss.backward()
        gnorm = clip_grad_norm(params, 1.0, net)
        if on_step is not None:
            on_step(it, loss.item(), gnorm.item())
        if torch.isfinite(loss) and torch.isfinite(gnorm):
            opt.step()
        else:                                                                                  # skip a non-finite step (keeps the weights finite)
            nskip += 1
            if verbose and (nskip <= 5 or nskip % 50 == 0):
                print(f"  iter {it}: non-finite loss / gradient, step skipped ({nskip})", flush=True)
        sched.step()
        if it % eval_every == 0 or it == iters - 1:
            with torch.no_grad():
                v = nse_score(model.simulate()[0], qz, okf_va, den_va, val_c).median().item()
            hist.append({"iter": it, "train_median": ntr.median().item(), "val_median": v})
            if v > best:
                best, best_state = v, {k: x.detach().clone() for k, x in net.state_dict().items()}
            if verbose and (it % (4 * eval_every) == 0 or it == iters - 1):
                print(f"  iter {it:5d}  train median NSE {ntr.median().item():.4f}  val {v:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    net.load_state_dict(best_state)
    return model, best, hist


def nse_np(sim, o):
    k = np.isfinite(o) & np.isfinite(sim)
    return 1 - np.sum((sim[k] - o[k]) ** 2) / np.sum((o[k] - o[k].mean()) ** 2) if k.sum() > 365 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(PRESETS), help="named configuration (config.py)")
    ap.add_argument("--dataset", required=True, choices=["gb", "us", "aus"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--snoise", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--fc_min", type=float, default=None, help="lower bound (mm/d) of the infiltration capacity F_c (presets with "
                    "infiltration excess); default: the dataset's setting (aus 1e-4, gb / us none = 0)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="", help="output directory (default runs/<dataset>_<model>_s<seed>)")
    args = ap.parse_args()
    cfg = config_for(args.model, args.dataset, **({} if args.fc_min is None else {"fc_min": args.fc_min}))
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)                                                                 # Triton kernels launch on the current device
    out = Path(args.out or f"runs/{args.dataset}_{args.model}_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    ds = data_mod.load(args.dataset)
    model, best, hist = train(ds, cfg, args.seed, args.iters, args.lr, args.snoise, args.eval_every, dev)
    with torch.no_grad():
        Q, ps, bal = model.simulate(TT=model.T)
    Qn = Q.cpu().numpy().T
    np.save(out / "sim.npy", Qn.astype(np.float32))
    torch.save(model.net.state_dict(), out / "net.pt")
    json.dump({"args": vars(args), "config": cfg.__dict__, "best_val_median": best, "history": hist}, open(out / "run.json", "w"), indent=1)
    b = {k: v.double().cpu().numpy() for k, v in bal.items()}
    resid = b["P"] + b["X"] - b["A"] - b["Q"] - b["dS"] - b["dR"] - b["dW"] - b["transit"]
    print(f"checks: water balance (float32 GPU states) max |residual|/P {np.abs(resid / b['P']).max():.2e}; median X/P {np.median(b['X'] / b['P']):+.3f}",
          flush=True)
    line = f"RESULT {args.dataset}_{args.model}_s{args.seed} |"
    for grp, msk in (("gauged", ds.gauged), ("ungauged", ds.ungauged)):
        if not msk.any():
            continue
        for pn in (("val", "test") if args.dataset in data_mod.US_LAYOUT else ("val", "2011-15", "2015-19")):
            m = ds.mask(pn)
            line += f" {grp} {pn} {np.nanmedian([nse_np(Qn[m, c], ds.q[m, c]) for c in np.where(msk)[0]]):.4f} |"
    print(line, flush=True)


if __name__ == "__main__":
    main()

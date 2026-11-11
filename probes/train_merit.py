"""Train CaRT+ (preset `cartplus`) on the MERIT gauges of the lumped dHBV setup and write the run.

Author: tbindas

What it does: builds the `cartplus` preset on the dataset of probes/merit_data.py, trains it and writes net.pt (mapping
network), sim.npy ([days, gauges] float32, mm/d, whole record 1980-10-01 .. 2010-09-30) and run.json to --out.

    python probes/train_merit.py --out /mnt/ssd1/data/cartplus/runs/<new name> [--K 50] [--iters 5000] [--seed 111111]

Experimental purpose: a CaRT+ run on the same gauges, forcing, attributes, periods and seed as the lumped dHBV run of
water_loss. Later sessions run the parameter learnability census of the equifinality paper on it.

Settings (user decisions of 2026-11-10): the CaRT+ defaults (AdamW, lr 2e-3, weight decay 1e-4, one-cycle with 5 % warm-up,
5000 iterations, gradient-norm clip 1, all gauges in each iteration, attribute noise SD 0.1), the CaRT+ loss (mean over
gauges of 1 - NSE) on 1981-10-01 .. 1995-09-30, seed 111111, and the weights of the LAST iteration (no validation period).

The decision "last iteration, no validation period" cannot be met with cart.train.train: that function scores a "val"
period and returns the best iteration. So train() below is a copy of cart.train.train (src/cart/train.py:58-107).
Lines that differ from the original:
  1. removed  okf_va, den_va, n_va = masks(ds.mask("val"))            (no validation masks)
  2. removed  val_c = gauged & (n_va > 365)
  3. removed  best, best_state bookkeeping and the whole `if it % eval_every == 0 ...` block (no validation forward pass,
              no "val_median" history, no copy of the best state)
  4. removed  net.load_state_dict(best_state)                         (the weights of the last iteration stay)
  5. removed  arguments eval_every and on_step
  6. added    per-iteration record: loss, gradient norm, median train NSE, wall time, whether the step was skipped
  7. added    torch.cuda.synchronize() before each time stamp and a print every `log_every` iterations
  8. changed  return value: (model, history, nskip, number of gauges in the loss) in place of (model, best, hist)
  9. changed  the start-up print (no validation count)
Nothing else differs: seed calls, masks, train_c (gauged and > 730 valid train days), optimizer, schedule, loss, clip and
the skip of a non-finite step are the original lines. nse_score and clip_grad_norm are imported from cart.train.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cart.config import config_for  # noqa: E402
from cart.model import CaRT  # noqa: E402
from cart.train import clip_grad_norm, nse_score  # noqa: E402
import merit_data  # noqa: E402

SEED = 111111                                                                                 # randomseed of the lumped dHBV script


def train(ds, cfg, seed, iters=5000, lr=2e-3, snoise=0.1, dev="cuda:0", log_every=50, verbose=True):
    """Copy of cart.train.train without the validation period. Returns (model with the weights of the last iteration,
    history [one dict per iteration], number of skipped steps, number of gauges in the loss)."""
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
    train_c = torch.from_numpy(ds.gauged).to(dev) & (n_tr > 730)                                # >= 2 years of training data

    params = net.train_parameters()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.05)
    hist, nskip = [], 0
    if verbose:
        print(f"{cfg.name} seed {seed}: gauges {len(ds.ids)}, in the loss {int(train_c.sum())}, training days simulated {model.Ttr}, "
              f"loss days {int(model.trm.sum())}; per-catchment parameters {len(net.static_par) + len(net.law_par)}", flush=True)
    torch.cuda.synchronize(); t0 = time.time()
    for it in range(iters):
        Q, _, _ = model.simulate(snoise)
        ntr = nse_score(Q, qz, okf_tr, den_tr, train_c)
        loss = (1 - ntr).mean()
        opt.zero_grad(); loss.backward()
        gnorm = clip_grad_norm(params, 1.0, net)
        skipped = not (torch.isfinite(loss) and torch.isfinite(gnorm))
        if not skipped:
            opt.step()
        else:                                                                                  # skip a non-finite step (keeps the weights finite)
            nskip += 1
            if verbose and (nskip <= 5 or nskip % 50 == 0):
                print(f"  iter {it}: non-finite loss / gradient, step skipped ({nskip})", flush=True)
        sched.step()
        torch.cuda.synchronize()
        hist.append({"iter": it, "loss": loss.item(), "gnorm": gnorm.item(), "train_median": ntr.median().item(),
                     "skipped": skipped, "t": time.time() - t0})
        if verbose and (it % log_every == 0 or it == iters - 1):
            print(f"  iter {it:5d}  loss {loss.item():.6f}  train median NSE {ntr.median().item():.4f}  ({hist[-1]['t']:.0f}s)", flush=True)
    return model, hist, nskip, int(train_c.sum())


def gpu_used_mib(gpu):
    """memory in use on the GPU as nvidia-smi reports it (all processes), MiB"""
    return int(subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory; it must not exist")
    ap.add_argument("--K", type=int, default=0, help="only the first K gauges (0 = all 2717)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--snoise", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; pass a new path (nothing is overwritten)")
    cfg = config_for("cartplus", "merit")
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)                                                                 # Triton kernels launch on the current device
    gpu_before = gpu_used_mib(args.gpu)
    ds = merit_data.load(args.K or None)
    out.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats(dev)
    model, hist, nskip, n_loss = train(ds, cfg, args.seed, args.iters, args.lr, args.snoise, dev, args.log_every)
    peak_train = {"allocated_GiB": torch.cuda.max_memory_allocated(dev) / 2 ** 30, "reserved_GiB": torch.cuda.max_memory_reserved(dev) / 2 ** 30}
    with torch.no_grad():
        Q, ps, bal = model.simulate(TT=model.T)
    peak_all = {"allocated_GiB": torch.cuda.max_memory_allocated(dev) / 2 ** 30, "reserved_GiB": torch.cuda.max_memory_reserved(dev) / 2 ** 30}
    Qn = Q.cpu().numpy().T
    np.save(out / "sim.npy", Qn.astype(np.float32))
    torch.save(model.net.state_dict(), out / "net.pt")
    t = np.array([h["t"] for h in hist])
    dt = np.diff(np.concatenate([[0.0], t]))
    timing = {"total_s": float(t[-1]), "first_iter_s": float(dt[0]),                            # the first iteration compiles the Triton kernels
              "median_s_per_iter": float(np.median(dt[1:])), "mean_s_per_iter_after_first": float(dt[1:].mean())}
    args_d = dict(vars(args), dataset="merit", model="cartplus")
    json.dump({"args": args_d, "config": cfg.__dict__, "selection": "last iteration (no validation period)",
               "periods": ds.periods, "store": merit_data.STORE, "n_gauges": len(ds.ids), "n_gauges_in_loss": n_loss,
               "sim_days_train": int(model.Ttr), "loss_days": int(model.trm.sum()), "sim_days_all": int(model.T),
               "nskip": nskip, "timing": timing, "peak_gpu_training": peak_train, "peak_gpu_with_final_forward": peak_all,
               "gpu_used_before_MiB": gpu_before, "torch": torch.__version__, "gpu_name": torch.cuda.get_device_name(dev),
               "history": hist}, open(out / "run.json", "w"), indent=1)
    b = {k: v.double().cpu().numpy() for k, v in bal.items()}
    resid = b["P"] + b["X"] - b["A"] - b["Q"] - b["dS"] - b["dR"] - b["dW"] - b["transit"]
    print(f"checks: water balance (float32 GPU states) max |residual|/P {np.abs(resid / b['P']).max():.2e}; skipped steps {nskip}", flush=True)
    print(f"timing: {timing['median_s_per_iter']:.3f} s/iter (median), first iter {timing['first_iter_s']:.1f} s; peak GPU in training "
          f"{peak_train['allocated_GiB']:.2f} GiB allocated, {peak_train['reserved_GiB']:.2f} GiB reserved; in use before the run {gpu_before} MiB", flush=True)
    print(f"RESULT merit_cartplus_s{args.seed} K={len(ds.ids)} iters={args.iters} -> {out}", flush=True)


if __name__ == "__main__":
    main()

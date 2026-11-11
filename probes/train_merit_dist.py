"""Train distributed CaRT+ on the MERIT unit catchments of the distributed dHBV setup and write the run.

Author: tbindas

What it does: runs the `cartplus` preset on each MERIT unit catchment ("divide"), sums the divide runoff to each gauge
with area weights, delays the sum with a gamma unit hydrograph at the gauge (probes/uh.py) and trains the mapping
network and the unit-hydrograph head on the gauge discharge. Writes to --out: net.pt (mapping network), uh_head.pt
(unit-hydrograph head), sim.npy ([days, gauges] float32, mm/d, whole record, after the unit hydrograph), sim_preuh.npy
(the same before the unit hydrograph), uh_gauge.npy ([gauges, 2]: a, b), chunk_plan.json, run.json and checkpoints.

    python probes/train_merit_dist.py --out /mnt/ssd1/data/cartplus/runs/<new name> --cap 2000 [--K 50] [--iters 5000]

Experimental purpose: the equifinality paper needs a CaRT+ run in the spatial layout of the distributed dHBV run
(CONUS2717_AORC2F_v3_gradaccum): same divides, forcing, attributes, gauges, periods and seed. The only planned
difference from lumped CaRT+ (probes/train_merit.py) is the spatial layout.

The model has two delay steps: each divide keeps the unit hydrograph of the `cartplus` preset (parameter X4), and the
gauge unit hydrograph acts on the summed runoff.

Gradient accumulation: one iteration is one pass over all gauges. The gauges are split into chunks
(merit_dist_data.chunk_plan). Each chunk runs forward and backward on its own; the gradients add up; the optimizer
takes one step per iteration. The loss of a chunk is sum over its gauges of (1 - NSE) / (number of gauges in the loss),
so the sum over the chunks is the mean over all gauges. A divide is simulated once per chunk. The attribute noise of
an iteration is drawn once for all divides and indexed by divide, so a divide has the same noise in each chunk.

train() follows the loop of train() in probes/train_merit.py. Lines that differ from that loop:
  1. changed  model = CaRT(ds, cfg, dev)  ->  net = MappingNetwork(...) built at the same place in the random stream,
              then ChunkCaRT (a subclass of cart.model.CaRT that takes its tensors from one chunk; simulate() is the
              inherited method) and the UHHead (built after the mapping network, so the start weights of the mapping
              network are those of the lumped run with the same seed).
  2. changed  Q, _, _ = model.simulate(snoise)  ->  a loop over the chunks: load the chunk, add the noise of its
              divides to the attributes, model.simulate(0.0, xs_in=noisy attributes), uh.route (area-weighted sum, unit
              hydrograph). The noise is snoise * randn, as in simulate().
  3. changed  loss = (1 - ntr).mean(); loss.backward()  ->  per chunk loss_c = (1 - ntr).sum() / n_loss;
              loss_c.backward() (gradients add up); loss = sum of loss_c. opt.zero_grad() runs before the chunk loop.
  4. changed  params = net.train_parameters()  ->  the same list plus the parameters of the UHHead (one AdamW, one clip).
  5. changed  the masks are indexed by the gauges of the chunk; train_c has no `gauged` factor (each gauge is gauged).
  6. added    a checkpoint (weights, optimizer, schedule, random state, history) each --ckpt_every iterations and
              --resume, because the run takes many hours.
  7. changed  return value and the start-up print.
Nothing else differs: seed calls, masks, the filter n_tr > 730, AdamW (lr 2e-3, weight decay 1e-4), one-cycle
(pct_start 0.05), clip 1 on the accumulated gradient, the skip of a non-finite step and the weights of the last
iteration. nse_score and clip_grad_norm are imported from cart.train.
"""
import argparse
import json
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
from cart.parameters import MappingNetwork  # noqa: E402
from cart.train import clip_grad_norm, nse_score  # noqa: E402
import merit_dist_data as mdd  # noqa: E402
import uh  # noqa: E402
from train_merit import SEED, gpu_used_mib  # noqa: E402


class ChunkCaRT(CaRT):
    """cart.model.CaRT with the forcing of one chunk of divides at a time and a mapping network that all chunks share.
    __init__ of CaRT is not called (it needs a Dataset of all columns); load() sets the attributes that __init__ sets,
    with the same values and dtypes (probes/tests/check_dist.py compares them with CaRT on a Dataset of the divides).
    simulate() and all other methods are inherited."""

    def __init__(self, cfg, dev, net, tr, Ttr):
        self.cfg, self.dev, self.ds = cfg, dev, None
        self.T = len(tr)
        self.tr = tr
        self.Ttr = Ttr
        self.trm = torch.from_numpy(tr).to(dev)
        self.nfftV = 1 << int(np.ceil(np.log2(self.T + 2048)))
        self.net = net

    def load(self, ch):
        dev = self.dev
        n = len(ch["rows"])
        self.Hdev = torch.zeros((n, 1), dtype=torch.float32, device=dev) / 1000.0              # one band: offset 0 (see merit_dist_data)
        self.Aband = torch.ones((n, 1), dtype=torch.float32, device=dev)
        self.Pbar_ie = ch["Pbar"]
        self.xs = ch["xs"]
        self.Pg = ch["P"].to(dev, non_blocking=True)
        self.Eg = ch["E"].to(dev, non_blocking=True)
        self.Tcl = ch["Tcl"].to(dev, non_blocking=True)


def build_chunks(dd, plan, dev, pin=True):
    """the tensors of each chunk. Forcing [divides, days] float32 stays on the CPU (pinned); small tensors go to the GPU."""
    chunks = []
    for c in plan:
        rows = np.asarray(c["rows"]); gauges = np.asarray(c["gauges"])
        Tcl, Pbar = mdd.model_inputs(dd, rows)
        cpu = lambda a: (torch.from_numpy(np.ascontiguousarray(a)).pin_memory() if pin else torch.from_numpy(np.ascontiguousarray(a)))
        chunks.append({"rows": rows, "gauges": gauges,
                       "P": cpu(dd.P[rows]), "E": cpu(dd.PET[rows]), "Tcl": cpu(Tcl),
                       "Pbar": torch.from_numpy(Pbar).to(dev)[:, None], "xs": torch.from_numpy(dd.static[rows]).to(dev),
                       "ri": torch.from_numpy(rows).to(dev), "gi": torch.from_numpy(gauges).to(dev),
                       "M": uh.membership(gauges, rows, dd.gmap, dd.frac, dev)})
    return chunks


class Setup:
    """everything train() and the tests need: mapping network, UH head, model, chunks, masks"""

    def __init__(self, dd, cfg, seed, K, cap, dev, plan=None, pin=True):
        torch.manual_seed(seed); np.random.seed(seed)
        self.dd, self.cfg, self.dev = dd, cfg, dev
        self.G = K or len(dd.ids)
        self.net = MappingNetwork(dd.static.shape[1], cfg, dev)                                # the first random draws, as in CaRT.__init__
        self.uhh = uh.UHHead(dd.static.shape[1], dev)
        tr = dd.mask("train")
        Ttr = int(np.where(dd.mask("val"))[0].max()) + 1
        self.model = ChunkCaRT(cfg, dev, self.net, tr, Ttr)
        self.plan = mdd.chunk_plan(dd.gmap, cap, self.G) if plan is None else plan
        self.chunks = build_chunks(dd, self.plan, dev, pin)
        qg = torch.from_numpy(dd.q[:self.G]).to(dev)
        okq = torch.isfinite(qg); qz = torch.where(okq, qg, torch.zeros_like(qg))

        def masks(mask):
            ok = okq & mask[None]; okf = ok.float()
            mean = (qz * okf).sum(1) / ok.sum(1).clamp(min=1)
            return okf, ((qz - mean[:, None]) * okf).pow(2).sum(1).clamp(min=1e-6), ok.sum(1)
        self.qz = qz
        self.okf_tr, self.den_tr, n_tr = masks(self.model.trm)
        self.train_c = n_tr > 730                                                              # >= 2 years of training data
        self.n_loss = int(self.train_c.sum())
        self.params = self.net.train_parameters() + list(self.uhh.parameters())
        self.n_div_all = dd.static.shape[0]

    def chunk_forward(self, ch, noise=None, TT=None):
        """-> (Q after the UH [gauges of the chunk, TT], summed runoff before it, gauge a, gauge b, bal of simulate)"""
        self.model.load(ch)
        xsn = ch["xs"] if noise is None else ch["xs"] + noise[ch["ri"]]
        Qd, _, bal = self.model.simulate(0.0, TT=TT, xs_in=xsn)
        a_div, b_div = self.uhh(xsn)
        Qr, Qg, a, b = uh.route(ch["M"], Qd, a_div, b_div)
        return Qr, Qg, a, b, bal

    def accumulate(self, snoise):
        """one pass over all chunks: gradients add up in .grad (the caller zeroes them). -> (loss, NSE of the gauges in the loss)"""
        dev = self.dev
        noise = snoise * torch.randn((self.n_div_all, self.dd.static.shape[1]), device=dev) if snoise > 0 else None
        loss = torch.zeros((), device=dev)
        ntrs = []
        for ch in self.chunks:
            gi = ch["gi"]
            Qr = self.chunk_forward(ch, noise)[0]
            ntr = nse_score(Qr, self.qz[gi], self.okf_tr[gi], self.den_tr[gi], self.train_c[gi])
            loss_c = (1 - ntr).sum() / self.n_loss
            loss_c.backward()
            loss = loss + loss_c.detach()
            ntrs.append(ntr.detach())
            del Qr, ntr, loss_c
        return loss, torch.cat(ntrs)

    def forward_all(self):
        """whole record, no noise, no gradient -> sim [G, T], sim before the UH [G, T], a [G], b [G], balance numbers of the divide stores"""
        T = self.model.T
        sim = np.empty((self.G, T), np.float32); pre = np.empty((self.G, T), np.float32)
        a_g = np.empty(self.G, np.float32); b_g = np.empty(self.G, np.float32)
        worst, n_dry, worst_dry = 0.0, 0, 0.0                                                    # dry: divides with P = 0 (NaN forcing filled with 0)
        with torch.no_grad():
            for ch in self.chunks:
                Qr, Qg, a, b, bal = self.chunk_forward(ch, None, TT=T)
                g = ch["gauges"]
                sim[g] = Qr.cpu().numpy(); pre[g] = Qg.cpu().numpy(); a_g[g] = a.cpu().numpy(); b_g[g] = b.cpu().numpy()
                bb = {k: v.double().cpu().numpy() for k, v in bal.items()}
                resid = bb["P"] + bb["X"] - bb["A"] - bb["Q"] - bb["dS"] - bb["dR"] - bb["dW"] - bb["transit"]
                wet = bb["P"] > 0
                worst = max(worst, float(np.abs(resid[wet] / bb["P"][wet]).max()))
                if (~wet).any():
                    n_dry += int((~wet).sum()); worst_dry = max(worst_dry, float(np.abs(resid[~wet]).max()))
        return sim, pre, a_g, b_g, {"divide_stores_max_abs_resid_over_P": worst, "columns_with_zero_P": n_dry,
                                    "zero_P_columns_max_abs_resid": worst_dry, "resid_unit_zero_P": "unit of the bal terms of simulate()"}


def train(dd, cfg, seed, K, cap, iters=5000, lr=2e-3, snoise=0.1, dev="cuda:0", log_every=50, verbose=True, out=None, ckpt_every=0,
          resume=None):
    """The loop of train_merit.train with gradient accumulation over gauge chunks (see the module docstring).
    Returns (Setup with the weights of the last iteration, history, number of skipped steps)."""
    st = Setup(dd, cfg, seed, K, cap, dev)
    net, params = st.net, st.params
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.05)
    hist, nskip, it0, t_before = [], 0, 0, 0.0
    if resume:
        ck = torch.load(resume, map_location=dev, weights_only=False)
        assert ck["key"] == [seed, K, cap, iters, lr, snoise], f"the checkpoint is of another run: {ck['key']}"
        net.load_state_dict(ck["net"]); st.uhh.load_state_dict(ck["uhh"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["rng_cpu"]); torch.cuda.set_rng_state(ck["rng_cuda"], dev)
        hist, nskip, it0 = ck["hist"], ck["nskip"], ck["iter"] + 1
        t_before = hist[-1]["t"]
    if verbose:
        s = mdd.plan_summary(st.plan, dd.gmap)
        print(f"{cfg.name} distributed, seed {seed}: gauges {st.G}, in the loss {st.n_loss}, chunks {s['n_chunks']} (cap {cap}), "
              f"divide columns per pass {s['columns_per_pass']} (unique divides {s['unique_divides']}), training days simulated {st.model.Ttr}, "
              f"loss days {int(st.model.trm.sum())}; per-divide parameters {len(net.static_par) + len(net.law_par)}; "
              f"UH head parameters {sum(p.numel() for p in st.uhh.parameters())}; start at iteration {it0}", flush=True)
    torch.cuda.synchronize(); t0 = time.time()
    for it in range(it0, iters):
        opt.zero_grad()
        loss, ntr = st.accumulate(snoise)
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
                     "skipped": skipped, "t": t_before + time.time() - t0})
        if verbose and (it % log_every == 0 or it == iters - 1):
            print(f"  iter {it:5d}  loss {loss.item():.6f}  train median NSE {ntr.median().item():.4f}  ({hist[-1]['t']:.0f}s)", flush=True)
        if out is not None and ckpt_every and (it + 1) % ckpt_every == 0 and it + 1 < iters:
            f = Path(out) / f"ckpt_iter{it:05d}.pt"                                            # a new file each time; nothing is overwritten
            torch.save({"key": [seed, K, cap, iters, lr, snoise], "iter": it, "net": net.state_dict(), "uhh": st.uhh.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state(dev), "hist": hist, "nskip": nskip}, f)
    return st, hist, nskip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory; it must not exist")
    ap.add_argument("--cap", type=int, required=True, help="largest number of divide columns in one chunk (GPU memory)")
    ap.add_argument("--K", type=int, default=0, help="only the first K gauges (0 = all 2717)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--snoise", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--ckpt_every", type=int, default=250, help="write a checkpoint each N iterations (0 = none)")
    ap.add_argument("--resume", default="", help="checkpoint of a run with the same arguments; --out must still be a new path")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; pass a new path (nothing is overwritten)")
    cfg = config_for("cartplus", "merit")
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)                                                                 # Triton kernels launch on the current device
    gpu_before = gpu_used_mib(args.gpu)
    dd = mdd.load()
    out.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats(dev)
    st, hist, nskip = train(dd, cfg, args.seed, args.K, args.cap, args.iters, args.lr, args.snoise, dev, args.log_every, out=out,
                            ckpt_every=args.ckpt_every, resume=args.resume or None)
    peak_train = {"allocated_GiB": torch.cuda.max_memory_allocated(dev) / 2 ** 30, "reserved_GiB": torch.cuda.max_memory_reserved(dev) / 2 ** 30}
    t_fw = time.time()
    sim, pre, a_g, b_g, worst = st.forward_all()
    t_fw = time.time() - t_fw
    peak_all = {"allocated_GiB": torch.cuda.max_memory_allocated(dev) / 2 ** 30, "reserved_GiB": torch.cuda.max_memory_reserved(dev) / 2 ** 30}
    np.save(out / "sim.npy", np.ascontiguousarray(sim.T))
    np.save(out / "sim_preuh.npy", np.ascontiguousarray(pre.T))
    np.save(out / "uh_gauge.npy", np.stack([a_g, b_g], 1))
    torch.save(st.net.state_dict(), out / "net.pt")
    torch.save(st.uhh.state_dict(), out / "uh_head.pt")
    summ = mdd.plan_summary(st.plan, dd.gmap)
    json.dump({"cap": args.cap, "n_gauges": st.G, "summary": summ, "chunks": st.plan}, open(out / "chunk_plan.json", "w"))
    t = np.array([h["t"] for h in hist])
    dt = np.diff(np.concatenate([[0.0], t]))
    timing = {"total_s": float(t[-1]), "first_iter_s": float(dt[0]),                            # the first iteration compiles the Triton kernels
              "median_s_per_iter": float(np.median(dt[1:])), "mean_s_per_iter_after_first": float(dt[1:].mean()),
              "final_forward_s": float(t_fw)}
    m_pre, m_uh = pre.astype(np.float64).mean(1), sim.astype(np.float64).mean(1)
    uh_vol = float(np.abs(m_uh / m_pre - 1).max())                                              # the UH keeps the volume (less the last 14 days)
    args_d = dict(vars(args), dataset="merit", model="cartplus")
    json.dump({"args": args_d, "config": cfg.__dict__, "selection": "last iteration (no validation period)", "layout": "distributed",
               "periods": dict(mdd.PERIODS), "store": mdd.STORE, "map": mdd.MAP, "n_gauges": st.G, "n_gauges_in_loss": st.n_loss,
               "n_divides": summ["unique_divides"], "chunk_plan": dict(summ, cap=args.cap),
               "uh_head": {"layers": f"Linear({dd.static.shape[1]}, {uh.HIDDEN}) - GELU - Linear({uh.HIDDEN}, 2) - sigmoid",
                           "n_parameters": sum(p.numel() for p in st.uhh.parameters()), "ranges_a_b": uh.ROUT_RANGE, "lenF": uh.LEN_F,
                           "start": "last layer zero: a = 1.45, b = 3.25 at each divide"},
               "uh_gauge": {"a_min_median_max": [float(a_g.min()), float(np.median(a_g)), float(a_g.max())],
                            "b_min_median_max": [float(b_g.min()), float(np.median(b_g)), float(b_g.max())]},
               "missing_values_before_fill": dd.nan,
               "sim_days_train": int(st.model.Ttr), "loss_days": int(st.model.trm.sum()), "sim_days_all": int(st.model.T),
               "nskip": nskip, "timing": timing, "peak_gpu_training": peak_train, "peak_gpu_with_final_forward": peak_all,
               "gpu_used_before_MiB": gpu_before, "torch": torch.__version__, "gpu_name": torch.cuda.get_device_name(dev),
               "balance": dict(worst, gauge_uh_max_abs_volume_ratio_minus_1=uh_vol),
               "history": hist}, open(out / "run.json", "w"), indent=1)
    print(f"checks: divide water balance (float32 GPU states) max |residual|/P {worst['divide_stores_max_abs_resid_over_P']:.2e} (columns with P = 0: {worst['columns_with_zero_P']}, max |residual| there {worst['zero_P_columns_max_abs_resid']:.2e}); gauge UH volume max |ratio - 1| {uh_vol:.2e}; "
          f"skipped steps {nskip}", flush=True)
    print(f"timing: {timing['median_s_per_iter']:.3f} s/iter (median), first iter {timing['first_iter_s']:.1f} s; peak GPU in training "
          f"{peak_train['allocated_GiB']:.2f} GiB allocated, {peak_train['reserved_GiB']:.2f} GiB reserved; in use before the run {gpu_before} MiB", flush=True)
    print(f"RESULT merit_cartplus_dist_s{args.seed} K={st.G} iters={args.iters} -> {out}", flush=True)


if __name__ == "__main__":
    main()

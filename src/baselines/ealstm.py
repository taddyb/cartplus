"""EA-LSTM benchmark (Kratzert et al. 2019, HESS 23, 5089-5110, "entity-aware LSTM") under the paper's unified protocol, on CAMELS-GB,
CAMELS-US and CAMELS-AUS, with exactly the inputs of the stateful LSTM benchmark (baselines/lstm_stateful.py).

Data (as the stateful LSTM; its functions are reused):
  GB: lstm_stateful.build_inputs_gb + stage_stats(..., 2008-12-31): days 1988-01-03 .. 2019-12-31 (1988 = history only), dynamic P, T,
      PET, static = 21 attributes + 5 band elevations.
  US: days 1980-01-01 .. 2019-12-31 (1980-1988 = history only), dynamic P, T, PET, static = 27 attributes + the 5 equal-area band-centre
      elevations (lstm_stateful.band_quantile_elevations).
  AUS: the US code path on CAMELS-AUS v2: 1980-2019, 27 + 5 static inputs.
  Forcing normalised with mean / std over the record start .. 2008-12-31 of the gauged catchments; target q normalised per basin as in the
  stateful LSTM ((q - gm p_mean) / (gs p_mean)); observed flows after 2010-12-31 are set to NaN on loading (test period never seen).
Model (EA-LSTM): input gate i = sigmoid(W_s s + b_s) from the static attributes only (constant over time); forget, cell and output gates
  from the dynamic inputs and h. Implemented as a cuDNN nn.LSTM over [dynamic, static] with the non-EA connections fixed at zero (weights
  initialised to 0 and their gradients zeroed, so Adam never moves them) -- mathematically the EA-LSTM cell. Initialisation as Kratzert's
  code (ealstm_regional_modeling): orthogonal input weights, identity recurrent weights per gate, zero biases, initial forget-gate bias 5.
  Hidden 256, dropout 0.4 on the last hidden state, linear head.
Training (Kratzert 2019, except 365-day windows in place of 270): sequence-to-one with 365-day input windows (which may reach into the pre-1989 history); samples = every
  (training basin, day in 1989-01-01 .. 2008-12-31) with an observed flow and a full window; each epoch draws a fresh random subset of
  1,745,928 samples (= the size of Kratzert 2019's epoch, 531 basins x 1999-10-01..2008-09-30) without replacement, batch 256; NSE* loss
  w_b (y - q)^2 with w_b = 1 / (s_b + 0.1)^2; Adam lr 1e-3, 5e-4 from epoch 11, 1e-4 from epoch 26; 30 epochs; gradient clipping 1.
Selection: after each epoch, median NSE over 2009-01-01 .. 2010-12-31 of the training basins (> 365 observed days); best epoch kept.
  Training basins = gauged with > 730 observed days in 1989-2008; held-out catchments never enter loss, selection or normalisation.

Usage (from src/): python -m baselines.ealstm --dataset gb|us|aus --seed 0 [--gpu 0] [--out DIR]
  -> sim.npy [T, N] (mm/d, clipped at 0; dataset days; days without a full 365-day window are NaN), model.pt, norm.npz, run.json
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cart import data as D
from . import lstm_stateful as G

SEQ, HIDDEN, DROP, BATCH, EPOCHS, FORGET_BIAS, CLIP = 365, 256, 0.4, 256, 30, 5.0, 1.0
LR = {1: 1e-3, 11: 5e-4, 26: 1e-4}                                         # epoch (1-based) -> learning rate
SAMPLES_PER_EPOCH = 1_745_928    # = Kratzert 2019's epoch (531 basins x 3288 days, 1999-10-01..2008-09-30); drawn afresh
                                 # without replacement each epoch from all ~4.4M training samples (same count GB and US)
END_SEL = date(2010, 12, 31)


class EALSTM(nn.Module):
    """Entity-aware LSTM on cuDNN: nn.LSTM over [x_dyn, x_static] with the non-EA connections held at zero."""

    def __init__(self, n_dyn, n_static):
        super().__init__()
        H, F = HIDDEN, n_dyn + n_static
        self.n_dyn = n_dyn
        self.lstm = nn.LSTM(F, H, batch_first=True)
        m_ih = torch.zeros(4 * H, F)                                         # PyTorch gate order i, f, g, o
        m_ih[:H, n_dyn:] = 1                                                # input gate <- static only
        m_ih[H:, :n_dyn] = 1                                                # f, g, o <- dynamic only
        m_hh = torch.ones(4 * H, H)
        m_hh[:H] = 0                                                        # input gate has no recurrent weights
        self.register_buffer("m_ih", m_ih)
        self.register_buffer("m_hh", m_hh)
        with torch.no_grad():                                               # Kratzert's reset_parameters
            w_ih = torch.zeros(4 * H, F)
            w_dyn = torch.empty(n_dyn, 3 * H); nn.init.orthogonal_(w_dyn)    # weight_ih [n_dyn, 3H] (f, o, g in his code)
            w_s = torch.empty(n_static, H); nn.init.orthogonal_(w_s)         # weight_sh [n_static, H]
            w_ih[:H, n_dyn:] = w_s.T
            w_ih[H:, :n_dyn] = w_dyn.T
            w_hh = torch.zeros(4 * H, H)
            w_hh[H:] = torch.eye(H).repeat(3, 1)                            # weight_hh = eye repeated per gate
            self.lstm.weight_ih_l0.copy_(w_ih)
            self.lstm.weight_hh_l0.copy_(w_hh)
            self.lstm.bias_ih_l0.zero_(); self.lstm.bias_hh_l0.zero_()
            self.lstm.bias_ih_l0[H:2 * H] = FORGET_BIAS
        self.lstm.weight_ih_l0.register_hook(lambda g: g * self.m_ih)
        self.lstm.weight_hh_l0.register_hook(lambda g: g * self.m_hh)
        self.drop = nn.Dropout(DROP)
        self.head = nn.Linear(H, 1)

    def forward(self, x):
        """x [B, SEQ, n_dyn + n_static] -> y [B] (last day)."""
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1])).squeeze(-1)

    def check_mask(self):
        return float((self.lstm.weight_ih_l0 * (1 - self.m_ih)).abs().max() + (self.lstm.weight_hh_l0 * (1 - self.m_hh)).abs().max())


def load_gb():
    ds = D.load("gb")
    data = {"ids": ds.ids, "basin_mask": ds.gauged, "ug_mask": ds.ungauged, "p_mean": ds.p_mean}
    days, off, raw, q, static_np, snames = G.build_inputs_gb(ds)           # q after 2010 already NaN
    pos = {d: i for i, d in enumerate(days)}
    x, qn, q_mean, q_std, w = G.stage_stats(raw, q, data, pos, date(2008, 12, 31))
    return dict(data=data, days=days, out0=off, x=x, q=q, qn=qn, q_mean=q_mean, q_std=q_std, w=w, static=static_np,
                snames=snames, gauged=data["basin_mask"], ug=data["ug_mask"], dataset="CAMELS-GB (inputs of baselines.lstm_stateful)")


def load_us(name="us"):
    """US code path (name = "us" or "aus", the datasets of cart.data.US_LAYOUT)"""
    assert name in D.US_LAYOUT
    ds = D.load(name)
    data = {"ids": ds.ids, "basin_mask": ds.gauged, "ug_mask": ds.ungauged}
    days, P, E, q, Tm = ds.days, ds.P, ds.E, ds.q, ds.Tm
    q = q.astype(np.float32)
    q[np.array([d > END_SEL for d in days])] = np.nan
    raw = np.stack([P, Tm, E], -1).astype(np.float32)
    gauged, ug = data["basin_mask"], data["ug_mask"]
    H = G.band_quantile_elevations(ds.ids, eb=(ds.band_elev_raw, ds.band_frac_raw))
    h_mean, h_std = H[gauged].mean(0), H[gauged].std(0) + 1e-6
    static_np = np.concatenate([ds.static, ((H - h_mean) / h_std).astype(np.float32)], 1)
    snames = [str(s) for s in ds.static_names] + [f"band_elev_{int(p)}" for p in (10, 30, 50, 70, 90)]
    assert np.isfinite(static_np).all()
    pos = {d: i for i, d in enumerate(days)}
    i_tr0, i_08 = pos[date(1989, 1, 1)], pos[date(2008, 12, 31)]
    fm = raw[:i_08 + 1][:, gauged].mean(axis=(0, 1)); fs = raw[:i_08 + 1][:, gauged].std(axis=(0, 1)) + 1e-6
    x = ((raw - fm) / fs).astype(np.float32)
    scale = ds.p_mean.astype(np.float32)
    qt = q[i_tr0:i_08 + 1] / scale[None]
    b_std = np.nanstd(qt, axis=0)
    usable = gauged & np.isfinite(b_std) & (b_std > 0)
    gm, gs = float(np.nanmean(qt[:, usable])), float(np.nanstd(qt[:, usable]))
    q_mean, q_std = (gm * scale).astype(np.float32), (gs * scale).astype(np.float32)
    s_b = np.where(usable, b_std / gs, 1.0)
    w = (1.0 / (s_b + 0.1) ** 2).astype(np.float32)
    w = np.where(usable, w / w[usable].mean(), 0.0).astype(np.float32)
    qn = np.nan_to_num((q - q_mean) / q_std).astype(np.float32)
    return dict(data=data, days=days, out0=0, x=x, q=q, qn=qn, q_mean=q_mean, q_std=q_std, w=w, static=static_np,
                snames=snames, gauged=gauged, ug=ug, f_mean=fm, f_std=fs, dataset={"us": "CAMELS-US (cart.data us: Daymet v4, 671 basins)", "aus": "CAMELS-AUS v2 (cart.data aus: 561 catchments)"}[name])


class Windows:
    """Builds [B, SEQ, F] input windows on the GPU from x [D, C, n_dyn] and static [C, S]."""

    def __init__(self, x, static, device):
        self.x = torch.from_numpy(x).to(device)
        self.s = torch.from_numpy(static).to(device)
        self.ar = torch.arange(-SEQ + 1, 1, device=device)

    def __call__(self, c, t):
        """c, t: LongTensor [B] (catchment, last day index) -> [B, SEQ, n_dyn + S]."""
        xd = self.x[t[:, None] + self.ar[None], c[:, None]]                 # [B, SEQ, n_dyn]
        return torch.cat([xd, self.s[c][:, None].expand(-1, SEQ, -1)], 2)


@torch.no_grad()
def simulate(model, win, cs, t0, t1, q_mean, q_std, n_days, bs=1024):
    """Simulated discharge (mm/d, clipped at 0) [n_days, len(cs)] for days t0..t1 (NaN elsewhere)."""
    model.eval()
    out = np.full((n_days, len(cs)), np.nan, np.float32)
    dev = win.x.device
    tt = torch.arange(t0, t1 + 1, device=dev)
    for j, c in enumerate(cs):
        ys = []
        for s in range(0, len(tt), bs):
            t = tt[s:s + bs]
            ys.append(model(win(torch.full_like(t, int(c)), t)).float())
        out[t0:t1 + 1, j] = np.clip(torch.cat(ys).cpu().numpy() * q_std[c] + q_mean[c], 0, None)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["gb", "us", "aus"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--smoke", type=int, default=0, help="smoke test: use only this many training basins, a 2 %% sample of "
                    "the training days per epoch, selection and output only for these basins")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    t_load = time.time()
    Dd = load_gb() if args.dataset == "gb" else load_us(args.dataset)
    days, q, w_np, gauged = Dd["days"], Dd["q"], Dd["w"], Dd["gauged"]
    pos = {d: i for i, d in enumerate(days)}
    i_tr0, i_08, i_09, i_10 = pos[date(1989, 1, 1)], pos[date(2008, 12, 31)], pos[date(2009, 1, 1)], pos[date(2010, 12, 31)]
    assert not np.isfinite(q[i_10 + 1:]).any()                              # test period never loaded
    n_tr = np.isfinite(q[i_tr0:i_08 + 1]).sum(0)
    train_c = np.where(gauged & (n_tr > 730))[0]
    assert not Dd["ug"][train_c].any()
    if args.smoke:
        train_c = train_c[:args.smoke]
    # training samples: (basin, day) with observed q in 1989-01-01 .. 2008-12-31 and a full window
    t_first = max(i_tr0, SEQ - 1)
    valid = np.isfinite(q[t_first:i_08 + 1][:, train_c])
    tt, cc = np.nonzero(valid)
    samp_t = torch.from_numpy((tt + t_first).astype(np.int64)).to(device)
    samp_c = torch.from_numpy(train_c[cc].astype(np.int64)).to(device)
    n_samp = len(samp_t)
    print(f"{Dd['dataset']}; days {days[0]} .. {days[-1]} ({len(days)}), output from {days[Dd['out0']]}; catchments {q.shape[1]} "
          f"(gauged {int(gauged.sum())}, ungauged {int(Dd['ug'].sum())}), training {len(train_c)}; dynamic P, T, PET + static "
          f"{Dd['static'].shape[1]}; training samples {n_samp} (first target day {days[t_first]}); load {time.time() - t_load:.0f}s", flush=True)
    win = Windows(Dd["x"], Dd["static"], device)
    qn_t = torch.from_numpy(Dd["qn"]).to(device)
    w_t = torch.from_numpy(w_np).to(device)
    model = EALSTM(Dd["x"].shape[2], Dd["static"].shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR[1])
    sel_obs = q[i_09:i_10 + 1][:, train_c]
    best, best_ep, best_state, hist = -np.inf, 0, None, []
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    for ep in range(1, args.epochs + 1):
        lr = [v for k, v in LR.items() if k <= ep][-1]
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        perm = torch.randperm(n_samp, device=device)
        perm = perm[:max(BATCH, n_samp // 50)] if args.smoke else perm[:SAMPLES_PER_EPOCH]
        for s in range(0, len(perm), BATCH):
            k = perm[s:s + BATCH]
            c, t = samp_c[k], samp_t[k]
            y = model(win(c, t))
            loss = (w_t[c] * (y - qn_t[t, c]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            tot += loss.item(); nb += 1
        t_tr = time.time() - t0
        sim = simulate(model, win, train_c, i_09, i_10, Dd["q_mean"], Dd["q_std"], i_10 + 1)[i_09:]
        vm = G.median_nse(sim, sel_obs, np.ones(len(train_c), bool))
        rec = {"epoch": ep, "lr": lr, "loss": tot / max(nb, 1), "sel_median_nse": vm, "train_s": round(t_tr, 1),
               "total_s": round(time.time() - t0, 1)}
        mark = ""
        if vm > best:
            best, best_ep, mark = vm, ep, "  *best*"
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        hist.append(rec)
        print(f"[{args.dataset} s{args.seed}] epoch {ep:2d}/{args.epochs}  lr {lr:.0e}  loss {rec['loss']:.4f}  train {t_tr:.0f}s  "
              f"total {rec['total_s']:.0f}s  sel median NSE {vm:.4f}{mark}  peak {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB",
              flush=True)
    assert model.check_mask() == 0.0, "EA-LSTM masked weights moved"
    minutes = (time.time() - t_start) / 60
    model.load_state_dict(best_state)
    out = Path(args.out or f"runs/{args.dataset}_ealstm_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    t0 = time.time()
    cs = train_c if args.smoke else np.arange(q.shape[1])
    full = simulate(model, win, cs, SEQ - 1, len(days) - 1, Dd["q_mean"], Dd["q_std"], len(days))[Dd["out0"]:]
    np.save(out / "sim.npy", full)
    np.savez(out / "norm.npz", q_mean=Dd["q_mean"], q_std=Dd["q_std"], w=w_np, static=Dd["static"], static_names=np.array(Dd["snames"]),
             ids=np.array(Dd["data"]["ids"]), train_c=train_c, **({"f_mean": Dd["f_mean"], "f_std": Dd["f_std"]} if "f_mean" in Dd else {}))
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    res = {"model": "EA-LSTM (Kratzert et al. 2019): static -> input gate only; dynamic -> forget/cell/output gates",
           "dataset": Dd["dataset"], "seed": args.seed, "smoke": args.smoke,
           "protocol": "unified: fit 1989-01-01..2008-12-31 (earlier days only as window history), select epoch on median NSE "
                       "2009-01-01..2010-12-31 of the gauged training catchments, keep best; test 2011-2019 not evaluated here; "
                       "ungauged catchments never in loss / selection / normalisation",
           "config": {"seq_len": SEQ, "hidden": HIDDEN, "dropout": DROP, "batch": BATCH, "epochs": args.epochs, "lr_schedule": LR,
                      "optimizer": "Adam", "clip_grad_norm": CLIP, "initial_forget_bias": FORGET_BIAS,
                      "init": "orthogonal input weights, identity recurrent weights per gate, zero biases (Kratzert code)",
                      "loss": "NSE*: w_b (y-q)^2, w_b = 1/(s_b+0.1)^2 on the per-basin p_mean-scaled, globally standardised target",
                      "samples_per_epoch": SAMPLES_PER_EPOCH if not args.smoke else "2% (smoke)", "implementation": "cuDNN nn.LSTM with masked EA connections"},
           "dynamic": ["P", "T", "PET"], "static": Dd["snames"], "n_train_catchments": int(len(train_c)),
           "n_gauged": int(gauged.sum()), "n_ungauged": int(Dd["ug"].sum()), "n_train_samples": int(n_samp),
           "sim_days": [str(days[Dd["out0"]]), str(days[-1])], "sim_shape": list(full.shape),
           "first_full_window_day": str(days[max(SEQ - 1, Dd["out0"])]),
           "best_epoch": best_ep, "sel_median_nse_2009_10": best, "train_minutes": round(minutes, 1),
           "sim_minutes": round((time.time() - t0) / 60, 1), "peak_gpu_mem_gib": round(peak, 2),
           "epoch_seconds_mean": round(float(np.mean([h["total_s"] for h in hist])), 1), "history": hist}
    json.dump(res, open(out / "run.json", "w"), indent=1, default=float)
    print(f"done {args.dataset} s{args.seed}: best epoch {best_ep}, selection median NSE {best:.4f}, training {minutes:.1f} min, "
          f"simulation {res['sim_minutes']} min, peak {peak:.2f} GiB, sim {full.shape} -> {out}", flush=True)


if __name__ == "__main__":
    main()

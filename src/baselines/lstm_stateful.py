"""Stateful LSTM benchmark (cuDNN LSTM, hidden 256, output dropout 0.3, forget-gate bias 3; truncated back-propagation through
time on 180-day chunks with a random offset, groups of 151 catchments, Adam 1e-3, gradient clip 1, NSE* loss (Kratzert et al. 2019) on
q / p_mean globally standardised; at most 300 epochs, the epoch with the best median NSE of the gauged training catchments over
2009-2010 is kept (patience 30)). Inputs = the information CaRT+ gets: daily P, mean air temperature, PET + the static attributes +
the 5 equal-area band elevations.

  GB: spin-up 1988-01-03 .. 1988-12-31 (no loss), fit
     1989-2008, select on 2009-2010; observed flows after 2010 are set to NaN on loading; band elevations = the CaRT bands,
     z-scored over the gauged catchments; 21 + 5 static inputs.
  US: spin-up 1980-1988, fit 1989-2008, select on 2009-2010; band elevations
     = elevations of the equal-area band centres of the CAMELS hypsometry (band_quantile_elevations), z-scored over the gauged basins;
     27 + 5 static inputs; continuous inference in 365-day calls.
  AUS: the US code path on CAMELS-AUS v2; band elevations from the AUS hypsometry (elev_bands_aus.npz).
Data are read with cart.data.

Usage (from src/): python -m baselines.lstm_stateful --dataset gb|us|aus --seed 0 [--gpu 0] [--out DIR]
  -> sim.npy [T, N] (mm/d, full record on the dataset days), model.pt, norm.npz (US, AUS), run.json
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

CHUNK, GROUP, HIDDEN, DROP, LR, EPOCHS, PATIENCE = 180, 151, 256, 0.3, 1e-3, 300, 30
END_DEV = date(2010, 12, 31)
CONT_CHUNK_US = 365                                                        # days per inference call on CAMELS-US and CAMELS-AUS


_DATASET = "gb"


class SLSTM(nn.Module):
    def __init__(self, n_f, n_s):
        super().__init__()
        self.lstm = nn.LSTM(n_f + n_s, HIDDEN, batch_first=True)
        with torch.no_grad():
            self.lstm.bias_hh_l0[HIDDEN:2 * HIDDEN] = 3.0                # forget gate (PyTorch order i, f, g, o)
        self.drop = nn.Dropout(DROP)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x, static, state=None):
        """x [C, T, F], static [C, S] -> y [C, T], state."""
        inp = torch.cat([x, static[:, None].expand(-1, x.shape[1], -1)], dim=2)
        out, state = self.lstm(inp, state)
        return self.head(self.drop(out)).squeeze(-1), state


def stage_stats(raw, q, data, pos, fit_end):
    """Normalisation from the stage's fitting days only: forcing (spin-up .. fit_end, gauged), global q / p_mean
    standardisation and NSE* weights (1989-01-01 .. fit_end)."""
    g = data["basin_mask"]
    i1 = pos[fit_end]
    fm = raw[:i1 + 1][:, g].mean(axis=(0, 1)); fs = raw[:i1 + 1][:, g].std(axis=(0, 1)) + 1e-6
    x = ((raw - fm) / fs).astype(np.float32)
    scale = data["p_mean"].astype(np.float32)                              # raw p_mean attribute (static_raw[:, p_mean])
    qt = q[pos[date(1989, 1, 1)]:i1 + 1] / scale[None]
    b_std = np.nanstd(qt, axis=0)
    usable = g & np.isfinite(b_std) & (b_std > 0)
    gm, gs = float(np.nanmean(qt[:, usable])), float(np.nanstd(qt[:, usable]))
    q_mean, q_std = (gm * scale).astype(np.float32), (gs * scale).astype(np.float32)
    s_b = np.where(usable, b_std / gs, 1.0)
    w = (1.0 / (s_b + 0.1) ** 2).astype(np.float32)
    w = np.where(usable, w / w[usable].mean(), 0.0).astype(np.float32)
    qn = np.nan_to_num((q - q_mean) / q_std).astype(np.float32)
    return x, qn, q_mean, q_std, w


@torch.no_grad()
def run_continuous(model, x, static, device, chunk=None):
    chunk = chunk or (CONT_CHUNK_US if _DATASET in D.US_LAYOUT else 2000)   # GB 2000-day calls, US (and AUS) 365-day calls
    model.eval()
    out, state = [], None
    for s in range(0, x.shape[0], chunk):
        y, state = model(torch.from_numpy(x[s:s + chunk]).transpose(0, 1).to(device), static, state)
        out.append(y.float().cpu().numpy().T)
    return np.concatenate(out)                                             # [days, C]


def median_nse(sim, obs, cmask, min_days=365):
    v = []
    for c in np.where(cmask)[0]:
        o, s = obs[:, c], sim[:, c]
        k = np.isfinite(o)
        if k.sum() > min_days:
            v.append(1 - np.sum((s[k] - o[k]) ** 2) / np.sum((o[k] - o[k].mean()) ** 2))
    return float(np.median(v))


def train_stage(seed, x, qn, q, valid, w_np, static, train_c, i_tr0, i_tr1, n_epochs, device, sel=None, run=None, log="", keep_best=False):
    """TBPTT training as train_stateful_lstm.py. sel = (i_v0, i_v1): select the epoch on that block (patience), else
    train exactly n_epochs. Returns model, best epoch, history."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = SLSTM(x.shape[2], static.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    w = torch.from_numpy(w_np).to(device)
    xt, qt, vt = torch.from_numpy(x), torch.from_numpy(qn), torch.from_numpy(valid)
    best, best_ep, hist, best_state = -np.inf, 0, [], None
    for ep in range(1, n_epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        perm = np.random.permutation(train_c)
        off = np.random.randint(CHUNK)
        for g in range(0, len(perm), GROUP):
            cc = torch.as_tensor(perm[g:g + GROUP])
            st = static[cc.to(device)]
            with torch.no_grad():                                          # spin-up 1988 (+ random offset)
                _, state = model(xt[:i_tr0 + off, cc].transpose(0, 1).to(device), st)
            for s in range(i_tr0 + off, i_tr1 + 1, CHUNK):
                e = min(s + CHUNK, i_tr1 + 1)
                y, state = model(xt[s:e, cc].transpose(0, 1).to(device), st, state)
                state = tuple(h.detach() for h in state)
                tq, tv = qt[s:e, cc].T.to(device), vt[s:e, cc].T.to(device)
                if tv.sum() == 0:
                    continue
                loss = ((w[cc.to(device)][:, None] * (y - tq) ** 2)[tv]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item(); nb += 1
        rec = {"epoch": ep, "loss": tot / max(nb, 1)}
        mark = ""
        if sel is not None:
            i_v0, i_v1 = sel
            yn = run_continuous(model, x[:i_v1 + 1], static, device)       # never beyond the selection block
            sim = np.clip(yn * q_std_g[None] + q_mean_g[None], 0, None)
            vm = median_nse(sim[i_v0:i_v1 + 1], q[i_v0:i_v1 + 1], np.isin(np.arange(q.shape[1]), train_c))
            rec["sel_median_nse"] = vm
            if vm > best:
                best, best_ep = vm, ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                mark = "  *best*"
        hist.append(rec)
        print(f"{log} epoch {ep:3d}/{n_epochs}  loss {rec['loss']:.4f}  {time.time() - t0:.0f}s"
              + (f"  sel median NSE {rec['sel_median_nse']:.4f}{mark}" if sel else ""), flush=True)
        if sel is not None and ep - best_ep >= PATIENCE:
            print(f"{log} early stop (best epoch {best_ep}, selection median NSE {best:.4f})", flush=True)
            break
    if keep_best and best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, best, hist


def band_elevations(ids, K=5, eb=None):
    """Equal-area groups of the hypsometry, as CaRT (cart.data): H [C, K] (m)."""
    be, bf = eb                                                            # (band_elev, band_frac) of the basins ids, from cart.data
    C = len(ids)
    H = np.zeros((C, K)); A_ = np.zeros((C, K))
    for i in range(C):
        e, a = be[i][bf[i] > 0], bf[i][bf[i] > 0]; o_ = np.argsort(e); e, a = e[o_], a[o_]
        g = np.minimum((np.cumsum(a) - 1e-9) // (1.0 / K), K - 1).astype(int)
        for j in range(K):
            if (g == j).any():
                H[i, j] = (a[g == j] * e[g == j]).sum() / a[g == j].sum(); A_[i, j] = a[g == j].sum()
    return H, A_


def band_quantile_elevations(ids, K=5, eb=None):
    """Elevations of the K equal-area band centres (10, 30, 50, 70, 90 % of the area for K = 5) from the hypsometric
    curve of elev_bands671.npz (100 m bands, area uniform within a band) -- the GB construction (interpolated elevation
    quantiles) applied to the US hypsometry. CaRT+'s grouping (band_elevations) assigns whole 100 m bands to groups and
    leaves ~1/3 of the groups empty (H = 0, area 0), which is not a usable network input."""
    be, bf = eb                                                            # (band_elev, band_frac) of the basins ids, from cart.data
    qm = (np.arange(K) + 0.5) / K
    out = np.zeros((len(ids), K))
    for i in range(len(ids)):
        k = bf[i] > 0; e, a = be[i][k], bf[i][k]; o_ = np.argsort(e); e, a = e[o_], a[o_] / a.sum()
        cdf = np.concatenate([[0.0], np.cumsum(a)])
        xs = np.ravel(np.column_stack([cdf[:-1], cdf[1:]])); ys = np.ravel(np.column_stack([e - 50.0, e + 50.0]))
        out[i] = np.interp(qm, xs, ys)
    return out


def build_inputs_gb(ds):
    """raw forcing [D, C, 3] (P, T, PET) on 1988-01-03 .. 2019-12-31, targets with everything after 2010 set
    to NaN, static [C, 26] and the offset of day 0 of the dataset in this day list."""
    C = ds.P.shape[1]
    q = ds.q.copy()
    q[np.array([d > END_DEV for d in ds.days])] = np.nan                   # withheld: never used, never looked at
    off = len(ds.spin_days)
    raw = np.concatenate([np.asarray(ds.spin_forcing, np.float64),
                          np.stack([ds.P, ds.Tm, ds.E], -1)]).astype(np.float32)   # [D, C, 3]: P, T, PET
    days = list(ds.spin_days) + list(ds.days)
    qq = np.concatenate([np.full((off, C), np.nan), q]).astype(np.float32)
    H = ds.band_H                                                          # the CaRT bands (5 equal-area band elevations)
    g = ds.gauged
    Hn = (H - H[g].mean(0)) / (H[g].std(0) + 1e-6)
    static = np.concatenate([ds.static.astype(np.float32), Hn.astype(np.float32)], 1)
    qm = 100.0 * (np.arange(5) + 0.5) / 5
    snames = list(ds.static_names) + [f"band_elev_{int(p)}" for p in qm]
    return days, off, raw, qq, static, snames


def main():
    global q_mean_g, q_std_g
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["gb", "us", "aus"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="maximum epochs (smoke tests only)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    out = Path(args.out or f"runs/{args.dataset}_lstm_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    global _DATASET
    _DATASET = args.dataset
    ds = D.load(args.dataset)
    data = {"ids": ds.ids, "basin_mask": ds.gauged, "ug_mask": ds.ungauged, "static": ds.static, "p_mean": ds.p_mean}
    t_start = time.time()
    if args.dataset == "gb":
        days, off, raw, q, static_np, snames = build_inputs_gb(ds)
        pos = {d: i for i, d in enumerate(days)}
        i_tr0, i_08, i_09, i_10 = pos[date(1989, 1, 1)], pos[date(2008, 12, 31)], pos[date(2009, 1, 1)], pos[date(2010, 12, 31)]
        gauged = ds.gauged
        n_tr = np.isfinite(q[i_tr0:i_08 + 1]).sum(0)
        train_c = np.where(gauged & (n_tr > 730))[0]
        static = torch.from_numpy(static_np).to(device)
        valid = np.isfinite(q) & gauged[None]
        x, qn, q_mean_g, q_std_g, w = stage_stats(raw, q, data, pos, date(2008, 12, 31))
        model, best_ep, best, hist = train_stage(args.seed, x, qn, q, valid, w, static, train_c, i_tr0, i_08, args.epochs, device,
                                                 sel=(i_09, i_10), log=f"[gb s{args.seed}]", keep_best=True)
        minutes = (time.time() - t_start) / 60
        yn = run_continuous(model, x, static, device)
        sim = np.clip(yn * q_std_g[None] + q_mean_g[None], 0, None)[off:].astype(np.float32)
        torch.save(model.state_dict(), out / "model.pt")
    else:
        days, P, E, Tm = ds.days, ds.P, ds.E, ds.Tm
        q = ds.q.astype(np.float32)
        q[np.array([d > END_DEV for d in days])] = np.nan                  # nothing after 2010 is used or looked at
        raw = np.stack([P, Tm, E], -1).astype(np.float32)                  # [D, C, 3]: P, T (K), PET
        gauged = ds.gauged
        H = band_quantile_elevations(ds.ids, eb=(ds.band_elev_raw, ds.band_frac_raw))
        H_cart, A_cart = band_elevations(ds.ids, eb=(ds.band_elev_raw, ds.band_frac_raw))                           # CaRT+ grouping, saved for the record
        h_mean, h_std = H[gauged].mean(0), H[gauged].std(0) + 1e-6
        static_np = np.concatenate([ds.static, ((H - h_mean) / h_std).astype(np.float32)], 1)
        snames = list(ds.static_names) + [f"band_elev_{int(p)}" for p in (10, 30, 50, 70, 90)]
        assert np.isfinite(static_np).all()
        pos = {d: i for i, d in enumerate(days)}
        i_tr0, i_08, i_09, i_10 = pos[date(1989, 1, 1)], pos[date(2008, 12, 31)], pos[date(2009, 1, 1)], pos[date(2010, 12, 31)]
        n_tr = np.isfinite(q[i_tr0:i_08 + 1]).sum(0)
        train_c = np.where(gauged & (n_tr > 730))[0]
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
        valid = np.isfinite(q) & gauged[None]
        q_mean_g, q_std_g = q_mean, q_std
        static = torch.from_numpy(static_np).to(device)
        model, best_ep, best, hist = train_stage(args.seed, x, qn, q, valid, w, static, train_c, i_tr0, i_08, args.epochs, device,
                                                 sel=(i_09, i_10), log=f"[{args.dataset} s{args.seed}]", keep_best=True)
        minutes = (time.time() - t_start) / 60
        torch.save(model.state_dict(), out / "model.pt")
        np.savez(out / "norm.npz", f_mean=fm, f_std=fs, q_mean=q_mean, q_std=q_std, static=static_np, static_names=np.array(snames),
                 ids=ds.ids, band_H=H, band_H_cartplus=H_cart, band_A_cartplus=A_cart, band_h_mean=h_mean, band_h_std=h_std,
                 inputs=np.array(["P", "T_mean_K", "PET"]))
        yn = run_continuous(model, x, static, device)
        sim = np.clip(yn * q_std[None] + q_mean[None], 0, None).astype(np.float32)
    np.save(out / "sim.npy", sim)
    json.dump({"dataset": args.dataset, "seed": args.seed, "dynamic": ["P", "T", "PET"], "static": snames,
               "n_train_catchments": int(len(train_c)), "best_epoch": best_ep, "sel_median_nse_2009_10": best,
               "train_minutes": round(minutes, 1), "history": hist}, open(out / "run.json", "w"), indent=1, default=float)
    print(f"done {args.dataset} s{args.seed}: best epoch {best_ep}, selection median NSE {best:.4f}, {minutes:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()

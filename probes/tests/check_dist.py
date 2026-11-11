"""Checks of distributed CaRT+ on the MERIT unit catchments (loader, chunk plan, gauge step, gradient accumulation, runs).

Author: tbindas

What it does. Stage `data` (CPU):
  gauges        gauge ids, order and discharge of the distributed store are bit-equal to the lumped store.
  forcing       P, Temp, PET of the loader are bit-equal to the store where the store is finite, and 0 where it is NaN.
                divide_dataset gives Tm = Temp + 273.15 and the [day, divide] layout.
  missing data  NaN counts before the fill equal the numbers of the brief; no NaN after the fill.
  map           each gauge has >= 1 divide, rows are in range and unique, the union is all 73,959 divides, fractions sum to 1.
  attributes    scaled attributes are bit-equal (float32) to scale.cal_stat + scale.trans_norm (loaded by file path) + NaN -> 0.
  chunk plan    each gauge in exactly 1 chunk, each chunk holds all divides of its gauges once, no chunk over the cap,
                the plan is the same when built twice.
  aggregation   uh.aggregate against a plain numpy loop; a gauge with 1 divide returns the runoff of that divide.
  UH copy       uh.UH_gamma and uh.UH_conv against the source functions of hydroDL/model/rnn.py. The source text of the
                two functions is read from the file and run as it is (hydroDL itself cannot be imported in this
                environment: rnn.py imports `kan`). Needs the GPU, because the source calls `.cuda`. Also against a
                numpy version written here.
  UH properties weights >= 0, sum 1, causal, volume of an impulse kept.
  lumped P      information: area-weighted divide P of 20 gauges against the P of the lumped store.
Stage `gpu`:
  model inputs  ChunkCaRT holds the same tensors, bit for bit, as cart.model.CaRT built on a Dataset of the same divides;
                the mapping network starts at the same weights; simulate() gives the same runoff.
  elevation     the runoff does not change when the elevation of each divide changes (one band).
  accumulation  loss and gradient of 50 gauges from 1 chunk equal those from >= 2 chunks (relative 1e-4), no noise,
                at the start weights and at perturbed weights.
Stage `run` (--run <run dir>, CPU):
  outputs       the files exist; sim_preuh.npy and uh_gauge.npy are finite; the saved chunk plan equals a new plan.
  gauge volume  mean of the UH output against the mean of the summed runoff per gauge.

    python probes/tests/check_dist.py --stage data|gpu|run [--run <run dir>] [--gpu 0]

Experimental purpose: evidence that distributed CaRT+ differs from lumped CaRT+ in the spatial layout only, and that
its gauge step is the gauge step of the distributed dHBV run, before the full run starts.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(PKG / "src"))
import merit_data  # noqa: E402
import merit_dist_data as mdd  # noqa: E402

WL = Path.home() / "projects/water_loss/dPLHBVrelease-master/hydroDL-dev"
SCALE_PY = WL / "hydroDL/data/scale.py"
RNN_PY = WL / "hydroDL/model/rnn.py"
BRIEF = {"forcing_all_nan_divides": 71, "attr_nan_divides": 548, "elev_nan_divides": 1, "n_gauges_with_any_nan_divide": 436,
         "attr_nan_per_column": {"snow_fraction": 43, "meanelevation": 1, "NDVI": 247, "Porosity": 276, "HWSD_clay": 11}}

fails = []


def check(name, ok, msg):
    print(f"{'PASSED' if ok else 'FAILED'}  {name}: {msg}", flush=True)
    if not ok:
        fails.append(name)


def info(name, msg):
    print(f"INFO    {name}: {msg}", flush=True)


def by_path(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def source_functions(path, names):
    """the functions `names` of a python file, run from their source text without an import of the file"""
    import torch
    import torch.nn.functional as F
    src = open(path).read()
    tree = ast.parse(src)
    ns = {"torch": torch, "F": F}
    lines = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
            lines[node.name] = (node.lineno, node.end_lineno)
    return [ns[n] for n in names], lines


def np_uh_gamma(a, b, lenF):
    """gamma UH weights in numpy float64: a, b [G] -> [G, lenF]"""
    from math import lgamma
    aa = np.maximum(a, 0) + 0.1; th = np.maximum(b, 0) + 0.5
    t = np.arange(0.5, lenF)[None]
    w = t ** (aa[:, None] - 1) * np.exp(-t / th[:, None]) / (np.exp([lgamma(x) for x in aa]) * th ** aa)[:, None]
    return w / w.sum(1, keepdims=True)


def np_uh_conv(x, w):
    """causal convolution in numpy: y[g, t] = sum_k w[g, k] x[g, t - k]"""
    y = np.zeros_like(x)
    for k in range(w.shape[1]):
        y[:, k:] += w[:, k:k + 1] * x[:, :x.shape[1] - k]
    return y


# ------------------------------------------------------------------------------------------------------------ data
def stage_data(gpu):
    import torch
    import uh
    z = zarr.open_group(mdd.STORE, mode="r")
    zl = zarr.open_group(merit_data.STORE, mode="r")
    dd = mdd.load()
    G, D, T = len(dd.ids), dd.P.shape[0], dd.P.shape[1]

    # gauges
    ids_l, q_l = zl["gage_staid"][:], zl["flow_mmday"][:]
    check("gauges", (D, T, G) == (mdd.N_DIVIDES, mdd.N_DAYS, mdd.N_GAUGES) and list(dd.ids) == list(ids_l) and same_bits(dd.q, q_l),
          f"{G} gauges, {D} divides, {T} days; ids and order equal to the lumped store: {list(dd.ids) == list(ids_l)}; discharge bit-equal "
          f"to the lumped store: {same_bits(dd.q, q_l)} (shape {dd.q.shape}, {dd.q.dtype}, NaN {int(np.isnan(dd.q).sum())})")
    del q_l

    # forcing
    ok_all, parts, nan_after = True, [], 0
    for k, arr in (("P", dd.P), ("Temp", dd.Temp), ("PET", dd.PET)):
        raw = z[k][:]
        nanm = np.isnan(raw)
        eq = arr.shape == raw.shape and arr.dtype == raw.dtype and bool((arr[~nanm] == raw[~nanm]).all()) and bool((arr[nanm] == 0).all())
        nan_after += int(np.isnan(arr).sum())
        ok_all &= eq
        parts.append(f"{k} {arr.shape} {arr.dtype} equal where finite and 0 where NaN: {eq} (NaN in the store {int(nanm.sum())})")
        del raw, nanm
    rows = np.array([0, 17, 5000, 73958])
    ds = mdd.divide_dataset(dd, rows)
    tm_ok = ds.Tm.dtype == np.float64 and ds.Tm.shape == (T, 4) and bool((ds.Tm == dd.Temp[rows].T.astype(np.float64) + 273.15).all()) \
        and bool((ds.P == dd.P[rows].T.astype(np.float64)).all()) and bool((ds.E == dd.PET[rows].T.astype(np.float64)).all())
    check("forcing", ok_all and tm_ok, "; ".join(parts) + f"; divide_dataset: Tm = Temp + 273.15 in float64, layout [day, divide]: {tm_ok}")

    # missing data
    n = dd.nan
    got = {k: n[k] for k in BRIEF}
    after = nan_after + int(np.isnan(dd.static).sum()) + int(np.isnan(dd.elev).sum()) + int(np.isnan(dd.catchsize).sum())
    check("missing data", got == BRIEF and after == 0 and n["forcing_any_nan_divides"] == n["forcing_all_nan_divides"],
          f"before the fill: divides with all-NaN forcing {n['forcing_all_nan_divides']} (brief 71; divides with part-NaN forcing "
          f"{n['forcing_any_nan_divides'] - n['forcing_all_nan_divides']}), attribute NaN per column {n['attr_nan_per_column']} "
          f"(brief {BRIEF['attr_nan_per_column']}), divides with a NaN attribute {n['attr_nan_divides']} (brief 548), NaN elevation "
          f"{n['elev_nan_divides']} (brief 1), gauges with a divide that has any NaN {n['n_gauges_with_any_nan_divide']} (brief 436); "
          f"gauges with a divide without forcing {n['n_gauges_with_nan_forcing']}; NaN after the fill (forcing, attributes, elevation, area) {after}")

    # map
    sizes = np.array([len(r) for r in dd.gmap])
    allr = np.concatenate(dd.gmap)
    uniq_in = all(len(np.unique(r)) == len(r) for r in dd.gmap)
    fsum = np.array([f.sum() for f in dd.frac])
    f_ref = [z["catchsize"][:][r].astype(np.float64) for r in dd.gmap[:200]]
    f_ok = all(np.allclose(f, a / a.sum(), rtol=0, atol=1e-15) for f, a in zip(dd.frac[:200], f_ref))
    ok = sizes.min() >= 1 and allr.min() >= 0 and allr.max() < D and uniq_in and len(np.unique(allr)) == D \
        and np.abs(fsum - 1).max() < 1e-6 and all((f > 0).all() for f in dd.frac) and f_ok
    check("map", ok, f"divides per gauge min {sizes.min()}, median {int(np.median(sizes))}, max {sizes.max()}, sum {sizes.sum()}; rows in "
          f"[{allr.min()}, {allr.max()}]; no divide twice in a gauge: {uniq_in}; union {len(np.unique(allr))} divides (store {D}); "
          f"fractions: max |sum - 1| {np.abs(fsum - 1).max():.1e}, all > 0, equal to catchsize / sum(catchsize): {f_ok}")

    # attributes against scale.cal_stat / trans_norm
    scale = by_path("wl_scale", SCALE_PY)
    attr2 = z["attrs"][:]
    names = list(z.attrs["attr_list"])
    stat = {a: scale.cal_stat(attr2[:, i]) for i, a in enumerate(names)}
    ref = scale.trans_norm(attr2, list(names), stat, to_norm=True)
    ref[ref != ref] = 0
    ref32 = ref.astype(np.float32)
    lump = merit_data.scale_attrs(zl["attrs"][:])
    check("attributes", same_bits(dd.static, ref32) and names == dd.static_names and not np.allclose(dd.attr_mean, lump[1]),
          f"scaled attributes {dd.static.shape} {dd.static.dtype} bit-equal to scale.cal_stat + scale.trans_norm + NaN -> 0 (float32): "
          f"{same_bits(dd.static, ref32)}; max |difference| {np.abs(dd.static - ref32).max():.1e}; statistics from the {D} divides "
          f"(mean of meanP {dd.attr_mean[0]:.3f}; the lumped store gives {lump[1][0]:.3f}); p_mean is the raw meanP column: "
          f"{same_bits(dd.p_mean, attr2[:, names.index('meanP')])}")

    # chunk plan
    for cap, K in ((1100, None), (2000, None), (3000, None), (700, 50)):
        plan = mdd.chunk_plan(dd.gmap, cap, K)
        again = mdd.chunk_plan(dd.gmap, cap, K)
        Gk = K or G
        gs = np.concatenate([c["gauges"] for c in plan])
        once = len(gs) == Gk and len(np.unique(gs)) == Gk and gs.min() == 0 and gs.max() == Gk - 1
        holds = all(set(np.concatenate([dd.gmap[g] for g in c["gauges"]]).tolist()) == set(c["rows"]) and len(set(c["rows"])) == len(c["rows"])
                    and c["rows"] == sorted(c["rows"]) for c in plan)
        s = mdd.plan_summary(plan, dd.gmap)
        check(f"chunk plan cap {cap}" + (f", first {K} gauges" if K else ""), once and holds and s["largest_chunk"] <= cap and plan == again,
              f"{s['n_chunks']} chunks; each gauge in exactly 1 chunk: {once}; each chunk holds the divides of its gauges, each once: {holds}; "
              f"largest chunk {s['largest_chunk']} <= {cap}; columns per pass {s['columns_per_pass']} (unique divides {s['unique_divides']}, "
              f"sum over gauges {s['sum_of_divides_over_gauges']}); same plan when built twice: {plan == again}")

    # aggregation
    dev = torch.device(f"cuda:{gpu}")
    plan = mdd.chunk_plan(dd.gmap, 2000, 50)
    c = plan[0]
    rows, gauges = np.asarray(c["rows"]), np.asarray(c["gauges"])
    rng = np.random.default_rng(0)
    Qd = rng.gamma(0.5, 2.0, size=(len(rows), 400))
    M = uh.membership(gauges, rows, dd.gmap, dd.frac, dev, torch.float64)
    got = uh.aggregate(M, torch.from_numpy(Qd).to(dev)).cpu().numpy()
    ref = np.zeros((len(gauges), 400))
    pos = {r: i for i, r in enumerate(rows.tolist())}
    for j, g in enumerate(gauges):
        for r, f in zip(dd.gmap[g].tolist(), dd.frac[g].tolist()):
            ref[j] += f * Qd[pos[r]]
    M32 = uh.membership(gauges, rows, dd.gmap, dd.frac, dev)
    got32 = uh.aggregate(M32, torch.from_numpy(Qd.astype(np.float32)).to(dev)).cpu().numpy()
    g1 = int(np.where(sizes == 1)[0][0])
    r1 = dd.gmap[g1]
    M1 = uh.membership([g1], r1, dd.gmap, dd.frac, dev)
    q1 = torch.from_numpy(Qd[:1].astype(np.float32)).to(dev)
    one = bool((uh.aggregate(M1, q1) == q1).all())
    d64, d32 = np.abs(got - ref).max(), np.abs(got32 - ref).max()
    check("aggregation", d64 < 1e-6 and d32 < 1e-4 * ref.max() and one,
          f"{len(gauges)} gauges, {len(rows)} divides, 400 days against a numpy loop: max |difference| float64 {d64:.1e} (limit 1e-6), "
          f"float32 {d32:.1e} (largest value {ref.max():.1f}); gauge {dd.ids[g1]} with 1 divide returns the runoff of that divide: {one}")

    # UH copy against the source functions
    (src_gamma, src_conv), lines = source_functions(RNN_PY, ["UH_gamma", "UH_conv"])
    gen = torch.Generator().manual_seed(0)
    ng, nt = 64, 300
    a = (torch.rand(ng, generator=gen) * 2.9).to(dev); b = (torch.rand(ng, generator=gen) * 6.5).to(dev)
    a[0], b[0], a[1], b[1] = 0.0, 0.0, 2.9, 6.5
    x = torch.rand(ng, nt, generator=gen).to(dev) * 10
    routa, routb = a.repeat(nt, 1).unsqueeze(-1), b.repeat(nt, 1).unsqueeze(-1)                # as waterlossv18_1.py:472-473
    w_src = src_gamma(routa, routb, lenF=15)
    w_new = uh.UH_gamma(routa, routb, lenF=15)
    y_src = src_conv(x.unsqueeze(1), w_src.permute([1, 2, 0]))
    y_new = uh.UH_conv(x.unsqueeze(1), w_new.permute([1, 2, 0]))
    Mi = torch.eye(ng, device=dev)
    y_route = uh.route(Mi, x, a, b)[0]
    w_np = np_uh_gamma(a.double().cpu().numpy(), b.double().cpu().numpy(), 15)
    y_np = np_uh_conv(x.double().cpu().numpy(), w_np)
    dw, dy = float((w_src - w_new).abs().max()), float((y_src - y_new).abs().max())
    dr = float((y_route - y_src[:, 0]).abs().max())
    dnw, dny = np.abs(w_new[:, :, 0].T.cpu().numpy() - w_np).max(), np.abs(y_route.cpu().numpy() - y_np).max()
    check("UH copy", dw < 1e-6 and dy < 1e-6 and dr < 1e-6 and dnw < 1e-6 and dny < 1e-4,
          f"against the source text of rnn.py (UH_gamma lines {lines['UH_gamma']}, UH_conv lines {lines['UH_conv']}), {ng} gauges, {nt} days, "
          f"a in [0, 2.9], b in [0, 6.5], lenF 15, float32 on the GPU: max |difference| weights {dw:.1e}, UH_conv {dy:.1e}, uh.route {dr:.1e} "
          f"(limit 1e-6); against a numpy float64 version: weights {dnw:.1e}, output {dny:.1e} (float32 limits, values up to 10)")

    # UH properties (float64)
    a64, b64 = a.double(), b.double()
    w = uh.UH_gamma(a64.repeat(15, 1).unsqueeze(-1), b64.repeat(15, 1).unsqueeze(-1), lenF=15)[:, :, 0].T        # [G, 15]
    imp = torch.zeros(ng, nt, dtype=torch.float64, device=dev); imp[:, 100] = 1.0
    yi = uh.route(Mi.double(), imp, a64, b64)[0]
    causal = float(yi[:, :100].abs().max())
    resp = float((yi[:, 100:115] - w).abs().max())
    vol = float((yi.sum(1) - 1).abs().max())
    x2 = x.double().clone(); x2[:, 200:] += 5.0
    future = float((uh.route(Mi.double(), x2, a64, b64)[0][:, :200] - uh.route(Mi.double(), x.double(), a64, b64)[0][:, :200]).abs().max())
    check("UH properties", float(w.min()) >= 0 and float((w.sum(1) - 1).abs().max()) < 1e-12 and causal == 0 and resp < 1e-12 and vol < 1e-5
          and future == 0,
          f"weights min {float(w.min()):.2e} (>= 0), max |sum - 1| {float((w.sum(1) - 1).abs().max()):.1e}; impulse on day 100: output before "
          f"day 100 max {causal:.1e} (causal), response equal to the weights (max |difference| {resp:.1e}), volume max |sum - 1| {vol:.1e} "
          f"(limit 1e-5); a change of the input from day 200 changes the output before day 200 by {future:.1e}")

    # lumped consistency (information)
    Pl = zl["P"][:20].astype(np.float64)
    Pd = np.stack([(dd.frac[g][:, None] * dd.P[dd.gmap[g]].astype(np.float64)).sum(0) for g in range(20)])
    corr = np.array([np.corrcoef(Pl[g], Pd[g])[0, 1] for g in range(20)])
    ratio = Pd.mean(1) / Pl.mean(1)
    info("lumped P", f"first 20 gauges, area-weighted divide P against the P of the lumped store: correlation min {corr.min():.4f}, median "
         f"{np.median(corr):.4f}; ratio of the means min {ratio.min():.4f}, median {np.median(ratio):.4f}, max {ratio.max():.4f}")


# ------------------------------------------------------------------------------------------------------------- gpu
def stage_gpu(gpu):
    import torch
    from cart.config import config_for
    from cart.model import CaRT
    import train_merit_dist as tmd
    dev = torch.device(f"cuda:{gpu}"); torch.cuda.set_device(dev)
    cfg = config_for("cartplus", "merit")
    dd = mdd.load()
    seed = tmd.SEED

    # model inputs: ChunkCaRT against CaRT on a Dataset of the same divides
    st = tmd.Setup(dd, cfg, seed, 20, 2000, dev, pin=False)
    ch = st.chunks[0]
    rows = ch["rows"]
    ds = mdd.divide_dataset(dd, rows)
    torch.manual_seed(seed); np.random.seed(seed)
    ref = CaRT(ds, cfg, dev)
    st.model.load(ch)
    names = ("Hdev", "Aband", "Pbar_ie", "xs", "Pg", "Eg", "Tcl", "trm")
    eq = {k: bool(getattr(st.model, k).dtype == getattr(ref, k).dtype and getattr(st.model, k).shape == getattr(ref, k).shape
                  and torch.equal(getattr(st.model, k), getattr(ref, k))) for k in names}
    scal = (st.model.T, st.model.Ttr, st.model.nfftV) == (ref.T, ref.Ttr, ref.nfftV)
    w_eq = all(torch.equal(p, q) for p, q in zip(st.net.state_dict().values(), ref.net.state_dict().values()))
    with torch.no_grad():
        Qa = st.model.simulate()[0]; Qb = ref.simulate()[0]
    q_eq = torch.equal(Qa, Qb)
    nan_rows = int(np.isin(rows, dd.nan["forcing_all_nan_rows"]).sum())
    check("model inputs", all(eq.values()) and scal and w_eq and q_eq and bool(torch.isfinite(Qa).all()),
          f"{len(rows)} divides of the first 20 gauges ({nan_rows} without forcing): ChunkCaRT tensors bit-equal to cart.model.CaRT on a Dataset "
          f"of the divides: {eq}; T, Ttr, nfftV equal: {scal} ({ref.T}, {ref.Ttr}, {ref.nfftV}); start weights of the mapping network equal "
          f"(seed {seed}): {w_eq}; runoff of simulate() bit-equal: {q_eq} (max |difference| {float((Qa - Qb).abs().max()):.1e}), finite: "
          f"{bool(torch.isfinite(Qa).all())}")

    # elevation has no effect with one band
    ds2 = mdd.divide_dataset(dd, rows)
    ds2.band_H = ds2.band_H + 1234.5
    torch.manual_seed(seed); np.random.seed(seed)
    ref2 = CaRT(ds2, cfg, dev)
    with torch.no_grad():
        Qc = ref2.simulate()[0]
    check("elevation", torch.equal(Qb, Qc) and bool((ref.Hdev == 0).all()) and bool((ref2.Hdev == 0).all()),
          f"one band per divide: Hdev is 0 for each divide ({bool((ref.Hdev == 0).all())}); runoff with each elevation + 1234.5 m is bit-equal: "
          f"{torch.equal(Qb, Qc)}. So the fill value of the 1 NaN elevation has no effect.")
    del ref, ref2, Qa, Qb, Qc, st
    torch.cuda.empty_cache()

    # gradient accumulation: 1 chunk against >= 2 chunks
    K = 50
    one = tmd.Setup(dd, cfg, seed, K, 2000, dev, pin=False)
    many = tmd.Setup(dd, cfg, seed, K, 700, dev, pin=False)
    msgs, ok = [], len(one.chunks) == 1 and len(many.chunks) >= 2

    def loss_grad(s):
        for p in s.params:
            p.grad = None
        loss, ntr = s.accumulate(0.0)
        return loss.item(), torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in s.params]).double(), ntr
    for label in ("start weights", "perturbed weights"):
        if label == "perturbed weights":
            gen = torch.Generator(device="cpu").manual_seed(1)
            with torch.no_grad():
                for p, q in zip(one.params, many.params):
                    d = 0.02 * torch.randn(p.shape, generator=gen).to(dev)
                    p.add_(d); q.add_(d)
        l1, g1, n1 = loss_grad(one)
        l2, g2, n2 = loss_grad(many)
        rl = abs(l1 - l2) / abs(l1)
        rg = float((g1 - g2).norm() / g1.norm())
        gu1 = torch.cat([p.grad.flatten() for p in one.uhh.parameters()]).double(); gu2 = torch.cat([p.grad.flatten() for p in many.uhh.parameters()]).double()
        ru = float((gu1 - gu2).norm() / gu1.norm())
        dn = float((n1.sort().values - n2.sort().values).abs().max())
        ok &= rl < 1e-4 and rg < 1e-4 and ru < 1e-4 and np.isfinite(l1) and float(g1.norm()) > 0 and float(gu1.norm()) > 0
        msgs.append(f"{label}: loss {l1:.7f} against {l2:.7f} (relative difference {rl:.1e}); gradient norm {float(g1.norm()):.4e}, relative "
                    f"difference of the gradient vectors {rg:.1e}; UH head gradient norm {float(gu1.norm()):.3e}, relative difference {ru:.1e}; "
                    f"max |difference| of the gauge NSE {dn:.1e}")
    s2 = mdd.plan_summary(many.plan, dd.gmap)
    check("accumulation", ok, f"first {K} gauges ({one.n_loss} in the loss), no noise, {len(one.chunks)} chunk of {len(one.chunks[0]['rows'])} divides "
          f"against {len(many.chunks)} chunks (cap 700, columns per pass {s2['columns_per_pass']}); limit 1e-4. " + "; ".join(msgs))


# ------------------------------------------------------------------------------------------------------------- run
def stage_run(run):
    run = Path(run)
    rj = json.load(open(run / "run.json"))
    need = ["net.pt", "uh_head.pt", "sim.npy", "sim_preuh.npy", "uh_gauge.npy", "chunk_plan.json", "run.json"]
    have = [f for f in need if (run / f).exists()]
    sim, pre, ab = np.load(run / "sim.npy"), np.load(run / "sim_preuh.npy"), np.load(run / "uh_gauge.npy")
    pj = json.load(open(run / "chunk_plan.json"))
    G = rj["n_gauges"]
    z = zarr.open_group(mdd.STORE, mode="r")
    gmap = mdd.read_map(z["gage_staid"][:])
    plan = mdd.chunk_plan(gmap, pj["cap"], G)
    fin = bool(np.isfinite(sim).all() and np.isfinite(pre).all() and np.isfinite(ab).all())
    rng = bool((ab[:, 0] >= 0).all() and (ab[:, 0] <= 2.9).all() and (ab[:, 1] >= 0).all() and (ab[:, 1] <= 6.5).all())
    check(f"outputs {run.name}", have == need and fin and rng and sim.shape == pre.shape == (10957, G) and sim.dtype == np.float32
          and ab.shape == (G, 2) and plan == pj["chunks"] and rj["chunk_plan"]["n_chunks"] == len(plan),
          f"files {len(have)} of {len(need)}; sim.npy {sim.shape} {sim.dtype}, sim_preuh.npy {pre.shape}, uh_gauge.npy {ab.shape}; all finite: {fin}; "
          f"sim min {sim.min():.3g}; gauge a in [{ab[:, 0].min():.3f}, {ab[:, 0].max():.3f}], b in [{ab[:, 1].min():.3f}, {ab[:, 1].max():.3f}] "
          f"(ranges [0, 2.9], [0, 6.5]): {rng}; saved chunk plan equal to a new plan (cap {pj['cap']}, {len(plan)} chunks, columns per pass "
          f"{rj['chunk_plan']['columns_per_pass']}): {plan == pj['chunks']}")
    m_pre, m_uh = pre.astype(np.float64).mean(0), sim.astype(np.float64).mean(0)
    ratio = m_uh / m_pre
    lost = pre[-14:].astype(np.float64).sum(0) / pre.astype(np.float64).sum(0)                  # most that the cut at the end of the record can remove
    bal = rj["balance"]
    check(f"gauge volume {run.name}", bool((np.abs(ratio - 1) <= lost + 1e-5).all()) and bal["divide_stores_max_abs_resid_over_P"] < 1e-5,
          f"{G} gauges, whole record: mean UH output / mean summed runoff min {ratio.min():.6f}, median {np.median(ratio):.6f}, max {ratio.max():.6f}; "
          f"the UH is cut at the end of the record, so the runoff of the last 14 days can leave: each |ratio - 1| is within that share + 1e-5 "
          f"(largest share {lost.max():.1e}); divide stores in the final forward pass: max |residual| / P {bal['divide_stores_max_abs_resid_over_P']:.1e} "
          f"(limit 1e-5), columns with P = 0: {bal['columns_with_zero_P']} (max |residual| {bal['zero_P_columns_max_abs_resid']:.1e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "gpu", "run"])
    ap.add_argument("--run", default="")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "data":
        stage_data(a.gpu)
    elif a.stage == "gpu":
        stage_gpu(a.gpu)
    else:
        stage_run(a.run)
    print(f"## check_dist {a.stage}: {'all checks passed' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

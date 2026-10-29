"""Gradient check of the GR4J production-store kernel (src/cart/kernels/grstores.py, GProdK / gprod_k) against float64 autograd.

The Triton kernel (float32, hand-written adjoint) is compared with torch autograd through a float64 loop of the same daily equations
as the reference:

    pn = max(p - e, 0), en = max(e - p, 0), s = S/C, ps = C (1 - s^2) tanh(pn/C) / (1 + s tanh(pn/C)),
    es = S (2 - s) tanh(en/C) / (1 + (1 - s) tanh(en/C)), S2 = S - es + ps, G = S2 [1 - (1 + kq (S2/C)^4)^(-1/4)], S+ = S2 - G,
    outputs D = pn - ps, G, AET = es + min(p, e).

The scalar loss is sum(wD D) + sum(wG G) + sum(wA AET) with random weights; the gradients with respect to p, e, C and kq are compared.
Cases: all three outputs weighted, and AET alone, each for several initial fillings S0/C.
Pass: relative max-abs error (max |kernel - reference| / max |reference|) < 1e-4 for every output and gradient.

    python tests/check_kernel_grads.py [--gpu 0] [--json OUT]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))
from cart.kernels.grstores import gprod_k  # noqa: E402

TOL = 1e-4


def reference(p, e, Cc, kq, S0):
    S, D, G, A = S0, [], [], []
    for t in range(p.shape[1]):
        pn, en = (p[:, t] - e[:, t]).clamp(min=0), (e[:, t] - p[:, t]).clamp(min=0)
        s = S / Cc
        tp, te = torch.tanh(pn / Cc), torch.tanh(en / Cc)
        ps = Cc * (1 - s ** 2) * tp / (1 + s * tp)
        es = S * (2 - s) * te / (1 + (1 - s) * te)
        S2 = S - es + ps
        g = S2 * (1 - (1 + kq * (S2 / Cc) ** 4) ** -0.25)
        D.append(pn - ps); G.append(g); A.append(es + torch.minimum(p[:, t], e[:, t])); S = S2 - g
    return torch.stack(D, -1), torch.stack(G, -1), torch.stack(A, -1)


def rel(u, v):
    u, v = u.double(), v.double()
    return ((u - v).abs().max() / v.abs().max().clamp(min=1e-12)).item()


def case(dev, seed, fill, weights, n=26, T=400):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64).to(dev)
    p = r(n, T) ** 4 * 40                                                  # intermittent rain, heavy tail (mm/d)
    e = r(n, T) * 4
    Cc = torch.tensor([25.0, 400.0] * (n // 2), dtype=torch.float64, device=dev) * (0.5 + r(n))
    kq = r(n) * 0.3 + 0.001
    S0 = Cc * fill
    w = [torch.randn(n, T, generator=g, dtype=torch.float64).to(dev) * k for k in weights]
    a = [v.clone().requires_grad_(True) for v in (p, e, Cc, kq)]
    ref = reference(*a, S0)
    sum((o * wi).sum() for o, wi in zip(ref, w)).backward()
    b = [v.float().clone().requires_grad_(True) for v in (p, e, Cc, kq)]
    out = gprod_k(b[0], b[1], b[2], b[3], S0.float())[:3]
    sum((o * wi.float()).sum() for o, wi in zip(out, w)).backward()
    res = {f"fwd_{k}": rel(o.detach(), r_.detach()) for k, o, r_ in zip("DGA", out, ref)}
    res.update({f"grad_{k}": rel(bb.grad, aa.grad) for k, aa, bb in zip(("p", "e", "C", "kq"), a, b)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(dev)
    rows, worst = [], 0.0
    for wname, weights in (("D+G+AET", (1.0, 1.0, 1.0)), ("AET only", (0.0, 0.0, 1.0))):
        for seed, fill in ((7, 0.6), (11, 0.1), (13, 0.95)):
            res = case(dev, seed, fill, weights)
            m = max(res.values()); worst = max(worst, m)
            rows.append({"loss": wname, "seed": seed, "S0_over_C": fill, "rel_err": res, "max_rel_err": m})
            print(f"  GProdK vs float64 autograd, loss {wname:8s}, S0/C {fill:4.2f}: " + ", ".join(f"{k} {v:.1e}" for k, v in res.items())
                  + f"  -> max {m:.1e} {'ok' if m < TOL else 'FAIL'}")
    ok = worst < TOL
    print(f"KERNEL GRADIENT CHECK (grstores.GProdK incl. AET): max relative error {worst:.1e} (tolerance {TOL:.0e}) -> {'PASSED' if ok else 'FAILED'}")
    if a.json:
        open(a.json, "a").write(json.dumps({"what": "kernel_grad", "kernel": "grstores.GProdK (gprod_k)", "reference": "float64 autograd loop",
                                            "tolerance": TOL, "max_rel_err": worst, "passed": ok, "cases": rows,
                                            "gpu": torch.cuda.get_device_name(dev)}) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

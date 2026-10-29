"""Generic stored-state adjoint for K-state daily recursions (shared by several of the fused store kernels).

A model supplies
  * a Triton forward kernel that runs x_t, out_t = step(x_{t-1}, u_t, theta) day by day and stores the states X [N, T, K] and the daily
    outputs (each [N, T]);
  * the same one-day step as an elementwise torch function
        step(xprev: list of K tensors [N, T], *inputs [N, T], *params [N, T]) -> (list of K tensors x_t, list of outputs out_t)
    (it must reproduce the kernel exactly; every operation elementwise over (N, T)).
backward_k() then returns the gradients of the inputs and parameters from the stored states:
  J_t = d x_t / d x_{t-1}  [K x K] and c_t = sum_m g_out_m,t d out_m,t / d x_{t-1}  by K + 1 vectorised autograd passes over all days,
  lambda_t = g_x,t + c_{t+1} + J_{t+1}^T lambda_{t+1}   (Triton kernel _adjk, backwards in time),
  parameter / input gradients = one VJP of sum_t [lambda_t . x_t + sum_m g_out_m,t out_m,t] with x_{t-1} held at the stored values.
The initial state is treated as a constant (it comes from a no-gradient spin-up).
"""

import torch
import triton
import triton.language as tl

BLOCK = 32


@triton.jit
def _adjk(J_ptr, g_ptr, lam_ptr, N, T, KP: tl.constexpr, BLOCK: tl.constexpr):
    """lam_t = g_t + Jn_t^T lam_{t+1} backwards, lam_T = 0. Jn [N, T, KP, KP] (Jn_t = J_{t+1}, row = new state, col = old state),
    g, lam [N, T, KP]."""
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    i = tl.arange(0, KP)
    mn = n < N
    lam = tl.zeros((BLOCK, KP), dtype=tl.float32)
    for tt in range(T):
        t = T - 1 - tt
        base = (n * T + t)
        g = tl.load(g_ptr + base[:, None] * KP + i[None, :], mask=mn[:, None], other=0.0)
        J = tl.load(J_ptr + base[:, None, None] * KP * KP + i[None, :, None] * KP + i[None, None, :], mask=mn[:, None, None], other=0.0)
        lam = g + tl.sum(J * lam[:, :, None], axis=1)                       # (J^T lam)_j = sum_i J_ij lam_i
        tl.store(lam_ptr + base[:, None] * KP + i[None, :], lam, mask=mn[:, None])


def _kp(K):
    kp = 1
    while kp < K:
        kp *= 2
    return max(kp, 2)


def backward_k(step, X, x0, gX, gouts, inputs, params):
    """X [N, T, K] stored states, x0 [N, K]; gX [N, T, K] or None; gouts list (None allowed) matching step's outputs;
    inputs: list of [N, T] tensors; params: list of [N] tensors. Returns (grads of inputs, grads of params) (None where unused)."""
    N, T, K = X.shape
    KP = _kp(K)
    Xp = torch.cat([x0[:, None, :], X[:, :-1, :]], 1).detach()
    xp = [Xp[..., k].contiguous().requires_grad_(True) for k in range(K)]
    ins = [v.detach().requires_grad_(True) for v in inputs]
    prm = [v.detach()[:, None].expand(N, T).contiguous().requires_grad_(True) for v in params]
    with torch.enable_grad():
        x, outs = step(xp, *ins, *prm)
        J = torch.zeros(N, T, KP, KP, device=X.device, dtype=torch.float32)
        for k in range(K):
            gk = torch.autograd.grad(x[k].sum(), xp, retain_graph=True, allow_unused=True)
            for j in range(K):
                if gk[j] is not None:
                    J[:, :, k, j] = gk[j]
        go = sum((g * o).sum() for g, o in zip(gouts, outs) if g is not None)
        c = torch.zeros(N, T, KP, device=X.device, dtype=torch.float32)
        if torch.is_tensor(go):
            gc = torch.autograd.grad(go, xp, retain_graph=True, allow_unused=True)
            for j in range(K):
                if gc[j] is not None:
                    c[:, :, j] = gc[j]
        gy = torch.zeros(N, T, KP, device=X.device, dtype=torch.float32)
        if gX is not None:
            gy[..., :K] = gX
        gy[:, :-1] += c[:, 1:]
        Jn = torch.zeros_like(J)
        Jn[:, :-1] = J[:, 1:]
        lam = torch.empty_like(gy)
        _adjk[(triton.cdiv(N, BLOCK),)](Jn.contiguous(), gy.contiguous(), lam, N, T, KP=KP, BLOCK=BLOCK)
        total = sum((lam[..., k].to(x[k].dtype) * x[k]).sum() for k in range(K)) + (go if torch.is_tensor(go) else 0.0)
        gr = torch.autograd.grad(total, ins + prm, allow_unused=True)
    gi = list(gr[:len(ins)])
    gp = [g.sum(1) if g is not None else None for g in gr[len(ins):]]
    return gi, gp

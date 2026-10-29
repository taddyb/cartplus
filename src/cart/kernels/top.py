"""Thin surface layer in series above the CaRT production store : fused Triton recursion with stored-state
adjoint (adjoint.backward_k).

State U (mm), capacity Cu, drainage time constant tau_u (per catchment). DAY (P liquid input, E demand):
    U1 = U + P;  eu = min(E, U1)            evaporation at the potential rate from the wet surface (interception, litter, top soil)
    U2 = U1 - eu;  ou = (U2 - Cu)+          overflow of the full layer -> direct runoff (saturation / infiltration excess of the layer)
    U3 = U2 - ou;  du = ku U3,  ku = 1 - exp(-1/tau_u)    drainage into the production store
    U+ = U3 - du
Outputs: ou (direct runoff), du (input of the production store), eu (evaporation), E - eu (remaining demand of the production store).
Limit tau_u -> 0 (ku -> 1), Cu -> inf: du = (P - E)+, remaining demand = (E - P)+, i.e. the GR4J same-day interception.
"""

import numpy as np
import torch
import triton
import triton.language as tl

from .adjoint import backward_k

BLOCK = 32


@triton.jit
def _top_fwd(p_ptr, e_ptr, cu_ptr, ku_ptr, x0_ptr, X_ptr, o_ptr, d_ptr, ev_ptr, N, T, BLOCK: tl.constexpr):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    cu = tl.load(cu_ptr + n, mask=m, other=1.0)
    ku = tl.load(ku_ptr + n, mask=m, other=1.0)
    U = tl.load(x0_ptr + n, mask=m, other=0.0)
    for t in range(T):
        p = tl.load(p_ptr + n * T + t, mask=m, other=0.0)
        e = tl.load(e_ptr + n * T + t, mask=m, other=0.0)
        U1 = U + p
        eu = tl.minimum(e, U1)
        U2 = U1 - eu
        ou = tl.maximum(U2 - cu, 0.0)
        U3 = U2 - ou
        du = ku * U3
        U = U3 - du
        tl.store(X_ptr + n * T + t, U, mask=m)
        tl.store(o_ptr + n * T + t, ou, mask=m)
        tl.store(d_ptr + n * T + t, du, mask=m)
        tl.store(ev_ptr + n * T + t, eu, mask=m)


def top_step(xp, p, e, cu, ku):
    (U,) = xp
    U1 = U + p
    eu = torch.minimum(e, U1)
    U2 = U1 - eu
    ou = torch.relu(U2 - cu)
    U3 = U2 - ou
    du = ku * U3
    return [U3 - du], [ou, du, eu]


class Top(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, e, cu, ku, x0):
        N, T = p.shape
        v = [t_.detach().float().contiguous() for t_ in (p, e, cu, ku, x0)]
        X, O, D, Ev = (torch.empty(N, T, device=p.device) for _ in range(4))
        _top_fwd[(triton.cdiv(N, BLOCK),)](*v, X, O, D, Ev, N, T, BLOCK=BLOCK)
        ctx.save_for_backward(v[0], v[1], v[2], v[3], v[4], X)
        return O, D, Ev, X

    @staticmethod
    def backward(ctx, gO, gD, gE, gX):
        p, e, cu, ku, x0, X = ctx.saved_tensors
        gi, gp = backward_k(top_step, X[..., None], x0[:, None], gX[..., None] if gX is not None else None, [gO, gD, gE], [p, e], [cu, ku])
        return gi[0], gi[1], gp[0], gp[1], None


def top_layer(p, e, cu, ku, U0):
    """p, e [N, T]; cu, ku, U0 [N] -> overflow, drainage, evaporation, U [N, T]"""
    return Top.apply(p, e, cu, ku, U0)

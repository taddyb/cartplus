"""Linear time-varying recursion y_t = a_t y_{t-1} + b_t as a Triton kernel with its adjoint (used by the HBV lower zone).
lin_rec(a, b, y0) -> y [N, T]; gradients by the adjoint mu_t = gy_t + a_{t+1} mu_{t+1}: dL/db_t = mu_t, dL/da_t = mu_t y_{t-1}.
"""

import torch
import triton
import triton.language as tl

BLOCK = 32


@triton.jit
def _adjoint(a_ptr, c_ptr, lamT_ptr, lam_ptr, N, T, BLOCK: tl.constexpr):
    """lambda_t = c_t + a_t lambda_{t+1}, lambda_T given -> lambda_0..lambda_{T-1}"""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    lam = tl.load(lamT_ptr + offs, mask=m, other=0.0)
    for tt in range(T):
        t = T - 1 - tt
        a = tl.load(a_ptr + offs * T + t, mask=m, other=0.0)
        c = tl.load(c_ptr + offs * T + t, mask=m, other=0.0)
        lam = c + a * lam
        tl.store(lam_ptr + offs * T + t, lam, mask=m)


@triton.jit
def _fwd(a_ptr, b_ptr, y0_ptr, y_ptr, N, T, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    y = tl.load(y0_ptr + offs, mask=m, other=0.0)
    for t in range(T):
        a = tl.load(a_ptr + offs * T + t, mask=m, other=0.0)
        b = tl.load(b_ptr + offs * T + t, mask=m, other=0.0)
        y = a * y + b
        tl.store(y_ptr + offs * T + t, y, mask=m)


class LinRec(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, y0):
        N, T = a.shape
        ac, bc, y0c = a.detach().float().contiguous(), b.detach().float().contiguous(), y0.detach().float().contiguous()
        y = torch.empty_like(ac)
        _fwd[(triton.cdiv(N, BLOCK),)](ac, bc, y0c, y, N, T, BLOCK=BLOCK)
        ctx.save_for_backward(ac, y0c, y)
        return y

    @staticmethod
    def backward(ctx, gy):
        a, y0, y = ctx.saved_tensors
        N, T = a.shape
        a_next = torch.cat([a[:, 1:], torch.zeros_like(a[:, :1])], 1).contiguous()
        mu = torch.empty_like(a)
        _adjoint[(triton.cdiv(N, BLOCK),)](a_next, gy.contiguous().float(), torch.zeros_like(y0), mu, N, T, BLOCK=BLOCK)
        yprev = torch.cat([y0[:, None], y[:, :-1]], 1)
        return mu * yprev, mu, a[:, 0] * mu[:, 0]


def lin_rec(a, b, y0):
    return LinRec.apply(a, b, y0)

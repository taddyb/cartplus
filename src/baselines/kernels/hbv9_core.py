"""HBV-9: the HBV-light model without the snow routine (9 parameters), as a benchmark for CaRT.

Equations (Seibert & Vis 2012, HESS 16, 3315, Eqs. 3-6; the order of operations within a day as in the open-source jHBV
implementation of SYMFLUENCE):
  soil:      F = I (min(SM/FC, 1))^BETA                    recharge (I = rain, no snow routine)
             SM = SM + I - F
             E_act = E_pot min(SM / (LP FC), 1), limited by SM;  SM = SM - E_act
  response:  SUZ = SUZ + F;  perc = min(PERC, SUZ);  SUZ = SUZ - perc;  SLZ = SLZ + perc
             Q0 = K0 max(SUZ - UZL, 0),  Q1 = K1 SUZ,  Q2 = K2 SLZ;  SUZ = SUZ - Q0 - Q1;  SLZ = SLZ - Q2
             Q_GW = Q0 + Q1 + Q2
  routing:   Q_sim(t) = sum_{i=1}^{ceil(MAXBAS)} c(i) Q_GW(t - i + 1),
             c(i) = integral_{i-1}^{min(i, MAXBAS)} [2/MAXBAS - |u - MAXBAS/2| 4/MAXBAS^2] du       (Eq. 6, "paper")
             (the "jhbv" option uses jHBV's point weights w(t), t = 1..L, normalised)
Parameters and ranges (RANGES): "hbv9" (LO, HI): FC 50-700 mm, LP 0.3-1, BETA 1-6, PERC 0-10 mm/d, UZL 0-100 mm,
K0 0.05-0.5, K1 0.01-0.3, K2 0.0001-0.1 /d, MAXBAS 1-7 d; "feng2022" (the regional HBV of the paper, baselines.hbv_regional
--ranges feng2022): FC 50-1000 mm, LP 0.2-1 and K2 0.001-0.2 /d as Feng et al. (2022), the others as "hbv9" (K0 and K1 keep their ranges, since the
simultaneous upper-zone update needs K0 + K1 < 1). Since K0 + K1 < 1 and E_act <= SM, no storage is ever clipped: the daily water balance
P = E_act + Q_GW + dSM + dSUZ + dSLZ is exact, and the routing buffer conserves Q_GW (sum c = 1).
Inputs: catchment-mean rain and PET (as CaRT); 2-year spin-up repeated once (states restart).
"""

import numpy as np
import torch

NAMES = ["FC", "LP", "BETA", "PERC", "UZL", "K0", "K1", "K2", "MAXBAS"]
LO = np.array([50.0, 0.3, 1.0, 0.0, 0.0, 0.05, 0.01, 0.0001, 1.0])
HI = np.array([700.0, 1.0, 6.0, 10.0, 100.0, 0.5, 0.3, 0.1, 7.0])
RANGES = {"hbv9": (LO, HI),                                                                    # --ranges of baselines.hbv_regional
          "feng2022": (np.array([50.0, 0.2, 1.0, 0.0, 0.0, 0.05, 0.01, 0.001, 1.0]), np.array([1000.0, 1.0, 6.0, 10.0, 100.0, 0.5, 0.3, 0.2, 7.0]))}
L_BUF = 8


def tri_cdf(x, M, lib=np):
    """CDF of the MAXBAS triangle on [0, M]."""
    x = lib.clip(x, 0, M) if lib is np else torch.minimum(torch.clamp(x, min=0), M)
    lo = 2 * x ** 2 / M ** 2
    hi = 1 - 2 * (M - x) ** 2 / M ** 2
    return lib.where(x <= M / 2, lo, hi)


def weights(M, kind="paper", lib=np):
    """M [...] -> c [..., L_BUF]; c[..., 0] applies to the current day."""
    if lib is np:
        M = np.asarray(M, np.float64)[..., None]
        i = np.arange(1, L_BUF + 1, dtype=np.float64)
    else:
        M = M[..., None]
        i = torch.arange(1, L_BUF + 1, dtype=M.dtype, device=M.device)
    if kind == "paper":
        return tri_cdf(i, M, lib) - tri_cdf(i - 1, M, lib)
    rising = lib.where(i <= M / 2, i / (M / 2), 0 * i)                             # jHBV point weights
    falling = lib.where((i > M / 2) & (i <= M), (M - i) / (M / 2), 0 * i)
    w = rising + falling
    return w / (w + 1e-10).sum(-1, keepdims=True) if lib is np else w / (w + 1e-10).sum(-1, keepdim=True)


def run_numpy(P, E, prm, kind="paper", state=None, keep=False):
    """P, E [T, N]; prm dict of [N] arrays. Returns Q [T, N], final state, (fluxes if keep)."""
    T, N = P.shape
    FC, LP, BETA, PERC, UZL, K0, K1, K2, MB = (np.asarray(prm[k], np.float64) for k in NAMES)
    c = weights(MB, kind)
    sm, suz, slz, buf = (state if state is not None else (0.5 * FC, np.full(N, 10.0), np.full(N, 10.0), np.zeros((N, L_BUF))))
    sm, suz, slz, buf = sm.copy(), suz.copy(), slz.copy(), buf.copy()
    Q = np.zeros((T, N)); AET = np.zeros(N); QGW = np.zeros(N)
    for t in range(T):
        I = P[t]
        F = I * np.minimum(sm / FC, 1.0) ** BETA
        sm = sm + I - F
        ea = np.minimum(E[t] * np.minimum(sm / (LP * FC), 1.0), sm)
        sm = sm - ea
        suz = suz + F
        perc = np.minimum(PERC, suz)
        suz = suz - perc; slz = slz + perc
        q0 = K0 * np.maximum(suz - UZL, 0.0); q1 = K1 * suz; q2 = K2 * slz
        suz = suz - q0 - q1; slz = slz - q2
        qgw = q0 + q1 + q2
        buf = buf + qgw[:, None] * c
        Q[t] = buf[:, 0]
        buf = np.concatenate([buf[:, 1:], np.zeros((N, 1))], 1)
        AET += ea; QGW += qgw
    return Q, (sm, suz, slz, buf), (AET, QGW)

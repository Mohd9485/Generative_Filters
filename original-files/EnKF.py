"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import torch
import time

from timing_utils import sync_clock


def EnKF(Y, X0, A, h, t, Noise, SIGMA=1e-6, device=None, timing_out=None):
    """
    Ensemble Kalman Filter (EnKF) function according to Algorithm 1 in
    [Al-Jarrah, M., Jin, N., Hosseini, B. and Taghvaei, A., 2024, July.
     Nonlinear Filtering with Brenier Optimal Transport Maps.
     In International Conference on Machine Learning (pp. 813-839). PMLR.]

    where:

    Y     : True observations with shape (NUM_SIM x T x dy),
    X0    : Initial particles with shape (NUM_SIM x L x N),
    A     : Deterministic dynamic model (without noise),
    h     : Deterministic observation model (without noise),
    t     : Time vector (e.g., t = 0.0, dt, 2*dt, ..., tf),
    Noise : A list [sigma, gamma] defining the noise levels in the dynamics (sigma) and observations (gamma),
    SIGMA : A small positive constant (nugget) added to maintain the invertibility of C_t^{yy}.
    device: Target compute device. When None, CUDA is used if available and the CPU
            otherwise; a string (e.g. 'cuda:0') or torch.device is accepted directly.
            The ensemble, its propagation, the empirical covariances and the Kalman
            gain are all formed as torch tensors on this device, so the analysis step
            executes on the GPU when one is available.
    timing_out : Optional dict. When supplied it is filled with 'step_times' —
            a (NUM_SIM x T-1) array of per-analysis-step wall-clock seconds —
            and 'total', the whole-run time. EnKF trains nothing, so its first
            column carries no spin-up cost; it is reported in the same layout
            as the learned filters purely so the comparison table has a
            baseline row that is read the same way.
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    NUM_SIM = X0.shape[0]
    L = X0.shape[1]
    N = X0.shape[2]

    T = Y.shape[1]
    dy = Y.shape[2]

    sigma = Noise[0]
    gamma = Noise[1]

    tau = t[1] - t[0]

    # Only filled when a caller asks for it, so the tuning sweeps pay no
    # synchronisation cost.
    _timing    = timing_out is not None
    step_times = np.full((NUM_SIM, T - 1), np.nan)

    start_time = time.time()
    X0_t = torch.as_tensor(X0, dtype=torch.float32, device=device)
    Y_t  = torch.as_tensor(Y,  dtype=torch.float32, device=device)
    # Nugget regularisation of C_t^{yy}, so the inverse below is well posed.
    I_dy = torch.eye(dy, dtype=torch.float32, device=device)

    X_EnKF = torch.zeros((NUM_SIM, T, N, L), dtype=torch.float32, device=device)

    for k in range(NUM_SIM):
        y = Y_t[k]

        X_EnKF[k, 0] = X0_t[k].T

        for i in range(T - 1):
            _t_step = sync_clock(device) if _timing else 0.0
            x_noise = sigma * torch.randn(N, L, dtype=torch.float32, device=device)
            x_hatEnKF = A(X_EnKF[k, i,].T, t[i]).T + x_noise            # (N x L)

            y_noise = gamma * torch.randn(N, dy, dtype=torch.float32, device=device)
            y_hatEnKF = h(x_hatEnKF.T).T + y_noise                      # (N x dy)

            X_hat = x_hatEnKF.mean(dim=0, keepdim=True)
            Y_hat = y_hatEnKF.mean(dim=0, keepdim=True)

            a = x_hatEnKF - X_hat
            b = y_hatEnKF - Y_hat

            # Empirical C_t^{xy} and C_t^{yy} of Algorithm 1, then its gain.
            C_xy = 1 / N * a.T @ b
            C_yy = 1 / N * b.T @ b

            K = C_xy @ torch.linalg.inv(C_yy + I_dy * SIGMA)

            X_EnKF[k, i + 1, :, :] = x_hatEnKF + (K @ (y[i + 1, :] - y_hatEnKF.T)).T
            if _timing: step_times[k, i] = sync_clock(device) - _t_step

    print("--- EnKF time : %s seconds ---" % (time.time() - start_time))

    if _timing:
        timing_out['step_times'] = step_times
        timing_out['total']      = time.time() - start_time

    # (NUM_SIM x T x L x N) on the host, the layout the drivers plot from.
    return X_EnKF.permute(0, 1, 3, 2).cpu().numpy()

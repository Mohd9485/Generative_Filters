"""
@author: Mohammad Al-Jarrah

Knothe–Rosenblatt Filter (KRF) — a nonlinear particle filter based
on conditional Knothe–Rosenblatt rearrangement, following the
Stochastic Map Filter (Algorithm 3.2) of

    Spantini, A., Baptista, R., and Marzouk, Y.,
    "Coupling techniques for nonlinear ensemble filtering",
    SIAM Review, 64(4):921-953, 2022.

KRF differs from the other filters in this collection in how it
realises the conditioning step.  Whereas OTF / FMF / SIF / SBF
rely on transport maps trained either by adversarial OT, flow matching,
stochastic interpolants, or Schrödinger bridges, KRF
learns a triangular MONOTONE map  S(y, x) : R^{dy + L} → R^L  whose
output  z = S(y, x)  is matched to a standard normal:

        z_k = S_k( [y, x_1, ..., x_{k-1}], x_k ),     k = 1, ..., L

with each component  S_k(·, x_k)  monotone non-decreasing in its last
argument by construction.  Once the map is fit, posterior samples for
the true observation  y*  are obtained by inverting

        z_i  =  S(y_i,  x_i^forecast),
        x_i^analysis  =  S(y*, ·)^{-1}( z_i ),

where the inverse is computed coordinate-wise by bisection because the
monotonicity ensures invertibility in 1-D along each coordinate.  This
realises the same block-triangular conditioning idea as the rest of the
filters: y is purely a conditioning input and is never moved.

Construction of  S_k :
        dS_k / dx_k  =  softplus( g_k(ctx, x_k) )  +  eps  >  0

with two MLPs
        c_k(ctx)        : shift network,
        g_k(ctx, x_k)   : pre-softplus slope-field network,

so that
        S_k(ctx, x_k)  =  c_k(ctx)  +  ∫_0^{x_k} dS_k/dt(ctx, t) dt

is monotone in  x_k  by construction.  The integral is evaluated by
Gauss–Legendre-style trapezoidal quadrature on  K = quad_points  nodes.

Training objective is maximum likelihood under z ~ N(0, I_L), which up
to constants reads

        L  =  E[  0.5 ‖z‖²  -  log |det(dS/dx)|  ],
        log |det(dS/dx)|  =  Σ_k  log( dS_k / dx_k ).

The network is retrained at every filter step on the pair
(forecast particles, synthetic observations) since the conditional
distribution shifts with the dynamics.  This matches the per-filter-
step training pattern of the OT / FM family.

Inputs and outputs match the rest of the filter functions exactly.
"""

import numpy as np
import os
import json
import hashlib
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from timing_utils import sync_clock


class _NonFiniteRun(Exception):
    """
    Raised internally when a run has provably failed and finishing it would
    only burn compute: the training loss went NaN/inf.  Caught in the main
    loop, which fills the remainder of the output with NaN and returns early.
    Not part of the public interface — callers see the NaN particles, not
    this exception.  Mirrors the same class in SIF.py and FMF.py.
    """


def KRF(Y, X0, A, h, t, Noise, parameters, device=None, timing_out=None,
        warm_start='off', warm_start_tag='', warm_start_dir='DATA/warm_start'):
    """
    Knothe–Rosenblatt Filter.

    Inputs — identical to the other filter functions
    ------------------------------------------------
    Y        : True observations,            shape (NUM_SIM × T × dy × 1)
    X0       : Initial particles,            shape (NUM_SIM × L × N)
    A        : Deterministic dynamic model
    h        : Deterministic observation model
    t        : Time vector
    Noise    : [sigma_proc, sigma_obs]
    parameters : dict with keys
        'normalization'    : 'None' | 'Standard' | 'MinMax'
        'INPUT_DIM'        : [L, dy]   (kept for interface symmetry; the
                                        triangular map reads L and dy
                                        directly from X0 and Y, so this
                                        is unused at runtime)
        'NUM_NEURON'       : hidden width of the c_k and g_k MLPs
        'BATCH_SIZE'       : mini-batch over the M = N forecast particles
        'LearningRate'     : Adam learning rate
        'ITERATION'        : initial training iterations per filter step
        'Final_Number_ITERATION' : floor for the iteration-halving schedule;
                              ITERATION is halved each step until this value
        'depth'            : number of hidden layers in c_k and g_k
        'quad_points'      : number of trapezoidal quadrature points used
                              to evaluate the monotone integral
        'BISECT_LO'        : (optional, default -10.0) lower bracket for
                              the inversion bisection on each coordinate
        'BISECT_HI'        : (optional, default +10.0) upper bracket
        'BISECT_ITERS'     : (optional, default 60) bisection iterations
        'EPS_SLOPE'        : (optional, default 1e-3) floor on the slope
                              dS_k/dx_k to keep the map strictly increasing

    timing_out : Optional dict. When supplied it is filled with
        'step_times' — a (NUM_SIM x T-1) array of per-analysis-step wall-clock
        seconds — and 'total', the whole-run time. Column 0 is the spin-up
        step, which trains from a random initialisation and can be done
        offline; the remaining columns are the online per-step cost. Left as
        None (the default) the timing is skipped entirely, so the tuning
        sweeps pay nothing for it.

    Output
    ------
    X_KRF : ndarray, shape (NUM_SIM × T × L × N)
    """

    # ------------------------------------------------------------------
    # Unpack dimensions
    # ------------------------------------------------------------------
    NUM_SIM = X0.shape[0]
    L       = X0.shape[1]
    N       = X0.shape[2]

    T  = Y.shape[1]
    dy = Y.shape[2]

    sigma_proc = Noise[0]
    sigma_obs  = Noise[1]

    # ------------------------------------------------------------------
    # Unpack hyperparameters
    # ------------------------------------------------------------------
    normalization = parameters.get('normalization', 'None')
    NUM_NEURON    = parameters['NUM_NEURON']
    BATCH_SIZE    = parameters['BATCH_SIZE']
    LearningRate  = parameters['LearningRate']
    ITERATION              = parameters['ITERATION']
    Final_Number_ITERATION = parameters['Final_Number_ITERATION']

    # ------------------------------------------------------------------
    # Improvements carried from the FMF / SIF / OTF studies. The triangular
    # monotone map, its maximum-likelihood objective and the block-triangular
    # conditioning are untouched; only training mechanics change. Defaults
    # reproduce the original algorithm's behaviour exactly.
    #   COLD_START  The default configuration builds a FRESH
    #       TriangularMonotoneMap inside fit_map, which is called once per
    #       analysis step — i.e. it is cold-start. In FMF this alone cost
    #       ~0.27 aa-SW2.
    #   SCHEDULE    The default configuration uses
    #       CosineAnnealingWarmRestarts(T_0=512,T_mult=2) stepped every
    #       iteration, so the LR is kicked back to full value at
    #       512, 1536, 3584, ...
    #   RESAMPLE_DATA / WARMUP_FRAC / GRAD_CLIP / OPTIMIZER / WEIGHT_DECAY
    #       as in the other studies.
    # Note KRF has no ODE, so the second-order-solver fix has no analogue: the
    # map is inverted directly.
    # ------------------------------------------------------------------
    COLD_START    = bool(parameters.get('COLD_START', True))
    SCHEDULE      = parameters.get('SCHEDULE', 'warm_restarts')
    WARMUP_FRAC   = float(parameters.get('WARMUP_FRAC', 0.0))
    GRAD_CLIP     = float(parameters.get('GRAD_CLIP', 0.0))
    OPTIMIZER     = parameters.get('OPTIMIZER', 'adam')
    WEIGHT_DECAY  = float(parameters.get('WEIGHT_DECAY', 0.0))
    depth         = parameters['depth']
    quad_points   = parameters['quad_points']

    # ------------------------------------------------------------------
    # Warm-start cache for the triangular monotone map. Off by default, so a
    # call that omits these arguments reproduces the default configuration
    # exactly.
    #
    #   warm_start      'off' | 'auto' | 'save' | 'load' | 'require'
    #   warm_start_tag  REQUIRED unless 'off'. Names the PROBLEM, which KRF
    #       cannot recover from A and h -- without it a checkpoint from another
    #       experiment sharing (L, dy) would load silently.
    #   warm_start_dir  default 'DATA/warm_start'.
    #
    # On a hit the spin-up runs on Final_Number_ITERATION rather than ITERATION.
    # Two consequences: all NUM_SIM simulations load the SAME checkpoint, so the
    # across-simulation spread is deflated for a reason that is not statistical;
    # and step_times[:, 0] is no longer the from-scratch spin-up it is
    # documented as below, so it is not comparable across warm and cold methods.
    #
    # Arguments rather than `parameters` keys: the dict is the tuned
    # configuration, reused across runs and by the sweeps, while where weights
    # are cached is a property of the CALL.
    # ------------------------------------------------------------------
    WARM_START     = str(warm_start).lower()
    WARM_START_DIR = str(warm_start_dir)
    WARM_START_TAG = str(warm_start_tag)
    if WARM_START not in ('off', 'auto', 'save', 'load', 'require'):
        raise ValueError(
            "warm_start must be one of 'off', 'auto', 'save', 'load', "
            "'require'; got %r" % warm_start)
    if WARM_START != 'off':
        if not WARM_START_TAG:
            raise ValueError(
                "warm_start=%r requires a non-empty warm_start_tag naming the "
                "problem (e.g. 'quadratic'). Without it, a checkpoint trained "
                "on a different system of the same dimension would be loaded "
                "silently." % warm_start)
        if COLD_START:
            raise ValueError(
                "warm_start=%r contradicts COLD_START=True, under which fit_map "
                "builds a fresh map at every analysis step and so discards the "
                "loaded weights after the spin-up." % warm_start)

    BISECT_LO     = parameters.get('BISECT_LO',     -10.0)
    BISECT_HI     = parameters.get('BISECT_HI',     +10.0)
    BISECT_ITERS  = parameters.get('BISECT_ITERS',  60)
    EPS_SLOPE     = parameters.get('EPS_SLOPE',     1e-3)

    # Honour an explicitly supplied `device`, otherwise default to CUDA when available.
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    # ==================================================================
    # Network definition
    # ==================================================================
    class MonotoneComponent(nn.Module):
        """
        One triangular component  S_k( ctx, x_k ) → z_k  with

            dS_k / dx_k  =  softplus( g_k(ctx, x_k) )  +  eps  >  0,
            S_k(ctx, x_k) =  c_k(ctx) + ∫_0^{x_k} dS_k/dt(ctx, t) dt,

        evaluated by uniform-grid trapezoidal quadrature with `quad_points`
        nodes.  The strict positivity of the slope guarantees monotonic
        invertibility in 1-D along x_k, which is exploited later by
        coordinate-wise bisection.
        """
        def __init__(self, ctx_dim, hidden, depth_, eps, quad_points_):
            super().__init__()
            self.ctx_dim     = ctx_dim
            self.eps         = eps
            self.quad_points = quad_points_

            # Shift network c_k(ctx)
            c_layers = [nn.Linear(ctx_dim, hidden), nn.Tanh()]
            for _ in range(depth_ - 1):
                c_layers += [nn.Linear(hidden, hidden), nn.Tanh()]
            c_layers += [nn.Linear(hidden, 1)]
            self.c_net = nn.Sequential(*c_layers)

            # Slope-field network g_k(ctx, x_k)  (pre-softplus)
            g_layers = [nn.Linear(ctx_dim + 1, hidden), nn.Tanh()]
            for _ in range(depth_ - 1):
                g_layers += [nn.Linear(hidden, hidden), nn.Tanh()]
            g_layers += [nn.Linear(hidden, 1)]
            self.g_net = nn.Sequential(*g_layers)

        def dsdx(self, ctx, xk):
            """Strictly positive slope dS_k / dx_k at  (ctx, x_k)."""
            inp = torch.cat([ctx, xk], dim=1)
            g   = self.g_net(inp)
            return torch.nn.functional.softplus(g) + self.eps

        def forward(self, ctx, xk):
            """
            ctx : (B, ctx_dim)
            xk  : (B, 1)
            returns z_k of shape (B, 1).
            """
            c = self.c_net(ctx)

            # Quadrature nodes  α ∈ [0, 1]  →  t = α · x_k
            K = self.quad_points
            alpha = torch.linspace(0.0, 1.0, K,
                                   device=xk.device).view(1, K, 1)   # (1, K, 1)

            x_exp   = xk.view(-1, 1, 1)                              # (B, 1, 1)
            ctx_exp = ctx.view(-1, 1, ctx.shape[1])                  # (B, 1, ctx_dim)
            t_grid  = alpha * x_exp                                  # (B, K, 1)

            ctx_flat = ctx_exp.expand(-1, K, -1).reshape(-1, ctx.shape[1])
            t_flat   = t_grid.reshape(-1, 1)
            d_flat   = self.dsdx(ctx_flat, t_flat).reshape(-1, K, 1) # (B, K, 1)

            dt       = x_exp / (K - 1)                               # (B, 1, 1)
            integral = (d_flat * dt).sum(dim=1)                      # (B, 1)
            return c + integral

        def log_diag_jac(self, ctx, xk):
            """log | dS_k / dx_k |  evaluated at  (ctx, x_k)."""
            return torch.log(self.dsdx(ctx, xk))

    class TriangularMonotoneMap(nn.Module):
        """
        Full block-triangular monotone map  S(y, x) : R^{dy + L} → R^L

            z_k = S_k( [y, x_1, ..., x_{k-1}], x_k ),    k = 1, ..., L,

        with each S_k monotone non-decreasing in x_k.  The Jacobian of S
        with respect to x is lower-triangular, so its log-determinant is
        the sum of diagonal log-slopes — the only quantity needed for the
        likelihood objective.  y is purely a conditioning input (no
        movement).
        """
        def __init__(self, n_state, n_obs, hidden, depth_, eps, quad_points_):
            super().__init__()
            self.n = n_state
            self.d = n_obs
            self.comps = nn.ModuleList()
            for k in range(self.n):
                ctx_dim = self.d + k          # y plus x_1, ..., x_{k-1}
                self.comps.append(
                    MonotoneComponent(ctx_dim=ctx_dim, hidden=hidden,
                                      depth_=depth_, eps=eps,
                                      quad_points_=quad_points_)
                )

        def forward(self, y, x):
            """
            y : (B, dy)
            x : (B, L)
            returns z : (B, L)
            """
            z_list = []
            for k in range(self.n):
                ctx = torch.cat([y, x[:, :k]], dim=1)
                xk  = x[:, k:k + 1]
                z_list.append(self.comps[k](ctx, xk))
            return torch.cat(z_list, dim=1)

        def log_abs_det_jacobian(self, y, x):
            """
            log |det(dS / dx)|  =  Σ_k  log( dS_k / dx_k ).  Returns (B, 1).
            """
            terms = []
            for k in range(self.n):
                ctx = torch.cat([y, x[:, :k]], dim=1)
                xk  = x[:, k:k + 1]
                terms.append(self.comps[k].log_diag_jac(ctx, xk))
            return torch.sum(torch.cat(terms, dim=1), dim=1, keepdim=True)

    # ==================================================================
    # Training — maximum likelihood under z ~ N(0, I_L)
    # ==================================================================
    def _state_dict_compatible(sd_ckpt, module):
        """
        True when sd_ckpt can be loaded into module without any surprises.

        Checked BEFORE load_state_dict rather than relying on it to raise,
        because load_state_dict copies tensors as it walks the module and only
        raises at the end: a mismatch part way through would leave some layers
        loaded and some not.
        """
        sd = module.state_dict()
        return (set(sd_ckpt) == set(sd)
                and all(sd_ckpt[k].shape == sd[k].shape for k in sd))

    def _warm_start_path():
        """
        Path of the checkpoint for THIS problem and THIS architecture.

        The readable part of the name carries what fixes the shape of the map's
        state dict -- tag, dimensions, width, depth, quadrature points -- so a
        stale file can be identified by eye. The trailing digest covers what
        leaves those shapes alone but changes what the trained map MEANS: the
        noise levels it was fitted at, the normalisation, and
        the slope floor of the monotone parameterisation. A mismatch in the
        first group would be caught by _state_dict_compatible; a mismatch in the
        second would not, which is exactly why it is hashed into the filename.

        The ensemble size is deliberately NOT in the key. The networks have the
        same shape at any N -- only the training data drawn from the ensemble
        changes -- so one checkpoint serves a whole particle sweep instead of
        forcing a fresh spin-up at every point. N is recorded in meta, so the
        size a checkpoint was fitted at stays recoverable.
        """
        semantics = {
            'sigma_proc':    np.asarray(sigma_proc, dtype=float).ravel().tolist(),
            'sigma_obs':     np.asarray(sigma_obs,  dtype=float).ravel().tolist(),
            'normalization': str(normalization),
            'EPS_SLOPE':     float(EPS_SLOPE),
        }
        digest = hashlib.sha1(
            json.dumps(semantics, sort_keys=True).encode('utf-8')).hexdigest()[:8]
        name   = 'krf_%s_L%d_dy%d_nn%d_d%d_q%d_%s.pt' % (
            WARM_START_TAG, L, dy, int(NUM_NEURON), int(depth),
            int(quad_points), digest)
        return os.path.join(WARM_START_DIR, name)

    def _warm_start_save(path, model):
        """
        Write the map's weights, plus the provenance needed to interpret them.

        Saved through a process-unique temporary file and os.replace so that a
        reader never sees a half-written checkpoint: the sweeps dispatch several
        filters concurrently and several runs may share the directory. Two
        processes missing the cache at once both train and both write; the last
        writer wins, which is harmless because the two files are
        interchangeable. Tensors go to the host so the file is loadable on any
        device.
        """
        meta = {
            'tag':           WARM_START_TAG,
            'L':             int(L),
            'dy':            int(dy),
            'N':             int(N),
            'sigma_proc':    np.asarray(sigma_proc, dtype=float).ravel().tolist(),
            'sigma_obs':     np.asarray(sigma_obs,  dtype=float).ravel().tolist(),
            'normalization': str(normalization),
            'NUM_NEURON':    int(NUM_NEURON),
            'depth':         int(depth),
            'quad_points':   int(quad_points),
            'EPS_SLOPE':     float(EPS_SLOPE),
            'ITERATION':     int(ITERATION),
            'source':        'simulation 0, analysis step 0',
        }
        payload = {
            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'meta':  meta,
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp = '%s.tmp%d' % (path, os.getpid())
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _warm_start_load(path):
        """
        Return a map loaded from path, or None on any failure.

        Unlike the other filters there is no network yet to load INTO: fit_map
        constructs the map lazily, and passing a pre-built one is exactly how
        KRF already warm-starts across analysis steps. So the map is built
        here -- which draws from the RNG. The RNG state is therefore saved and
        restored when the load fails, so a rejected checkpoint leaves the stream
        exactly where a cold run would have left it and the fallback run is the
        cold run.

        Every failure mode (absent file, unreadable file, shape mismatch) is a
        cache miss and is reported, never raised: a missing warm start costs
        iterations, not correctness.
        """
        if not os.path.exists(path):
            return None
        rng_state  = torch.get_rng_state()
        cuda_state = (torch.cuda.get_rng_state_all()
                      if torch.cuda.is_available() else None)
        try:
            ckpt  = torch.load(path, map_location=device, weights_only=True)
            model = TriangularMonotoneMap(n_state=L, n_obs=dy, hidden=NUM_NEURON,
                                          depth_=depth, eps=EPS_SLOPE,
                                          quad_points_=quad_points).to(device)
            if not _state_dict_compatible(ckpt['model'], model):
                raise ValueError('checkpoint does not fit this architecture')
            model.load_state_dict(ckpt['model'])
            return model
        except Exception as exc:
            # Under 'require' a checkpoint that does not fit is exactly the
            # failure the mode exists to surface, so it propagates instead of
            # degrading to a cold start.
            if WARM_START == 'require':
                raise
            torch.set_rng_state(rng_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            print('[KRF] warm start: ignoring %s (%s: %s)'
                  % (path, type(exc).__name__, exc))
            return None

    def fit_map(y_train, x_train, iterations, lr, batch_size, hidden,
                depth_, quad_points_, ts_filter, T_filter, k_sim, K_sim,
                model=None):
        """
        Train  S(y, x)  by minimising the negative log-likelihood under
        z = S(y, x) ~ N(0, I_L), which up to constants is

            L = E[ 0.5 ‖S(y, x)‖²  -  log |det(dS/dx)| ].
        """
        M, _  = y_train.shape
        _, n  = x_train.shape
        _, d  = y_train.shape

        if model is None:
            model = TriangularMonotoneMap(n_state=n, n_obs=d, hidden=hidden,
                                          depth_=depth_, eps=EPS_SLOPE,
                                          quad_points_=quad_points_).to(device)
        if OPTIMIZER == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr,
                                    weight_decay=WEIGHT_DECAY)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
        if SCHEDULE == 'warm_restarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=512, T_mult=2,
                                                    eta_min=lr * 1e-3)
        elif SCHEDULE == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, iterations), eta_min=lr * 1e-3)
        else:
            scheduler = None
        n_warm = int(WARMUP_FRAC * iterations)

        for i in range(iterations):
            idx = torch.randint(0, M, (batch_size,), device=device)
            yb  = y_train[idx]
            xb  = x_train[idx]

            z_b      = model(yb, xb)                                   # (bs, L)
            logdet_b = model.log_abs_det_jacobian(yb, xb)              # (bs, 1)

            loss = (0.5 * (z_b ** 2).sum(dim=1, keepdim=True)
                    - logdet_b).mean()

            if n_warm and i < n_warm:
                for g in optimizer.param_groups:
                    g['lr'] = lr * (i + 1) / n_warm
            optimizer.zero_grad()
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            if scheduler is not None and not (n_warm and i < n_warm):
                scheduler.step()

            if (i + 1) == iterations:
                with torch.no_grad():
                    z_full = model(y_train, x_train)
                    z_mean = z_full.mean().item()
                    z_std  = z_full.std().item()
                print("Simu#%d/%d, Step:%d/%d, KR iter:%d/%d, "
                      "NLL=%+.4f  z mean=%+.3f  z std=%.3f" %
                      (k_sim + 1, K_sim, ts_filter, T_filter - 1,
                       i + 1, iterations, loss.item(), z_mean, z_std))

                # A NaN/inf loss means the run has diverged and cannot recover,
                # so abandon it rather than train the remaining analysis steps.
                # Tested here because loss is already computed and already
                # synced to the host for the log above.
                if not torch.isfinite(loss):
                    raise _NonFiniteRun(
                        "training loss became %s at simulation %d, "
                        "analysis step %d" % (loss.item(), k_sim + 1, ts_filter))

        return model

    # ==================================================================
    # Coordinate-wise bisection for the inverse  S(y*, ·)^(-1)
    # ==================================================================
    @torch.no_grad()
    def invert_at_y_star(model, y_star_row, z_target,
                         x_lo_init, x_hi_init, n_iters):
        """
        Solve  z = S(y*, x)  for x in batch, coordinate by coordinate.
        Each coordinate is solved by bisection because S_k is strictly
        monotone in x_k by construction.

        y_star_row : (1, dy)
        z_target   : (M, L)
        returns x  : (M, L)
        """
        M  = z_target.shape[0]
        n  = model.n
        yv = y_star_row.expand(M, -1).to(device)
        x  = torch.zeros((M, n), dtype=torch.float32, device=device)

        for k in range(n):
            lo = torch.full((M, 1), x_lo_init,
                            dtype=torch.float32, device=device)
            hi = torch.full((M, 1), x_hi_init,
                            dtype=torch.float32, device=device)

            # ctx_k = [y*, x_1,...,x_{k-1}] is fixed for the entire bisection
            # of coordinate k — compute once and call comp[k] directly instead
            # of running the full n-component forward pass each evaluation.
            ctx_k  = torch.cat([yv, x[:, :k]], dim=1)
            comp_k = model.comps[k]
            zk     = z_target[:, k:k + 1]

            def Sk_at(xk):
                return comp_k(ctx_k, xk)

            slo = Sk_at(lo)
            shi = Sk_at(hi)

            # Expand bracket if any sample is not bracketed
            for _ in range(20):
                ok = (slo <= zk) & (zk <= shi)
                if ok.all():
                    break
                lo  = torch.where(ok, lo, lo * 2.0)
                hi  = torch.where(ok, hi, hi * 2.0)
                slo = Sk_at(lo)
                shi = Sk_at(hi)

            # Bisection
            for _ in range(n_iters):
                mid  = 0.5 * (lo + hi)
                smid = Sk_at(mid)
                go_left = smid > zk
                hi = torch.where(go_left, mid, hi)
                lo = torch.where(go_left, lo, mid)

            x[:, k:k + 1] = mid

        return x

    # ==================================================================
    # Main filter loop
    # ==================================================================
    # Per-analysis-step wall-clock, (NUM_SIM x T-1). Column 0 is the spin-up:
    # it trains from a random initialisation for ITERATION iterations, while
    # every later step refines the warm network for Final_Number_ITERATION, so
    # the two costs are not comparable and averaging over them hides both.
    # Preallocated rather than appended so that an aborted run leaves NaN in
    # the steps it never reached instead of a ragged result. Only filled when
    # a caller asks for it, so the tuning sweeps pay no synchronisation cost.
    _timing    = timing_out is not None
    step_times = np.full((NUM_SIM, T - 1), np.nan)

    # Resolved once: the path depends only on the problem and the architecture,
    # both fixed for the whole call.
    #
    # Availability is snapshotted HERE, before any training, and that is the
    # point: the run that BUILDS the cache would otherwise warm-start its own
    # later simulations from the checkpoint its first simulation had just
    # written, and would be neither the cold baseline nor a warm run. With the
    # snapshot a run is cold throughout or warm throughout, so the
    # cache-building run reproduces the uncached behaviour exactly and every
    # run after it is uniform.
    _warm_path         = _warm_start_path() if WARM_START != 'off' else None
    _warm_available    = (WARM_START in ('auto', 'load', 'require')
                          and _warm_path is not None
                          and os.path.exists(_warm_path))
    # Raised at entry, before any compute: a run that MEANT to warm start and
    # silently trained cold is worse than one that stopped, because nothing
    # downstream shows which trials inherited the weights.
    if WARM_START == 'require' and not _warm_available:
        raise FileNotFoundError(
            "warm_start='require' but no checkpoint exists at %s. Either the "
            "architecture/problem settings differ from the ones it was saved "
            "under (they are encoded in that filename), or it was never "
            "written -- run once with warm_start='save' first."
            % _warm_path)

    # 'require' never writes: it consumes a checkpoint, it does not produce one.
    _warm_save_pending = (WARM_START == 'save'
                          or (WARM_START == 'auto' and not _warm_available))

    start_time = time.time()
    X_KRF = np.zeros((NUM_SIM, T, N, L), dtype=np.float32)

    # Abort early via _NonFiniteRun rather than finishing a run that has
    # already diverged: a NaN loss cannot recover, and for a tuning sweep the
    # remaining analysis steps are pure waste. The test lives at the terminal
    # loss log, so it costs nothing — the loss is already computed and synced
    # there — but the failing step pays its full iteration budget first.
    # Whatever was computed before the failure is kept; everything from the
    # failing step onward is NaN. Mirrors SIF.py and FMF.py.
    aborted = None
    try:
        for k_sim in range(NUM_SIM):

            y_obs = Y[k_sim]                                   # (T, dy, 1)
            ITERS = ITERATION
            _warm_model = None      # carried across steps when COLD_START=False

            # The loaded map is handed to fit_map through the channel KRF
            # already uses to carry a map across analysis steps. Every simulation
            # reads the same file, so on a hit none pays the ITERATION spin-up.
            if _warm_available:
                _warm_model = _warm_start_load(_warm_path)
                if _warm_model is not None:
                    # The cache exists precisely so the spin-up need not be
                    # paid, so step 0 is charged the ordinary online budget.
                    ITERS = Final_Number_ITERATION
                    print('[KRF] Simu#%d/%d: warm start from %s '
                          '(spin-up %d -> %d iterations)'
                          % (k_sim + 1, NUM_SIM, _warm_path, ITERATION, ITERS))

            # Initialise particles at t = 0
            X_KRF[k_sim, 0] = X0[k_sim].T                     # (N, L)

            for i in range(T - 1):
                _t_step = sync_clock(device) if _timing else 0.0

                # ----------------------------------------------------------
                # 1. Propagate particles through the dynamics
                # ----------------------------------------------------------
                x_noise = sigma_proc * np.random.multivariate_normal(
                    np.zeros(L), np.eye(L), N)
                x_forecast = A(X_KRF[k_sim, i].T, t[i]).T + x_noise   # (N, L)

                # ----------------------------------------------------------
                # 2. Synthetic (pseudo) observations for each particle
                # ----------------------------------------------------------
                y_noise = sigma_obs * np.random.multivariate_normal(
                    np.zeros(dy), np.eye(dy), N)
                y_pseudo = h(x_forecast.T).T + y_noise                  # (N, dy)

                # ----------------------------------------------------------
                # 3. Optional normalisation, fit per filter step
                # ----------------------------------------------------------
                if normalization == 'Standard':
                    scaler_X = StandardScaler().fit(x_forecast)
                    scaler_Y = StandardScaler().fit(y_pseudo)
                    x_train  = scaler_X.transform(x_forecast)
                    y_train  = scaler_Y.transform(y_pseudo)
                elif normalization == 'MinMax':
                    scaler_X = MinMaxScaler().fit(x_forecast)
                    scaler_Y = MinMaxScaler().fit(y_pseudo)
                    x_train  = scaler_X.transform(x_forecast)
                    y_train  = scaler_Y.transform(y_pseudo)
                else:
                    scaler_X = scaler_Y = None
                    x_train  = x_forecast
                    y_train  = y_pseudo

                x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
                y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)

                # ----------------------------------------------------------
                # 4. Fit the triangular monotone map  S(y, x)
                # ----------------------------------------------------------
                model = fit_map(
                    model       = None if COLD_START else _warm_model,
                    y_train     = y_train_t,
                    x_train     = x_train_t,
                    iterations  = ITERS,
                    lr          = LearningRate,
                    batch_size  = BATCH_SIZE,
                    hidden      = NUM_NEURON,
                    depth_      = depth,
                    quad_points_= quad_points,
                    ts_filter   = i + 1,
                    T_filter    = T,
                    k_sim       = k_sim,
                    K_sim       = NUM_SIM,
                )

                # Saved after fit_map returns, so a diverged run
                # (_NonFiniteRun) never reaches this line and cannot poison the
                # cache.
                if _warm_save_pending and k_sim == 0 and i == 0:
                    _warm_start_save(_warm_path, model)
                    _warm_save_pending = False
                    print('[KRF] warm start: saved %s' % _warm_path)

                _warm_model = model
                ITERS = Final_Number_ITERATION
                # ----------------------------------------------------------
                # 5. Compute z_i = S(y_pseudo_i, x_forecast_i)
                # ----------------------------------------------------------
                with torch.no_grad():
                    z_t = model(y_train_t, x_train_t)                  # (N, L)

                # ----------------------------------------------------------
                # 6. Prepare the true observation y_{i+1} and apply scaler
                # ----------------------------------------------------------
                y_star = y_obs[i + 1, :].reshape(1, dy)                # (1, dy)
                if scaler_Y is not None:
                    y_star = scaler_Y.transform(y_star)
                y_star_t = torch.tensor(y_star, dtype=torch.float32, device=device)

                # ----------------------------------------------------------
                # 7. Invert  z_i = S(y*, x)  for x  by coordinate bisection
                # ----------------------------------------------------------
                x_analysis_t = invert_at_y_star(
                    model       = model,
                    y_star_row  = y_star_t,
                    z_target    = z_t,
                    x_lo_init   = BISECT_LO,
                    x_hi_init   = BISECT_HI,
                    n_iters     = BISECT_ITERS,
                )
                x_analysis = x_analysis_t.cpu().numpy()                # (N, L)

                # ----------------------------------------------------------
                # 8. Reverse normalisation if applied
                # ----------------------------------------------------------
                if scaler_X is not None:
                    x_analysis = scaler_X.inverse_transform(x_analysis)

                X_KRF[k_sim, i + 1] = x_analysis
                if _timing: step_times[k_sim, i] = sync_clock(device) - _t_step

    except _NonFiniteRun as exc:
        aborted = exc
        # Remainder of the failing simulation, then every simulation after it.
        X_KRF[k_sim, i + 1:] = np.nan
        X_KRF[k_sim + 1:]    = np.nan
        print("[KRF] ABORT: %s" % exc)
        print("[KRF] returning NaN from simulation %d, analysis step %d onward "
              "(%d of %d analysis steps completed)"
              % (k_sim + 1, i + 1, k_sim * (T - 1) + i, NUM_SIM * (T - 1)))

    print("--- KRF time : %s seconds ---%s"
          % (time.time() - start_time, " (ABORTED)" if aborted else ""))

    # Hand the per-step breakdown back through the caller's dict. The total is
    # kept alongside it because it also covers the per-simulation network
    # construction, which sits outside the step loop -- step_times.sum() is
    # deliberately smaller than the total, and the gap is worth seeing.
    if _timing:
        timing_out['step_times'] = step_times
        timing_out['total']      = time.time() - start_time

    # Match the (NUM_SIM × T × L × N) layout used by the other filters
    return X_KRF.transpose(0, 1, 3, 2)

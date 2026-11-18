"""
@author: Mohammad Al-Jarrah


INVARIANT (never violated by any option below)
----------------------------------------------
The block-triangular structure of OTF is preserved exactly:

    source  =  P_X (x) P_Y   (independent coupling: x_0 drawn independently of y)
    target  =  P_{X,Y}       (joint coupling: x_1 paired with its own y)
    y never moves along the flow — it enters only as a conditioning input,

so integrating dx/dt = v_θ(x_t, t, y) from t=0 to t=1 at a FIXED true y maps the
prior ensemble onto the posterior P_{X|Y=y}. Every switch in this file changes
how that map is represented, trained or integrated. None changes what is being
transported, and none lets y evolve.

One option deserves an explicit note against that invariant. OT_COUPLING pairs
x_0 with x_1 by minibatch optimal transport instead of by a random shuffle. It
re-pairs samples WITHIN the batch, so the source marginal is still P_X taken
independently of y and the target marginal is still the joint — the endpoints
are untouched and only the paths between them straighten (Tong et al., 2023,
"Improving and generalizing flow-based generative models with minibatch optimal
transport"). It is off by default and reported separately in the ablation.

Compatibility
-------------
With every switch left at its default this file reproduces the plain Flow
Matching Filter exactly:
concatenated time conditioning, Adam, CosineAnnealingWarmRestarts stepped from
the halfway point, uniform t, random-shuffle coupling, a fixed training set, and
Euler integration. The baseline can therefore be measured through the same code
path as the variants, which is what makes the ablation honest.

Options (all read with .get, so existing FMF parameter dicts work unchanged)
---------------------------------------------------------------------------
Architecture
    COND_MODE     'concat' (default) | 'film'
        'concat' appends the time encoding and y to the state, the original
        construction.
        At the tuned D = 50 with L = 10 and dy = 5 that leaves the trunk seeing
        77% time, 8% observation and 15% state. 'film' instead projects (t, y)
        to per-layer scale/shift parameters that modulate the hidden activations
        (FiLM, Perez et al. 2018; the adaLN conditioning of DiT, Peebles & Xie
        2023), so the input carries the state alone and conditioning acts
        multiplicatively at every block.
    USE_LAYERNORM  False | True — LayerNorm inside each residual block.
    ZERO_INIT_OUT  False | True — zero the output layer so the field starts at
        v = 0 (the identity map) rather than at Xavier noise.
Training
    EMA_DECAY      0.0 (off) | e.g. 0.999 — exponential moving average of the
        weights, evaluated at inference. Standard in diffusion/flow training.
    OPTIMIZER      'adam' | 'adamw';  WEIGHT_DECAY  (adamw only)
    SCHEDULE       'warm_restarts' (default) | 'cosine' | 'none'
    WARMUP_FRAC    0.0 — fraction of iterations spent linearly warming the LR.
    GRAD_CLIP      0.0 (off) | max global grad norm.
    TIME_SAMPLING  'uniform' | 'logitnormal' — the SD3 recommendation (Esser et
        al. 2024), which concentrates t near 1/2 where the field is hardest.
    OT_COUPLING    False | True — see the note above.
    RESAMPLE_DATA  False | True — redraw the training pairs every iteration from
        the predictive distribution (fresh process noise on the propagated
        ensemble, fresh observation noise on h(x)) instead of reusing one fixed
        set of N pairs. Both draws are exactly the ones that built the fixed set,
        so this changes only how many samples the network sees, not what they
        are distributed as.
    SIGMA_MIN      0.0 — width of Gaussian noise around the linear interpolant.
    STANDARDIZE    False | True — train in units of the ensemble's own mean/std
        (an affine change of variables, undone before the particles are stored).
Inference
    SOLVER         'euler' (default) | 'heun' | 'rk4'
Filtering
    COLD_START     True (default) — re-initialise the network every analysis
        step. False restores the warm start used across analysis steps before
        these switches were added.
"""

import math
import time
import os
import json
import hashlib

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from timing_utils import sync_clock


class _NonFiniteRun(Exception):
    """Raised when training goes non-finite; the run is abandoned and NaN-filled."""


# ======================================================================
# Time encoding
# ======================================================================
def timestep_embedding(t_scalar, dim, max_period):
    """Sinusoidal encoding of a continuous position."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=t_scalar.device)
    args = t_scalar.float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResidualBlock(nn.Module):
    """Pre-activation residual block, optionally LayerNorm'd and FiLM-modulated."""

    def __init__(self, hidden_dim, activation, use_layernorm=False, cond_dim=0):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.activation = activation
        self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else None
        # FiLM: the conditioning vector produces a per-channel scale and shift.
        # Zero-initialised so the block starts as the unmodulated identity, which
        # keeps the film variant's first steps as stable as concat's.
        if cond_dim:
            self.film = nn.Linear(cond_dim, 2 * hidden_dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        else:
            self.film = None

    def forward(self, x, cond=None):
        identity = x
        out = self.linear1(x)
        if self.norm is not None:
            out = self.norm(out)
        if self.film is not None and cond is not None:
            scale, shift = self.film(cond).chunk(2, dim=-1)
            out = out * (1.0 + scale) + shift
        out = self.activation(out)
        out = self.linear2(out)
        out = self.activation(out + identity)
        return out


class VectorField(nn.Module):
    """
    Conditional vector field v_θ(x_t, t, y) -> R^L.

    'concat' feeds concat(x, emb(t), y) to the trunk, the original construction.
    'film'   feeds x alone and modulates every block with an embedding of
             (emb(t), y), so conditioning is multiplicative and the trunk's
             input is not crowded out by the time encoding.
    """

    def __init__(self, state_dim, obs_dim, hidden_dim, num_resblocks=2,
                 time_embed_dim=32, time_scale=300.0, time_max_period=50.0,
                 cond_mode='concat', use_layernorm=False, zero_init_out=False):
        super().__init__()
        self.activation = nn.SiLU()
        self.time_embed_dim = time_embed_dim
        self.time_scale = time_scale
        self.time_max_period = time_max_period
        self.cond_mode = cond_mode

        if cond_mode == 'film':
            cond_in = time_embed_dim + obs_dim
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_in, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim))
            self.layer_input = nn.Linear(state_dim, hidden_dim, bias=False)
            cond_dim = hidden_dim
        else:
            self.cond_mlp = None
            self.layer_input = nn.Linear(state_dim + time_embed_dim + obs_dim,
                                         hidden_dim, bias=False)
            cond_dim = 0

        self.resblocks = nn.ModuleList([
            ResidualBlock(hidden_dim, self.activation, use_layernorm, cond_dim)
            for _ in range(num_resblocks)])
        self.layer_out = nn.Linear(hidden_dim, state_dim, bias=False)
        self.zero_init_out = zero_init_out

    def _t_feat(self, t_scalar):
        if self.time_embed_dim > 1:
            return timestep_embedding(t_scalar * self.time_scale,
                                      self.time_embed_dim, self.time_max_period)
        return t_scalar

    def forward(self, x, t_scalar, y):
        t_feat = self._t_feat(t_scalar)
        if self.cond_mode == 'film':
            cond = self.cond_mlp(torch.cat([t_feat, y], dim=1))
            out = self.layer_input(x)
            for block in self.resblocks:
                out = block(out, cond)
        else:
            out = self.layer_input(torch.cat([x, t_feat, y], dim=1))
            for block in self.resblocks:
                out = block(out)
        out = self.activation(out)
        return self.layer_out(out)


def make_init_weights(zero_init_out=False):
    """Xavier init; optionally zeroing the output layer."""
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
    return init_weights


class EMA:
    """Exponential moving average of parameters, swapped in for inference."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def store_and_apply(self, model):
        sd = model.state_dict()
        self.backup = {k: sd[k].detach().clone() for k in self.shadow}
        sd.update({k: v for k, v in self.shadow.items()})
        model.load_state_dict(sd)

    def restore(self, model):
        if self.backup:
            sd = model.state_dict()
            sd.update(self.backup)
            model.load_state_dict(sd)
            self.backup = {}


def _ot_pair(x0, x1):
    """
    Re-pair a minibatch by squared-euclidean optimal transport.

    Endpoints and marginals are unchanged — only which x_0 travels to which x_1.
    Falls back to the identity pairing if POT is unavailable or the solve fails,
    so a missing dependency degrades to the default coupling instead of crashing.
    """
    try:
        import ot as pot
        M = torch.cdist(x0, x1) ** 2
        a = np.ones(x0.shape[0]) / x0.shape[0]
        b = np.ones(x1.shape[0]) / x1.shape[0]
        G = pot.emd(a, b, M.detach().cpu().numpy())
        idx = np.argmax(G, axis=1)          # each source to its transported target
        return x0, x1[torch.as_tensor(idx, device=x1.device)]
    except Exception:
        return x0, x1


def FMF(Y, X0, A, h, t, Noise, parameters, device=None, timing_out=None,
        warm_start='off', warm_start_tag='', warm_start_dir='DATA/warm_start'):
    """
    Flow Matching Filter, instrumented with the ablation switches documented
    in the module docstring above. Signature identical to the plain filter's
    apart from the three warm-start arguments below.

    Output: ndarray (NUM_SIM x T x L x N)

    warm_start : 'off' (the default) | 'auto' | 'save' | 'load' | 'require'
        Disk cache for the trained velocity field v_theta(x_t, t, y). 'save'
        trains the spin-up from a random initialisation and writes its weights
        to disk. 'load' reads them when a matching file exists and trains from
        scratch when it does not, never writing. 'auto' does both: load on a
        hit, else train and save. 'require' is 'load' with the fallback removed
        -- it raises when no usable checkpoint is found, which is what a sweep
        wants when every trial is meant to start from the same weights and a
        silent cold start would go unnoticed. Left at 'off' the whole mechanism
        is inert and this function reproduces the plain Flow Matching Filter
        exactly.
        Whether the checkpoint exists is decided ONCE, at entry, so a run is
        warm for all of its simulations or cold for all of them: the run that
        creates the cache does not warm-start its own later simulations from it,
        and so is itself a clean cold baseline.
        The spin-up (analysis step 0) is the expensive step -- ITERATION
        iterations against Final_Number_ITERATION for every later one -- so on a
        cache HIT step 0 runs on Final_Number_ITERATION too. Skipping that cost
        is the whole point of the cache.
        Requires the warm start FMF already supports across analysis steps, so
        COLD_START=True (this module's default, though not the tuned config's)
        is rejected rather than silently ignored.
    warm_start_tag : str, REQUIRED whenever warm_start != 'off'
        Names the problem: 'quadratic', 'lorenz96', ... FMF receives A and h as
        opaque closures and so cannot tell two different systems of the same
        dimension apart; without an explicit tag a checkpoint from one
        experiment would be loaded into another that merely shares (L, dy), and
        the result would be a plausible-looking wrong posterior rather than an
        error. A missing tag is therefore an error, not a default.
    warm_start_dir : str
        Directory holding the checkpoints (default 'DATA/warm_start').

    Two consequences of the cache worth knowing before reading its results.
    First, ALL NUM_SIM simulations load the SAME checkpoint, so the runs no
    longer differ in their initialisation and the across-simulation spread --
    the error bars in the sweep figures -- is deflated for a reason that is not
    statistical. Second, step_times[:, 0] is the from-scratch spin-up cost; on a
    cache hit it is nothing of the kind, and is not comparable with the spin-up
    column of the other filters, none of which have a cache.
    """
    NUM_SIM, L, N = X0.shape
    T, dy = Y.shape[1], Y.shape[2]
    sigma, gamma = Noise[0], Noise[1]

    P = parameters
    NUM_NEURON = P['NUM_NEURON']
    BATCH_SIZE = P['BATCH_SIZE']
    LearningRate = P['LearningRate']
    ITERATION = P['ITERATION']
    Final_Number_ITERATION = P['Final_Number_ITERATION']
    num_resblocks = P['num_resblocks']
    ODE_STEPS = P['ODE_STEPS']

    TIME_EMBED_DIM = int(P.get('TIME_EMBED_DIM', 30))
    TIME_SCALE = float(P.get('TIME_SCALE', 300.0))
    TIME_MAX_PERIOD = float(P.get('TIME_MAX_PERIOD', 50.0))

    COND_MODE = P.get('COND_MODE', 'concat')
    USE_LAYERNORM = bool(P.get('USE_LAYERNORM', False))
    ZERO_INIT_OUT = bool(P.get('ZERO_INIT_OUT', False))
    EMA_DECAY = float(P.get('EMA_DECAY', 0.0))
    OPTIMIZER = P.get('OPTIMIZER', 'adam')
    WEIGHT_DECAY = float(P.get('WEIGHT_DECAY', 0.0))
    SCHEDULE = P.get('SCHEDULE', 'warm_restarts')
    WARMUP_FRAC = float(P.get('WARMUP_FRAC', 0.0))
    GRAD_CLIP = float(P.get('GRAD_CLIP', 0.0))
    TIME_SAMPLING = P.get('TIME_SAMPLING', 'uniform')
    OT_COUPLING = bool(P.get('OT_COUPLING', False))
    RESAMPLE_DATA = bool(P.get('RESAMPLE_DATA', False))
    SIGMA_MIN = float(P.get('SIGMA_MIN', 0.0))
    STANDARDIZE = bool(P.get('STANDARDIZE', False))
    SOLVER = P.get('SOLVER', 'euler')
    COLD_START = bool(P.get('COLD_START', True))
    VERBOSE = bool(P.get('VERBOSE', True))
    # Multiplicative covariance inflation of the posterior ensemble about its own
    # mean, applied once per analysis step (Anderson & Anderson 1999) — the
    # standard ensemble-DA correction for a filter that is systematically
    # under-dispersed. It is here because that is precisely what a diagnostic
    # run measured: the ensemble std relative to the reference falls ~2% per step
    # (1.00 at step 1 -> 0.44 at step 45, and 0.98^45 ~ 0.4), the signature of an
    # MSE-regressed velocity field smoothing toward the conditional mean. It
    # rescales spread only and leaves the ensemble mean untouched.
    INFLATION = float(P.get('INFLATION', 1.0))
    # Half-width, in robust sigmas, of the interval the posterior particles are
    # clamped to after each analysis step. None/0 disables it. See the note at
    # the clamp itself for why this is the right place to break the instability.
    PARTICLE_CLAMP = P.get('PARTICLE_CLAMP', None)
    PARTICLE_CLAMP = float(PARTICLE_CLAMP) if PARTICLE_CLAMP else None
    # Extra conditioning FEATURES built from the fixed observation y and the
    # current state x_t. This does not change what is conditioned on — y is
    # still the same fixed vector and still never moves along the flow — it only
    # changes how y is presented to the network, exactly as the sinusoidal
    # encoding does for t. Motivated by a follow-up diagnostic: with the spread
    # collapse fixed, the residual error is mean drift, the signature of
    # mis-assigned mass between the two posterior modes at x ~ +/- sqrt(y).
    #   'y'     the observation itself (default; alone this is the old behaviour)
    #   'sqrt'  sqrt(relu(y)) — the modes sit at +/- this value, so it hands the
    #           network the mode LOCATION instead of making it invert a square
    #   'resid' h(x_t) - y — the observation-space mismatch of the particle the
    #           field is being evaluated at, i.e. how far this particle still is
    #           from explaining the data
    COND_FEATURES = P.get('COND_FEATURES', ['y'])
    if isinstance(COND_FEATURES, str):
        COND_FEATURES = [COND_FEATURES]
    n_cond = dy * len(COND_FEATURES)

    # ------------------------------------------------------------------
    # Arguments rather than `parameters` keys: the dict is the tuned
    # configuration, reused across runs and by the sweeps, while where weights
    # are cached is a property of the CALL. So get_config stays untouched and a
    # tuning run cannot inherit a cache setting by accident.
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
                "warm_start=%r contradicts COLD_START=True, which rebuilds the "
                "network at every analysis step and so discards the loaded "
                "weights after the spin-up." % warm_start)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    init_weights = make_init_weights(ZERO_INIT_OUT)

    def build_net():
        vf = VectorField(L, n_cond, NUM_NEURON, num_resblocks, TIME_EMBED_DIM,
                         TIME_SCALE, TIME_MAX_PERIOD, COND_MODE, USE_LAYERNORM,
                         ZERO_INIT_OUT).to(device)
        vf.apply(init_weights)
        if ZERO_INIT_OUT:
            nn.init.zeros_(vf.layer_out.weight)
        return vf

    def _warm_start_path():
        """
        Path of the checkpoint for THIS problem and THIS architecture.

        The readable part of the name carries what fixes the shape of the
        network's state dict -- tag, dimensions, width, depth, time-embedding
        width and conditioning mode -- so a stale file can be identified by eye.
        The trailing digest covers what leaves those shapes alone but changes
        what the trained field MEANS: the noise levels it was fitted at,
        whether it was trained in standardised units, the time
        encoding's scale, the interpolant noise, and which features the
        conditioning vector is built from. A mismatch in the first group would
        fail loudly at load_state_dict; a mismatch in the second would not,
        which is exactly why it is hashed into the filename instead.

        The ensemble size is deliberately NOT in the key. The networks have the
        same shape at any N -- only the training data drawn from the ensemble
        changes -- so one checkpoint serves a whole particle sweep instead of
        forcing a fresh spin-up at every point. N is recorded in meta, so the
        size a checkpoint was fitted at stays recoverable.
        """
        semantics = {
            'sigma':         np.asarray(sigma, dtype=float).ravel().tolist(),
            'gamma':         np.asarray(gamma, dtype=float).ravel().tolist(),
            'STANDARDIZE':   bool(STANDARDIZE),
            'SIGMA_MIN':     float(SIGMA_MIN),
            'TIME_SCALE':    float(TIME_SCALE),
            'TIME_MAX_PERIOD': float(TIME_MAX_PERIOD),
            'COND_FEATURES': list(COND_FEATURES),
            'USE_LAYERNORM': bool(USE_LAYERNORM),
            'n_cond':        int(n_cond),
        }
        digest = hashlib.sha1(
            json.dumps(semantics, sort_keys=True).encode('utf-8')).hexdigest()[:8]
        name   = 'fmf_%s_L%d_dy%d_nn%d_rb%d_te%d_%s_%s.pt' % (
            WARM_START_TAG, L, dy, int(NUM_NEURON), int(num_resblocks),
            int(TIME_EMBED_DIM), COND_MODE, digest)
        return os.path.join(WARM_START_DIR, name)

    def _state_dict_compatible(sd_ckpt, module):
        """
        True when sd_ckpt can be loaded into module without any surprises.

        Checked BEFORE load_state_dict rather than relying on it to raise,
        because load_state_dict copies tensors as it walks the module and only
        raises at the end: a mismatch part way through leaves some layers loaded
        and some not. Verifying first means a rejected checkpoint applies
        NOTHING, so the fallback is the cold-start network the run would have
        had anyway -- identical down to the RNG stream, since no re-initialisation
        is needed to undo a partial load.
        """
        sd = module.state_dict()
        return (set(sd_ckpt) == set(sd)
                and all(sd_ckpt[k].shape == sd[k].shape for k in sd))

    def _warm_start_save(path, vf):
        """
        Write the velocity field's weights, plus the provenance needed to
        interpret them.

        Saved through a process-unique temporary file and os.replace so that a
        reader never sees a half-written checkpoint: the sweeps dispatch several
        filters concurrently, and several runs may share the directory. Two
        processes missing the cache at once both train and both write; the last
        writer wins, which is harmless because the two files are
        interchangeable. Tensors go to the host so the file is loadable on any
        device.

        Only the network is stored. FMF's EMA shadow is built inside
        train_flow and discarded when it returns, so unlike OTF's there is no
        cross-step EMA state for a checkpoint to carry.
        """
        meta = {
            'tag':             WARM_START_TAG,
            'L':               int(L),
            'dy':              int(dy),
            'N':               int(N),
            'sigma':           np.asarray(sigma, dtype=float).ravel().tolist(),
            'gamma':           np.asarray(gamma, dtype=float).ravel().tolist(),
            'NUM_NEURON':      int(NUM_NEURON),
            'num_resblocks':   int(num_resblocks),
            'TIME_EMBED_DIM':  int(TIME_EMBED_DIM),
            'TIME_SCALE':      float(TIME_SCALE),
            'TIME_MAX_PERIOD': float(TIME_MAX_PERIOD),
            'COND_MODE':       str(COND_MODE),
            'COND_FEATURES':   list(COND_FEATURES),
            'USE_LAYERNORM':   bool(USE_LAYERNORM),
            'ZERO_INIT_OUT':   bool(ZERO_INIT_OUT),
            'STANDARDIZE':     bool(STANDARDIZE),
            'SIGMA_MIN':       float(SIGMA_MIN),
            'ITERATION':       int(ITERATION),
            'source':          'simulation 0, analysis step 0',
        }
        payload = {
            'vf':   {k: v.detach().cpu() for k, v in vf.state_dict().items()},
            'meta': meta,
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp = '%s.tmp%d' % (path, os.getpid())
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _warm_start_load(path, vf):
        """
        Load a checkpoint into the velocity field, or return None on any failure.

        Every failure mode (absent file, unreadable file, shape mismatch) is a
        cache miss and is reported, never raised: a missing warm start costs
        iterations, not correctness. Compatibility is checked BEFORE anything is
        applied, so a rejected checkpoint leaves the freshly built network
        exactly as build_net made it.
        """
        if not os.path.exists(path):
            return None
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
            if not _state_dict_compatible(ckpt['vf'], vf):
                raise ValueError('checkpoint does not fit this architecture')
            vf.load_state_dict(ckpt['vf'])
            return ckpt.get('meta', {})
        except Exception as exc:
            # Under 'require' a checkpoint that does not fit is exactly the
            # failure the mode exists to surface, so it propagates instead of
            # degrading to a cold start.
            if WARM_START == 'require':
                raise
            print('[FMF] warm start: ignoring %s (%s: %s)'
                  % (path, type(exc).__name__, exc))
            return None

    def sample_t(bs):
        if TIME_SAMPLING == 'logitnormal':
            return torch.sigmoid(torch.randn(bs, 1, device=device))
        return torch.rand(bs, 1, device=device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_flow(vf, X_target, Y_cond, iterations, lr, batch_size, draw_fn,
                   ts, k, cond_fn):
        vf.train()
        if OPTIMIZER == 'adamw':
            optimizer = torch.optim.AdamW(vf.parameters(), lr=lr,
                                          weight_decay=WEIGHT_DECAY)
        else:
            optimizer = torch.optim.Adam(vf.parameters(), lr=lr)

        if SCHEDULE == 'warm_restarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=512, T_mult=2,
                                                    eta_min=lr * 1e-3)
        elif SCHEDULE == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, iterations), eta_min=lr * 1e-3)
        else:
            scheduler = None

        ema = EMA(vf, EMA_DECAY) if EMA_DECAY > 0 else None
        batch_size = min(batch_size, X_target.shape[0])
        n_warm = int(WARMUP_FRAC * iterations)

        for i in range(iterations):
            if n_warm and i < n_warm:
                for g in optimizer.param_groups:
                    g['lr'] = lr * (i + 1) / n_warm

            x1_batch, y_batch, x0_batch = draw_fn(batch_size)
            if OT_COUPLING:
                x0_batch, x1_batch = _ot_pair(x0_batch, x1_batch)

            t_batch = sample_t(x1_batch.shape[0])
            x_t = (1.0 - t_batch) * x0_batch + t_batch * x1_batch
            if SIGMA_MIN > 0:
                x_t = x_t + SIGMA_MIN * torch.randn_like(x_t)
            u_target = x1_batch - x0_batch
            u_pred = vf(x_t, t_batch, cond_fn(x_t, y_batch))
            loss = ((u_pred - u_target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(vf.parameters(), GRAD_CLIP)
            optimizer.step()
            if ema is not None:
                ema.update(vf)

            if scheduler is not None and (SCHEDULE != 'warm_restarts'
                                          or i >= iterations / 2):
                if not (n_warm and i < n_warm):
                    scheduler.step()

            if (i + 1) == iterations:
                vf.eval()
                with torch.no_grad():
                    t_eval = sample_t(X_target.shape[0])
                    x0_eval = X_target[torch.randperm(X_target.shape[0])]
                    x_t_eval = (1.0 - t_eval) * x0_eval + t_eval * X_target
                    u_eval = vf(x_t_eval, t_eval, cond_fn(x_t_eval, Y_cond))
                    loss_log = ((u_eval - (X_target - x0_eval)) ** 2).mean()
                    if VERBOSE:
                        print('Simu#%d, Step:%d/%d, Iter:%d, FM loss=%.4f'
                              % (k + 1, ts, T - 1, iterations, loss_log.item()))
                    if not torch.isfinite(loss_log):
                        raise _NonFiniteRun('loss %s at sim %d step %d'
                                            % (loss_log.item(), k + 1, ts))
        return ema

    # ------------------------------------------------------------------
    # Integration — y is FIXED throughout, only x moves
    # ------------------------------------------------------------------
    def integrate(vf, x_init, y_obs, n_steps, cond_fn):
        """y_obs is held FIXED for the whole integration; only x moves."""
        dt = 1.0 / n_steps
        x = x_init.clone()
        vf.eval()

        def f(x_, t_):
            return vf(x_, t_, cond_fn(x_, y_obs))

        with torch.no_grad():
            for step in range(n_steps):
                tv = step * dt
                tt = torch.full((x.shape[0], 1), tv, dtype=torch.float32,
                                device=device)
                if SOLVER == 'euler':
                    x = x + dt * f(x, tt)
                elif SOLVER == 'heun':
                    v1 = f(x, tt)
                    tt2 = torch.full_like(tt, tv + dt)
                    v2 = f(x + dt * v1, tt2)
                    x = x + dt * 0.5 * (v1 + v2)
                elif SOLVER == 'rk4':
                    t2 = torch.full_like(tt, tv + dt / 2)
                    t3 = torch.full_like(tt, tv + dt)
                    k1 = f(x, tt)
                    k2 = f(x + dt / 2 * k1, t2)
                    k3 = f(x + dt / 2 * k2, t2)
                    k4 = f(x + dt * k3, t3)
                    x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                else:
                    raise ValueError(f'unknown SOLVER {SOLVER}')
        return x

    # ------------------------------------------------------------------
    # Filter loop
    # ------------------------------------------------------------------
    _timing = timing_out is not None
    step_times = np.full((NUM_SIM, T - 1), np.nan)

    # Resolved once: the path depends only on the problem and the architecture,
    # both fixed for the whole call.
    #
    # Availability is snapshotted HERE, before any training, and that is the
    # point: the run that BUILDS the cache would otherwise warm-start its own
    # later simulations from the checkpoint its first simulation had just
    # written, and would be neither the cold baseline nor a warm run. With the
    # snapshot a run is cold throughout or warm throughout, so the
    # cache-building run reproduces the plain Flow Matching Filter exactly and
    # every run after it is uniform.
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
    X_OUT = torch.zeros((NUM_SIM, T, N, L), device=device, dtype=torch.float32)

    aborted = None
    try:
        for k in range(NUM_SIM):
            y = Y[k]
            X_OUT[k, 0] = torch.from_numpy(X0[k].T).to(torch.float32).to(device)
            ITERS = ITERATION
            vf = build_net()

            # Every simulation reads the same file, so on a hit none pays the
            # ITERATION spin-up. A rejected checkpoint applies nothing, so the
            # network build_net just made is still the right cold start.
            if _warm_available and _warm_start_load(_warm_path, vf) is not None:
                # Step 0 is charged the ordinary online budget instead.
                ITERS = Final_Number_ITERATION
                print('[FMF] Simu#%d/%d: warm start from %s '
                      '(spin-up %d -> %d iterations)'
                      % (k + 1, NUM_SIM, _warm_path, ITERATION, ITERS))

            for i in range(T - 1):
                _t_step = sync_clock(device) if _timing else 0.0
                if COLD_START:
                    vf = build_net()

                # ---- predictive ensemble and its synthetic observations ----
                # x ~ p(x_i+1 | y_1:i)  and  y = h(x) + noise, i.e. a joint draw.
                Xprev = X_OUT[k, i]                                  # (N x L)
                AX = A(Xprev.T, t[i]).T                              # (N x L)
                X1 = AX + sigma * torch.randn(N, L, device=device)
                Y1 = h(X1.T).T + gamma * torch.randn(N, dy, device=device)

                # ---- optional standardisation (affine change of variables) ----
                if STANDARDIZE:
                    mx, sx = X1.mean(0, keepdim=True), X1.std(0, keepdim=True) + 1e-6
                    my, sy = Y1.mean(0, keepdim=True), Y1.std(0, keepdim=True) + 1e-6
                else:
                    mx = torch.zeros(1, L, device=device)
                    sx = torch.ones(1, L, device=device)
                    my = torch.zeros(1, dy, device=device)
                    sy = torch.ones(1, dy, device=device)
                X1s, Y1s = (X1 - mx) / sx, (Y1 - my) / sy

                # ---- minibatch draw ----
                # Fixed set (default) reuses the N pairs above; RESAMPLE_DATA
                # redraws process and observation noise every iteration, which
                # samples the SAME predictive/joint law with fresh realisations.
                def draw_fn(bs):
                    if RESAMPLE_DATA:
                        idx = torch.randint(0, N, (bs,), device=device)
                        xb = AX[idx] + sigma * torch.randn(bs, L, device=device)
                        yb = h(xb.T).T + gamma * torch.randn(bs, dy, device=device)
                        xb, yb = (xb - mx) / sx, (yb - my) / sy
                        # x_0 independent of y: a second, unpaired predictive draw
                        j = torch.randint(0, N, (bs,), device=device)
                        x0b = AX[j] + sigma * torch.randn(bs, L, device=device)
                        x0b = (x0b - mx) / sx
                        return xb, yb, x0b
                    idx = torch.randperm(N, device=device)[:bs]
                    perm = torch.randperm(N, device=device)[:bs]
                    return X1s[idx], Y1s[idx], X1s[perm]

                # Conditioning vector: y itself plus any requested features of
                # (x_t, y). y is a FIXED input here — nothing below evolves it.
                def cond_fn(x_s, y_s):
                    if len(COND_FEATURES) == 1 and COND_FEATURES[0] == 'y':
                        return y_s
                    feats = []
                    y_o = y_s * sy + my
                    for name in COND_FEATURES:
                        if name == 'y':
                            feats.append(y_s)
                        elif name == 'sqrt':
                            # Modes of the posterior sit at x ~ +/- sqrt(y).
                            feats.append(torch.sqrt(torch.clamp(y_o, min=0.0)))
                        elif name == 'resid':
                            x_o = x_s * sx + mx
                            feats.append((h(x_o.T).T - y_o) / sy)
                        else:
                            raise ValueError(f'unknown COND_FEATURE {name}')
                    return torch.cat(feats, dim=1)

                ema = train_flow(vf, X1s, Y1s, ITERS, LearningRate, BATCH_SIZE,
                                 draw_fn, i + 1, k, cond_fn)

                # Saved after train_flow returns, so a diverged run
                # (_NonFiniteRun) never reaches this line and cannot poison the
                # cache.
                if _warm_save_pending and k == 0 and i == 0:
                    _warm_start_save(_warm_path, vf)
                    _warm_save_pending = False
                    print('[FMF] warm start: saved %s' % _warm_path)
                ITERS = Final_Number_ITERATION

                # ---- inference at the TRUE observation, y held fixed ----
                y_true = torch.from_numpy(
                    np.repeat(y[i + 1, :].reshape(1, dy), N, axis=0)
                ).to(torch.float32).to(device)
                y_true_s = (y_true - my) / sy

                if ema is not None:
                    ema.store_and_apply(vf)
                X_mapped = integrate(vf, X1s, y_true_s, ODE_STEPS, cond_fn)
                if ema is not None:
                    ema.restore(vf)

                X_post = (X_mapped * sx + mx).detach()

                # Robust outlier projection. The 8192-iteration configuration
                # fails through a FEEDBACK LOOP, not a training blow-up: a few
                # particles leave the ensemble during ODE integration, become
                # part of the next step's forecast A(X), and so enter the next
                # step's TRAINING TARGETS — at which point the loss goes NaN
                # (observed at simulation 5, step 43). Tightening GRAD_CLIP does
                # not touch this and in fact made it worse (max|X| = 4.7e12).
                # Clamping each coordinate to a robust interval around the
                # ensemble median breaks the loop at its source. Median and MAD
                # are used rather than mean and std precisely because the few
                # runaway particles would otherwise set the bound themselves.
                # PARTICLE_CLAMP is the half-width in robust sigmas; None = off.
                if PARTICLE_CLAMP:
                    med = X_post.median(dim=0, keepdim=True).values
                    mad = (X_post - med).abs().median(dim=0, keepdim=True).values
                    scale = 1.4826 * mad + 1e-8          # MAD -> sigma estimate
                    X_post = torch.clamp(X_post,
                                         med - PARTICLE_CLAMP * scale,
                                         med + PARTICLE_CLAMP * scale)

                if INFLATION != 1.0:
                    mu = X_post.mean(dim=0, keepdim=True)
                    X_post = mu + INFLATION * (X_post - mu)
                X_OUT[k, i + 1] = X_post
                if _timing:
                    step_times[k, i] = sync_clock(device) - _t_step

    except _NonFiniteRun as exc:
        aborted = exc
        X_OUT[k, i + 1:] = float('nan')
        X_OUT[k + 1:] = float('nan')
        print('[FMF] ABORT: %s' % exc)

    print('--- FMF time : %s seconds ---%s'
          % (time.time() - start_time, ' (ABORTED)' if aborted else ''))
    if _timing:
        timing_out['step_times'] = step_times
        timing_out['total'] = time.time() - start_time

    return X_OUT.cpu().numpy().transpose(0, 1, 3, 2)

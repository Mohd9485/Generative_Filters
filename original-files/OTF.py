"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import time
import os
import json
import hashlib
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, CosineAnnealingWarmRestarts
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


def OTF(Y, X0, A, h, t, Noise, parameters, device=None, timing_out=None,
        warm_start='off', warm_start_tag='', warm_start_dir='DATA/warm_start'):
    """
    Optimal transport filter (OTF) function according to Algorithm 3 in
    [Al-Jarrah, M., Jin, N., Hosseini, B. and Taghvaei, A., 2024, July. 
     Nonlinear Filtering with Brenier Optimal Transport Maps. 
     In International Conference on Machine Learning (pp. 813-839). PMLR.]
    
     --Side note: In the paper, the algorithm was called optimal transport 
     particle filter (OTPF) which is not an approrate name for the algorithm 
     since we aren't working with weighted particles--
     
    where:
        
    Y     : True observations with shape (NUM_SIM x T x dy),
    X0    : Initial particles with shape (NUM_SIM x L x N),
    A     : Deterministic dynamic model (without noise),
    h     : Deterministic observation model (without noise),
    t     : Time vector (e.g., t = 0.0, dt, 2*dt, ..., tf),
    Noise : A list [sigma, gamma] defining the noise levels in the dynamics (sigma) and observations (gamma),
    param : Dictionary of hyperparameters for the neural network components.
    timing_out : Optional dict. When supplied it is filled with
        'step_times' — a (NUM_SIM x T-1) array of per-analysis-step wall-clock
        seconds — and 'total', the whole-run time. Column 0 is the spin-up
        step, which trains from a random initialisation and can be done
        offline; the remaining columns are the online per-step cost. Left as
        None (the default) the timing is skipped entirely, so the tuning
        sweeps pay nothing for it.

    warm_start : 'off' (the default) | 'auto' | 'save' | 'load' | 'require'
        Disk cache for the two trained networks -- the critic f and the map T.
        'save' trains the spin-up from a random initialisation and writes both
        sets of weights to disk. 'load' reads them when a matching file exists
        and trains from scratch when it does not, never writing. 'auto' does
        both: load on a hit, else train and save. 'require' is 'load' with the
        fallback removed -- it raises when no usable checkpoint is found, which
        is what a sweep wants when every trial is meant to start from the same
        weights and a silent cold start would go unnoticed. Left at 'off' the
        whole mechanism is inert and this function reproduces the baseline
        algorithm exactly.
        Whether the checkpoint exists is decided ONCE, at entry, so a run is
        warm for all of its simulations or cold for all of them: the run that
        creates the cache does not warm-start its own later simulations from
        it, and so is itself a clean cold baseline.
        The spin-up (analysis step 0) is the expensive step -- ITERATION
        iterations against Final_Number_ITERATION for every later one -- so on
        a cache HIT step 0 runs on Final_Number_ITERATION too. Skipping that
        cost is the whole point of the cache.
    warm_start_tag : str, REQUIRED whenever warm_start != 'off'
        Names the problem: 'quadratic', 'lorenz96', ... OTF receives A and h as
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
    the error bars in the sweep figures -- is deflated for a reason that is
    not statistical. Second, step_times[:, 0] is documented below as the
    from-scratch spin-up cost; on a cache hit it is nothing of the kind, and
    is not comparable with the spin-up column of the other filters, none of
    which have a cache.
    """

    NUM_SIM = X0.shape[0]
    L = X0.shape[1]
    N = X0.shape[2]
    
    T = Y.shape[1]
    dy = Y.shape[2]

    INPUT_DIM = [L, dy]
    sigma = Noise[0]
    gamma = Noise[1]
    
    tau = t[1] - t[0]
    
    normalization = parameters['normalization']
    NUM_NEURON = parameters['NUM_NEURON']
    BATCH_SIZE =  parameters['BATCH_SIZE']
    LearningRate = parameters['LearningRate']
    ITERATION = parameters['ITERATION']
    Final_Number_ITERATION = parameters['Final_Number_ITERATION']

    # ------------------------------------------------------------------
    # Improvements carried over from the FMF and SIF studies. All are training
    # mechanics: the max-min OT objective, the critic/map pair and the
    # block-triangular coupling (T pushes the INDEPENDENT coupling
    # (X, Y_shuffled) onto the joint, y fixed) are untouched. Defaults
    # reproduce the baseline algorithm exactly.
    #
    # Note what does NOT carry over. OTF already warm-starts — init_weights is
    # applied once per simulation, outside the time loop — which is why it
    # performs well on a 64-iteration online budget where FMF needed 2048. And
    # it has no ODE, so the second-order-solver fix that was worth ~0.2 to both
    # FMF and SIF has no analogue here: T(x, y) is a single forward pass.
    # That leaves the schedule, the training data, and the budget.
    #
    #   SCHEDULE      'warm_restarts' (the default) | 'cosine' | 'none'
    #       The default configuration uses CosineAnnealingWarmRestarts
    #       (T_0=512, T_mult=2) on BOTH optimisers, stepped every iteration, so
    #       the learning rates are kicked back to full value at iterations
    #       512, 1536, 3584, ...  In an adversarial max-min loop that also
    #       desynchronises the critic and the map, which is a second reason to
    #       prefer a monotone decay.
    #   WARMUP_FRAC / GRAD_CLIP   stability.
    #   RESAMPLE_DATA  redraw (X, Y) from the predictive law every iteration.
    #   COLD_START    True re-initialises both nets every analysis step; kept
    #       only for symmetry with the other two studies, since OTF's default
    #       (and better) behaviour is the warm start it already does.
    # ------------------------------------------------------------------
    SCHEDULE      = parameters.get('SCHEDULE', 'warm_restarts')
    WARMUP_FRAC   = float(parameters.get('WARMUP_FRAC', 0.0))
    GRAD_CLIP     = float(parameters.get('GRAD_CLIP', 0.0))
    RESAMPLE_DATA = bool(parameters.get('RESAMPLE_DATA', False))
    COLD_START    = bool(parameters.get('COLD_START', False))
    # --- OTF-specific ideas (all default to the original behaviour) ---
    #   T_RESIDUAL   parameterise the map as T(x,y) = x + Delta(x,y).
    #       The OT term 0.5||x - T(x,y)||^2 in loss_T explicitly penalises
    #       departure from the identity, so the network is being asked to
    #       reproduce x and then correct it. Predicting the CORRECTION instead
    #       makes the identity exact at initialisation and lets the whole
    #       capacity model the deviation, which is what the objective pays for.
    #   ACT          'relu' (the default) | 'silu' — smoother field for the map.
    #   EMA_DECAY    exponential moving average of the MAP weights, used at
    #       inference. Standard for adversarial training, where the iterate
    #       oscillates around the saddle rather than converging to it.
    #   CRITIC_GP    gradient-penalty weight on the critic f (WGAN-GP,
    #       Gulrajani et al. 2017). The max-min OT dual needs f in a bounded
    #       family; the default configuration leaves it unconstrained.
    T_RESIDUAL    = bool(parameters.get('T_RESIDUAL', False))
    ACT           = parameters.get('ACT', 'relu')
    EMA_DECAY     = float(parameters.get('EMA_DECAY', 0.0))
    CRITIC_GP     = float(parameters.get('CRITIC_GP', 0.0))
    #   The stabiliser block developed for FMF and carried to SIF. In FMF it
    #   turned an unusable budget into the best FMF result (max|X| 1e9 -> 12.6,
    #   aa-SW2 0.4092 -> 0.3713 at NUM_SIM=10); in SIF, which was already
    #   stable, it still bought 7-8% accuracy. Untested on OTF until now.
    #   Note ZERO_INIT_OUT means something different here: with T_RESIDUAL=False
    #   zeroing the map's output layer starts it at T(x,y) = 0, which maps the
    #   whole ensemble to the origin. The sensible identity start for an OT map
    #   is T_RESIDUAL=True together with ZERO_INIT_OUT, giving T(x,y) = x at
    #   initialisation — which is also what the OT cost 0.5||x - T||^2 prefers.
    USE_LAYERNORM = bool(parameters.get('USE_LAYERNORM', False))
    ZERO_INIT_OUT = bool(parameters.get('ZERO_INIT_OUT', False))
    OPTIMIZER     = parameters.get('OPTIMIZER', 'adam')
    WEIGHT_DECAY  = float(parameters.get('WEIGHT_DECAY', 0.0))
    K_in = parameters['K_in'] 
    num_resblocks = parameters['num_resblocks']

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
                "warm_start=%r contradicts COLD_START=True, which re-initialises "
                "both networks at every analysis step and so discards the loaded "
                "weights after the spin-up." % warm_start)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif not isinstance(device, torch.device):
        device = torch.device(device)


    class ResidualBlock(nn.Module):
        def __init__(self, hidden_dim, activation):
            super(ResidualBlock, self).__init__()
            self.linear1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.activation = activation
            self.norm = nn.LayerNorm(hidden_dim) if USE_LAYERNORM else None

        def forward(self, x):
            identity = x  # save input for skip connection
            out = self.linear1(x)
            if self.norm is not None:
                out = self.norm(out)
            out = self.activation(out)
            out = self.linear2(out)
            out = self.activation(out + identity)
            return out

    class f_NeuralNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_resblocks=2):
            """
            Parameters:
                input_dim (tuple): A tuple where input_dim[0] is the output dimension and 
                                   input_dim[1] is the second part of the input dimension.
                hidden_dim (int): The number of neurons in the hidden layers.
                num_resblocks (int): Number of residual blocks to use.
            """
            super(f_NeuralNet, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.activation = nn.SiLU() if ACT == 'silu' else nn.ReLU()
            
            self.layer_input = nn.Linear(self.input_dim[0] + self.input_dim[1], self.hidden_dim, bias=False)
            
            self.resblocks = nn.ModuleList([
                ResidualBlock(self.hidden_dim, self.activation) for _ in range(num_resblocks)
            ])
            
            self.layer_out = nn.Linear(self.hidden_dim, 1, bias=False)
            
        def forward(self, x, y):
            inp = torch.concat((x, y), dim=1)
            inp = self.layer_input(inp)
            
            out = inp
            
            for block in self.resblocks:
                out = block(out)
            
            out = self.activation(out)
            out = self.layer_out(out)
            return out
    
    class T_NeuralNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_resblocks=2):
            """
            Parameters:
                input_dim (tuple): A tuple where input_dim[0] is the output dimension and 
                                   input_dim[1] is the second part of the input dimension.
                hidden_dim (int): The number of neurons in the hidden layers.
                num_resblocks (int): Number of residual blocks to use.
            """
            super(T_NeuralNet, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.activation = nn.SiLU() if ACT == 'silu' else nn.ReLU()
            
            self.layer_input = nn.Linear(self.input_dim[0] + self.input_dim[1], self.hidden_dim, bias=False)
            
            self.resblocks = nn.ModuleList([
                ResidualBlock(self.hidden_dim, self.activation) for _ in range(num_resblocks)
            ])
            
            self.layer_out = nn.Linear(self.hidden_dim, self.input_dim[0], bias=False)
            
        def forward(self, x, y):
            inp = torch.concat((x, y), dim=1)
            inp = self.layer_input(inp)
            
            out = inp
            
            for block in self.resblocks:
                out = block(out)
            
            out = self.activation(out)
            out = self.layer_out(out)
            # T(x,y) = x + Delta(x,y): the OT cost measures displacement from
            # x, so the network predicts that displacement directly.
            return x + out if T_RESIDUAL else out

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.1)

    _ema = {}          # EMA shadow of the MAP weights, keyed by parameter name

    def _ema_update(T_net):
        with torch.no_grad():
            for name, v in T_net.state_dict().items():
                if not v.dtype.is_floating_point:
                    continue
                if name not in _ema:
                    _ema[name] = v.detach().clone()
                else:
                    _ema[name].mul_(EMA_DECAY).add_(v.detach(), alpha=1 - EMA_DECAY)

    def _ema_swap_in(T_net):
        sd = T_net.state_dict()
        backup = {n: sd[n].detach().clone() for n in _ema}
        sd.update(_ema)
        T_net.load_state_dict(sd)
        return backup

    def _ema_restore(T_net, backup):
        sd = T_net.state_dict()
        sd.update(backup)
        T_net.load_state_dict(sd)

    def _pair(v):
        """Widths and depths are per-network ([critic, map]); accept a scalar too."""
        return tuple(v) if isinstance(v, (list, tuple)) else (v, v)

    def _warm_start_path():
        """
        Path of the checkpoint for THIS problem and THIS architecture.

        The readable part of the name carries what determines the shape of the
        two state dicts -- tag, dimensions, widths, depths -- so a stale file
        can be identified by eye. The trailing digest covers what leaves the
        shapes alone but changes what the trained weights MEAN: the noise
        levels the map was fitted at, the normalisation, and
        the two architectural switches that alter the parameterisation without
        altering the parameter count. A mismatch in the first group would fail
        loudly at load_state_dict; a mismatch in the second would not, which is
        exactly why it is hashed into the filename instead.

        The ensemble size is deliberately NOT in the key. The networks have the
        same shape at any N -- only the training data drawn from the ensemble
        changes -- so one checkpoint serves a whole particle sweep instead of
        forcing a fresh spin-up at every point. N is recorded in meta, so the
        size a checkpoint was fitted at stays recoverable.
        """
        nn_c, nn_m = _pair(NUM_NEURON)
        rb_c, rb_m = _pair(num_resblocks)
        semantics  = {
            'sigma':         np.asarray(sigma, dtype=float).ravel().tolist(),
            'gamma':         np.asarray(gamma, dtype=float).ravel().tolist(),
            'normalization': str(normalization),
            'T_RESIDUAL':    bool(T_RESIDUAL),
            'ACT':           str(ACT),
            'USE_LAYERNORM': bool(USE_LAYERNORM),
        }
        digest = hashlib.sha1(
            json.dumps(semantics, sort_keys=True).encode('utf-8')).hexdigest()[:8]
        name   = 'otf_%s_L%d_dy%d_nn%d-%d_rb%d-%d_%s.pt' % (
            WARM_START_TAG, L, dy, int(nn_c), int(nn_m), int(rb_c), int(rb_m), digest)
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

    def _warm_start_save(path, f_net, T_net):
        """
        Write both state dicts, plus the provenance needed to interpret them.

        Saved through a process-unique temporary file and os.replace so that a
        reader never sees a half-written checkpoint: the sweeps dispatch several
        filters concurrently, and several runs may share the directory. Two
        processes missing the cache at once both train and both write; the last
        writer wins, which is harmless because the two files are interchangeable.
        Tensors go to the host so the file is loadable on any device.
        """
        meta = {
            'tag':            WARM_START_TAG,
            'L':              int(L),
            'dy':             int(dy),
            'N':              int(N),
            'sigma':          np.asarray(sigma, dtype=float).ravel().tolist(),
            'gamma':          np.asarray(gamma, dtype=float).ravel().tolist(),
            'normalization':  str(normalization),
            'T_RESIDUAL':     bool(T_RESIDUAL),
            'ACT':            str(ACT),
            'USE_LAYERNORM':  bool(USE_LAYERNORM),
            'ZERO_INIT_OUT':  bool(ZERO_INIT_OUT),
            'ITERATION':      int(ITERATION),
            'K_in':           int(K_in),
            'source':         'simulation 0, analysis step 0',
        }
        payload = {
            'f':    {k: v.detach().cpu() for k, v in f_net.state_dict().items()},
            'T':    {k: v.detach().cpu() for k, v in T_net.state_dict().items()},
            'ema':  {k: v.detach().cpu() for k, v in _ema.items()},
            'meta': meta,
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp = '%s.tmp%d' % (path, os.getpid())
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _warm_start_load(path, f_net, T_net):
        """
        Load a checkpoint into both networks, or return None on any failure.

        Every failure mode (absent file, unreadable file, shape mismatch) is a
        cache miss and is reported, never raised: a missing warm start costs
        iterations, not correctness. Both state dicts are checked for
        compatibility BEFORE either is applied, so a map that does not fit
        cannot leave a warm critic paired with a random map, and a rejected
        checkpoint leaves both networks exactly as they were.
        """
        if not os.path.exists(path):
            return None
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
            if not (_state_dict_compatible(ckpt['f'], f_net)
                    and _state_dict_compatible(ckpt['T'], T_net)):
                raise ValueError('checkpoint does not fit this architecture')
            f_net.load_state_dict(ckpt['f'])
            T_net.load_state_dict(ckpt['T'])
            if EMA_DECAY > 0 and ckpt.get('ema'):
                _ema.clear()
                _ema.update({k: v.to(device) for k, v in ckpt['ema'].items()})
            return ckpt.get('meta', {})
        except Exception as exc:
            # Under 'require' a checkpoint that does not fit is exactly the
            # failure the mode exists to surface, so it propagates instead of
            # degrading to a cold start.
            if WARM_START == 'require':
                raise
            print("[OTF] warm start: ignoring %s (%s: %s)"
                  % (path, type(exc).__name__, exc))
            return None

    def train(f, T_net, X_Train, Y_Train, iterations, learning_rate, ts, Ts, batch_size, k, K, K_in,
              draw_fn=None):
        f.train()
        T_net.train()

        if OPTIMIZER == 'adamw':
            optimizer_f = torch.optim.AdamW(f.parameters(), lr=learning_rate[0],
                                            weight_decay=WEIGHT_DECAY)
            optimizer_T = torch.optim.AdamW(T_net.parameters(), lr=learning_rate[1],
                                            weight_decay=WEIGHT_DECAY)
        else:
            optimizer_f = torch.optim.Adam(f.parameters(), lr=learning_rate[0])
            optimizer_T = torch.optim.Adam(T_net.parameters(), lr=learning_rate[1])

        # Configure learning rate schedulers using cosine annealing with warm
        # restarts: fixed base period T_0=512 with T_mult=2, so each
        # successive cycle doubles in length (512, 1024, 2048, ...), matching
        # an ITERATION budget of T_0*(2^k - 1) for k complete cycles.
        if SCHEDULE == 'warm_restarts':
            T_0 = 512
            scheduler_f = CosineAnnealingWarmRestarts(optimizer_f, T_0=T_0, T_mult=2,
                                                      eta_min=learning_rate[0] * 1e-3)
            scheduler_T = CosineAnnealingWarmRestarts(optimizer_T, T_0=T_0, T_mult=2,
                                                      eta_min=learning_rate[1] * 1e-3)
        elif SCHEDULE == 'cosine':
            scheduler_f = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_f, T_max=max(1, iterations), eta_min=learning_rate[0] * 1e-3)
            scheduler_T = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_T, T_max=max(1, iterations), eta_min=learning_rate[1] * 1e-3)
        else:
            scheduler_f = scheduler_T = None
        n_warm = int(WARMUP_FRAC * iterations)

        inner_iterations = K_in
        Y_Train_shuffled = Y_Train[torch.randperm(Y_Train.shape[0], device=Y_Train.device)].view(Y_Train.shape)
        for i in range(iterations):
            if n_warm and i < n_warm:
                for g in optimizer_f.param_groups:
                    g['lr'] = learning_rate[0] * (i + 1) / n_warm
                for g in optimizer_T.param_groups:
                    g['lr'] = learning_rate[1] * (i + 1) / n_warm

            if draw_fn is not None:
                # Fresh forecast particles and their simulated observations,
                # drawn from the same laws as the fixed set.
                X_train, Y_train = draw_fn(batch_size)
            else:
                idx = torch.randperm(X_Train.shape[0], device=X_Train.device)[:batch_size]
                X_train = X_Train[idx].clone().detach()
                Y_train = Y_Train[idx].clone().detach()

            # Randomly shuffle Y_train for training the transport mapping.
            # This is the independent coupling (X, Y_shuffled) ~ P_X (x) P_Y.
            Y_shuffled = Y_train[torch.randperm(Y_train.shape[0], device=Y_train.device)].view(Y_train.shape)
            for j in range(inner_iterations):
                map_T = T_net.forward(X_train, Y_shuffled)
                f_of_map_T = f.forward(map_T, Y_shuffled)
                loss_T = - f_of_map_T.mean() + 0.5 * ((X_train - map_T) ** 2).sum(axis=1).mean()

                optimizer_T.zero_grad()
                loss_T.backward()
                if GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(T_net.parameters(), GRAD_CLIP)
                optimizer_T.step()

            f_of_xy = f.forward(X_train, Y_train)
            map_T = T_net.forward(X_train, Y_shuffled)
            f_of_map_T = f.forward(map_T, Y_shuffled)
            loss_f = - f_of_xy.mean() + f_of_map_T.mean()

            if CRITIC_GP > 0:
                # WGAN-GP: penalise ||grad_x f|| away from 1 on the segment
                # between the joint samples and the pushforward. The max-min OT
                # dual is only meaningful for f in a bounded family, which the
                # default configuration does not enforce.
                eps_gp = torch.rand(X_train.shape[0], 1, device=X_train.device)
                x_int = (eps_gp * X_train + (1 - eps_gp) * map_T).detach().requires_grad_(True)
                f_int = f.forward(x_int, Y_train)
                g = torch.autograd.grad(f_int.sum(), x_int, create_graph=True)[0]
                loss_f = loss_f + CRITIC_GP * ((g.norm(2, dim=1) - 1.0) ** 2).mean()

            optimizer_f.zero_grad()
            loss_f.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(f.parameters(), GRAD_CLIP)
            optimizer_f.step()

            if EMA_DECAY > 0:
                _ema_update(T_net)

            if scheduler_f is not None and not (n_warm and i < n_warm):
                scheduler_f.step()
                scheduler_T.step()

            if (i + 1) == iterations:
                f.eval()
                T_net.eval()
                with torch.no_grad():
                    f_of_xy = f.forward(X_Train, Y_Train)
                    map_T = T_net.forward(X_Train, Y_Train_shuffled)
                    f_of_map_T = f.forward(map_T, Y_Train_shuffled)
                    loss_f = f_of_xy.mean() - f_of_map_T.mean()
                    loss = loss_f + 0.5 * ((X_Train - map_T) ** 2).sum(axis=1).mean()
                    print("Simu#%d/%d, Time Step:%d/%d, Iteration: %d/%d, OTF loss = %.4f" %
                          (k + 1, K, ts, Ts - 1, i + 1, iterations, loss.item()))

                    # A NaN/inf loss means the run has diverged and cannot
                    # recover, so abandon it rather than train the remaining
                    # analysis steps.  Tested here because loss is already
                    # computed and already synced to the host for the log
                    # above — no extra pass, no extra device transfer.
                    if not torch.isfinite(loss):
                        raise _NonFiniteRun(
                            "training loss became %s at simulation %d, "
                            "analysis step %d" % (loss.item(), k + 1, ts))

    # Per-analysis-step wall-clock, (NUM_SIM x T-1). Column 0 is the spin-up:
    # it trains from a random initialisation for ITERATION iterations, while
    # every later step refines the warm network for Final_Number_ITERATION, so
    # the two costs are not comparable and averaging over them hides both.
    # Preallocated rather than appended so that an aborted run leaves NaN in
    # the steps it never reached instead of a ragged result. Only filled when
    # a caller asks for it, so the tuning sweeps pay no synchronisation cost.
    _timing    = timing_out is not None
    step_times = np.full((NUM_SIM, T - 1), np.nan)

    # Availability is snapshotted HERE, before any training. Without that, the
    # run BUILDING the cache would warm-start its own later simulations from the
    # checkpoint its first simulation just wrote, and would be neither the cold
    # baseline nor a warm run.
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
    # (NUM_SIM x T x N x L): one slot per simulation, time step, particle, state.
    X_OTF = torch.zeros((NUM_SIM, T, N, L), device=device, dtype=torch.float32)

    # Abort early via _NonFiniteRun rather than finishing a run that has
    # already diverged: a NaN loss cannot recover, and for a tuning sweep the
    # remaining analysis steps are pure waste. The test lives at the terminal
    # loss log, so it costs nothing — the loss is already computed and synced
    # there — but the failing step pays its full iteration budget first.
    # Whatever was computed before the failure is kept; everything from the
    # failing step onward is NaN. Mirrors SIF.py and FMF.py.
    aborted = None
    try:
        for k in range(NUM_SIM):
            y = Y[k]

            X_OTF[k, 0] = torch.from_numpy(X0[k].T).to(torch.float32).to(device)

            ITERS = ITERATION
            LR = LearningRate

            f     = f_NeuralNet(INPUT_DIM, NUM_NEURON[0], num_resblocks[0])
            MAP_T = T_NeuralNet(INPUT_DIM, NUM_NEURON[1], num_resblocks[1])

            f.to(device)
            MAP_T.to(device)

            # Every simulation reads the same file, so on a hit none pays the
            # ITERATION spin-up -- and none differs from the others in its
            # starting weights either.
            _warm_meta = _warm_start_load(_warm_path, f, MAP_T) if _warm_available else None
            if _warm_meta is None:
                f.apply(init_weights)
                MAP_T.apply(init_weights)
                if ZERO_INIT_OUT:
                    nn.init.zeros_(MAP_T.layer_out.weight)
                    nn.init.zeros_(f.layer_out.weight)
            else:
                # Step 0 is charged the ordinary online budget instead.
                ITERS = Final_Number_ITERATION
                print("[OTF] Simu#%d/%d: warm start from %s (spin-up %d -> %d iterations)"
                      % (k + 1, NUM_SIM, _warm_path, ITERATION, ITERS))
            for i in range(T - 1):
                if COLD_START:
                    f.apply(init_weights)
                    MAP_T.apply(init_weights)
                _t_step = sync_clock(device) if _timing else 0.0
                x_noise = torch.distributions.MultivariateNormal(
                    torch.zeros(L, device=device), covariance_matrix=torch.eye(L, device=device))
                # Deterministic part of the forecast, kept so RESAMPLE_DATA can
                # redraw the process noise without re-propagating.
                AX = A(X_OTF[k, i].T, t[i]).T.to(device)
                X1 = AX + sigma * x_noise.sample((N,))

                y_noise = torch.distributions.MultivariateNormal(
                    torch.zeros(dy, device=device), covariance_matrix=torch.eye(dy, device=device))
                Y1 = h(X1.T).T.to(device) + gamma * y_noise.sample((N,))

                if normalization == 'Standard':
                    scaler_X = StandardScaler()
                    scaler_Y = StandardScaler()

                    X1 = torch.tensor(scaler_X.fit_transform(X1.cpu()), device=device, dtype=torch.float32)
                    Y1 = torch.tensor(scaler_Y.fit_transform(Y1.cpu()), device=device, dtype=torch.float32)

                elif normalization == 'MinMax':
                    scaler_X = MinMaxScaler()
                    scaler_Y = MinMaxScaler()

                    X1 = torch.tensor(scaler_X.fit_transform(X1.cpu()), device=device, dtype=torch.float32)
                    Y1 = torch.tensor(scaler_Y.fit_transform(Y1.cpu()), device=device, dtype=torch.float32)

                if RESAMPLE_DATA:
                    if normalization != 'None':
                        raise ValueError(
                            "RESAMPLE_DATA requires normalization='None'; the "
                            "redrawn pairs are in raw units and would not match "
                            "a scaler fitted to the fixed ensemble.")

                    def _draw(bs):
                        j = torch.randint(0, N, (bs,), device=device)
                        xb = AX[j] + sigma * torch.randn(bs, L, device=device)
                        yb = h(xb.T).T + gamma * torch.randn(bs, dy, device=device)
                        return xb, yb
                else:
                    _draw = None

                train(f, MAP_T, X1, Y1, ITERS, LR, i + 1, T, BATCH_SIZE, k, NUM_SIM, K_in,
                      draw_fn=_draw)

                # Saved after train() returns, so a diverged run
                # (_NonFiniteRun) never reaches this line and cannot poison the
                # cache.
                if _warm_save_pending and k == 0 and i == 0:
                    _warm_start_save(_warm_path, f, MAP_T)
                    _warm_save_pending = False
                    print("[OTF] warm start: saved %s" % _warm_path)
                ITERS = Final_Number_ITERATION

                Y1_true = y[i + 1, :]                         # (dy x 1)
                Y1_true = Y1_true.repeat(N, 1).T              # broadcast to all N particles: (N x dy)
                if normalization in ('Standard', 'MinMax'):
                    Y1_true = scaler_Y.transform(Y1_true)

                Y1_true = torch.from_numpy(Y1_true).to(torch.float32).to(device)

                _bk = _ema_swap_in(MAP_T) if (EMA_DECAY > 0 and _ema) else None
                X_mapped = MAP_T.forward(X1, Y1_true)
                if _bk is not None:
                    _ema_restore(MAP_T, _bk)

                if normalization in ('Standard', 'MinMax'):
                    X_mapped = torch.tensor(scaler_X.inverse_transform(X_mapped.cpu().detach().numpy()))
                X_OTF[k, i + 1] = X_mapped.detach()
                if _timing: step_times[k, i] = sync_clock(device) - _t_step

    except _NonFiniteRun as exc:
        aborted = exc
        # Remainder of the failing simulation, then every simulation after it.
        X_OTF[k, i + 1:] = float('nan')
        X_OTF[k + 1:]    = float('nan')
        print("[OTF] ABORT: %s" % exc)
        print("[OTF] returning NaN from simulation %d, analysis step %d onward "
              "(%d of %d analysis steps completed)"
              % (k + 1, i + 1, k * (T - 1) + i, NUM_SIM * (T - 1)))

    print("--- OTF time : %s seconds ---%s"
          % (time.time() - start_time, " (ABORTED)" if aborted else ""))

    # Hand the per-step breakdown back through the caller's dict. The total is
    # kept alongside it because it also covers the per-simulation network
    # construction, which sits outside the step loop -- step_times.sum() is
    # deliberately smaller than the total, and the gap is worth seeing.
    if _timing:
        timing_out['step_times'] = step_times
        timing_out['total']      = time.time() - start_time

    # Rearrange the dimensions of the output to (NUM_SIM x T x L x N) for convenient plotting.
    return X_OTF.cpu().numpy().transpose(0, 1, 3, 2)
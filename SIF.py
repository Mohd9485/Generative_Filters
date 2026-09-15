"""
@author: Mohammad Al-Jarrah

Stochastic Interpolant Filter (SIF) — a nonlinear particle filter based on
conditional stochastic interpolants.

Filtering context.  For the hidden Markov model  X_t ~ a(·|X_{t-1}),
Y_t ~ h(·|X_t), the posterior obeys the two-step recursion

        π_{t|t-1} = A[π_{t-1}]        (forecast)
        π_t       = B_{Y_t}[π_{t|t-1}]  (analysis / Bayes)

and SIF replaces the analysis operator B_y by a learned transport map
T : R^n x R^m -> R^n satisfying the consistency condition

        T(·, y)_# π_{t|t-1} = B_y[π_{t|t-1}]     for all y,        (Eq. 3)

equivalently, in the joint form that is actually trainable,

        ( T(X̄, Y), Y ) ~ P_{X,Y},    (X̄, Y) ~ P_X ⊗ P_Y,          (Eq. 4)

where X is a forecast particle, Y ~ h(·|X) its simulated observation, and
X̄ an independent copy of the state — realized on the ensemble by permuting
the forecast particles relative to their simulated observations.  Particles
are then updated by  X_t^i = T(X_{t|t-1}^i, Y_t), giving a uniformly
weighted posterior ensemble with no importance weights and no likelihood
evaluations.

SIF generalizes the Flow Matching Filter (FMF): it interpolates between
prior samples and joint samples along a NOISY linear bridge

        X_τ = (1 - τ) X̄  +  τ X  +  γ(τ) Z,     Z ~ N(0, I_n)     (Eq. 6)

with  γ(τ) = ε · sqrt( τ (1-τ) )  (a Brownian-bridge schedule, zero at
both endpoints) and τ ∈ [0,1] the interpolation ("artificial") time, which
is distinct from the filtering time t.  The bridge runs from the
independent source X̄ at τ = 0 to the paired target X at τ = 1.  Two fields
are learned from the same trunk network:

        b_θ(x,τ,y) = E[ ∂_τ X_τ | X_τ = x, Y = y ]  — velocity (probability flow)
        η_θ(x,τ,y) = E[    Z    | X_τ = x, Y = y ]  — denoiser (score = -η_θ/γ)

trained jointly by the mean-squared regression

    min_θ E[ ||b_θ(X_τ,τ,Y) - ∂_τ X_τ||²  +  λ ||η_θ(X_τ,τ,Y) - Z||² ],  (Eq. 7)

        ∂_τ X_τ = (X - X̄) + γ̇(τ) Z,        τ ~ Unif[0,1],

in which λ ≥ 0 (`DENOISER_WEIGHT` below) weights the denoiser loss relative
to the velocity loss.  λ only matters when ε > 0: the score -η_θ/γ enters
inference through the SDE sampler alone, so the η head is dead weight for
the deterministic ODE path and is dropped entirely when ε = 0.

At inference the user may choose deterministic (ODE) or stochastic (SDE)
sampling.  The SDE is marginal-preserving: at every τ it samples from
the same conditional marginal p_τ(· | y) as the ODE.  It is stochastic at
inference and is observed to be advantageous for multimodal posteriors.

Key limits:
    ε = 0          ── SIF reduces to FMF behaviorally (interpolant is noiseless):
                      the bridge (Eq. 6) becomes linear, the denoiser head is
                      dropped (λ = 0), and (Eq. 7) collapses to the conditional
                      flow-matching regression  E||b_θ(X_τ,τ,Y) - (X - X̄)||²
    ε > 0, 'ode'   ── FMF-like transport but with noise-regularized training
    ε > 0, 'sde'   ── diffusion-style stochastic sampling conditional on y

Reference:
    Albergo, M. S., and Vanden-Eijnden, E., "Building Normalizing Flows with
    Stochastic Interpolants", ICLR, 2023.

    Albergo, M. S., Boffi, N. M., and Vanden-Eijnden, E.,
    "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions",
    2023.

    Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M.,
    "Flow Matching for Generative Modeling", ICLR, 2023.  (ε = 0 limit)

The block-triangular structure is preserved: y is purely a conditioning
input — it is never interpolated, diffused, or moved.  Equivalently, the
augmented map S(x,y) = (T(x,y), y) holds the observation coordinate fixed,
so its first block transports the forecast to the conditional
P_{X|Y}(· | y).
"""

import numpy as np
import math
import time
import os
import json
import hashlib
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
sys.path.insert(0, '/home/mohd9485/Tutorial_project_dynamics')
from timing_utils import sync_clock


class _NonFiniteRun(Exception):
    """
    Raised internally when a run has provably failed and finishing it would
    only burn compute: the training loss went NaN/inf, or the mapped particles
    did.  Caught in the main loop, which fills the remainder of the output
    with NaN and returns early.  Not part of the public interface — callers
    see the NaN particles, not this exception.
    """


def SIF(Y, X0, A, h, t, Noise, parameters, device=None, timing_out=None,
        warm_start='off', warm_start_tag='', warm_start_dir='DATA/warm_start'):
    """
    Stochastic Interpolant Filter (SIF).

    Inputs — identical to OTF / FMF
    --------------------------------
    Y        : True observations,          shape (NUM_SIM x T x dy x 1)
    X0       : Initial particles,          shape (NUM_SIM x L x N)
    A        : Deterministic dynamic model (without noise)
    h        : Deterministic observation model (without noise)
    t        : Time vector  [0, dt, 2dt, ..., tf]  — filtering time, not the
               interpolation time τ ∈ [0,1] of the bridge (Eq. 6)
    Noise    : [sigma_proc, sigma_obs] — process / observation noise std
                                         (σ and γ of the state-space model)
    parameters : dict with keys
        'normalization'          : 'None' | 'Standard' | 'MinMax'
        'INPUT_DIM'              : [L, dy]
        'NUM_NEURON'             : hidden width
        'BATCH_SIZE'             : mini-batch size
        'LearningRate'           : scalar learning rate
        'ITERATION'              : initial training iterations per filter step
        'Final_Number_ITERATION' : floor for iteration-halving schedule
        'num_resblocks'          : int, number of residual blocks (shared
                                   residual-network backbone, common to all
                                   the neural transport filters)
        'ODE_STEPS'              : integrator steps used at inference
                                   (10–50 forward-Euler steps in the paper)
        The interpolation time τ enters the network through a sinusoidal
        encoding rather than as a raw scalar.  Writing D = TIME_EMBED_DIM,
        S = TIME_SCALE and P = TIME_MAX_PERIOD, the scalar τ is replaced by

            [ cos(ω_k S τ), sin(ω_k S τ) ]_{k < D/2},   ω_k = P^(-2k/D),

        the transformer positional encoding of Vaswani et al. (2017)
        evaluated at a continuous position, as used for the diffusion
        timestep by Ho et al. (2020).  The frequency ladder spans
        ω ∈ [S/P, S] rad, and the three keys are best read as choosing that
        BAND rather than as three independent knobs:

            S   sets the top of the band  — the finest structure in τ the
                network can represent
            S/P sets the bottom           — the coarsest
            D   sets how densely the band is sampled

        'TIME_EMBED_DIM'         : (optional, default 32) D above; must be
                                   even.  Set to 1 to recover the raw scalar τ
                                   this filter used previously.  Widening D
                                   past the point where the band is resolved
                                   buys little: with the defaults below the
                                   encoding already has effective rank ≈ D,
                                   so every channel is doing work and extra
                                   ones are near-duplicates.  Note SIF
                                   CONCATENATES the encoding into layer_input,
                                   so D also sets how much of the input the
                                   trunk sees is time rather than state: at
                                   n = 5 (L = 10, dy = 5), D = 32 is 68% of
                                   the input width and D = 128 would be 90%.
        'TIME_SCALE'             : (optional, default 300.0) S above.  This
                                   rescaling is REQUIRED, not cosmetic: at
                                   S = 1 every ω_k S τ ≤ 1 rad, so cos ≈ 1 and
                                   sin ≈ ω_k τ, the D channels collapse to a
                                   constant plus a scaled copy of τ, and the
                                   encoding is no richer than the scalar it
                                   replaced.  The default puts the fastest
                                   channel at ≈48 cycles over τ ∈ [0,1], i.e.
                                   one cycle per ~2 steps of the ODE_STEPS
                                   integration grid — the finest structure
                                   that grid can actually use.  Larger S
                                   (DDPM's 1000 gives ≈159 cycles) buys
                                   resolution the integrator cannot see.
        'TIME_MAX_PERIOD'        : (optional, default 50.0) P above; must
                                   be > 1.  SBF.py and the image-diffusion
                                   codebases use P = 10000 because their τ is
                                   an INTEGER index running to the hundreds
                                   or thousands.  Here τ ∈ [0,1], so P = 10000
                                   would put the bottom of the ladder at
                                   S/10000 rad — flat across the whole
                                   interval for any sane S — and roughly half
                                   the channels would be constants.  P = 50
                                   with S = 300 spans ω ∈ [6, 300]: one cycle
                                   across the interval at the low end, one per
                                   ~2 integrator steps at the high end.

        # SIF-specific
        'NOISE_LEVEL'            : ε ≥ 0 in γ(τ) = ε sqrt(τ(1-τ)), Eq. (6) —
                                   the noise level of the interpolant.
                                   0.0  → FMF-equivalent (noiseless interpolant)
                                   >0.0 → stochastic interpolant
        'INFERENCE_MODE'         : 'ode' | 'sde'
                                   'ode' = probability-flow, deterministic
                                   'sde' = Euler-Maruyama, marginal-preserving
        'DENOISER_WEIGHT'        : λ ≥ 0 in Eq. (7) — the weight of the
                                   denoiser loss ||η_θ(X_τ,τ,Y) - Z||²
                                   relative to the velocity loss
                                   ||b_θ(X_τ,τ,Y) - ∂_τ X_τ||².
                                   λ = 0 gives velocity-only training (pure
                                   flow matching on a noisy bridge); λ > 0
                                   additionally shapes the shared trunk.
                                   Inert when ε = 0, and REQUIRED to be > 0
                                   when INFERENCE_MODE='sde', since the score
                                   -η_θ/γ is consumed by the SDE sampler.

    The admissible (ε, λ, mode) combinations, and the one that is rejected,
    are enumerated in the validation block below.

    timing_out : Optional dict. When supplied it is filled with
        'step_times' — a (NUM_SIM x T-1) array of per-analysis-step wall-clock
        seconds — and 'total', the whole-run time. Column 0 is the spin-up
        step, which trains from a random initialisation and can be done
        offline; the remaining columns are the online per-step cost. Left as
        None (the default) the timing is skipped entirely, so the tuning
        sweeps pay nothing for it.

    Output
    ------
    X_SIF : ndarray, shape (NUM_SIM x T x L x N)
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
    normalization          = parameters['normalization']
    INPUT_DIM              = parameters['INPUT_DIM']
    NUM_NEURON             = parameters['NUM_NEURON']
    BATCH_SIZE             = parameters['BATCH_SIZE']
    LearningRate           = parameters['LearningRate']
    ITERATION              = parameters['ITERATION']
    Final_Number_ITERATION = parameters['Final_Number_ITERATION']
    num_resblocks          = parameters['num_resblocks']
    ODE_STEPS              = parameters['ODE_STEPS']

    # Sinusoidal encoding of τ — the ladder spans ω ∈ [S/P, S] rad, see above.
    # All three default, so existing callers need no change.
    TIME_EMBED_DIM         = int(parameters.get('TIME_EMBED_DIM', 30))
    TIME_SCALE             = float(parameters.get('TIME_SCALE', 300.0))
    TIME_MAX_PERIOD        = float(parameters.get('TIME_MAX_PERIOD', 50.0))

    # SIF-specific
    NOISE_LEVEL            = parameters['NOISE_LEVEL']
    INFERENCE_MODE         = parameters['INFERENCE_MODE']
    DENOISER_WEIGHT        = parameters['DENOISER_WEIGHT']

    # ------------------------------------------------------------------
    # Improvements carried over from the FMF study. All four are
    # training/inference mechanics: the interpolant of Eq. (6), the two-head
    # loss of Eq. (7), the source/target coupling and the samplers are
    # untouched, so the block-triangular structure — independent coupling
    # P_X (x) P_Y transported onto the joint at a FIXED y — is exactly SIF's.
    # Every switch below defaults to the original algorithm's behaviour, so a
    # call that omits these keys reproduces it exactly.
    #
    #   SCHEDULE      'warm_restarts' (the default) | 'cosine' | 'none'.
    #       The default uses CosineAnnealingWarmRestarts(T_0=512, T_mult=2)
    #       stepped EVERY iteration, so the learning rate is kicked back to
    #       full value at iterations 512, 1536, 3584, ... In the FMF study
    #       that schedule drove a warm-started network to NaN in every run
    #       with a budget above ~2048 iterations/step, and replacing it was
    #       worth more than any other single change (0.7389 -> 0.4877) while
    #       collapsing the seed-to-seed spread from +/-0.14 to +/-0.001.
    #   WARMUP_FRAC / GRAD_CLIP   stability, needed at the larger budgets.
    #   RESAMPLE_DATA  redraw (X, Y) from the predictive law every iteration
    #       instead of reusing one fixed set of N pairs. Same law, fresh
    #       realisations; without it, high budgets memorise and diverge.
    #   COLD_START    False warm-starts the net across analysis steps, as OTF
    #       does. The default re-initialises every step (its
    #       `net.apply(init_weights)` inside the step loop), which the FMF
    #       study found costs ~0.27.
    # ------------------------------------------------------------------
    SCHEDULE      = parameters.get('SCHEDULE', 'warm_restarts')
    WARMUP_FRAC   = float(parameters.get('WARMUP_FRAC', 0.0))
    GRAD_CLIP     = float(parameters.get('GRAD_CLIP', 0.0))
    RESAMPLE_DATA = bool(parameters.get('RESAMPLE_DATA', False))
    COLD_START    = bool(parameters.get('COLD_START', True))
    #   SOLVER  'euler' (the default forward Euler / Euler-Maruyama) | 'heun'
    #       Second order on both inference paths — see ode_integrate and
    #       sde_integrate below.
    SOLVER        = parameters.get('SOLVER', 'euler')
    #   The stabiliser set that made FMF's largest budget usable. There, the
    #   8192-iteration configuration scored well but carried particles at 1e9
    #   and went NaN at NUM_SIM = 10; LayerNorm + zero-init heads + AdamW
    #   weight decay fixed it AND improved accuracy (0.4092 -> 0.3713 at
    #   NUM_SIM = 10). Whether SIF needs them is an open question — its 8192
    #   runs completed — so they default to off.
    #   USE_LAYERNORM  LayerNorm inside each residual block. In FMF this was the
    #       stabiliser that actually mattered, because it bounds activations
    #       throughout training rather than only at initialisation.
    #   ZERO_INIT_OUT  zero BOTH output heads, so the velocity field starts at
    #       b = 0 (identity transport) and the denoiser at eta = 0. In FMF this
    #       was harmful ALONE but worth ~0.09 once LayerNorm was present.
    #   OPTIMIZER / WEIGHT_DECAY   'adam' (the default) | 'adamw' + decay.
    USE_LAYERNORM = bool(parameters.get('USE_LAYERNORM', False))
    ZERO_INIT_OUT = bool(parameters.get('ZERO_INIT_OUT', False))
    OPTIMIZER     = parameters.get('OPTIMIZER', 'adam')
    WEIGHT_DECAY  = float(parameters.get('WEIGHT_DECAY', 0.0))

    # ------------------------------------------------------------------
    # Warm-start cache for the interpolant network. Off by default, so a call
    # that omits these arguments reproduces the original algorithm exactly.
    #
    #   warm_start      'off' | 'auto' | 'save' | 'load' | 'require'
    #   warm_start_tag  REQUIRED unless 'off'. Names the PROBLEM, which SIF
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
                "warm_start=%r contradicts COLD_START=True, which re-initialises "
                "the network at every analysis step and so discards the loaded "
                "weights after the spin-up." % warm_start)

    # ------------------------------------------------------------------
    # Configuration validation
    #
    # The (ε, λ, mode) grid collapses to four distinct methods, forming an
    # ablation ladder in which each rung adds exactly one ingredient:
    #
    #   ε = 0 | any λ | any mode ─► FMF        — noiseless bridge, η head
    #                                            dropped, ODE only; λ and
    #                                            INFERENCE_MODE are inert
    #   ε > 0 | λ = 0 | 'ode'    ─► SIF (ODE)  — noisy bridge, velocity-only
    #                                            training  [+ interpolant noise]
    #   ε > 0 | λ > 0 | 'ode'    ─► SIF (ODE)  — + auxiliary denoiser loss on
    #                                            the shared trunk  [+ η loss]
    #   ε > 0 | λ > 0 | 'sde'    ─► SIF (SDE)  — full stochastic interpolant
    #                                            [+ stochastic inference]
    #
    # The remaining cell is rejected below:
    #
    #   ε > 0 | λ = 0 | 'sde'    ─► INVALID    — the SDE drift consumes the
    #                                            score -η_θ/γ, but λ = 0 gives
    #                                            the η head zero gradient, so
    #                                            it stays at its Xavier init
    #                                            and the injected score is
    #                                            meaningless (silent garbage).
    # ------------------------------------------------------------------
    if INFERENCE_MODE not in ('ode', 'sde'):
        raise ValueError("INFERENCE_MODE must be 'ode' or 'sde'.")
    if NOISE_LEVEL < 0.0:
        raise ValueError("NOISE_LEVEL (ε) must be >= 0.")
    if DENOISER_WEIGHT < 0.0:
        raise ValueError("DENOISER_WEIGHT (λ) must be >= 0.")
    if TIME_EMBED_DIM < 1:
        raise ValueError("TIME_EMBED_DIM must be >= 1.")
    if TIME_SCALE <= 0.0:
        raise ValueError("TIME_SCALE must be > 0.")
    if TIME_MAX_PERIOD <= 1.0:
        raise ValueError("TIME_MAX_PERIOD must be > 1 (it is the ratio between "
                         "the fastest and slowest channel of the frequency "
                         "ladder; at 1 every channel is identical).")
    if TIME_EMBED_DIM > 1 and TIME_EMBED_DIM % 2:
        raise ValueError("TIME_EMBED_DIM must be even (it is split into "
                         "cos/sin halves); got %d." % TIME_EMBED_DIM)

    if INFERENCE_MODE == 'sde' and NOISE_LEVEL > 0.0 and DENOISER_WEIGHT <= 0.0:
        raise ValueError(
            "Invalid SIF configuration: INFERENCE_MODE='sde' with "
            "DENOISER_WEIGHT=0. The SDE drift needs the score -η_θ/γ, but "
            "λ = 0 leaves the denoiser head untrained at its initialization. "
            "Set DENOISER_WEIGHT > 0, or use INFERENCE_MODE='ode'."
        )

    if INFERENCE_MODE == 'sde' and NOISE_LEVEL <= 0.0:
        print("[SIF] WARNING: INFERENCE_MODE='sde' with NOISE_LEVEL=0 "
              "degenerates to ODE — no noise will be injected.")
    if NOISE_LEVEL <= 0.0 and DENOISER_WEIGHT > 0.0:
        print("[SIF] WARNING: NOISE_LEVEL=0 makes DENOISER_WEIGHT inert — the "
              "denoiser head is dropped and the filter reduces to FMF.")

    # Resolved method, echoed once so the run is identifiable in the logs.
    if NOISE_LEVEL <= 0.0:
        METHOD = "FMF (ε=0)"
    elif INFERENCE_MODE == 'sde':
        METHOD = "SIF-SDE"
    elif DENOISER_WEIGHT > 0.0:
        METHOD = "SIF (ODE, with denoiser loss)"
    else:
        METHOD = "SIF-ODE"
    _temb = ("raw scalar τ" if TIME_EMBED_DIM <= 1
             else "sinusoidal τ, dim=%d, ω∈[%.3g, %.3g] rad"
                  % (TIME_EMBED_DIM, TIME_SCALE / TIME_MAX_PERIOD, TIME_SCALE))
    print("[SIF] method: %s  |  ε=%.4g, λ=%.4g, mode='%s'  |  %s"
          % (METHOD, NOISE_LEVEL, DENOISER_WEIGHT, INFERENCE_MODE, _temb))

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    # Honour an explicitly supplied `device`, otherwise default to CUDA when available.
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    # ------------------------------------------------------------------
    # Interpolant noise schedule  (γ of Eq. 6; τ is written `t` below)
    #    γ(τ)  = ε · sqrt( τ (1-τ) )
    #    γ'(τ) = ε · (1 - 2τ) / ( 2 sqrt( τ (1-τ) ) )
    #
    # Brownian-bridge schedule: γ(0) = γ(1) = 0, so the interpolant hits
    # X̄ and X exactly at the endpoints and only the interior is noised.
    #
    # The derivative diverges at τ ∈ {0, 1}; training-time τ is clipped
    # to [T_BOUND_EPS, 1 - T_BOUND_EPS] to avoid numerical issues.
    # ------------------------------------------------------------------
    T_BOUND_EPS = 1e-3

    def noise_schedule(t_tensor):
        return NOISE_LEVEL * torch.sqrt(t_tensor * (1.0 - t_tensor))

    def noise_schedule_dot(t_tensor):
        safe = torch.sqrt(t_tensor * (1.0 - t_tensor)) + 1e-8
        return NOISE_LEVEL * (1.0 - 2.0 * t_tensor) / (2.0 * safe)

    # ==================================================================
    # Sinusoidal encoding of the interpolation time τ
    # ==================================================================
    def timestep_embedding(t_scalar, dim, max_period):
        """
        Transformer positional encoding of a CONTINUOUS position.

        For k = 0 ... dim/2 - 1 and ω_k = max_period^(-2k/dim),

            emb(τ) = [ cos(ω_0 τ), ..., cos(ω_{D/2-1} τ),
                       sin(ω_0 τ), ..., sin(ω_{D/2-1} τ) ]

        i.e. the geometric frequency ladder of Vaswani et al. (2017), reused
        for the diffusion timestep by Ho et al. (2020).  The ladder spans
        ω ∈ [max_period^-1, 1] before the caller's TIME_SCALE, so it resolves
        τ at many scales at once instead of handing the trunk a single
        linear feature.

        Same construction as _timestep_embedding in SBF.py; the only
        difference is that max_period is a parameter here rather than pinned
        at 10000, because SIF's τ lives on [0,1] rather than on an integer
        index range (see TIME_MAX_PERIOD above).

        t_scalar : (B, 1) — ALREADY rescaled by TIME_SCALE, see caller.
        returns  : (B, dim)
        """
        half  = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t_scalar.device)
        args = t_scalar.float() * freqs[None]                  # (B, half)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # ==================================================================
    # Network definition — single trunk, two output heads
    # ==================================================================
    class ResidualBlock(nn.Module):
        def __init__(self, hidden_dim, activation):
            super().__init__()
            self.linear1    = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.linear2    = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.activation = activation
            self.norm       = nn.LayerNorm(hidden_dim) if USE_LAYERNORM else None

        def forward(self, x):
            identity = x
            out = self.linear1(x)
            if self.norm is not None:
                out = self.norm(out)
            out = self.activation(out)
            out = self.linear2(out)
            out = self.activation(out + identity)
            return out

    class InterpolantNet(nn.Module):
        """
        Shared trunk with two output heads (the "velocity + denoiser"
        learned object of Table I):

            b_head : velocity                   b_θ(x, τ, y) ∈ R^L
            η_head : conditional noise estimate η_θ(x, τ, y) ∈ R^L

        The score is obtained at inference by  score = -η_θ / γ(τ).

        The observation y enters only as a conditioning input concatenated
        to (x, τ) — never as a state to be transported — which is what makes
        the induced map block-triangular in the sense of Eq. (4).

        τ enters through a sinusoidal encoding of width time_embed_dim
        rather than as a raw scalar; layer_input then serves as the learned
        projection of that encoding, so no extra module is needed.
        """
        def __init__(self, state_dim, obs_dim, hidden_dim, num_resblocks,
                     time_embed_dim, time_scale, time_max_period):
            super().__init__()
            self.activation       = nn.SiLU()
            self.time_embed_dim   = time_embed_dim
            self.time_scale       = time_scale
            self.time_max_period  = time_max_period
            # x_t | emb(τ) | y   — time_embed_dim = 1 restores the raw scalar
            input_width          = state_dim + time_embed_dim + obs_dim

            self.layer_input   = nn.Linear(input_width, hidden_dim, bias=False)
            self.resblocks     = nn.ModuleList([
                ResidualBlock(hidden_dim, self.activation)
                for _ in range(num_resblocks)
            ])
            self.head_velocity = nn.Linear(hidden_dim, state_dim, bias=False)
            self.head_denoiser = nn.Linear(hidden_dim, state_dim, bias=False)

        def forward(self, x, t_scalar, y):
            if self.time_embed_dim > 1:
                # Rescale τ ∈ [0,1] onto the integer-index range the encoding
                # was designed for before embedding (see TIME_SCALE above).
                t_feat = timestep_embedding(t_scalar * self.time_scale,
                                            self.time_embed_dim,
                                            self.time_max_period)
            else:
                t_feat = t_scalar
            inp = torch.cat([x, t_feat, y], dim=1)
            feat = self.layer_input(inp)
            for block in self.resblocks:
                feat = block(feat)
            feat = self.activation(feat)
            return self.head_velocity(feat), self.head_denoiser(feat)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.1)

    # ==================================================================
    # Training — MSE regression on both heads
    # ==================================================================
    def _state_dict_compatible(sd_ckpt, module):
        """
        True when sd_ckpt can be loaded into module without any surprises.

        Checked BEFORE load_state_dict rather than relying on it to raise,
        because load_state_dict copies tensors as it walks the module and only
        raises at the end: a mismatch part way through would leave some layers
        loaded and some not. Verifying first means a rejected checkpoint applies
        NOTHING, so the fallback is the cold-start network the run would have had
        anyway -- identical down to the RNG stream, since no re-initialisation is
        needed to undo a partial load.
        """
        sd = module.state_dict()
        return (set(sd_ckpt) == set(sd)
                and all(sd_ckpt[k].shape == sd[k].shape for k in sd))

    def _warm_start_path():
        """
        Path of the checkpoint for THIS problem and THIS architecture.

        The readable part of the name carries what fixes the shape of the state
        dict -- variant, tag, dimensions, width, depth, time-embedding width --
        so a stale file can be identified by eye. The trailing digest covers
        what leaves those shapes alone but changes what the trained heads MEAN:
        the noise levels the interpolant was fitted at, the normalisation, the
        interpolant's own noise level eps and denoiser weight lambda, the time
        encoding's scale, and LayerNorm.

        INFERENCE_MODE is in the digest too, even though it changes only how the
        trained net is sampled from and not how it is trained. That is
        deliberate: it keeps the two SIF variants (and the third, unused
        DENOISER_WEIGHT>0 + INFERENCE_MODE='ode' combination this function can
        still be called with) on SEPARATE files, so none of them silently
        inherits another's spin-up, and it keeps the concurrently dispatched
        SIF jobs of a sweep off one path.

        The variant itself -- sif_ode, sif_sde, or the sif_ode_den fallback --
        leads the name in the clear, same priority order as the METHOD label
        above: 'sde' mode first, then whether the denoiser loss is active.

        The ensemble size is deliberately NOT in the key. The networks have the
        same shape at any N -- only the training data drawn from the ensemble
        changes -- so one checkpoint serves a whole particle sweep instead of
        forcing a fresh spin-up at every point. N is recorded in meta, so the
        size a checkpoint was fitted at stays recoverable.
        """
        semantics = {
            'sigma_proc':      np.asarray(sigma_proc, dtype=float).ravel().tolist(),
            'sigma_obs':       np.asarray(sigma_obs,  dtype=float).ravel().tolist(),
            'normalization':   str(normalization),
            'NOISE_LEVEL':     float(NOISE_LEVEL),
            'DENOISER_WEIGHT': float(DENOISER_WEIGHT),
            'INFERENCE_MODE':  str(INFERENCE_MODE),
            'TIME_SCALE':      float(TIME_SCALE),
            'TIME_MAX_PERIOD': float(TIME_MAX_PERIOD),
            'USE_LAYERNORM':   bool(USE_LAYERNORM),
        }
        digest = hashlib.sha1(
            json.dumps(semantics, sort_keys=True).encode('utf-8')).hexdigest()[:8]
        if INFERENCE_MODE == 'sde':
            variant = 'sif_sde'
        elif DENOISER_WEIGHT > 0.0:
            variant = 'sif_ode_den'
        else:
            variant = 'sif_ode'
        name = '%s_%s_L%d_dy%d_nn%d_rb%d_te%d_%s.pt' % (
            variant, WARM_START_TAG, L, dy, int(NUM_NEURON), int(num_resblocks),
            int(TIME_EMBED_DIM), digest)
        return os.path.join(WARM_START_DIR, name)

    def _warm_start_save(path, net):
        """
        Write the interpolant network's weights, plus the provenance needed to
        interpret them.

        Saved through a process-unique temporary file and os.replace so that a
        reader never sees a half-written checkpoint: the sweeps dispatch several
        filters concurrently -- two of them SIF -- and several runs may share
        the directory. Two processes missing the cache at once both train and
        both write; the last writer wins, which is harmless because the two
        files are interchangeable. Tensors go to the host so the file is
        loadable on any device.
        """
        meta = {
            'tag':             WARM_START_TAG,
            'L':               int(L),
            'dy':              int(dy),
            'N':               int(N),
            'sigma_proc':      np.asarray(sigma_proc, dtype=float).ravel().tolist(),
            'sigma_obs':       np.asarray(sigma_obs,  dtype=float).ravel().tolist(),
            'normalization':   str(normalization),
            'NUM_NEURON':      int(NUM_NEURON),
            'num_resblocks':   int(num_resblocks),
            'TIME_EMBED_DIM':  int(TIME_EMBED_DIM),
            'TIME_SCALE':      float(TIME_SCALE),
            'TIME_MAX_PERIOD': float(TIME_MAX_PERIOD),
            'NOISE_LEVEL':     float(NOISE_LEVEL),
            'DENOISER_WEIGHT': float(DENOISER_WEIGHT),
            'INFERENCE_MODE':  str(INFERENCE_MODE),
            'USE_LAYERNORM':   bool(USE_LAYERNORM),
            'ZERO_INIT_OUT':   bool(ZERO_INIT_OUT),
            'ITERATION':       int(ITERATION),
            'source':          'simulation 0, analysis step 0',
        }
        payload = {
            'net':  {k: v.detach().cpu() for k, v in net.state_dict().items()},
            'meta': meta,
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp = '%s.tmp%d' % (path, os.getpid())
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _warm_start_load(path, net):
        """
        Load a checkpoint into the network, or return None on any failure.

        Every failure mode (absent file, unreadable file, shape mismatch) is a
        cache miss and is reported, never raised: a missing warm start costs
        iterations, not correctness. Compatibility is checked before anything is
        applied, so a rejected checkpoint leaves the network exactly as it was.
        """
        if not os.path.exists(path):
            return None
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
            if not _state_dict_compatible(ckpt['net'], net):
                raise ValueError('checkpoint does not fit this architecture')
            net.load_state_dict(ckpt['net'])
            return ckpt.get('meta', {})
        except Exception as exc:
            # Under 'require' a checkpoint that does not fit is exactly the
            # failure the mode exists to surface, so it propagates instead of
            # degrading to a cold start.
            if WARM_START == 'require':
                raise
            print('[SIF] warm start: ignoring %s (%s: %s)'
                  % (path, type(exc).__name__, exc))
            return None

    def train_interpolant(net, X_source, X_target, Y_cond,
                          iterations, lr, ts, Ts, batch_size, k, K,
                          draw_fn=None):
        """
        Joint regression of Eq. (7), with x_0 = X̄ (independent source),
        x_1 = X (paired target), and τ ~ Unif[0,1]:

            b_target = ∂_τ X_τ = (x_1 - x_0) + γ'(τ) z
            η_target = z

            loss = ||b_θ - b_target||²  +  λ ||η_θ - η_target||²

        where λ = DENOISER_WEIGHT ≥ 0 balances the two heads.

        When NOISE_LEVEL = 0 the denoiser head's target would be undefined
        / unused at inference, so the η loss is dropped (λ = 0 effectively)
        and the network behaves identically to FMF's single-head velocity
        regression  E||b_θ(X_τ,τ,Y) - (X - X̄)||².
        """
        net.train()
        if OPTIMIZER == 'adamw':
            optimizer = torch.optim.AdamW(net.parameters(), lr=lr,
                                          weight_decay=WEIGHT_DECAY)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        if SCHEDULE == 'warm_restarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=512, T_mult=2,
                                                    eta_min=lr * 1e-3)
        elif SCHEDULE == 'cosine':
            # Monotone decay over the whole budget: no mid-training restart, so
            # a warm-started network is never kicked back to the full LR.
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, iterations), eta_min=lr * 1e-3)
        else:
            scheduler = None
        n_warm = int(WARMUP_FRAC * iterations)

        # The η head is trained only when it is both defined (ε > 0) and
        # weighted (λ > 0).  With λ = 0 the term 0·loss_eta contributes no
        # gradient anyway, so skipping it changes nothing numerically while
        # avoiding a wasted backward and a flat, misleading η loss in the log.
        use_denoiser = (NOISE_LEVEL > 0.0) and (DENOISER_WEIGHT > 0.0)

        # An ensemble smaller than the configured batch yields only N indices from
        # randperm below, so clamp batch_size to keep t_batch's row count matched.
        batch_size = min(batch_size, X_target.shape[0])

        for i in range(iterations):
            if n_warm and i < n_warm:
                for g in optimizer.param_groups:
                    g['lr'] = lr * (i + 1) / n_warm

            if draw_fn is not None:
                # Fresh (X̄, X, Y) from the same predictive/joint law — the
                # coupling is unchanged, only the realisations are new.
                x1_batch, y_batch, x0_batch = draw_fn(batch_size)
            else:
                idx      = torch.randperm(X_target.shape[0])[:batch_size]
                x0_batch = X_source[idx]              # X̄ — independent copy
                x1_batch = X_target[idx]              # X  — paired with y_batch
                y_batch  = Y_cond[idx]                # Y ~ h(·|X)

            # Clipped τ ~ Unif[0,1] — avoids γ'(τ) divergence at τ ∈ {0,1}
            t_batch = T_BOUND_EPS + (1.0 - 2.0 * T_BOUND_EPS) \
                      * torch.rand(batch_size, 1, device=device)

            # Noise augmentation — Z ~ N(0, I_n), independent of (X̄, X, Y)
            z = torch.randn_like(x1_batch)

            gam     = noise_schedule(t_batch)         # γ(τ)   (bs, 1)
            gam_dot = noise_schedule_dot(t_batch)     # γ'(τ)  (bs, 1)

            # Noisy interpolant  X_τ = (1-τ)X̄ + τX + γ(τ)Z            (Eq. 6)
            x_t = (1.0 - t_batch) * x0_batch + t_batch * x1_batch + gam * z

            # Regression targets:  ∂_τ X_τ = (X - X̄) + γ'(τ)Z,  and Z
            b_target   = (x1_batch - x0_batch) + gam_dot * z
            eta_target = z

            # Forward
            b_pred, eta_pred = net(x_t, t_batch, y_batch)

            # Loss — the two terms of Eq. (7), weighted by λ = DENOISER_WEIGHT
            loss_b = ((b_pred - b_target) ** 2).mean()
            if use_denoiser:
                loss_eta = ((eta_pred - eta_target) ** 2).mean()
                loss = loss_b + DENOISER_WEIGHT * loss_eta
            else:
                loss = loss_b

            optimizer.zero_grad()
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            optimizer.step()
            if scheduler is not None and not (n_warm and i < n_warm):
                scheduler.step()

            # Terminal log
            if (i + 1) == iterations:
                net.eval()
                with torch.no_grad():
                    t_eval = T_BOUND_EPS + (1.0 - 2.0 * T_BOUND_EPS) \
                             * torch.rand(X_target.shape[0], 1, device=device)
                    z_eval       = torch.randn_like(X_target)
                    gam_eval     = noise_schedule(t_eval)
                    gam_dot_eval = noise_schedule_dot(t_eval)
                    x_t_eval     = ((1.0 - t_eval) * X_source
                                    + t_eval * X_target
                                    + gam_eval * z_eval)
                    b_pr, eta_pr = net(x_t_eval, t_eval, Y_cond)
                    lb = ((b_pr - ((X_target - X_source) + gam_dot_eval * z_eval)) ** 2).mean()
                    if use_denoiser:
                        le = ((eta_pr - z_eval) ** 2).mean()
                        print("Simu#%d/%d, Step:%d/%d, Iter:%d/%d, "
                              "SIF b loss=%.4f, η loss=%.4f" %
                              (k + 1, K, ts, Ts - 1, i + 1, iterations,
                               lb.item(), le.item()))
                    else:
                        # No η loss to report: either ε = 0 (head dropped) or
                        # λ = 0 (velocity-only training, head left at init).
                        tag = "ε=0" if NOISE_LEVEL <= 0.0 else "λ=0"
                        print("Simu#%d/%d, Step:%d/%d, Iter:%d/%d, "
                              "SIF b loss=%.4f (%s)" %
                              (k + 1, K, ts, Ts - 1, i + 1, iterations,
                               lb.item(), tag))

                    # A NaN/inf loss means the run has diverged and cannot
                    # recover, so abandon it rather than train the remaining
                    # analysis steps.  Tested here because lb is already
                    # computed and already synced to the host for the log
                    # above — no extra pass, no extra device transfer.
                    if not torch.isfinite(lb):
                        raise _NonFiniteRun(
                            "training loss became %s at simulation %d, "
                            "analysis step %d" % (lb.item(), k + 1, ts))

    # ==================================================================
    # Inference integrators
    # ==================================================================
    def ode_integrate(net, x_init, y_obs, n_steps, dev):
        """
        Deterministic probability-flow ODE:  dx/dτ = b_θ(x, τ, Y_t),
        integrated over τ ∈ [0,1] from each forecast particle with a
        forward Euler scheme (n_steps = ODE_STEPS, 10–50 in the paper).
        The denoiser head is unused on this path.
        """
        dt = 1.0 / n_steps
        x  = x_init.clone()
        net.eval()
        with torch.no_grad():
            for step in range(n_steps):
                t_val    = step * dt
                t_tensor = torch.full((x.shape[0], 1), t_val,
                                      dtype=torch.float32, device=dev)
                b, _ = net(x, t_tensor, y_obs)
                if SOLVER == 'heun':
                    # Trapezoidal corrector. Forward Euler systematically
                    # undershoots on a curved field, which contracts the
                    # ensemble a little at every analysis step and compounds
                    # over the filter run; in the FMF study this one change was
                    # worth ~0.2 aa-SW2.
                    t2 = torch.full_like(t_tensor, t_val + dt)
                    b2, _ = net(x + dt * b, t2, y_obs)
                    x = x + dt * 0.5 * (b + b2)
                else:
                    x = x + dt * b
        return x

    def sde_integrate(net, x_init, y_obs, n_steps, dev):
        """
        Euler-Maruyama for the marginal-preserving SDE

            dx = [ b_θ(x,τ,y)  +  (γ(τ)²/2) · s_θ(x,τ,y) ] dτ  +  γ(τ) dW
               = [ b_θ(x,τ,y)  -  (γ(τ)/2)  · η_θ(x,τ,y) ] dτ  +  γ(τ) dW

        using the score  s_θ = -η_θ / γ(τ)  recovered from the denoiser
        head.  This is the stochastic inference mode of Table I, preferred
        for multimodal posteriors; it shares the marginals p_τ(· | y) of
        the probability-flow ODE, hence "marginal-preserving".

        Since γ(0) = γ(1) = 0, the endpoints are deterministic: no noise
        is injected at τ = 0 (preserves P_X exactly) or τ = 1 (posterior).
        """
        dt       = 1.0 / n_steps
        sqrt_dt  = np.sqrt(dt)
        x        = x_init.clone()
        net.eval()
        with torch.no_grad():
            for step in range(n_steps):
                t_val    = step * dt
                t_tensor = torch.full((x.shape[0], 1), t_val,
                                      dtype=torch.float32, device=dev)
                b, eta   = net(x, t_tensor, y_obs)
                gam      = noise_schedule(t_tensor)       # (N, 1)
                drift    = b - 0.5 * gam * eta
                noise    = torch.randn_like(x)
                if SOLVER == 'heun':
                    # Stochastic Heun: predictor-corrector on the DRIFT with the
                    # same Wiener increment (Karras et al. 2022). Euler-Maruyama
                    # is only weak order 1, so the drift error dominates here
                    # exactly as it does on the ODE path.
                    x_pred = x + drift * dt + gam * sqrt_dt * noise
                    t2 = torch.full_like(t_tensor, t_val + dt)
                    b2, eta2 = net(x_pred, t2, y_obs)
                    gam2 = noise_schedule(t2)
                    drift2 = b2 - 0.5 * gam2 * eta2
                    x = x + 0.5 * (drift + drift2) * dt + gam * sqrt_dt * noise
                else:
                    x = x + drift * dt + gam * sqrt_dt * noise
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
    # cache-building run reproduces the original algorithm exactly and every
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

    X_SIF = torch.zeros((NUM_SIM, T, N, L), device=device, dtype=torch.float32)

    # The loops below abort early via _NonFiniteRun rather than finishing a run
    # that has already diverged: a NaN loss cannot recover, and for a tuning
    # sweep the remaining analysis steps are pure waste. The test lives at the
    # terminal-loss log inside train_interpolant, so it costs nothing — lb is
    # already computed and synced there — but it does mean the failing step
    # pays its full ITERATION before the run stops. Whatever was computed
    # before the failure is kept; everything from the failing step onward is
    # NaN, so the caller sees exactly how far the run got. To mark the whole
    # run instead, replace the two fills below with  X_SIF[:] = nan.
    aborted = None
    try:
        for k in range(NUM_SIM):

            y = Y[k]   # (T x dy x 1)

            X_SIF[k, 0] = torch.from_numpy(X0[k].T).to(torch.float32).to(device)

            ITERS = ITERATION

            # Single network per simulation; warm-started across time steps.
            net = InterpolantNet(
                state_dim      = L,
                obs_dim        = dy,
                hidden_dim     = NUM_NEURON,
                num_resblocks   = num_resblocks,
                time_embed_dim  = TIME_EMBED_DIM,
                time_scale      = TIME_SCALE,
                time_max_period = TIME_MAX_PERIOD,
            ).to(device)
            net.apply(init_weights)
            if ZERO_INIT_OUT:
                nn.init.zeros_(net.head_velocity.weight)
                nn.init.zeros_(net.head_denoiser.weight)

            # Every simulation reads the same file, so on a hit none pays the
            # ITERATION spin-up. A rejected checkpoint applies nothing, so the
            # initialisation just performed is still the right cold start.
            if _warm_available and _warm_start_load(_warm_path, net) is not None:
                # Step 0 is charged the ordinary online budget instead.
                ITERS = Final_Number_ITERATION
                print('[SIF] Simu#%d/%d: warm start from %s '
                      '(spin-up %d -> %d iterations)'
                      % (k + 1, NUM_SIM, _warm_path, ITERATION, ITERS))

            for i in range(T - 1):
                # COLD_START=True reproduces the original algorithm (fresh
                # weights every step); False warm-starts across steps, which
                # the FMF study found is worth ~0.27 aa-SW2 and is what OTF does.
                if COLD_START:
                    net.apply(init_weights)
                    if ZERO_INIT_OUT:
                        nn.init.zeros_(net.head_velocity.weight)
                        nn.init.zeros_(net.head_denoiser.weight)
                _t_step = sync_clock(device) if _timing else 0.0

                # ----------------------------------------------------------
                # 1. Propagate particles through dynamics
                #    Forecast step  π_{t|t-1} = A[π_{t-1}]  of the recursion:
                #    X1 holds the forecast particles {X^i} ~ P_X.
                # ----------------------------------------------------------
                x_noise = torch.distributions.MultivariateNormal(
                    torch.zeros(L), covariance_matrix=torch.eye(L))
                # Deterministic part of the forecast, kept so RESAMPLE_DATA can
                # redraw the process noise without re-propagating.
                AX = A(X_SIF[k, i].T, t[i]).T
                X1 = AX + sigma_proc * x_noise.sample((N,)).to(device)

                # ----------------------------------------------------------
                # 2. Simulate synthetic observations
                #    Y^i ~ h(·|X^i), so that (X1, Y1) are samples of the joint
                #    law P_{X,Y} — the target side of the coupling in Eq. (4).
                # ----------------------------------------------------------
                y_noise = torch.distributions.MultivariateNormal(
                    torch.zeros(dy), covariance_matrix=torch.eye(dy))
                Y1 = h(X1.T).T + sigma_obs * y_noise.sample((N,)).to(device)

                # ----------------------------------------------------------
                # 3. Optional normalization
                # ----------------------------------------------------------
                if normalization == 'Standard':
                    scaler_X = StandardScaler()
                    scaler_Y = StandardScaler()
                    X1 = torch.tensor(scaler_X.fit_transform(X1.cpu()),
                                      device=device, dtype=torch.float32)
                    Y1 = torch.tensor(scaler_Y.fit_transform(Y1.cpu()),
                                      device=device, dtype=torch.float32)
                elif normalization == 'MinMax':
                    scaler_X = MinMaxScaler()
                    scaler_Y = MinMaxScaler()
                    X1 = torch.tensor(scaler_X.fit_transform(X1.cpu()),
                                      device=device, dtype=torch.float32)
                    Y1 = torch.tensor(scaler_Y.fit_transform(Y1.cpu()),
                                      device=device, dtype=torch.float32)

                # ----------------------------------------------------------
                # 4. Independent coupling  (x_0 = shuffled X1, x_1 = X1)
                #    Permuting the forecast particles relative to their own
                #    simulated observations realizes (X̄, Y) ~ P_X ⊗ P_Y, the
                #    source side of Eq. (4).  This construction is shared by
                #    every filter in the comparison.
                # ----------------------------------------------------------
                X_source = X1[torch.randperm(N)]

                # ----------------------------------------------------------
                # 5. Train interpolant network — regression (7) on the
                #    couplings (X̄, X, Y) assembled above.  The net is trained
                #    afresh at every analysis step, warm-started from the
                #    previous step's weights.
                # ----------------------------------------------------------
                # Fresh draws from the same laws the fixed set above was drawn
                # from: X ~ forecast, Y ~ h(·|X), and X̄ a SECOND, unpaired
                # forecast draw so (X̄, Y) is still the independent coupling.
                if RESAMPLE_DATA:
                    if normalization != 'None':
                        raise ValueError(
                            "RESAMPLE_DATA requires normalization='None'; the "
                            "redrawn pairs are in raw units and would not match "
                            "a scaler fitted to the fixed ensemble.")

                    def _draw(bs):
                        j1 = torch.randint(0, N, (bs,), device=device)
                        xb = AX[j1] + sigma_proc * torch.randn(bs, L, device=device)
                        yb = h(xb.T).T + sigma_obs * torch.randn(bs, dy, device=device)
                        j0 = torch.randint(0, N, (bs,), device=device)
                        x0b = AX[j0] + sigma_proc * torch.randn(bs, L, device=device)
                        return xb, yb, x0b
                else:
                    _draw = None

                train_interpolant(
                    net        = net,
                    X_source   = X_source,
                    X_target   = X1,
                    Y_cond     = Y1,
                    iterations = ITERS,
                    lr         = LearningRate,
                    ts         = i + 1,
                    Ts         = T,
                    batch_size = BATCH_SIZE,
                    k          = k,
                    K          = NUM_SIM,
                    draw_fn    = _draw,
                )

                # Saved after train_interpolant returns, so a diverged run
                # (_NonFiniteRun) never reaches this line and cannot poison the
                # cache.
                if _warm_save_pending and k == 0 and i == 0:
                    _warm_start_save(_warm_path, net)
                    _warm_save_pending = False
                    print('[SIF] warm start: saved %s' % _warm_path)

                # ----------------------------------------------------------
                # 6. Iteration halving
                #    Iteration-halving schedule shared with the other filters:
                #    the first analysis step pays ITERATION iterations, later
                #    steps decay towards the Final_Number_ITERATION floor.
                # ----------------------------------------------------------
                # if ITERS > Final_Number_ITERATION and i >= 1:
                #     ITERS = int(ITERS / 2)
                ITERS = Final_Number_ITERATION

                # ----------------------------------------------------------
                # 7. Prepare true observation  y_{i+1}  (broadcast to N particles)
                #    The realized Y_t at which the conditional map T(·, Y_t)
                #    is evaluated; it is held fixed for the whole integration.
                # ----------------------------------------------------------
                Y1_true = y[i + 1, :]                # (dy, 1) numpy
                Y1_true = Y1_true.repeat(N, 1).T     # (N, dy)
                if normalization in ('Standard', 'MinMax'):
                    Y1_true = scaler_Y.transform(Y1_true)
                Y1_true = torch.from_numpy(Y1_true).to(torch.float32).to(device)

                # ----------------------------------------------------------
                # 8. Inference — ODE or SDE, user's choice
                #    Analysis step  X_t^i = T(X_{t|t-1}^i, Y_t): each forecast
                #    particle is flowed from τ = 0 to τ = 1 conditioned on
                #    Y1_true, giving a uniformly weighted posterior ensemble.
                # ----------------------------------------------------------
                if INFERENCE_MODE == 'ode':
                    X_mapped = ode_integrate(net, X1, Y1_true, ODE_STEPS, device)
                else:  # 'sde'
                    X_mapped = sde_integrate(net, X1, Y1_true, ODE_STEPS, device)

                # ----------------------------------------------------------
                # 9. Reverse normalization if applied
                # ----------------------------------------------------------
                if normalization in ('Standard', 'MinMax'):
                    X_mapped = torch.tensor(
                        scaler_X.inverse_transform(X_mapped.cpu().detach().numpy()),
                        dtype=torch.float32,
                    )

                X_SIF[k, i + 1] = X_mapped.detach()
                if _timing: step_times[k, i] = sync_clock(device) - _t_step

    except _NonFiniteRun as exc:
        aborted = exc
        # Remainder of the failing simulation, then every simulation after it.
        X_SIF[k, i + 1:] = float('nan')
        X_SIF[k + 1:]    = float('nan')
        print("[SIF] ABORT: %s" % exc)
        print("[SIF] returning NaN from simulation %d, analysis step %d onward "
              "(%d of %d analysis steps completed)"
              % (k + 1, i + 1, k * (T - 1) + i, NUM_SIM * (T - 1)))

    print("--- SIF time : %s seconds ---%s"
          % (time.time() - start_time, " (ABORTED)" if aborted else ""))

    # Hand the per-step breakdown back through the caller's dict. The total is
    # kept alongside it because it also covers the per-simulation network
    # construction, which sits outside the step loop -- step_times.sum() is
    # deliberately smaller than the total, and the gap is worth seeing.
    if _timing:
        timing_out['step_times'] = step_times
        timing_out['total']      = time.time() - start_time

    return X_SIF.cpu().numpy().transpose(0, 1, 3, 2)

"""
@author: Mohammad Al-Jarrah

Stage 2 of the two-stage SMAC tuner (see TWO_STAGE_TUNING.md): tunes the
online-refinement hyperparameters for one method at a time, warm-started from
the weights stage 1's incumbent produced -- so what is searched here is how a
fixed, warm map should be refined at each analysis step, not the map itself.

Every trial starts from param_one_step.get_config(tune, L, dy), i.e. stage 1's
incumbent, and pins the architecture knobs stage 1 already fixed (nns, kin,
sde, ted, tsc, ode, NOISE_LEVEL, ...) as Constants: the warm-start checkpoint's
filename encodes the architecture, so searching them here would look for a
checkpoint that does not exist. What is actually searched is the online budget
(fitr, unless FINAL_ITER pins it directly) and the optimiser knobs (wd, clip,
warm).

Warm start
----------
warm_start='require': every trial loads the checkpoint stage 1's incumbent
wrote (via warm_start='save'), rather than 'load', so a missing checkpoint is
an immediate named failure instead of a trial that silently trains cold and
differs from its siblings in both initialisation and cost. This tuner never
writes a checkpoint -- stage 1 owns the weights, stage 2 only consumes them.

Multiplier convention
----------------------
Every integer hyperparameter is a small multiple of a fixed base:

    nns   x64    -> NUM_NEURON             bs    x64   -> BATCH_SIZE
    itr   x1024  -> ITERATION (spin-up)    ode   x50   -> ODE_STEPS
    ted   x10    -> TIME_EMBED_DIM         tsc   x50   -> TIME_SCALE
    kin   x5     -> K_in        (OTF)      sde   x20   -> SDE_STEPS   (SBF)
    fitr  -> Final_Number_ITERATION, multiplier varies by method (x256 for
             FMF/SIF, x16 for OTF, x8 for KRF/SBF) -- see each Constant("fitr", ...)
    SBF's itr is the one exception at x256 rather than x1024.

FINAL_ITER, when set, pins fitr directly instead of searching it, and
namespaces the scenario/incumbent/log by it, so several online budgets can be
swept one after another without SMAC offering to resume a different budget's
history.

Cost
----
Tune one method at a time (set `tune` below, or via the TUNE env var).

Outputs
-------
smac3_output/StageTwo_<tune>[_fi<FINAL_ITER>]_sim<NUM_SIM>/    scenario
    directory -- distinct from stage 1's OneStep_<tune>_sim<NUM_SIM>, which
    tuning_stage_one.py owns; the two objectives must never resume each
    other's history.
logs/best_multi_steps_<tune>[_fi<FINAL_ITER>]_sim<NUM_SIM>.json  the incumbent
"""

tune = 'otf'  # 'fmf' | 'otf' | 'sif_ode' | 'sif_sde' | 'sbf' | 'krf'

N_TRIALS  = 100  # trials per method
N_WORKERS = 1    # dask workers (parallel trials)

# ----------------------------------------------------------------------
# Warm start -- this is the STAGE-2 tuner
# ----------------------------------------------------------------------
# The checkpoint must already exist, written by running stage 1's incumbent
# once with warm_start='save'. Because its filename encodes the architecture,
# every knob that feeds it must be pinned to stage 1's winner in the search
# space below (nns*, nbs*, and for the other methods ted/tsc/depth/
# quad_points/NOISE_LEVEL) -- a mismatch is a missing file, not a silent
# fallback.
#
# Set FINAL_ITER to a positive number to pin the pre-multiplier `fitr` value
# directly (Final_Number_ITERATION is still fitr times its method's multiplier
# -- see the module docstring -- so FINAL_ITER itself is NOT the iteration
# count) and namespace the scenario/incumbent/log by it, so several budgets can
# be swept one after another without SMAC offering to overwrite the previous
# run's history. 0 leaves fitr searched.
FINAL_ITER = 4

WARM_START     = 'require'         # 'require' | 'load' | 'off'
WARM_START_TAG = 'quadratic'       # must match what stage 1 saved under
WARM_START_DIR = 'DATA/warm_start'

#%%
import os

# Env overrides so a method can be swept without editing this file, e.g.:
#   TUNE=sif_sde N_WORKERS=2 N_TRIALS=100 FINAL_ITER=8 python tuning_stage_two.py
tune           = os.environ.get('TUNE', tune)
N_TRIALS       = int(os.environ.get('N_TRIALS',  N_TRIALS))
N_WORKERS      = int(os.environ.get('N_WORKERS', N_WORKERS))
FINAL_ITER     = int(os.environ.get('FINAL_ITER', FINAL_ITER))
WARM_START     = os.environ.get('WARM_START',     WARM_START)
WARM_START_TAG = os.environ.get('WARM_START_TAG', WARM_START_TAG)
WARM_START_DIR = os.environ.get('WARM_START_DIR', WARM_START_DIR)

import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib

from SIR import SIR
from SBF import SBF                      
from FMF import FMF
from SIF import SIF
from OTF import OTF
from KRF import KRF
from param_one_step import get_config as get_sota

from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical, Constant
from smac import Scenario, HyperparameterOptimizationFacade as HPO

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=16)
plt.close('all')

np.random.seed(0)
torch.manual_seed(0)

# List the CUDA device indices available to pick from; the first one found is
# used. Add more here on a multi-GPU machine to make a different GPU available
# without touching anything else below.
GPU_IDS = [0]

if torch.cuda.is_available():
    n_visible = torch.cuda.device_count()
    _ids = [g for g in GPU_IDS if g < n_visible] or [0]
    device = torch.device(f'cuda:{_ids[0]}')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')


def h(x):
    return x[::2] * x[::2]  # observe every other state (0, 2, ...); n obs for L = 2n states

def A(x, t=0):
    try:
        return F @ x
    except Exception:
        return torch.from_numpy(F).to(dtype=torch.float32, device=x.device) @ x

def Gen_True_Data(L, dy, T, sigma0, sigma, gamma, tau):
    """
    Generates true state and observation data using the Linear-Quadratic model.

    For a given number of time steps, the true state is evolved by the
    Linear model. The observations are generated using an observation rule
    (function h) with added Gaussian noise.

    Parameters
    ----------
    L : int
        Dimension of the state space.
    dy : int
        Dimension of the observation space.
    T : int
        Number of time steps.
    sigma0 : float
        Standard deviation for the initial state distribution.
    sigma : float
        Standard deviation for the process noise.
    gamma : float
        Standard deviation for the observation noise.
    tau : float
        Time step size.

    Returns
    -------
    x : ndarray
        True state evolution with shape (T x L x 1).
    y : ndarray
        Observations with shape (T x dy x 1).
    """
    x = np.zeros((T, L, 1))
    y = np.zeros((T, dy, 1))

    x[0,] = np.random.multivariate_normal(np.zeros(L), sigma0 * sigma0 * np.eye(L), 1).T

    for i in range(T - 1):
        x[i + 1, :] = A(x[i, :]) + np.random.multivariate_normal(np.zeros(L), sigma * sigma * np.eye(L), 1).T
        y[i + 1, :] = h(x[i + 1, :]) + np.random.multivariate_normal(np.zeros(dy), gamma * gamma * np.eye(dy), 1).T

    return x, y


def aa_sw2(X, X_ref):
    """
    Axis-aligned sliced 2-Wasserstein distance between a particle ensemble and
    the reference posterior, resolved per (simulation, time step).

    Each state coordinate is one axis-aligned slice, so the L-dimensional
    comparison reduces to L one-dimensional W2 distances that are averaged.
    Each 1-D distance is evaluated from the empirical inverse CDFs on a common
    midpoint quantile grid; when both ensembles have the same particle count
    this reduces exactly to the L2 distance between the sorted samples.

    This is the same function Quadratic.py uses to score the filters, kept
    identical here so the tuning objective and the reported metric agree.

    Parameters
    ----------
    X     : ndarray (NUM_SIM x T x L x N)      — filter particles
    X_ref : ndarray (T x L x N_true)           — reference posterior particles,
                                                 shared by every simulation

    Returns
    -------
    ndarray (NUM_SIM x T) — per-simulation, per-time-step aa-SW2
    """
    n_x, n_r = X.shape[-1], X_ref.shape[-1]
    M        = min(n_x, n_r)
    u        = (np.arange(M) + 0.5) / M          # midpoint quantile grid

    # Empirical inverse CDF: index the sorted samples at the grid quantiles.
    # For M == n this returns arange(n), i.e. the order statistics themselves.
    i_x = np.minimum((u * n_x).astype(int), n_x - 1)
    i_r = np.minimum((u * n_r).astype(int), n_r - 1)

    Xs = np.sort(X,     axis=-1)[..., i_x]       # (NUM_SIM, T, L, M)
    Rs = np.sort(X_ref, axis=-1)[..., i_r]       # (T, L, M)

    w2 = np.sqrt(np.mean((Xs - Rs[None]) ** 2, axis=-1))   # (NUM_SIM, T, L)
    return w2.mean(axis=-1)                                 # (NUM_SIM, T)

#%%
# Simulation parameters.
n   = 5
L   = n * 2            # number of states
tau = 1e-1              # time step
T   = int(2 / tau)      # number of time steps (T = 20 -> t up to ~2s at tau=0.1)
dy  = n                 # number of states observed
t   = np.arange(0.0, tau * T, tau)

# Dynamical system: n decoupled 2D rotation blocks.
alpha   = 0.9
a       = alpha * 0.99
b       = np.sqrt(1 - alpha ** 2)
F_block = np.array([[a, -b], [b, a]])
F       = np.kron(np.eye(int(n)), F_block)

noise  = np.sqrt(1e-1)   # noise level std
sigma  = noise           # process noise
sigma0 = 1               # initial-state noise
gamma  = noise           # observation noise
Noise  = [sigma, gamma]

N       = int(1e4)   # ensemble particles
NUM_SIM = 2           # independent simulations
NUM_SIM = int(os.environ.get('NUM_SIM', NUM_SIM))

X_true = np.zeros((NUM_SIM, T, L, 1))
Y_True = np.zeros((NUM_SIM, T, dy, 1))
X0     = np.zeros((NUM_SIM, L, N))

for k in range(NUM_SIM):
    X_true[k,], Y_True[k,] = Gen_True_Data(L, dy, T, sigma0, sigma, gamma, tau)
    X0[k,] = np.random.multivariate_normal(np.zeros(L), sigma0 * sigma0 * np.eye(L), N).T


def A_block(x, t=0):
    """Block dynamics for one decoupled 2D subsystem (uses the 2x2 F_block)."""
    try:
        return F_block @ x
    except Exception:
        return torch.from_numpy(F_block).to(dtype=torch.float32, device=x.device) @ x

# Reference posterior: n independent 2D SIR filters, one per decoupled block of
# F = kron(I_n, F_block). Each block's posterior is exact -- dynamics, noise and
# prior all factorise across blocks, and block j is observed only through y[j] --
# so running SIR in 2D rather than the full L = 2n space sidesteps the curse of
# dimensionality. Results are stacked back into the full-state reference.
N_true   = int(1e5)  # particles SIR runs with, for an accurate posterior
n_sample = int(2e3)  # particles kept in X_True (all aa_sw2 uses downstream)
X_True   = np.zeros((NUM_SIM, T, L, n_sample))
for j in range(n):
    X0_blk = np.zeros((NUM_SIM, 2, N_true))
    for k in range(NUM_SIM):
        X0_blk[k,] = np.random.multivariate_normal(np.zeros(2), sigma0 * sigma0 * np.eye(2), N_true).T
    Y_blk = Y_True[:, :, j:j + 1, :]                       # (NUM_SIM, T, 1, 1)
    X_blk = SIR(Y_blk, X0_blk, A_block, h, t, Noise, device=device)  # (NUM_SIM, T, 2, N_true)
    X_True[:, :, 2 * j:2 * j + 2, :] = X_blk[:, :, :, :n_sample]


# ============================================================
# SMAC hyperparameter tuning via aa-SW2 validation loss
# ============================================================
cs = ConfigurationSpace(seed=42)

# One cs.add block per method. Every range/choice is written as a literal in
# the block it belongs to, rather than shared through module-level constants,
# so widening or narrowing one method's search space cannot move another's.
# Architecture knobs are pinned (Constant) to stage 1's winner rather than
# searched, since the warm-start checkpoint's filename encodes them; fitr is
# either pinned to FINAL_ITER or searched over a small Categorical.
SIF_VARIANTS = ('sif_ode', 'sif_sde')

if tune == 'fmf':
    cs.add([
        Float(   "lr",         (1e-6, 1e-3), log=True, default=0.000748376559),
        Constant("nns",        5),
        Constant("nbs",        1),
        Integer( "bs",         (1, 12),            default=9),    # x64  -> 64-768
        Constant("itr",        64),
        (Constant("fitr", FINAL_ITER) if FINAL_ITER else
            Categorical("fitr", [1, 4, 16], default=4)),          # x256 -> 256-4096
        Constant("ode",        4),
        Constant("ted",        2),
        Constant("tsc",        2),
        Categorical("wd",      [0.0, 1e-3, 1e-2, 1e-1], default=0.0),
        Categorical("clip",    [0.25, 0.5, 1.0, 2.0],   default=0.5),
        Categorical("warm",    [0.0, 0.02, 0.05, 0.10], default=0.02),
    ])
# The two SIF rungs get one block each, so every value they search -- defaults
# included -- lives where that rung is defined. They differ only in NOISE_LEVEL
# and the two Constants that DEFINE the rung: DENOISER_WEIGHT (whether the
# denoiser head trains at all) and INFERENCE_MODE (which sampler runs).
elif tune == 'sif_ode':
    # epsilon > 0, lambda = 0: velocity head alone, no denoiser loss on the trunk.
    cs.add([
        Float(   "lr",         (1e-6, 1e-3), log=True, default=0.0008260079241),
        Constant("nns",        12),
        Constant("nbs",        1),
        Integer( "bs",         (1, 12),            default=9),    # x64  -> 64-768
        Constant("itr",        8),
        (Constant("fitr", FINAL_ITER) if FINAL_ITER else
            Categorical("fitr", [1, 4, 16], default=4)),          # x256 -> 256-4096
        Constant("NOISE_LEVEL", 0.1433125539948),
        Constant("ode",        4),
        Constant("ted",        3),
        Constant("tsc",        3),
        Categorical("wd",      [0.0, 1e-3, 1e-2, 1e-1], default=1e-2),
        Categorical("clip",    [0.25, 0.5, 1.0, 2.0],   default=2.0),
        Categorical("warm",    [0.0, 0.02, 0.05, 0.10], default=0.05),
        Constant("INFERENCE_MODE",  'ode'),
        Constant("DENOISER_WEIGHT", 0.0),
    ])
elif tune == 'sif_sde':
    # epsilon > 0, lambda = 1, SDE sampler: the full stochastic interpolant.
    # Wants more integrator steps than the ODE rung -- its error is dominated
    # by the noise discretisation.
    cs.add([
        Float(   "lr",         (1e-6, 1e-3), log=True, default=0.000730013106),
        Constant("nns",        9),
        Constant("nbs",        1),
        Integer( "bs",         (1, 12),            default=11),   # x64  -> 64-768
        Constant("itr",        32),
        (Constant("fitr", FINAL_ITER) if FINAL_ITER else
            Categorical("fitr", [1, 4, 16], default=4)),          # x256 -> 256-4096
        Constant("NOISE_LEVEL", 0.1814779544991),
        Constant("ode",        2),
        Constant("ted",        4),
        Constant("tsc",        1),
        Categorical("wd",      [0.0, 1e-3, 1e-2, 1e-1], default=1e-2),
        Categorical("clip",    [0.25, 0.5, 1.0, 2.0],   default=1.0),
        Categorical("warm",    [0.0, 0.02, 0.05, 0.10], default=0.0),
        Constant("INFERENCE_MODE",  'sde'),
        Constant("DENOISER_WEIGHT", 1.0),
    ])
elif tune == 'otf':
    # OTF resisted almost everything, so its space stays close to the published
    # incumbent. fitr uses the x16 base: OTF's online optimum is 64, and it
    # degrades steeply above it (512 -> 0.5973, 2048 -> 0.9204).
    cs.add([
        Float(   "lr1",        (1e-6, 1e-3), log=True, default=0.0002421871224),
        Float(   "lr2",        (1e-6, 1e-3), log=True, default=0.00034122139),
        Constant("nns1",       4),
        Constant("nns2",       8),
        Constant("nbs1",       1),
        Constant("nbs2",       1),
        Integer( "bs",         (1, 10),            default=9),
        Constant("kin",        3),
        Constant("itr",        2),
        # Searched normally; pinned to a placeholder when FINAL_ITER sets the
        # budget directly, since target_fun then ignores it.
        (Constant("fitr", FINAL_ITER) if FINAL_ITER else
            Categorical("fitr", [1, 4, 16], default=4)),          # x16  -> 16-256
        # AdamW over the same value sets FMF/SIF use, so tuners stay comparable.
        # OTF.py applies one WEIGHT_DECAY to both the critic f and the map T_net.
        Categorical("wd",      [0.0, 1e-3, 1e-2, 1e-1], default=1e-3),
        Categorical("clip",    [0.25, 0.5, 1.0, 2.0],   default=0.25),
        Categorical("warm",    [0.0, 0.02, 0.05, 0.10], default=0.05),
    ])
elif tune == 'krf':
    # KRF is the one method whose error RISES with budget, so fitr stops at 512
    # rather than sweeping into the thousands.
    cs.add([
        Float(   "lr",         (1e-6, 1e-3), log=True, default=3.2608338e-06),
        Constant("nns",        12),
        Integer( "bs",         (1, 12),            default=12),
        Constant("itr",        4),
        (Constant("fitr", FINAL_ITER) if FINAL_ITER else
            Categorical("fitr", [1, 4, 16], default=4)),          # x8   -> 8-128
        Constant("depth",       4),
        Constant("quad_points", 4),
        Categorical("wd",      [0.0, 1e-3, 1e-2, 1e-1], default=1e-1),
        Categorical("clip",    [0.25, 0.5, 1.0, 2.0],   default=1.0),
        Categorical("warm",    [0.0, 0.02, 0.05, 0.10], default=0.1),
    ])
elif tune == 'sbf':
    # SBF's work is num_itr * NUM_STAGES * 2 directions * SDE_STEPS rows through
    # a (nns*64)-wide net under second-order autograd, at every T-1 step.
    cs.add([
        Float(   "lr",         (1e-6, 1e-3), log=True, default=0.0002392084557),
        Constant("nns",        10),
        Constant("ted",        5),
        Constant("nbs",        1),
        Constant("sde",        4),
        Constant("itr",        4),
        (Constant("fitr", FINAL_ITER) if FINAL_ITER else
            Categorical("fitr", [1, 4, 16], default=4)),          # x8   -> 8-128
        Constant("NUM_STAGES",  3),
        Integer( "bs",         (1, 12),            default=5),
        # No 'warm' knob: SBF drives a global CosineAnnealingWarmRestarts
        # schedule rather than the per-step one FMF/SIF use.
        Categorical("wd",      [0.0, 1e-3, 1e-2, 1e-1], default=1e-3),
        Categorical("clip",    [0.25, 0.5, 1.0, 2.0],   default=0.25),
    ])
else:
    raise SystemExit(f"unknown tune={tune!r}")


def target_fun(config, seed: int = 0) -> float:
    """
    Train methods with the given config and return the aa-SW2 validation loss.
    SMAC minimises this value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Start from the current state of the art and override only what is being
    # searched, so every fixed improvement stays present in every trial and the
    # default configuration equals the incumbent.
    params = get_sota(tune, L, dy)

    if tune == 'fmf':
        params.update({
            'NUM_NEURON':             int(config["nns"] * 64),
            'BATCH_SIZE':             int(config["bs"] * 64),
            'LearningRate':           float(config["lr"]),
            'num_resblocks':          int(config["nbs"]),
            'ITERATION':              int(config["itr"] * 1024),
            'Final_Number_ITERATION': int(config["fitr"] * 256),
            'ODE_STEPS':              int(config["ode"] * 50),
            'TIME_EMBED_DIM':         int(config["ted"] * 10),
            'TIME_SCALE':             float(config["tsc"] * 50),
            'WEIGHT_DECAY':           float(config["wd"]),
            'GRAD_CLIP':              float(config["clip"]),
            'WARMUP_FRAC':            float(config["warm"]),
        })
    elif tune in SIF_VARIANTS:
        params.update({
            'NUM_NEURON':             int(config["nns"] * 64),
            'BATCH_SIZE':             int(config["bs"] * 64),
            'LearningRate':           float(config["lr"]),
            'num_resblocks':          int(config["nbs"]),
            'ITERATION':              int(config["itr"] * 1024),
            'Final_Number_ITERATION': int(config["fitr"] * 256),
            'ODE_STEPS':              int(config["ode"] * 50),
            'NOISE_LEVEL':            float(config["NOISE_LEVEL"]),         # epsilon
            'INFERENCE_MODE':         config["INFERENCE_MODE"],
            'DENOISER_WEIGHT':        float(config["DENOISER_WEIGHT"]),     # lambda
            'TIME_EMBED_DIM':         int(config["ted"] * 10),
            'TIME_SCALE':             float(config["tsc"] * 50),
            'WEIGHT_DECAY':           float(config["wd"]),
            'GRAD_CLIP':              float(config["clip"]),
            'WARMUP_FRAC':            float(config["warm"]),
        })
    elif tune == 'otf':
        params.update({
            'NUM_NEURON':             [int(config["nns1"]) * 64, int(config["nns2"]) * 64],
            'num_resblocks':          [int(config["nbs1"]), int(config["nbs2"])],
            'LearningRate':           [float(config["lr1"]), float(config["lr2"])],
            'BATCH_SIZE':             int(config["bs"] * 64),
            'K_in':                   int(config["kin"] * 5),
            'ITERATION':              int(config["itr"] * 1024),
            'Final_Number_ITERATION': int(config["fitr"] * 16),
            'OPTIMIZER':              'adamw',
            'WEIGHT_DECAY':           float(config["wd"]),
            'GRAD_CLIP':              float(config["clip"]),
            'WARMUP_FRAC':            float(config["warm"]),
        })
    elif tune == 'krf':
        params.update({
            'NUM_NEURON':             int(config["nns"] * 64),
            'BATCH_SIZE':             int(config["bs"] * 64),
            'LearningRate':           float(config["lr"]),
            'ITERATION':              int(config["itr"] * 1024),
            'Final_Number_ITERATION': int(config["fitr"] * 8),
            'depth':                  int(config["depth"]),
            'quad_points':            int(config["quad_points"]),
            'OPTIMIZER':              'adamw',
            'WEIGHT_DECAY':           float(config["wd"]),
            'GRAD_CLIP':              float(config["clip"]),
            'WARMUP_FRAC':            float(config["warm"]),
        })
    elif tune == 'sbf':
        params.update({
            'NUM_NEURON':             int(config["nns"] * 64),
            'TIME_EMBED_DIM':         int(config["ted"] * 10),
            'num_resblocks':          int(config["nbs"]),
            'SDE_STEPS':              int(config["sde"] * 20),
            'ITERATION':              int(config["itr"] * 256),
            'Final_Number_ITERATION': int(config["fitr"] * 8),
            'NUM_STAGES':             int(config["NUM_STAGES"]),
            'BATCH_SIZE':             int(config["bs"] * 64),
            'LR':                     float(config["lr"]),
            'OPTIMIZER':              'adamw',
            'WEIGHT_DECAY':           float(config["wd"]),
            'GRAD_CLIP':              float(config["clip"]),
        })

    if params.get('BATCH_SIZE', 0) > N:
        params['BATCH_SIZE'] = N  # a mini-batch cannot exceed the ensemble it is drawn from

    # Penalty for a trial that cannot be scored. Must be finite: SMAC fits its
    # surrogate model on the recorded costs, and a NaN/inf aborts the whole run
    # rather than just discarding the trial.
    FAIL_LOSS = 1e6

    try:
        FILTER = {
            'fmf':     FMF,
            'sif_sde': SIF,
            'sif_ode': SIF,
            'otf':     OTF,
            'krf':     KRF,
            'sbf':     SBF,
        }[tune]
        # warm_start is passed to every method here: all six accept it, and
        # EnKF/SIR are not tuned by this script. On 'require' the filter raises
        # if no checkpoint matches, which the except below turns into a
        # penalised trial with the wanted path recorded in
        # logs/failures_<tune>.log.
        X_val = FILTER(Y_True, X0, A, h, t, Noise, params, device=device,
                       warm_start=WARM_START, warm_start_tag=WARM_START_TAG,
                       warm_start_dir=WARM_START_DIR)

        # Axis-aligned sliced W2 -- the same metric Quadratic.py reports, so the
        # configuration SMAC selects is the one that wins on the published score.
        # Each simulation is scored against its own reference posterior X_True[k];
        # t = 0 is skipped since every configuration shares the same prior there.
        #
        # A diverged run is not an exception: the filters abort internally and
        # fill the remaining particles with NaN, so X_val comes back well-shaped
        # but partly NaN and the loss below would silently become NaN. Caught
        # here and given the same finite penalty a raised exception gets.
        if not np.isfinite(X_val).all():
            raise ValueError(
                "%s returned non-finite particles (%d of %d entries) — the run "
                "diverged and aborted internally"
                % (tune, int((~np.isfinite(X_val)).sum()), X_val.size))

        sw2  = np.stack([aa_sw2(X_val[k:k+1], X_True[k])[0] for k in range(X_val.shape[0])])
        loss = sw2[:, 1:].mean()      # mean over sims and time steps (t > 0)

        if not np.isfinite(loss):
            raise ValueError(f"aa-SW2 loss is not finite ({loss})")
    except Exception as e:
        # Trial output goes to the dask worker's stdout, not the parent's, so a
        # print here is invisible in logs/tune_<tune>.log; append to a file too,
        # so failures are always diagnosable.
        print(f"[SMAC] Trial failed: {e}")
        try:
            import traceback
            os.makedirs("logs", exist_ok=True)
            with open(os.path.join("logs", f"failures_{tune}.log"), "a") as fh:
                fh.write(f"=== {type(e).__name__}: {e}\n"
                         f"    device={device} config={dict(config)}\n"
                         f"{traceback.format_exc()}\n")
        except Exception:
            pass
        loss = FAIL_LOSS

    loss = float(loss)
    print(f"[SMAC] config={dict(config)}  aa-SW2={loss:.6f}")
    return loss


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Warm-start preflight
    # ------------------------------------------------------------------
    # With warm_start='require' a missing checkpoint does NOT stop the sweep on
    # its own: every trial raises, target_fun's except records the penalty, and
    # hours later the run has "finished" with nothing but failures. Far cheaper
    # to look for the file once, here, before any worker starts.
    #
    # This only proves that SOMETHING was saved for this method and tag. The
    # filename also encodes the architecture, so a pinned knob that does not
    # match stage 1's winner still surfaces on the first trial -- but there the
    # error names the exact path it wanted, which is what makes the mismatch
    # obvious.
    if WARM_START in ('require', 'load'):
        import glob
        _prefix = 'sif' if tune.startswith('sif') else tune
        _pattern = os.path.join(WARM_START_DIR, f'{_prefix}_{WARM_START_TAG}_*.pt')
        _found = sorted(glob.glob(_pattern))
        if not _found:
            raise SystemExit(
                f"warm_start={WARM_START!r} but no checkpoint matches {_pattern}. "
                f"Stage 2 consumes weights it does not create: run stage 1's "
                f"incumbent once with warm_start='save' (and the same "
                f"warm_start_tag) before tuning here.")
        print(f"Warm start ({WARM_START}) — candidate checkpoints for {tune}:")
        for _f in _found:
            print(f"    {_f}")
        print()

    scenario = Scenario(
        configspace   = cs,
        # Namespaced per TUNER, not just per method: tuning_stage_one.py owns
        # OneStep_<tune>, and sharing that name here would make SMAC offer to
        # overwrite or rename stage 1's run history -- fatal non-interactively,
        # since the prompt reads from stdin -- and the two objectives (T = 2
        # from scratch vs T = 20 warm-started) must never resume each other's
        # history in any case.
        name          = (f"StageTwo_{tune}_fi{FINAL_ITER}_sim{NUM_SIM}"
                         if FINAL_ITER else
                         f"StageTwo_{tune}_sim{NUM_SIM}"),
        # The objective is NOT deterministic and telling SMAC otherwise made it
        # evaluate every configuration exactly once, so the incumbent was the
        # minimum of 100 noisy draws -- optimistically biased by construction.
        # Measured: repeated evaluations of one configuration spread over
        # sigma = 0.0172 aa-SW2, and individual simulations occasionally land at
        # 0.6 instead of 0.35, so the minimum was partly selecting the trials
        # where nothing misbehaved. With deterministic=False SMAC evaluates a
        # promising configuration on several seeds and ranks it on the average,
        # which is what makes the reported incumbent mean something.
        #
        # Note n_trials now counts (configuration, seed) evaluations, so a given
        # budget covers fewer distinct configurations than before.
        deterministic = False,
        n_trials      = N_TRIALS,
        n_workers     = N_WORKERS,
    )

    smac = HPO(
        scenario        = scenario,
        target_function = target_fun,
        overwrite       = False,   # True overwrites a previous run with this scenario name
    )

    print(f"\n=== Starting SMAC optimisation for method: {tune} ===\n")
    print(f"Number of CPU cores: {os.cpu_count()}")
    print(f"Using n_workers={scenario.n_workers} for parallel evaluation.\n")
    try:
        incumbent = smac.optimize()
    finally:
        try:
            smac._runner.close()
        except Exception:
            pass

    # Read the incumbent's cost from the run history instead of calling
    # smac.validate(incumbent), which would re-run the method (and print all of
    # its training output) after this summary. With deterministic=False this is
    # the MEAN over the seeds SMAC actually evaluated for the incumbent, which is
    # the number worth reporting -- unlike a single evaluation, it is not
    # selected for having avoided the objective's bad tail.
    incumbent_cost = smac.runhistory.get_cost(incumbent)
    print("\n=== Best configuration found ===")
    print(incumbent)
    print(f"Validation aa-SW2 loss: {incumbent_cost:.6f}")
    print("Method: " + tune)

    # Persist the incumbent so it can be written into the param file without
    # parsing stdout. Categorical values come back as np.int64; tag them so the
    # param file's existing np.int64(...) style can be reproduced on load.
    #
    # Stage 1's incumbent lives in best_one_step_<tune>_sim<NUM_SIM>.json and is
    # the file the frozen Constants above are pasted from, so stage 2 must not
    # write over it.
    os.makedirs("logs", exist_ok=True)
    best_path = os.path.join(
        "logs", f"best_multi_steps_{tune}_fi{FINAL_ITER}_sim{NUM_SIM}.json"
        if FINAL_ITER else f"best_multi_steps_{tune}_sim{NUM_SIM}.json")
    with open(best_path, "w") as fh:
        json.dump({
            "method": tune,
            "cost":   float(incumbent_cost),
            "config": {k: ({"__np_int64__": int(v)} if isinstance(v, np.integer) else v)
                       for k, v in dict(incumbent).items()},
        }, fh, indent=2)
    print(f"Wrote {best_path}")

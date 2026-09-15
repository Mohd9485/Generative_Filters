"""
@author: Mohammad Al-Jarrah

Stage 1 of the two-stage SMAC tuner (see TWO_STAGE_TUNING.md): tunes the cold
spin-up hyperparameters (architecture, learning rate, batch size, schedule
knobs) for one method at a time, starting from param_one_step.get_config() and
searching only the knobs added to the cs.add() block for that method below.

Multiplier convention
----------------------
Every integer hyperparameter is a small multiple of a fixed base, which is what
keeps the search space small:

    nns   x64    -> NUM_NEURON             bs    x64   -> BATCH_SIZE
    itr   x1024  -> ITERATION (spin-up)    ode   x50   -> ODE_STEPS
    ted   x10    -> TIME_EMBED_DIM         tsc   x50   -> TIME_SCALE
    kin   x5     -> K_in        (OTF)      sde   x20   -> SDE_STEPS   (SBF)
    fitr  -> Final_Number_ITERATION, multiplier varies by method (x256 for
             FMF/SIF, x16 for OTF, x8 for KRF/SBF) -- see each Constant("fitr", ...)
    SBF's itr is the one exception at x256 rather than x1024.

Ranges come from what earlier tuning runs measured, not first principles; each
cs.add() block below notes why its bounds sit where they do.

Cost
----
The tuned methods are expensive per trial, so tune one method at a time (set
`tune` below, or via the TUNE env var) and lower N_TRIALS for the slower ones.

Outputs
-------
smac3_output/StageOne_<tune>_sim<NUM_SIM>/    SMAC scenario directory
logs/best_one_step_<tune>_sim<NUM_SIM>.json  the incumbent config
"""

tune = 'otf'  # 'fmf' | 'otf' | 'sif_ode' | 'sif_sde' | 'sbf' | 'krf'

N_TRIALS  = 100  # trials per method
N_WORKERS = 1    # dask workers (parallel trials)

#%%
import os

# Env overrides so a method can be swept without editing this file, e.g.:
#   TUNE=sif_sde N_WORKERS=2 N_TRIALS=100 python tuning_stage_one.py
tune      = os.environ.get('TUNE', tune)
N_TRIALS  = int(os.environ.get('N_TRIALS',  N_TRIALS))
N_WORKERS = int(os.environ.get('N_WORKERS', N_WORKERS))

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

    x[0,] = np.ones((L, 1))

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
T   = 2                 # number of time steps (T = 5 s would be int(5/tau))
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
SIF_VARIANTS = ('sif_ode', 'sif_sde')

if tune == 'fmf':
    cs.add([
        Float(   "lr",         (1e-6, 1e-3), log=True, default=1e-4),
        Integer( "nns",        (2, 12),            default=6),    # x64  -> 128-768
        Constant("nbs",        1),
        Integer( "bs",         (1, 12),            default=6),    # x64  -> 64-768
        Categorical("itr",     [8, 16, 32, 64],    default=16),   # x1024
        Constant("fitr",       0),                                # x256
        Integer( "ode",        (1, 7),             default=4),    # x50  -> 50-350
        Integer( "ted",        (1, 5),             default=3),    # x10  -> 10-50
        Integer( "tsc",        (1, 5),             default=3),    # x50  -> 50-250
        Categorical("wd",      [0.0, 1e-3, 1e-2, 1e-1], default=1e-2),
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
        Float(   "lr",         (1e-6, 1e-3), log=True, default=1e-4),
        Integer( "nns",        (2, 12),            default=6),
        Constant("nbs",        1),
        Integer( "bs",         (1, 12),            default=6),
        Categorical("itr",     [8, 16, 32, 64],    default=16),   # x1024
        Constant("fitr",       0),                                # x256
        Integer( "ode",        (1, 7),             default=4),    # x50 -> 50-400
        Float(   "NOISE_LEVEL", (1e-2, 0.3),       default=0.1),
        Integer( "ted",        (1, 5),             default=3),
        Integer( "tsc",        (1, 5),             default=3),
        Categorical("wd",      [0.0, 1e-3, 1e-2, 1e-1], default=1e-2),
        Categorical("clip",    [0.25, 0.5, 1.0, 2.0],   default=0.5),
        Categorical("warm",    [0.0, 0.02, 0.05, 0.10], default=0.02),
        Constant("INFERENCE_MODE",  'ode'),
        Constant("DENOISER_WEIGHT", 0.0),
    ])
elif tune == 'sif_sde':
    # epsilon > 0, lambda = 1, SDE sampler: the full stochastic interpolant.
    # Wants more integrator steps than the ODE rung -- its error is dominated
    # by the noise discretisation.
    cs.add([
        Float(   "lr",         (1e-6, 1e-3), log=True, default=1e-4),
        Integer( "nns",        (2, 12),            default=6),
        Constant("nbs",        1),
        Integer( "bs",         (1, 12),            default=6),
        Categorical("itr",     [8, 16, 32, 64],    default=16),   # x1024
        Constant("fitr",       0),                                # x256
        Integer( "ode",        (1, 7),             default=4),    # x50 -> 50-400
        Float(   "NOISE_LEVEL", (1e-2, 0.3),       default=0.1),
        Integer( "ted",        (1, 5),             default=3),
        Integer( "tsc",        (1, 5),             default=3),
        Categorical("wd",      [0.0, 1e-3, 1e-2, 1e-1], default=1e-2),
        Categorical("clip",    [0.25, 0.5, 1.0, 2.0],   default=0.5),
        Categorical("warm",    [0.0, 0.02, 0.05, 0.10], default=0.02),
        Constant("INFERENCE_MODE",  'sde'),
        Constant("DENOISER_WEIGHT", 1.0),
    ])
elif tune == 'otf':
    # OTF resisted almost everything, so its space stays close to the published
    # incumbent. fitr uses the x16 base: OTF's online optimum is 64, and it
    # degrades steeply above it (512 -> 0.5973, 2048 -> 0.9204).
    cs.add([
        Float(   "lr1",        (1e-6, 1e-3), log=True, default=1e-4),
        Float(   "lr2",        (1e-6, 1e-3), log=True, default=1e-4),
        Integer( "nns1",       (1, 10),            default=5),    # x64 -> critic 64-384
        Integer( "nns2",       (1, 10),            default=5),    # x64 -> map 128-640
        Constant("nbs1",       1),
        Constant("nbs2",       1),
        Integer( "bs",         (1, 10),            default=5),
        Integer( "kin",        (2, 4),             default=2),    # x5  -> K_in 5-30
        Categorical("itr",     [1, 2, 4, 8],       default=4),    # x1024
        Constant("fitr",       0),                                # x16
        # AdamW over the same value sets FMF/SIF use, so tuners stay comparable.
        # OTF.py applies one WEIGHT_DECAY to both the critic f and the map T_net.
        Categorical("wd",      [0.0, 1e-3, 1e-2, 1e-1], default=1e-2),
        Categorical("clip",    [0.25, 0.5, 1.0, 2.0],   default=0.5),
        Categorical("warm",    [0.0, 0.02, 0.05, 0.10], default=0.02),
    ])
elif tune == 'krf':
    # KRF is the one method whose error RISES with budget, so fitr stops at 512
    # rather than sweeping into the thousands.
    cs.add([
        Float(   "lr",         (1e-6, 1e-3), log=True, default=1e-04),
        Integer( "nns",        (2, 12),            default=6),    # x64 -> 448
        Integer( "bs",         (1, 12),            default=6),
        Categorical("itr",     [1, 2, 4, 8],       default=4),    # x1024
        Constant("fitr",       0),                                 # x8
        Integer( "depth",      (1, 5),             default=3),
        Integer( "quad_points", (2, 5),            default=4),
        Categorical("wd",      [0.0, 1e-3, 1e-2, 1e-1], default=1e-2),
        Categorical("clip",    [0.25, 0.5, 1.0, 2.0],   default=0.5),
        Categorical("warm",    [0.0, 0.02, 0.05, 0.10], default=0.02),
    ])
elif tune == 'sbf':
    # SBF's work is num_itr * NUM_STAGES * 2 directions * SDE_STEPS rows through
    # a (nns*64)-wide net under second-order autograd, at every T-1 step.
    cs.add([
        Float(   "lr",         (1e-6, 1e-3), log=True, default=1e-4),
        Integer( "nns",        (1, 10),            default=5),
        Integer( "ted",        (1, 5),             default=3),
        Constant("nbs",        1),
        Integer( "sde",        (1, 5),             default=3),    # x20 -> 20-100
        Categorical("itr",     [1, 2, 4, 8],       default=4),    # x256
        Constant("fitr",       0),                                 # x8
        Categorical("NUM_STAGES", [1, 2, 3],       default=2),
        Integer( "bs",         (1, 10),            default=5),
        # No 'warm' knob: SBF drives a global CosineAnnealingWarmRestarts
        # schedule rather than the per-step one FMF/SIF use.
        Categorical("wd",      [0.0, 1e-3, 1e-2, 1e-1], default=1e-2),
        Categorical("clip",    [0.25, 0.5, 1.0, 2.0],   default=0.5),
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
            'fmf':         FMF,
            'sif_sde':     SIF,
            'sif_ode':     SIF,
            'otf':         OTF,
            'krf':         KRF,
            'sbf':         SBF,
        }[tune]
        X_val = FILTER(Y_True, X0, A, h, t, Noise, params, device=device)

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
    scenario = Scenario(
        configspace   = cs,
        name          = f"StageOne_{tune}_sim{NUM_SIM}",  # namespaced away from the stage-2 tuner
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
    # its training output) after this summary.
    incumbent_cost = smac.runhistory.get_cost(incumbent)
    print("\n=== Best configuration found ===")
    print(incumbent)
    print(f"Validation aa-SW2 loss: {incumbent_cost:.6f}")
    print("Method: " + tune)

    # Persist the incumbent so it can be written into the param file without
    # parsing stdout. Categorical values come back as np.int64; tag them so the
    # param file's existing np.int64(...) style can be reproduced on load.
    os.makedirs("logs", exist_ok=True)
    best_path = os.path.join("logs", f"best_one_step_{tune}_sim{NUM_SIM}.json")
    with open(best_path, "w") as fh:
        json.dump({
            "method": tune,
            "cost":   float(incumbent_cost),
            "config": {k: ({"__np_int64__": int(v)} if isinstance(v, np.integer) else v)
                       for k, v in dict(incumbent).items()},
        }, fh, indent=2)
    print(f"Wrote {best_path}")

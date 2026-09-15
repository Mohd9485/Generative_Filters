"""
@author: Mohammad Al-Jarrah

Full particle-trajectory run of the Linear-Quadratic benchmark at one online
budget (FI): every learned filter is warm-started from a stage-1 spin-up
(Phase 1, trained here if missing) and then run at the stage-2 budget
(Phase 2). This is the data source for the trajectory and aa-SW2 figures, and
for the density figures in import_data.py.
"""

import os
import sys
import time
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# ----------------------------------------------------------------------
# Importing the filters from THIS directory, and proving that it happened
# ----------------------------------------------------------------------
# OTF.py, FMF.py, SIF.py and KRF.py each execute, at module scope,
#
#     sys.path.insert(0, '/home/mohd9485/Tutorial_project_dynamics')
#
# so that they can reach `timing_utils`. Whichever of them is imported first
# therefore puts an OLDER copy of this repository AHEAD of this one on the
# path, and every filter imported after it resolves there instead -- a run
# silently built from two different versions of the same code. Pinning this
# directory at the front of sys.path for each import, then checking where the
# module actually came from, turns that mistake into an error at line one.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _import_local(module_name, func_name):
    """
    Import one filter from the directory this script lives in.

    Parameters
    ----------
    module_name, func_name : str — e.g. ('SBF', 'SBF')

    Returns
    -------
    callable — the filter function

    Raises
    ------
    ImportError — when the module resolved to some other directory, which means
        sys.path is ordered such that a different copy of the repository wins.
    """
    saved = list(sys.path)
    sys.path.insert(0, _HERE)
    try:
        module = importlib.import_module(module_name)
    finally:
        # Restores the path exactly, undoing both the insert above and any
        # insert the imported module itself performed.
        sys.path[:] = saved

    where = os.path.dirname(os.path.abspath(module.__file__))
    if where != _HERE:
        raise ImportError(
            f'{module_name} resolved to {module.__file__!r}, not to {_HERE!r}. '
            f'Another copy of this repository is ahead on sys.path; the filters '
            f'there are a different version and must not be mixed with these.')
    return getattr(module, func_name)


EnKF = _import_local('EnKF', 'EnKF')
SIR  = _import_local('SIR',  'SIR')
OTF  = _import_local('OTF',  'OTF')
FMF  = _import_local('FMF',  'FMF')
SIF  = _import_local('SIF',  'SIF')
SBF  = _import_local('SBF',  'SBF')
KRF  = _import_local('KRF',  'KRF')

get_config          = _import_local('param_multi_steps_fi', 'get_config')
get_config_one_step = _import_local('param_one_step',       'get_config')

# ----------------------------------------------------------------------
# WHICH ONLINE BUDGET THIS RUN USES  —  set this before running
# ----------------------------------------------------------------------
# 1 | 4 | 16, resolving to a different absolute budget per method (fi x
# param_multi_steps_fi.FI_BASE[method]), since stage 2 re-searched the whole
# optimiser block -- not just the iteration count -- at every level:
#
#     fi        fmf/sif      otf      krf/sbf
#      1            256       16             8
#      4           1024       64            32
#     16           4096      256           128
FI = 4

# ----------------------------------------------------------------------
# The two phases
# ----------------------------------------------------------------------
#     Phase 1  param_one_step.get_config(method, L, dy)
#              WARM_START = 'auto', T_SPINUP steps, missing methods only
#              -> writes DATA/warm_start/<method>_quadratic_L<L>_dy<dy>_...pt
#
#     Phase 2  param_multi_steps_fi.get_config(method, L, dy, FI)
#              WARM_START = 'require', full T steps
#              -> loads exactly those weights, then runs the online budget
#
# The join is safe because the two parameter files agree on every
# architecture-defining key -- NUM_NEURON, num_resblocks, TIME_EMBED_DIM,
# depth, quad_points, NOISE_LEVEL, INFERENCE_MODE, DENOISER_WEIGHT -- for
# every learned method, and on the (sigma, gamma, normalization) triple that
# goes into the filename digest.
#
# 'auto' rather than 'save' for Phase 1: 'save' always retrains from scratch
# then overwrites, so a re-run would pay the offline cost again AND replace
# the weights Phase 2 was measured against. 'auto' trains only when nothing is
# on disk for that method, and the decision is made by the filter itself
# against its own _warm_start_path(), so it accounts for the digest too.
#
# 'require' rather than 'load' for Phase 2: the settings it runs were tuned
# against an already-trained network, so a silent from-scratch start would
# test them in a regime they were never fitted for. A missing or mismatched
# checkpoint is a loud, named failure on the first filter call instead.
WARM_START_PHASE1 = 'auto'      # 'off'  no cache, train from scratch
WARM_START_PHASE2 = 'require'   # 'auto' load if a checkpoint exists, else train and save
                                 # 'save' always train from scratch, then overwrite
                                 # 'load' load if a checkpoint exists, never write one
WARM_START_TAG    = 'quadratic'
WARM_START_DIR    = 'DATA/warm_start'

# Every filter writes its checkpoint at (k == 0, i == 0) and nowhere else, so
# two time steps -- one analysis step -- is all Phase 1 needs.
T_SPINUP = 2

# Retrain and overwrite EVERY checkpoint regardless of what is on disk. Set
# this after editing param_one_step.py, or to replace weights of unknown
# provenance.
FORCE_SPINUP = False

DATA_DIR = 'DATA'
os.makedirs(DATA_DIR, exist_ok=True)

matplotlib.rcParams['pdf.fonttype'] = 42   # embed TrueType fonts, which journals require
matplotlib.rcParams['ps.fonttype']  = 42
plt.rc('font', size=16)
plt.close('all')

randint = 390   # fixed seed; np.random.randint(0, 1000) draws a fresh one instead
print(randint)
np.random.seed(randint)
torch.manual_seed(randint)

# List the CUDA device indices to use. Defaults to a single GPU (or CPU if
# none is available); add more indices to dispatch filters concurrently
# across them.
GPU_IDS = [0]

if torch.cuda.is_available():
    n_visible = torch.cuda.device_count()
    devices = [torch.device(f'cuda:{g}') for g in GPU_IDS if g < n_visible] \
              or [torch.device('cuda:0')]
else:
    devices = [torch.device('cpu')]

print(f'Using device(s): {[str(d) for d in devices]}')


def _timed(func, *args):
    t0 = time.time()
    return func(*args), time.time() - t0


def h(x):
    return x[::2] * x[::2]   # observe every other state (0, 2, ...); n obs for L = 2n states

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


def A_block(x, t=0):
    """Dynamics of one decoupled 2D subsystem (the 2x2 F_block of F = kron(I_n, F_block))."""
    try:
        return F_block @ x
    except Exception:
        return torch.from_numpy(F_block).to(dtype=torch.float32, device=x.device) @ x


def aa_sw2(X, X_ref):
    """
    Axis-aligned sliced 2-Wasserstein distance between a particle ensemble and
    the reference posterior, resolved per (simulation, time step).

    Each state coordinate is one axis-aligned slice, so the L-dimensional
    comparison reduces to L one-dimensional W2 distances that are averaged.
    Each 1-D distance is evaluated from the empirical inverse CDFs on a common
    midpoint quantile grid; when both ensembles have the same particle count
    this reduces exactly to the L2 distance between the sorted samples, which
    is the objective the SMAC tuning scripts minimise.

    Parameters
    ----------
    X     : ndarray (NUM_SIM x T x L x N)      — filter particles
    X_ref : ndarray (NUM_SIM x T x L x N_true) — reference posterior particles,
                                                 one per simulation

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
    Rs = np.sort(X_ref, axis=-1)[..., i_r]       # (NUM_SIM, T, L, M)

    w2 = np.sqrt(np.mean((Xs - Rs) ** 2, axis=-1))   # (NUM_SIM, T, L)
    return w2.mean(axis=-1)                          # (NUM_SIM, T)


def qslice(A, M):
    """
    The M quantile-spaced order statistics of A along its last axis.

    aa_sw2 compares two ensembles on a shared grid of min(n_x, n_r) midpoint
    quantiles, so a reference kept as the FIRST M particles of a much larger SIR
    run discards everything the extra particles knew: its quantiles carry the
    Monte-Carlo error of an M-sample, not of the run that produced them.
    Evaluating the full run's empirical quantile function at the same M levels
    keeps the accuracy of all n particles while storing only M of them. Measured
    on the bimodal benchmark this drops the metric's sampling floor by ~2.6x
    (0.109 -> 0.042), and it lowers the reported error of the BEST methods most,
    since those sit closest to the floor.

    Parameters
    ----------
    A : ndarray — particles along the last axis, any leading shape.
    M : int     — number of quantile levels to keep (N_true / n_sample).

    Returns
    -------
    ndarray — A with its last axis replaced by M sorted order statistics.

    Note
    ----
    The result is SORTED along the last axis, so particle identity across time
    and across coordinates is destroyed. That is harmless for everything the
    reference is used for -- aa_sw2 is axis-aligned and the density figures are
    marginal -- but it is why the per-method ensembles are never passed through
    here: those are drawn as trajectories, which need identity preserved.
    """
    n = A.shape[-1]
    u = (np.arange(M) + 0.5) / M
    i = np.minimum((u * n).astype(int), n - 1)
    return np.sort(A, axis=-1)[..., i]

#%%
# Simulation parameters.
n   = 5
L   = n * 2             # number of states
tau = 1e-1               # time step
T   = int(5 / tau)       # number of time steps, 50 steps = 5 s — matches steps 4, 5, 6
dy  = n                  # number of states observed
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
NUM_SIM = 1            # independent simulations

parameters_otf     = get_config('otf',     L, dy, FI)
parameters_fmf     = get_config('fmf',     L, dy, FI)
parameters_sbf     = get_config('sbf',     L, dy, FI)
parameters_krf     = get_config('krf',     L, dy, FI)
parameters_sif_ode = get_config('sif_ode', L, dy, FI)  # eps>0, lambda=0, ODE inference
parameters_sif_sde = get_config('sif_sde', L, dy, FI)  # eps>0, lambda=1, SDE inference

# Method key -> (archive label used for its results/timing entries below, the
# filter function). Only the six LEARNED filters appear: EnKF and SIR train
# nothing and have no weights to spin up.
LEARNED = {
    'sbf':     ('SBF',         SBF),
    'krf':     ('KRF',         KRF),
    'otf':     ('OTF',         OTF),
    'fmf':     ('FMF',         FMF),
    'sif_ode': ('SIF_ODE',     SIF),
    'sif_sde': ('SIF_SDE',     SIF),
}

# Archive label -> the name shown in prints, plot titles and legends.
DISPLAY = {
    'EnKF': 'EnKF', 'SIR': 'SIR', 'OTF': 'OTF', 'FMF': 'FMF',
    'SIF_ODE': 'SIF-ODE', 'SIF_SDE': 'SIF-SDE',
    'SBF': 'SBF', 'KRF': 'KRF',
}

# Heaviest first, so the slowest jobs are not left for last in the log.
SPINUP_ORDER = ('sbf', 'krf', 'otf', 'fmf', 'sif_ode', 'sif_sde')

# Phase 1's parameters. The stage-2 dicts above are Phase 2's.
params_one_step = {m: get_config_one_step(m, L, dy) for m in LEARNED}

# Phase 2's, collected by the same key so the two phases can be indexed alike.
# These are the SAME dict objects as parameters_otf and friends, not copies, so
# the clamp below reaches both names.
params_multi_step = {
    'otf': parameters_otf,         'fmf': parameters_fmf,
    'sbf': parameters_sbf,         'krf': parameters_krf,
    'sif_ode': parameters_sif_ode, 'sif_sde': parameters_sif_sde,
}

# A mini-batch cannot exceed the ensemble it is drawn from — the tuned configs
# use BATCH_SIZE up to 768, above the small end of the N range. Both phases are
# clamped: a Phase 1 config that raised would leave Phase 2 with no checkpoint.
for phase_name, phase in (('one-step', params_one_step),
                          ('multi-step', params_multi_step)):
    for m, p in phase.items():
        if p['BATCH_SIZE'] > N:
            print(f'[warn] {phase_name} {m}: BATCH_SIZE {p["BATCH_SIZE"]} > N {N}, '
                  f'clamped — this is NOT the tuned configuration')
            p['BATCH_SIZE'] = N


# ----------------------------------------------------------------------
# NUM_SIM independent true trajectories, one filter run each
# ----------------------------------------------------------------------
# Every simulation draws its OWN state/observation trajectory, so the NUM_SIM
# runs of each filter differ in the data they see as well as in their initial
# ensemble X0 and their internal sampling noise. The spread across runs is
# therefore the end-to-end variability of the method over the data-generating
# distribution, not just filter variability at fixed data. Each run needs its
# own reference posterior, which is what the block below builds.
X_True = np.zeros((NUM_SIM, T, L,  1))
Y_True = np.zeros((NUM_SIM, T, dy, 1))
for k in range(NUM_SIM):
    X_True[k], Y_True[k] = Gen_True_Data(L, dy, T, sigma0, sigma, gamma, tau)

# Independent initial ensembles, one per run.
X0 = np.zeros((NUM_SIM, L, N))
for k in range(NUM_SIM):
    X0[k,] = np.random.multivariate_normal(np.zeros(L), sigma0 * sigma0 * np.eye(L), N).T


# ----------------------------------------------------------------------
# Reference posteriors — one per simulation
# ----------------------------------------------------------------------
# Build each reference posterior with n independent 2D SIR filters — one per
# decoupled block of F = kron(I_n, F_block). Running SIR in 2D rather than the
# full L = 2n space sidesteps the curse of dimensionality; the per-block
# posteriors are exact because the dynamics, process/observation noise, and
# initial prior all factorise across blocks, and block j is observed only
# through y index j (h observes state index 2j). The 2D block results are
# stacked back into the full-state reference. SIR's leading axis is the
# simulation axis, so all NUM_SIM trajectories are filtered in one call per
# block and run k gets its own reference.
#
# SIR is run with N_true_run particles for accuracy but only N_true
# quantile-spaced order statistics are kept (see qslice) — that is all the
# aa-SW2 comparison uses downstream.
N_true_run = int(1e5)   # particles used to *run* the reference SIR
# Matches the sweeps'. aa_sw2 resolves both ensembles at min(N, N_true)
# quantiles, so N_true sets the resolution -- and the finite-sample bias -- of
# the metric itself.
N_true = int(2e3)       # particles *kept* from it

X_ref = np.zeros((NUM_SIM, T, L, N_true))
for j in range(n):
    # Block j occupies state indices [2j, 2j+1] and is observed by y index j.
    X0_blk = np.zeros((NUM_SIM, 2, N_true_run))
    for k in range(NUM_SIM):
        X0_blk[k] = np.random.multivariate_normal(
            np.zeros(2), sigma0 * sigma0 * np.eye(2), N_true_run).T
    Y_blk = Y_True[:, :, j:j+1, :]                      # (NUM_SIM, T, 1, 1)
    X_blk = SIR(Y_blk, X0_blk, A_block, h, t, Noise, device=devices[0])  # (NUM_SIM, T, 2, N_true_run)
    # Quantile-spaced over ALL N_true_run particles, not the first N_true: see qslice.
    X_ref[:, :, 2*j:2*j+2, :] = qslice(X_blk, N_true)


#%%
# ----------------------------------------------------------------------
# PHASE 1 — spin-up with the stage-1 parameters, weights written to disk
# ----------------------------------------------------------------------
# Only the six LEARNED filters take part. The run is cut to T_SPINUP steps
# because each filter writes its checkpoint at (k == 0, i == 0) and never
# again -- everything after the first analysis step would be paid twice for no
# additional weights.
#
# Timing is not collected here: Phase 1 is the offline stage, and the cost the
# comparison reports is Phase 2's.
_ws1 = dict(warm_start=WARM_START_PHASE1, warm_start_tag=WARM_START_TAG,
            warm_start_dir=WARM_START_DIR)

os.makedirs(WARM_START_DIR, exist_ok=True)

Y_spin = Y_True[:, :T_SPINUP]
t_spin = t[:T_SPINUP]


def _tag_files():
    """Basename -> mtime for every checkpoint of this problem currently on disk."""
    return {f: os.path.getmtime(os.path.join(WARM_START_DIR, f))
            for f in os.listdir(WARM_START_DIR)
            if f'_{WARM_START_TAG}_' in f}


def _phase1_job(method):
    """Build one (label, runner) pair; a function so each closure keeps its own
    parameter dict rather than the loop variable's last value."""
    label, filter_fn = LEARNED[method]
    p = params_one_step[method]
    return (label, lambda d: filter_fn(Y_spin, X0, A, h, t_spin, Noise, p,
                                     device=d, **_ws1))


def _dispatch(jobs):
    """
    Run each job on one of the selected devices, cycling round-robin when
    there are more jobs than devices. Sequential when only one device is
    selected; concurrent, via a thread pool, when more than one is.

    Parameters
    ----------
    jobs : list of (name, runner) — runner takes one torch.device argument

    Returns
    -------
    (dict name -> filter output, dict name -> wall-clock seconds)
    """
    if len(devices) == 1:
        out, secs = {}, {}
        for name, runner in jobs:
            out[name], secs[name] = _timed(runner, devices[0])
        return out, secs

    out, secs = {}, {}
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = {name: executor.submit(_timed, runner, devices[i % len(devices)])
                   for i, (name, runner) in enumerate(jobs)}
        for name, future in futures.items():
            out[name], secs[name] = future.result()
    return out, secs


if FORCE_SPINUP:
    # 'auto' would find the existing files and merely load them, which is the
    # opposite of what forcing means; only 'save' retrains unconditionally.
    _ws1['warm_start'] = 'save'
    print('\nFORCE_SPINUP: retraining and overwriting every checkpoint.')

print(f'\n=== PHASE 1: stage-1 spin-up, {T_SPINUP} steps, '
      f'WARM_START={_ws1["warm_start"]!r}, tag={WARM_START_TAG!r} ===')
print('Checkpoints on disk before Phase 1:')
for fn in sorted(_tag_files()) or ['    (none)']:
    print('    %s' % fn)

# Every method is dispatched every run, and 'auto' decides PER METHOD whether
# that means training or loading, based on the filter's own _warm_start_path()
# -- which accounts for the digest (noise levels, normalisation, LayerNorm),
# not just the architecture. A file whose architecture matches but whose
# digest does not would otherwise be mistaken for a usable checkpoint, Phase 1
# would skip it, and Phase 2 would fail -- or, worse, quietly load it and
# produce wrong numbers.
_before   = _tag_files()
_t_phase1 = time.time()
_, phase1_runtimes = _dispatch([_phase1_job(m) for m in SPINUP_ORDER])
_after    = _tag_files()

# Whatever appeared or changed while Phase 1 ran is what it actually trained;
# everything else was loaded. Read back off the directory rather than
# inferred, so it stays correct whatever the filters choose to name their files.
phase1_written = sorted(f for f, mt in _after.items() if _before.get(f) != mt)

print(f'Phase 1 finished in {time.time() - _t_phase1:.1f} s')
for name, secs in phase1_runtimes.items():
    print(f'    {DISPLAY.get(name, name):12s}: {secs:8.2f} s')
if phase1_written:
    print('  trained and saved this run:')
    for fn in phase1_written:
        print('    %s' % fn)
else:
    print('  nothing written: every method loaded an existing checkpoint.')

print('\nCheckpoints in %s:' % WARM_START_DIR)
for fn in sorted(_tag_files()):
    print('    %s' % fn)


#%%
# ----------------------------------------------------------------------
# PHASE 2 — the stage-2 configurations at FI, warm-started from Phase 1
# ----------------------------------------------------------------------
# Per-method timing bucket, filled in place by each filter with a (NUM_SIM x T-1)
# array of per-analysis-step seconds. A defaultdict so each dict exists at the
# moment its closure is defined below, and so every job owns a separate one.
timings = defaultdict(dict)

# Warm-start arguments go to every LEARNED filter; EnKF and SIR train nothing and
# do not accept them. Under 'require' each one loads the checkpoint Phase 1 just
# wrote -- the one whose name carries this problem's tag, L, dy and architecture
# -- and raises FileNotFoundError naming the path if it is absent, so step 0 is a
# load, not a from-scratch fit, and the archived ensembles are the tuned filters'
# output.
_ws2 = dict(warm_start=WARM_START_PHASE2, warm_start_tag=WARM_START_TAG,
            warm_start_dir=WARM_START_DIR)

# Apply filtering methods using the true observations and initial particles.
# The expected data structure is: (NUM_SIM x T x L x N). Each entry pairs an
# archive label with a runner closure that takes one torch.device argument.
jobs_phase2 = [
    ('SBF',         lambda d: SBF(Y_True,  X0, A, h, t, Noise, parameters_sbf,     device=d, timing_out=timings['SBF'], **_ws2)),
    ('KRF',         lambda d: KRF(Y_True,  X0, A, h, t, Noise, parameters_krf,     device=d, timing_out=timings['KRF'], **_ws2)),
    ('OTF',         lambda d: OTF(Y_True,  X0, A, h, t, Noise, parameters_otf,     device=d, timing_out=timings['OTF'], **_ws2)),
    ('FMF',         lambda d: FMF(Y_True,  X0, A, h, t, Noise, parameters_fmf,     device=d, timing_out=timings['FMF'], **_ws2)),
    ('SIF_ODE',     lambda d: SIF(Y_True,  X0, A, h, t, Noise, parameters_sif_ode, device=d, timing_out=timings['SIF_ODE'], **_ws2)),
    ('SIF_SDE',     lambda d: SIF(Y_True,  X0, A, h, t, Noise, parameters_sif_sde, device=d, timing_out=timings['SIF_SDE'], **_ws2)),
    ('EnKF',        lambda d: EnKF(Y_True, X0, A, h, t, Noise, SIGMA=1e-6,         device=d, timing_out=timings['EnKF'])),
    ('SIR',         lambda d: SIR(Y_True,  X0, A, h, t, Noise,                     device=d, timing_out=timings['SIR'])),
]

results, runtimes = _dispatch(jobs_phase2)

# Unpack the results back into the per-method arrays expected downstream.
X_EnKF        = results['EnKF']
X_SIR         = results['SIR']
X_OTF         = results['OTF']
X_FMF         = results['FMF']
X_SIF_ODE     = results['SIF_ODE']
X_SIF_SDE     = results['SIF_SDE']
X_SBF         = results['SBF']
X_KRF         = results['KRF']

# Collect the per-method runtimes (seconds) into named variables.
time_EnKF        = runtimes['EnKF']
time_SIR         = runtimes['SIR']
time_OTF         = runtimes['OTF']
time_FMF         = runtimes['FMF']
time_SIF_ODE     = runtimes['SIF_ODE']
time_SIF_SDE     = runtimes['SIF_SDE']
time_SBF         = runtimes['SBF']
time_KRF         = runtimes['KRF']

# Column 0 of Phase 2 is the first analysis step. For the six learned filters
# it is a WARM start -- the checkpoint is loaded and the step is charged the
# ordinary online budget -- so it is not the from-scratch spin-up cost it would
# be under warm_start='off'. The genuine offline cost of those methods is Phase
# 1's runtime, printed above. EnKF and SIR have no cache, so their column 0
# still means what it always did. Every later step refines an already-warm
# network, which is the cost a deployed filter actually pays; averaging over
# both -- as a total divided by the step count does -- reports a number no
# step ever took.
steps      = {name: timings[name]['step_times'] for name in results}
first_step = {name: np.nanmean(s[:, 0])  for name, s in steps.items()}
online     = {name: np.nanmean(s[:, 1:]) for name, s in steps.items()}
online_std = {name: np.nanstd(s[:, 1:])  for name, s in steps.items()}

# The phase-1 column is what THIS run spent in Phase 1. For a method that
# trained, that is the real offline cost; for one that loaded an existing
# checkpoint it is a load plus a single warm analysis step, which is not the
# offline cost of those weights -- that was paid by the run named in
# phase1_written. The archives below store the raw per-step arrays; every
# reduction here is re-derivable from them.
print('\nComputational time (Phase 2; step 0 is a warm-started step for the learned '
      'filters, so the offline cost is Phase 1 above):')
for name, runtime in runtimes.items():
    if name in phase1_runtimes:
        offline = f'{phase1_runtimes[name]:8.2f} s'
    else:
        offline = '     n/a  '      # EnKF and SIR train nothing
    print(f'    {DISPLAY.get(name, name):12s}: phase-1 {offline}  |  step-0 {first_step[name]:7.4f} s  |  '
          f'online {online[name]:7.4f} +/- {online_std[name]:.4f} s/step  |  '
          f'phase-2 total {runtime:8.2f} s')

# ----------------------------------------------------------------------
# Axis-aligned sliced W2 against the reference posterior
# ----------------------------------------------------------------------
# One (NUM_SIM x T) array per method, each run scored against the reference
# posterior of its own trajectory -- so the spread over NUM_SIM combines filter
# variability with trajectory variability.
SW2 = {name: aa_sw2(X, X_ref) for name, X in results.items()}

print(f'\nTime-averaged aa-SW2 vs the reference posterior '
      f'(N_true={N_true} of {N_true_run} SIR particles, {NUM_SIM} run(s)):')
for name, sw2 in SW2.items():
    # Skip t = 0: every method starts from the same prior ensemble, before any update.
    per_sim = sw2[:, 1:].mean(axis=1)
    print(f'    {DISPLAY.get(name, name):12s}: {per_sim.mean():.4f} +/- {per_sim.std():.4f}')

sw2_EnKF        = SW2['EnKF']
sw2_SIR         = SW2['SIR']
sw2_OTF         = SW2['OTF']
sw2_FMF         = SW2['FMF']
sw2_SIF_ODE     = SW2['SIF_ODE']
sw2_SIF_SDE     = SW2['SIF_SDE']
sw2_SBF         = SW2['SBF']
sw2_KRF         = SW2['KRF']

# Persist the particle trajectories, true state, observations, and per-method
# runtimes. steps_<NAME> is the raw (NUM_SIM x T-1) per-analysis-step timing;
# column 0 is the offline spin-up and the rest is the online cost. Only the
# raw array is stored, not the spin-up/online reductions above -- they are one
# nanmean away, and a stored copy would be free to drift from the array it
# claims to summarise.
np.savez_compressed(os.path.join(DATA_DIR,
    f'DATA_file_Quadratic_FI_{FI}_n_{n}_NUM_SIM_{NUM_SIM}_randint_{randint}_N_{N}.npz'),
    randint=randint, t=t, Noise=Noise, tau=tau, n=n, L=L, dy=dy, N=N,
    # The budget this run used. Stored so an archive is self-describing: two runs
    # at different FI are different experiments, and the level alone does not say
    # what a method actually ran -- it resolves per method through FI_BASE.
    fi=FI,
    Final_Number_ITERATION=np.array(
        [p['Final_Number_ITERATION'] for p in
         (parameters_otf, parameters_fmf, parameters_sbf, parameters_krf,
          parameters_sif_ode, parameters_sif_sde)],
        dtype=np.int64),
    Final_Number_ITERATION_keys=np.array(
        ['otf', 'fmf', 'sbf', 'krf', 'sif_ode', 'sif_sde']),
    **{f'steps_{name}': s for name, s in steps.items()},
    # phase1_seconds_<NAME> is the offline cost the same method paid in Phase 1,
    # and is NaN for a method whose checkpoint was already on disk -- the cost
    # was real but was paid by an earlier run, so recording a zero would be a lie
    # and omitting the key would make the archive's shape depend on the cache.
    **{f'phase1_seconds_{lbl}': phase1_runtimes.get(lbl, np.nan)
       for lbl, _ in LEARNED.values()},
    # The checkpoint files Phase 1 actually wrote in this run; empty when every
    # method loaded one that already existed.
    phase1_written=np.array(phase1_written, dtype='<U128'),
    X0=X0, Y_true=Y_True, X_true=X_True,
    X_ref=X_ref, N_true=N_true, N_true_run=N_true_run,
    sw2_EnKF=sw2_EnKF,               sw2_SIR=sw2_SIR,
    sw2_OTF=sw2_OTF,                 sw2_FMF=sw2_FMF,
    sw2_SIF_ODE=sw2_SIF_ODE, sw2_SIF_SDE=sw2_SIF_SDE,
    sw2_SBF=sw2_SBF,                 sw2_KRF=sw2_KRF,
    X_EnKF=X_EnKF, time_EnKF=time_EnKF,
    X_SIR=X_SIR,   time_SIR=time_SIR,
    X_OTF=X_OTF,   time_OTF=time_OTF,
    X_FMF=X_FMF,   time_FMF=time_FMF,
    X_SIF_ODE=X_SIF_ODE, time_SIF_ODE=time_SIF_ODE,
    X_SIF_SDE=X_SIF_SDE,         time_SIF_SDE=time_SIF_SDE,
    X_SBF=X_SBF,   time_SBF=time_SBF,
    X_KRF=X_KRF,   time_KRF=time_KRF)


#%%
# Plot the results for each filtering method alongside the true state.
labeling = True  # set False to hide all axis labels

# (archive label, particle array, plot colour), in figure order. Display text
# comes from DISPLAY above, so a plot title/legend never drifts from a print.
METHODS = [
    ('EnKF',        X_EnKF,        'C0'),
    ('SIR',         X_SIR,         'C1'),
    ('OTF',         X_OTF,         'C2'),
    ('FMF',         X_FMF,         'C3'),
    ('SIF_ODE',     X_SIF_ODE,     'C4'),
    ('SIF_SDE',     X_SIF_SDE,     'C8'),
    ('SBF',         X_SBF,         'C6'),
    ('KRF',         X_KRF,         'C7'),
]

methods       = len(METHODS)
plot_particle = 500
k             = 0    # which of the NUM_SIM simulations (and its trajectory) to plot
plt.figure(figsize=(26, 16))
for col, (key, X, color) in enumerate(METHODS):
    for l in range(L):
        plt.subplot(L, methods, methods * l + col + 1)
        plt.plot(t, X[k, :, l, :plot_particle], color=color, alpha=0.1, rasterized=True)
        plt.plot(t, X_True[k, :, l], color='k', linestyle='--', label='True state')
        if labeling: plt.xlabel('time')
        if l == 0:
            plt.title(DISPLAY[key])
        if col == 0 and labeling:
            plt.ylabel(f'X({l + 1})')
        if l <= L-2:
            plt.gca().get_xaxis().set_visible(False)
        if col > 0:
            plt.gca().get_yaxis().set_visible(False)
        plt.ylim([-5,5])

plt.tight_layout()
plt.savefig(f'figs/Quadratic_FI_{FI}_n_{n}_NUM_SIM_{NUM_SIM}_randint_{randint}_N_{N}.pdf', bbox_inches='tight')

#%%
# Estimation error as a function of time: aa-SW2 against the reference
# posterior, averaged over the NUM_SIM independent trajectories.
fontsize = 16

plt.figure(figsize=(10, 6))
for key, _, color in METHODS:
    plt.plot(t, SW2[key].mean(axis=0), color=color, lw=2.5, label=DISPLAY[key])
if labeling: plt.xlabel('time',    fontsize=fontsize)
if labeling: plt.ylabel('aa-SW2',  fontsize=fontsize)
plt.yscale('log')
plt.legend(fontsize=fontsize - 4, ncol=2)
plt.tight_layout()
plt.savefig(f'figs/Quadratic_FI_{FI}_sw2_n_{n}_NUM_SIM_{NUM_SIM}_randint_{randint}_N_{N}.pdf', bbox_inches='tight')
plt.show()

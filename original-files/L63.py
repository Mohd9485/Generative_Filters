"""
@author: Mohammad Al-Jarrah

Lorenz-63 transfer test for the two-stage tuned configurations.

Every hyper-parameter this script runs was tuned on the linear-quadratic
benchmark (Quadratic.py, L = 10, dy = 5) and is used here UNCHANGED on a
different problem: a chaotic, fully coupled 3-D system observed through a
single linear component. Nothing is re-searched. The question the script
answers is how far the stage-2 incumbents transfer off the problem they were
fitted on.

Structure -- the two stages run back to back inside this one file:

    Phase 1  param_one_step.get_config(method, L, dy)
             WARM_START = 'auto', T_SPINUP steps, missing methods only
             -> writes DATA/warm_start/<method>_lorenz63_L3_dy1_<arch>_<digest>.pt

    Phase 2  param_multi_steps_fi.get_config(method, L, dy, FI)
             WARM_START = 'require', full T steps
             -> loads exactly those weights, then runs the online budget

The join is safe because the two files agree on every architecture-defining
key -- NUM_NEURON, num_resblocks, TIME_EMBED_DIM, depth, quad_points,
NOISE_LEVEL, INFERENCE_MODE, DENOISER_WEIGHT -- for all six learned methods,
and on the (sigma, gamma, normalization) triple that goes into the filename
digest. Phase 2 therefore finds the checkpoint Phase 1 wrote; 'require' rather
than 'load' makes any drift in that agreement a loud failure instead of a
silent from-scratch retrain.

Phase 1 stops after T_SPINUP steps on purpose: every filter writes its
checkpoint at (k == 0, i == 0) and never again, so the spin-up weights are
complete after the first analysis step and the remaining steps would only be
paid twice.

Phase 1 runs on every invocation, but under WARM_START = 'auto' it TRAINS only
what is missing: a method whose checkpoint is already on disk loads it and the
file is left untouched, so re-running this script re-uses the spin-up rather
than retraining and overwriting it. FORCE_SPINUP overrides that.
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

get_config_one_step    = _import_local('param_one_step',       'get_config')
get_config_multi_steps = _import_local('param_multi_steps_fi', 'get_config')

#%%
# ======================================================================
#                            CONFIGURATION
# ======================================================================
# Every constant this script reads, gathered here so a run can be set up
# without scrolling. Nothing below this block is a knob: the rest of the file
# is dynamics, dispatch, metric and figures, and reads these names as globals.
#
# Order inside the block matters in three places only -- T is derived from tau,
# t from both, and the assert checks the RK4 substeps against tau -- so those
# stay adjacent. Everything else is independent.

# ----------------------------------------------------------------------
# WHICH ONLINE BUDGET PHASE 2 USES
# ----------------------------------------------------------------------
# The level selects a whole tuned configuration, not just an iteration count:
# stage 2 re-searched the optimiser block at every level, so changing FI here
# changes the learning rate, batch size, weight decay, clip and warmup too.
#
# 1 | 4 | 16, resolving to a different absolute budget per method
# (fi x param_multi_steps_fi.FI_BASE[method]):
#
#     fi        fmf/sif      otf      krf/sbf
#      1            256       16             8
#      4           1024       64            32
#     16           4096      256           128
FI = 4

# ----------------------------------------------------------------------
# Warm start
# ----------------------------------------------------------------------
# The tag names the PROBLEM -- the one thing the filters cannot infer, since
# they receive A and h as opaque closures and would otherwise happily load a
# quadratic checkpoint into a Lorenz run that merely shared (L, dy).
WARM_START_TAG = 'lorenz63'
WARM_START_DIR = 'DATA/warm_start'

# Phase 1 trains the spin-up and writes the checkpoints; Phase 2 must find them.
#
# 'auto' rather than 'save'. 'save' means "always train from a random
# initialisation, then overwrite", so every re-run of this script paid the full
# offline cost again AND replaced the weights Phase 2 had already been measured
# against -- two runs of the same file would not be the same experiment.
# 'auto' writes only when nothing is on disk for that method, and on a hit
# loads and leaves the file alone. The decision is per method and is made by the
# filter itself against its own _warm_start_path(), so it accounts for the
# digest -- noise levels, normalisation, LayerNorm -- not just the architecture.
WARM_START_PHASE1 = 'save' #'auto'
WARM_START_PHASE2 = 'require'

# Retrain and overwrite EVERY checkpoint regardless of what is on disk -- the
# old 'save' behaviour. Set this when the spin-up itself must be redone: after
# editing param_one_step.py, or to replace weights of unknown provenance.
FORCE_SPINUP = False

# Every filter writes its checkpoint at (k == 0, i == 0) and nowhere else, so
# two time steps -- one analysis step -- is the whole of what Phase 1 needs.
T_SPINUP = 2

# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
DATA_DIR = 'DATA'    # every .npz archive; one place for the loaders to look
FIGS_DIR = 'figs'    # both PDFs

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
# The first line is the "draw a fresh seed" form; the second pins it. Delete
# the pin to randomise. Seeding itself happens in the SETUP block below.
randint = np.random.randint(0, 1000)  # random seed drawn at runtime
randint = 390

# ----------------------------------------------------------------------
# The Lorenz-63 system
# ----------------------------------------------------------------------
# Classic parameter set, at which the system is chaotic with a largest Lyapunov
# exponent of about 0.9 -- so one Lyapunov time is roughly 1.1 time units and
# the T*tau = 5 units below span some 4.5 of them.
SIGMA_L63 = 10.0
RHO_L63   = 28.0
BETA_L63  = 8.0 / 3.0

# One analysis interval is tau; it is covered by N_SUB Runge-Kutta 4 substeps of
# length DT_SUB. Splitting the interval matters here in a way it did not for the
# linear-quadratic problem: a single explicit step of tau = 0.1 on a system with
# these time scales is not accurate enough for the propagated ensemble to be a
# forecast of the same system the truth was drawn from.
DT_SUB = 1e-2
N_SUB  = 10

# ----------------------------------------------------------------------
# Simulation parameters
# ----------------------------------------------------------------------
L   = 3         # number of states in the Lorenz system
dy  = 1         # only the third component is observed
tau = 1e-1      # analysis / observation interval = N_SUB * DT_SUB
T   = int(5 / tau)               # 50 analysis steps, 5 time units
t   = np.arange(0.0, tau * T, tau)

assert abs(N_SUB * DT_SUB - tau) < 1e-12, \
    'the RK4 substeps must tile exactly one analysis interval'

# Noise parameters, carried over from the previous version of this file.
sigma  = np.sqrt(10) / 10   # process noise std ASSUMED by the filters
gamma  = np.sqrt(10) * 1    # observation noise std
Noise  = [sigma, gamma]

sigma0 = 10                 # std of the initial state distribution

# Above every BATCH_SIZE the tuned configurations carry (the largest is KRF's
# 768), so the clamp further down never fires and the stage-2 incumbents run
# exactly as they were searched. That is the whole point of the transfer test: a
# clamped batch size would mean measuring a configuration nobody ever tuned.
N       = int(1000) # particles per filter
NUM_SIM = 1

# ----------------------------------------------------------------------
# Reference posterior
# ----------------------------------------------------------------------
# Lorenz-63 does not factorise, so unlike the quadratic benchmark the reference
# is one SIR run in the full state rather than an exact per-block posterior.
# Why that is the best available yardstick, and what it costs, is set out where
# the reference is actually built.
N_true_run = int(1e6)   # particles used to *run* the reference SIR
N_true     = int(2e3)   # particles *kept* from it, matching the quadratic study

# ----------------------------------------------------------------------
# The methods
# ----------------------------------------------------------------------
# Every learned method, in the two forms the script needs: METHOD_KEYS drives
# the parameter lookups, LEARNED maps each key to its archive label and its
# filter function. EnKF and SIR appear in neither -- they train nothing.
METHOD_KEYS = ('otf', 'fmf', 'sif_ode', 'sif_sde', 'sbf', 'krf')

# Method key -> (archive label used for its results/timing entries below, the
# filter function).
LEARNED = {
    'sbf':     ('SBF',       SBF),
    'krf':     ('KRF',       KRF),
    'otf':     ('OTF',       OTF),
    'fmf':     ('FMF',       FMF),
    'sif_ode': ('SIF_ODE',   SIF),
    'sif_sde': ('SIF_SDE',   SIF),
}

# Archive label -> the name shown in prints, plot titles and legends.
DISPLAY = {
    'EnKF': 'EnKF', 'SIR': 'SIR', 'OTF': 'OTF', 'FMF': 'FMF',
    'SIF_ODE': 'SIF-ODE', 'SIF_SDE': 'SIF-SDE',
    'SBF': 'SBF', 'KRF': 'KRF',
}

# Heaviest first, so the slowest jobs are not left for last in the log.
SPINUP_ORDER = ('sbf', 'krf', 'otf', 'fmf', 'sif_ode', 'sif_sde')

# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
plot_particle = 500    # particle trajectories drawn per panel
PLOT_SIM      = 0      # which of the NUM_SIM simulations to plot
labeling      = True   # set False to hide all axis labels
fontsize      = 16

# ======================================================================
#                               SETUP
# ======================================================================
# The side effects the constants above imply. Kept apart from them so the
# configuration block stays a list of values and nothing else.

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(WARM_START_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# fonttype 42 embeds TrueType fonts, which the journals require.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=fontsize)

plt.close('all')

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


# ----------------------------------------------------------------------
# The Lorenz-63 system
# ----------------------------------------------------------------------
def L63_drift(x):
    """
    The Lorenz-63 vector field dx/dt, evaluated for a whole ensemble at once.

    Parameters
    ----------
    x : ndarray or torch.Tensor, shape (3, N)
        States as COLUMNS -- the layout every filter in this repository uses
        when it calls the dynamics, i.e. A(X.T, t).T with X stored (N x 3).

    Returns
    -------
    ndarray or torch.Tensor, shape (3, N) — the time derivative, same type,
    dtype and device as x.
    """
    d = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)

    d[0] = SIGMA_L63 * (x[1] - x[0])
    d[1] = x[0] * (RHO_L63 - x[2]) - x[1]
    d[2] = x[0] * x[1] - BETA_L63 * x[2]
    return d


def A(x, t=0):
    """
    Advance the state one analysis interval with classical RK4.

    Replaces the explicit Euler step the file used previously. The signature is
    the one every filter expects -- A(state, time) with the state as (L x N) --
    and the return is the state at t + tau, NOT a derivative: the filters add
    the process noise themselves.

    Parameters
    ----------
    x : ndarray or torch.Tensor, shape (3, N)
    t : float — current time; the vector field is autonomous, so it is accepted
        for interface compatibility and not used.

    Returns
    -------
    Same type and shape as x — the state advanced by tau = N_SUB * DT_SUB.
    """
    for _ in range(N_SUB):
        k1 = L63_drift(x)
        k2 = L63_drift(x + 0.5 * DT_SUB * k1)
        k3 = L63_drift(x + 0.5 * DT_SUB * k2)
        k4 = L63_drift(x + DT_SUB * k3)
        x  = x + (DT_SUB / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def h(x):
    """
    Observation operator: the third state component alone.

    dy = 1 against L = 3, so two of the three coordinates are never observed
    directly and must be recovered through the coupling in the dynamics. That
    is a considerably harder inverse problem than the quadratic benchmark's,
    where every 2-D block carried its own observation.

    Parameters
    ----------
    x : ndarray or torch.Tensor, shape (3, N)

    Returns
    -------
    Same type, shape (1, N).
    """
    return x[2, :].reshape(dy, -1)


def Gen_True_Data(L, dy, T, sigma0, sigma, gamma, tau):
    """
    Generate the true state trajectory and the observations.

    The truth is propagated NOISELESSLY through the RK4 integrator -- a
    perfect-model Lorenz orbit -- while the filters below all assume process
    noise of level sigma. That mismatch is deliberate and is the usual
    convention for L63 filtering studies; its consequence is that the reference
    posterior computed further down is the posterior of the ASSUMED model, not
    of the process that actually generated the data.

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
        Standard deviation of the process noise the FILTERS assume; unused
        here, since the truth is noiseless. Kept in the signature so the
        function reads the same as its counterpart in Quadratic.py.
    gamma : float
        Standard deviation for the observation noise.
    tau : float
        Analysis interval (the amount A advances per call).

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
        x[i + 1, :] = A(x[i, :], t[i])
        y[i + 1, :] = h(x[i + 1, :]) + np.random.multivariate_normal(
            np.zeros(dy), gamma * gamma * np.eye(dy), 1).T

    return x, y


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


def qslice(A_arr, M):
    """
    The M quantile-spaced order statistics of A_arr along its last axis.

    aa_sw2 compares two ensembles on a shared grid of min(n_x, n_r) midpoint
    quantiles, so a reference kept as the FIRST M particles of a much larger SIR
    run discards everything the extra particles knew: its quantiles carry the
    Monte-Carlo error of an M-sample, not of the run that produced them.
    Evaluating the full run's empirical quantile function at the same M levels
    keeps the accuracy of all n particles while storing only M of them.

    Parameters
    ----------
    A_arr : ndarray — particles along the last axis, any leading shape.
    M     : int     — number of quantile levels to keep.

    Returns
    -------
    ndarray — A_arr with its last axis replaced by M sorted order statistics.

    Note
    ----
    The result is SORTED along the last axis, so particle identity across time
    and across coordinates is destroyed. That is harmless for everything the
    reference is used for -- aa_sw2 is axis-aligned and the density figures are
    marginal -- but it is why the per-method ensembles are never passed through
    here: those are drawn as trajectories, which need identity preserved.
    """
    n = A_arr.shape[-1]
    u = (np.arange(M) + 0.5) / M
    i = np.minimum((u * n).astype(int), n - 1)
    return np.sort(A_arr, axis=-1)[..., i]


#%%
# ----------------------------------------------------------------------
# The two parameter sets
# ----------------------------------------------------------------------
# Phase 1 (stage 1) and Phase 2 (stage 2 at FI). They agree on every key that
# fixes the shape or the meaning of the weights, which is what lets Phase 2 load
# what Phase 1 wrote; they differ in the optimiser block and in the online
# iteration budget, which is what stage 2 re-searched.
params_one_step   = {m: get_config_one_step(m, L, dy)        for m in METHOD_KEYS}
params_multi_step = {m: get_config_multi_steps(m, L, dy, FI) for m in METHOD_KEYS}

# A mini-batch cannot exceed the ensemble it is drawn from. FMF, SIF and SBF
# take the batch as randperm(N)[:BATCH_SIZE] but size other tensors from
# BATCH_SIZE directly, so BATCH_SIZE > N raises a shape mismatch. At N = 1024
# nothing is clamped; the guard stays so a smaller N fails visibly rather than
# deep inside a filter.
for phase_name, phase in (('one-step', params_one_step), ('multi-step', params_multi_step)):
    for m, p in phase.items():
        if p['BATCH_SIZE'] > N:
            print(f'[warn] {phase_name} {m}: BATCH_SIZE {p["BATCH_SIZE"]} > N {N}, '
                  f'clamped — this is NOT the tuned configuration')
            p['BATCH_SIZE'] = N

# ----------------------------------------------------------------------
# Truth, observations and the initial ensembles
# ----------------------------------------------------------------------
X_True = np.zeros((NUM_SIM, T, L,  1))
Y_True = np.zeros((NUM_SIM, T, dy, 1))
for k in range(NUM_SIM):
    X_True[k], Y_True[k] = Gen_True_Data(L, dy, T, sigma0, sigma, gamma, tau)

X0 = np.zeros((NUM_SIM, L, N))
for k in range(NUM_SIM):
    X0[k,] = np.random.multivariate_normal(np.zeros(L), sigma0 * sigma0 * np.eye(L), N).T


#%%
# ----------------------------------------------------------------------
# Reference posterior
# ----------------------------------------------------------------------
# Lorenz-63 is fully coupled, so the block decomposition that made the quadratic
# benchmark's reference EXACT has no analogue here: there is no sub-system whose
# posterior factorises. The reference is therefore a single SIR run in the full
# 3-D state with a very large ensemble -- accurate, but not exact, and worth
# reading with two caveats in mind.
#
# First, dy = 1 against L = 3 over a chaotic orbit is where a bootstrap particle
# filter degenerates fastest; the effective sample size after resampling is well
# below N_true_run and falls further as the trajectory separates. Second, the
# reference assumes the same process noise sigma the filters do, while the truth
# was propagated noiselessly -- so it is the posterior of the assumed model.
#
# Both caveats affect every method identically, which is what the comparison
# needs: aa-SW2 here ranks methods against a common yardstick rather than
# measuring absolute distance to a known posterior.
X0_ref = np.zeros((NUM_SIM, L, N_true_run))
for k in range(NUM_SIM):
    X0_ref[k] = np.random.multivariate_normal(
        np.zeros(L), sigma0 * sigma0 * np.eye(L), N_true_run).T

print(f'Reference posterior: SIR in {L}-D with {N_true_run} particles ...')
_t_ref = time.time()
X_ref_run = SIR(Y_True, X0_ref, A, h, t, Noise, device=devices[0])
# Quantile-spaced over ALL N_true_run particles, not the first N_true: same
# N_true columns stored, far sharper quantiles. See qslice.
X_ref = qslice(X_ref_run, N_true)
del X_ref_run
print(f'    done in {time.time() - _t_ref:.1f} s')


#%%
# ----------------------------------------------------------------------
# PHASE 1 — spin-up with the stage-1 parameters, weights written to disk
# ----------------------------------------------------------------------
# Only the six LEARNED filters take part; EnKF and SIR train nothing and have
# no weights to save. The run is cut to T_SPINUP steps because each filter
# writes its checkpoint at (k == 0, i == 0) and never again -- everything after
# the first analysis step would be paid twice for no additional weights.
#
# Timing is not collected here: Phase 1 is the offline stage, and the cost the
# comparison reports is Phase 2's.
_ws1 = dict(warm_start=WARM_START_PHASE1, warm_start_tag=WARM_START_TAG,
            warm_start_dir=WARM_START_DIR)

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
# everything else was loaded. Read back off the directory rather than inferred,
# so it stays correct whatever the filters choose to name their files.
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


# The checkpoints Phase 2 is about to require.
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

# 'require' rather than 'load': the stage-2 settings were tuned against an
# already-trained network, so a silent from-scratch start would exercise them in
# a regime they were never fitted for, and nothing downstream would show which
# methods those were. A missing or mismatched checkpoint is a named failure on
# the first filter call instead.
_ws2 = dict(warm_start=WARM_START_PHASE2, warm_start_tag=WARM_START_TAG,
            warm_start_dir=WARM_START_DIR)

# Each entry pairs an archive label with a runner closure that takes one torch.device argument.
jobs_phase2 = [
    ('SBF',      lambda d: SBF(Y_True,  X0, A, h, t, Noise, params_multi_step['sbf'],     device=d, timing_out=timings['SBF'], **_ws2)),
    ('KRF',      lambda d: KRF(Y_True,  X0, A, h, t, Noise, params_multi_step['krf'],     device=d, timing_out=timings['KRF'], **_ws2)),
    ('OTF',      lambda d: OTF(Y_True,  X0, A, h, t, Noise, params_multi_step['otf'],     device=d, timing_out=timings['OTF'], **_ws2)),
    ('FMF',      lambda d: FMF(Y_True,  X0, A, h, t, Noise, params_multi_step['fmf'],     device=d, timing_out=timings['FMF'], **_ws2)),
    ('SIF_ODE',  lambda d: SIF(Y_True,  X0, A, h, t, Noise, params_multi_step['sif_ode'], device=d, timing_out=timings['SIF_ODE'], **_ws2)),
    ('SIF_SDE',  lambda d: SIF(Y_True,  X0, A, h, t, Noise, params_multi_step['sif_sde'], device=d, timing_out=timings['SIF_SDE'], **_ws2)),
    ('EnKF',     lambda d: EnKF(Y_True, X0, A, h, t, Noise, SIGMA=1e-6,                   device=d, timing_out=timings['EnKF'])),
    ('SIR',      lambda d: SIR(Y_True,  X0, A, h, t, Noise,                               device=d, timing_out=timings['SIR'])),
]

print(f'\n=== PHASE 2: stage-2 configurations at FI={FI}, {T} steps, '
      f'WARM_START={WARM_START_PHASE2!r} ===')
results, runtimes = _dispatch(jobs_phase2)

# Unpack the results back into the per-method arrays expected downstream.
X_EnKF    = results['EnKF']
X_SIR     = results['SIR']
X_OTF     = results['OTF']
X_FMF     = results['FMF']
X_SIF_ODE = results['SIF_ODE']
X_SIF_SDE = results['SIF_SDE']
X_SBF     = results['SBF']
X_KRF     = results['KRF']

# Collect the per-method runtimes (seconds) into named variables.
time_EnKF    = runtimes['EnKF']
time_SIR     = runtimes['SIR']
time_OTF     = runtimes['OTF']
time_FMF     = runtimes['FMF']
time_SIF_ODE = runtimes['SIF_ODE']
time_SIF_SDE = runtimes['SIF_SDE']
time_SBF     = runtimes['SBF']
time_KRF     = runtimes['KRF']

# Column 0 is Phase 2's first analysis step. For the six learned filters this
# is a warm start -- the checkpoint is loaded and the step is charged the
# ordinary online budget -- so, unlike in Quadratic.py, it is NOT a from-scratch
# spin-up cost. The genuine offline cost of these methods is Phase 1's runtime,
# printed above. EnKF and SIR have no cache, so their column 0 means what it
# always did.
steps      = {name: timings[name]['step_times'] for name in results}
first_step = {name: np.nanmean(s[:, 0])  for name, s in steps.items()}
online     = {name: np.nanmean(s[:, 1:]) for name, s in steps.items()}
online_std = {name: np.nanstd(s[:, 1:])  for name, s in steps.items()}

# The phase-1 column is what THIS run spent in Phase 1. For a method that
# trained, that is the real offline cost; for one that loaded an existing
# checkpoint it is a load plus a single warm analysis step, which is a far
# smaller number and is not the offline cost of those weights -- that was paid
# by the run named in phase1_written. The two cases are told apart by that list,
# not by this column.
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
SW2 = {name: aa_sw2(X, X_ref) for name, X in results.items()}

print(f'\nTime-averaged aa-SW2 vs the reference posterior '
      f'(N_true={N_true} of {N_true_run} SIR particles, {NUM_SIM} run(s)):')
for name, sw2 in SW2.items():
    # Skip t = 0: every method starts from the same prior ensemble, before any update.
    per_sim = sw2[:, 1:].mean(axis=1)
    print(f'    {DISPLAY.get(name, name):12s}: {per_sim.mean():.4f} +/- {per_sim.std():.4f}')

sw2_EnKF    = SW2['EnKF']
sw2_SIR     = SW2['SIR']
sw2_OTF     = SW2['OTF']
sw2_FMF     = SW2['FMF']
sw2_SIF_ODE = SW2['SIF_ODE']
sw2_SIF_SDE = SW2['SIF_SDE']
sw2_SBF     = SW2['SBF']
sw2_KRF     = SW2['KRF']

# Persist the particle trajectories, true state, observations, and per-method
# runtimes. steps_<NAME> is the raw (NUM_SIM x T-1) per-analysis-step timing
# from Phase 2; phase1_seconds_<NAME> is the offline cost the same method paid
# in Phase 1, and is NaN for a method whose checkpoint was already on disk --
# the cost was real but was paid by an earlier run, so recording a zero would
# be a lie and omitting the key would make the archive's shape depend on the
# cache. Only the raw arrays are stored, not the reductions above -- they are
# one nanmean away, and a stored copy would be free to drift from the array it
# summarises.
np.savez_compressed(os.path.join(DATA_DIR,
    f'DATA_file_L63_FI_{FI}_NUM_SIM_{NUM_SIM}_randint_{randint}_N_{N}.npz'),
    randint=randint, t=t, Noise=Noise, tau=tau, L=L, dy=dy, N=N,
    # The Lorenz-63 setup, so an archive is self-describing.
    sigma_l63=SIGMA_L63, rho_l63=RHO_L63, beta_l63=BETA_L63,
    dt_sub=DT_SUB, n_sub=N_SUB, integrator='rk4',
    # The budget this run used. Two runs at different FI are different
    # experiments, and the level alone does not say what a method actually ran --
    # it resolves per method through FI_BASE.
    fi=FI,
    warm_start_tag=WARM_START_TAG,
    Final_Number_ITERATION=np.array(
        [params_multi_step[m]['Final_Number_ITERATION'] for m in METHOD_KEYS],
        dtype=np.int64),
    Final_Number_ITERATION_keys=np.array(METHOD_KEYS),
    **{f'steps_{name}': s for name, s in steps.items()},
    **{f'phase1_seconds_{lbl}': phase1_runtimes.get(lbl, np.nan)
       for lbl, _ in LEARNED.values()},
    # The checkpoint files Phase 1 actually wrote in this run; empty when every
    # method loaded one that already existed.
    phase1_written=np.array(phase1_written, dtype='<U128'),
    X0=X0, Y_true=Y_True, X_true=X_True,
    X_ref=X_ref, N_true=N_true, N_true_run=N_true_run,
    sw2_EnKF=sw2_EnKF,           sw2_SIR=sw2_SIR,
    sw2_OTF=sw2_OTF,             sw2_FMF=sw2_FMF,
    sw2_SIF_ODE=sw2_SIF_ODE,     sw2_SIF_SDE=sw2_SIF_SDE,
    sw2_SBF=sw2_SBF,             sw2_KRF=sw2_KRF,
    X_EnKF=X_EnKF, time_EnKF=time_EnKF,
    X_SIR=X_SIR,   time_SIR=time_SIR,
    X_OTF=X_OTF,   time_OTF=time_OTF,
    X_FMF=X_FMF,   time_FMF=time_FMF,
    X_SIF_ODE=X_SIF_ODE, time_SIF_ODE=time_SIF_ODE,
    X_SIF_SDE=X_SIF_SDE, time_SIF_SDE=time_SIF_SDE,
    X_SBF=X_SBF,   time_SBF=time_SBF,
    X_KRF=X_KRF,   time_KRF=time_KRF)


#%%
# Plot the results for each filtering method alongside the true state.
# (archive label, particle array, plot colour), in figure order. Display text
# comes from DISPLAY above, so a plot title/legend never drifts from a print.
METHODS = [
    ('EnKF',    X_EnKF,    'C0'),
    ('SIR',     X_SIR,     'C1'),
    ('OTF',     X_OTF,     'C2'),
    ('FMF',     X_FMF,     'C3'),
    ('SIF_ODE', X_SIF_ODE, 'C4'),
    ('SIF_SDE', X_SIF_SDE, 'C8'),
    ('SBF',     X_SBF,     'C6'),
    ('KRF',     X_KRF,     'C7'),
]

methods = len(METHODS)

# One y-range per state, taken from the truth and padded. Unlike the quadratic
# figures there is no single sensible limit for all three rows: x1 and x2 live on
# roughly [-20, 20] while x3 is positive and reaches ~50, so a shared limit would
# flatten two rows to save the third.
ylims = []
for l in range(L):
    lo, hi = X_True[PLOT_SIM, :, l].min(), X_True[PLOT_SIM, :, l].max()
    pad    = 0.6 * (hi - lo) + 1e-9
    ylims.append((lo - pad, hi + pad))

plt.figure(figsize=(26, 16))
for col, (key, X, color) in enumerate(METHODS):
    for l in range(L):
        plt.subplot(L, methods, methods * l + col + 1)
        plt.plot(t, X[PLOT_SIM, :, l, :plot_particle], color=color, alpha=0.1, rasterized=True)
        plt.plot(t, X_True[PLOT_SIM, :, l], color='k', linestyle='--', label='True state')
        if labeling: plt.xlabel('time')
        if l == 0:
            plt.title(DISPLAY[key])
        if col == 0 and labeling:
            plt.ylabel(f'X({l + 1})')
        if l <= L - 2:
            plt.gca().get_xaxis().set_visible(False)
        if col > 0:
            plt.gca().get_yaxis().set_visible(False)
        plt.ylim(ylims[l])

plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/L63_FI_{FI}_NUM_SIM_{NUM_SIM}_randint_{randint}_N_{N}.pdf',
            bbox_inches='tight')

#%%
# Estimation error as a function of time: aa-SW2 against the reference
# posterior, averaged over the NUM_SIM independent trajectories.
plt.figure(figsize=(10, 6))
for key, _, color in METHODS:
    plt.plot(t, SW2[key].mean(axis=0), color=color, lw=2.5, label=DISPLAY[key])
if labeling: plt.xlabel('time',   fontsize=fontsize)
if labeling: plt.ylabel('aa-SW2', fontsize=fontsize)
plt.yscale('log')
plt.legend(fontsize=fontsize - 4, ncol=2)
plt.tight_layout()
plt.savefig(f'{FIGS_DIR}/L63_FI_{FI}_sw2_NUM_SIM_{NUM_SIM}_randint_{randint}_N_{N}.pdf',
            bbox_inches='tight')

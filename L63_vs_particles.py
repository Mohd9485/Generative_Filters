"""
@author: Mohammad Al-Jarrah

Lorenz-63: error and cost against ensemble size, at one online budget.

The Lorenz-63 counterpart of Quadratic_vs_particles.py, and the same experiment:
only `N` changes, at **test time only**. Nothing is retuned, and no new weights
are trained -- every hyper-parameter is a stage-2 incumbent searched on the
linear-quadratic benchmark (`L = 10, dy = 5`) and carried over unchanged, exactly
as in L63.py. This file adds the `N` axis to that transfer test.

✅ **No weight re-save is needed, and none is done here.** The checkpoint digest
deliberately excludes `N` and the ensemble size -- the network has the same shape
at any `N`, only the training data drawn from the ensemble changes -- so ONE set
of `lorenz63_L3_dy1` checkpoints serves the whole sweep. They are the ones
`L63.py` Phase 1 wrote, and this script loads them under `warm_start='require'`
without ever writing: it is a consumer of that cache, not a producer.

⚠️ **Run `L63.py` once before this file.** Under `'require'` a missing checkpoint
is a loud failure on the first filter call rather than a silent from-scratch
retrain, and the guard below turns it into a failure at line one instead -- after
the reference posterior has been paid for would be a poor time to find out.

What differs from the quadratic sweep, and why
----------------------------------------------
* **The reference posterior is not exact.** The quadratic benchmark factorises
  into decoupled 2-D blocks, each with its own observation, so its reference is a
  per-block SIR that is exact up to Monte Carlo. Lorenz-63 does not factorise and
  is observed through one component of three, so the reference here is a single
  SIR run in the full 3-D state with `N_true_run` particles. It is the best
  available yardstick, not ground truth.
* **The reference is quantile-spaced, not sliced.** `qslice` evaluates the full
  run's empirical quantile function at `N_true` levels, which keeps the accuracy
  of all `N_true_run` particles while storing only `N_true`. Taking the first
  `N_true` instead would throw that away -- see qslice's docstring.
* **The sampling floor needs a raw draw.** The floor must be an INDEPENDENT
  N-particle sample, so it cannot be quantile-spaced (that would sharpen it and
  report a floor no real ensemble attains). The reference run is therefore
  partitioned: the last `max(N_LIST)` particles are reserved as raw floor draws,
  and the reference is `qslice` of everything before them. The two are disjoint
  by construction.
* **The truth is propagated noiselessly** through RK4, while the filters assume
  process noise `sigma`. That mismatch is the usual L63 convention and is
  inherited from L63.py; its consequence is that the reference is the posterior
  of the ASSUMED model, not of the process that generated the data.

Memory
------
The binding constraint is `NUM_SIM x N_true_run`: the reference SIR materialises
`(NUM_SIM, T, L, N_true_run)` on the GPU and again on the host before `qslice`
reduces it. At the defaults below that is ~3 GB on the device and ~6 GB on the
host, transient. Raising NUM_SIM for tighter error bars scales both linearly;
lower `N_true_run` to compensate if a card is small.

Outputs
-------
DATA_file_L63_vs_particles_fi{FI}_N_{N}.npz : per-size archive — per-time-step
    aa-SW2, runtimes, the raw per-analysis-step timings steps_<KEY> of shape
    (NUM_SIM x T-1), and that size's sampling floor. Particles are omitted
    unless SAVE_PARTICLES.
DATA_file_L63_vs_particles_fi{FI}.npz       : the sweep summary — sw2_mean,
    sw2_std, runtime, spinup, online, online_std as
    (len(METHODS) x len(N_LIST)) matrices in method_keys row order, plus
    sw2_floor of shape (len(N_LIST),).
figs/L63_sw2_vs_particles_fi{FI}.pdf        : aa-SW2 against the ensemble size.
"""

import os
import sys
import time
import importlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# ----------------------------------------------------------------------
# Importing the filters from THIS directory, and proving that it happened
# ----------------------------------------------------------------------
# OTF.py, FMF.py, SIF.py and KRF.py each execute, at module scope,
#
#     sys.path.insert(0, '/home/mohd9485/Tutorial_project_dynamics')
#
# so whichever is imported first can put another copy of this repository ahead
# of this one on the path, and every filter imported after it resolves there
# instead. The module names match, so nothing complains -- a run built from two
# versions of the same code. Pinning this directory per import and checking
# where each module actually came from turns that into an error at line one.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _import_local(module_name, func_name):
    """
    Import one name from the directory this script lives in.

    Raises
    ------
    ImportError — when the module resolved elsewhere, which means sys.path is
        ordered such that a different copy of the repository wins.
    """
    saved = list(sys.path)
    sys.path.insert(0, _HERE)
    try:
        module = importlib.import_module(module_name)
    finally:
        sys.path[:] = saved
    where = os.path.dirname(os.path.abspath(module.__file__))
    if where != _HERE:
        raise ImportError(
            f'{module_name} resolved to {module.__file__!r}, not to {_HERE!r}. '
            f'Another copy of this repository is ahead on sys.path.')
    return getattr(module, func_name)


EnKF = _import_local('EnKF', 'EnKF')
SIR  = _import_local('SIR',  'SIR')
OTF  = _import_local('OTF',  'OTF')
FMF  = _import_local('FMF',  'FMF')
SIF  = _import_local('SIF',  'SIF')
SBF  = _import_local('SBF',  'SBF')
KRF  = _import_local('KRF',  'KRF')

get_config = _import_local('param_multi_steps_fi', 'get_config')

# Every .npz archive this script writes goes here.
DATA_DIR = 'DATA'
os.makedirs(DATA_DIR, exist_ok=True)

# fonttype 42 embeds TrueType fonts, which the journals require.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=16)
plt.close('all')

fontsize = 16
labeling = True  # set False to hide all axis labels

# ----------------------------------------------------------------------
# The online budget this sweep is held at
# ----------------------------------------------------------------------
# One level for the whole sweep: the point here is the ensemble size, so the
# tuned configuration is held fixed and only N varies. The level selects a whole
# configuration, not just an iteration count -- stage 2 re-searched the optimiser
# block at every level.
FI = 4

# ----------------------------------------------------------------------
# Warm start — consumed, never written
# ----------------------------------------------------------------------
# 'require' rather than 'load': the stage-2 settings were tuned against an
# already-trained network, so a silent from-scratch start would exercise them in
# a regime they were never fitted for. The tag must match what L63.py Phase 1
# saved under.
WARM_START     = 'require'
WARM_START_TAG = 'lorenz63'
WARM_START_DIR = 'DATA/warm_start'

# --- Reproducibility ---
randint = np.random.randint(0, 1000)
randint = 390
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


# ----------------------------------------------------------------------
# The Lorenz-63 system
# ----------------------------------------------------------------------
# Classic parameter set, at which the system is chaotic with a largest Lyapunov
# exponent of about 0.9 -- so one Lyapunov time is roughly 1.1 time units and
# the T*tau = 5 units below span some 4.5 of them.
SIGMA_L63 = 10.0
RHO_L63   = 28.0
BETA_L63  = 8.0 / 3.0

# One analysis interval is tau, covered by N_SUB Runge-Kutta 4 substeps of
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

# Noise parameters, carried over from L63.py unchanged.
sigma  = np.sqrt(10) / 10   # process noise std ASSUMED by the filters
gamma  = np.sqrt(10) * 1    # observation noise std
Noise  = [sigma, gamma]

sigma0 = 10                 # std of the initial state distribution

# ----------------------------------------------------------------------
# Sweep configuration
# ----------------------------------------------------------------------
# Starts at 1000 for the same reason the quadratic sweep does: several methods
# carry a tuned BATCH_SIZE up to 768, and below that the batch would have to be
# clamped, silently testing a configuration nobody tuned. The clamp below still
# exists as a guard and prints when it fires.
N_LIST = [100, 250, 500, 1000, 5000]

# More simulations buy error bars on every point; the cost is linear here AND in
# the reference SIR, which is the expensive part. See the Memory note above.
NUM_SIM = 20

# ----------------------------------------------------------------------
# Reference posterior
# ----------------------------------------------------------------------
# Lorenz-63 does not factorise, so unlike the quadratic benchmark the reference
# is one SIR run in the full state rather than an exact per-block posterior.
N_true_run = int(1e5)   # particles used to *run* the reference SIR
N_true     = int(2e3)   # quantile levels *kept* from it, matching the other studies

# Raw particles reserved from the tail of the reference run as independent draws
# for the sampling floor. Must cover the largest N in the sweep.
N_FLOOR_POOL = max(N_LIST)

# Particles are the bulk of an archive and nothing in the error/time figures
# reads them.
SAVE_PARTICLES = False

# Method labels, archive keys, and plot colours, in figure order.
METHODS = [
    ('EnKF',    'EnKF',    'C0'),
    ('SIR',     'SIR',     'C1'),
    ('OTF',     'OTF',     'C2'),
    ('SBF',     'SBF',     'C6'),
    ('KRF',     'KRF',     'C7'),
    ('FMF',     'FMF',     'C3'),
    ('SIF-ODE', 'SIF_ODE', 'C4'),
    ('SIF-SDE', 'SIF_SDE', 'C8'),
]

# A recorded runtime spans all NUM_SIM simulations and all T-1 steps of each.
PER_STEP = 1.0 / (NUM_SIM * (T - 1))


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


# ----------------------------------------------------------------------
# The checkpoints this sweep consumes
# ----------------------------------------------------------------------
# Checked before any compute. Under 'require' a missing file would raise on the
# first filter call anyway, but that call comes AFTER the reference posterior has
# been paid for -- minutes of SIR at N_true_run particles, thrown away because a
# prerequisite was not met. The digest is architecture-dependent so the names
# cannot be predicted here; the count is what is checked, and the filters
# themselves verify the exact file.
_ck = [f for f in os.listdir(WARM_START_DIR)
       if f'_{WARM_START_TAG}_L{L}_dy{dy}_' in f and f.endswith('.pt')] \
      if os.path.isdir(WARM_START_DIR) else []
if len(_ck) < 6:
    raise SystemExit(
        f'Found {len(_ck)} {WARM_START_TAG} checkpoints for L={L}, dy={dy} in '
        f'{WARM_START_DIR}, expected 6 (one per learned method).\n'
        f'Run L63.py once first: its Phase 1 writes them under the stage-1 '
        f'configuration, and this sweep only consumes them.')
print(f'{len(_ck)} {WARM_START_TAG} checkpoints found for L={L}, dy={dy}')


#%%
# ----------------------------------------------------------------------
# True trajectories and reference posteriors — generated once
# ----------------------------------------------------------------------
# Neither the trajectories nor their references depend on N, so building them
# outside the loop both saves the cost and guarantees that every N is scored
# against exactly the same targets. Only the initial ensemble X0 and the filters'
# internal randomness change with N.
X_True = np.zeros((NUM_SIM, T, L,  1))
Y_True = np.zeros((NUM_SIM, T, dy, 1))
for k in range(NUM_SIM):
    X_True[k], Y_True[k] = Gen_True_Data(L, dy, T, sigma0, sigma, gamma, tau)

# One SIR in the full 3-D state. Two things about it are worth stating plainly.
# First, it is a SAMPLE-based reference, not an exact posterior: L63 does not
# factorise, so there is no per-block construction to fall back on, and its
# effective sample size sits below N_true_run and falls as the trajectory
# separates. Second, it is the same object every method is scored against, so it
# biases no method relative to another.
X0_ref = np.zeros((NUM_SIM, L, N_true_run))
for k in range(NUM_SIM):
    X0_ref[k] = np.random.multivariate_normal(
        np.zeros(L), sigma0 * sigma0 * np.eye(L), N_true_run).T

print(f'Reference posterior: SIR in {L}-D with {N_true_run} particles, '
      f'{NUM_SIM} run(s) ...')
_t_ref = time.time()
X_ref_run = SIR(Y_True, X0_ref, A, h, t, Noise, device=devices[0])
print(f'    done in {time.time() - _t_ref:.1f} s')

# Partition, so the floor draws are independent of the reference rather than
# resampled from it. The tail N_FLOOR_POOL particles are kept RAW -- a floor has
# to be an honest N-particle draw, and quantile-spacing it would sharpen it into
# a bound no real ensemble attains. Everything before them is quantile-spaced to
# N_true, which keeps the accuracy of all those particles while storing N_true.
X_floor_pool = X_ref_run[:, :, :, -N_FLOOR_POOL:].copy()
X_ref        = qslice(X_ref_run[:, :, :, :-N_FLOOR_POOL], N_true)
del X_ref_run
print(f'    reference {X_ref.shape[-1]} quantile levels, '
      f'floor pool {X_floor_pool.shape[-1]} raw particles (disjoint)')

#%%
# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------
sw2_mean   = np.zeros((len(METHODS), len(N_LIST)))
sw2_std    = np.zeros((len(METHODS), len(N_LIST)))
runtime    = np.zeros((len(METHODS), len(N_LIST)))
spinup     = np.zeros((len(METHODS), len(N_LIST)))
online     = np.zeros((len(METHODS), len(N_LIST)))
online_std = np.zeros((len(METHODS), len(N_LIST)))
sw2_floor  = np.zeros(len(N_LIST))

# Row order of the matrices is method_keys; column order is N_list.
method_keys   = [key   for _, key, _ in METHODS]
method_labels = [label for label, _, _ in METHODS]


def write_summary(n_done):
    """
    Write the sweep summary covering the first n_done ensemble sizes.

    Called after every point rather than once after the loop, so a sweep that is
    interrupted still leaves a loadable summary holding the points that did
    finish. Only the completed columns are written -- the preallocated matrices
    are still zero beyond n_done, and a zero column is indistinguishable from a
    measurement once it is in the archive.
    """
    s = slice(None, n_done)
    np.savez_compressed(
        os.path.join(DATA_DIR, f'DATA_file_L63_vs_particles_fi{FI}.npz'),
        randint=randint, N_list=np.array(N_LIST[:n_done]), L=L, dy=dy,
        method_keys=np.array(method_keys), method_labels=np.array(method_labels),
        sw2_mean=sw2_mean[:, s], sw2_std=sw2_std[:, s], runtime=runtime[:, s],
        spinup=spinup[:, s], online=online[:, s], online_std=online_std[:, s],
        sw2_floor=sw2_floor[s], fi=FI,
        NUM_SIM=NUM_SIM, T=T, tau=tau, N_true=N_true, N_true_run=N_true_run,
        sigma=sigma, gamma=gamma, sigma0=sigma0,
        complete=bool(n_done == len(N_LIST)))
    print(f'  [summary] {n_done} of {len(N_LIST)} points written'
          f'{" (complete)" if n_done == len(N_LIST) else ""}')


for i_N, N in enumerate(N_LIST):

    # Re-seeded per ensemble size so any single N can be reproduced on its own.
    # The shared trajectories and reference are already built and unaffected, so
    # every N is still scored against the same target.
    np.random.seed(randint + N)
    torch.manual_seed(randint + N)

    print(f'\n{"="*70}\n N = {N} particles   (L = {L}, dy = {dy})\n{"="*70}')

    params = {m: get_config(m, L, dy, FI) for m in
              ('otf', 'fmf', 'sbf', 'krf',
               'sif_ode', 'sif_sde')}

    # A mini-batch cannot exceed the ensemble it is drawn from. At N >= 1000 no
    # tuned BATCH_SIZE comes close, so this should never fire; it prints if it
    # does, because a clamped batch means the cell being measured is not the
    # tuned one.
    for m, p in params.items():
        if p['BATCH_SIZE'] > N:
            print(f'[warn] {m}: BATCH_SIZE {p["BATCH_SIZE"]} > N {N}, clamped '
                  f'— this is NOT the tuned configuration')
            p['BATCH_SIZE'] = N

    # Initial ensembles at this size — independent across the NUM_SIM runs.
    X0 = np.zeros((NUM_SIM, L, N))
    for k in range(NUM_SIM):
        X0[k,] = np.random.multivariate_normal(
            np.zeros(L), sigma0 * sigma0 * np.eye(L), N).T

    # Per-method timing bucket, filled in place by each filter with a
    # (NUM_SIM x T-1) array of per-analysis-step seconds. A defaultdict so each
    # dict exists when its closure is defined, and so every job owns a separate one.
    timings = defaultdict(dict)

    # Warm-start arguments go to every LEARNED filter; EnKF and SIR train nothing
    # and do not accept them. Every point of this sweep resolves to the SAME
    # checkpoint per method -- N is not part of the key.
    _ws = dict(warm_start=WARM_START, warm_start_tag=WARM_START_TAG,
               warm_start_dir=WARM_START_DIR)

    jobs = [
        ('SBF',      lambda d: SBF(Y_True,  X0, A, h, t, Noise, params['sbf'],     device=d, timing_out=timings['SBF'],     **_ws)),
        ('KRF',      lambda d: KRF(Y_True,  X0, A, h, t, Noise, params['krf'],     device=d, timing_out=timings['KRF'],     **_ws)),
        ('OTF',      lambda d: OTF(Y_True,  X0, A, h, t, Noise, params['otf'],     device=d, timing_out=timings['OTF'],     **_ws)),
        ('FMF',      lambda d: FMF(Y_True,  X0, A, h, t, Noise, params['fmf'],     device=d, timing_out=timings['FMF'],     **_ws)),
        ('SIF_ODE',  lambda d: SIF(Y_True,  X0, A, h, t, Noise, params['sif_ode'], device=d, timing_out=timings['SIF_ODE'], **_ws)),
        ('SIF_SDE',  lambda d: SIF(Y_True,  X0, A, h, t, Noise, params['sif_sde'], device=d, timing_out=timings['SIF_SDE'], **_ws)),
        ('EnKF',     lambda d: EnKF(Y_True, X0, A, h, t, Noise, SIGMA=1e-6,        device=d, timing_out=timings['EnKF'])),
        ('SIR',      lambda d: SIR(Y_True,  X0, A, h, t, Noise,                    device=d, timing_out=timings['SIR'])),
    ]

    results, runtimes = _dispatch(jobs)

    # ------------------------------------------------------------------
    # Score against the reference posterior
    # ------------------------------------------------------------------
    # t = 0 is skipped: every method still holds its prior ensemble there,
    # before any Bayesian update.
    SW2      = {name: aa_sw2(X, X_ref) for name, X in results.items()}
    SW2_time = {name: s[:, 1:].mean(axis=1) for name, s in SW2.items()}

    # Sampling floor: the first N of the raw particles reserved above, which are
    # disjoint from everything the reference was built from. This is what a
    # perfect sampler scores at this N, averaged over the NUM_SIM trajectories
    # exactly as the filter scores are.
    X_floor        = X_floor_pool[:, :, :, :N]
    sw2_floor[i_N] = float(aa_sw2(X_floor, X_ref)[:, 1:].mean())

    steps = {key: timings[key]['step_times'] for _, key, _ in METHODS}

    print(f'\n Time-averaged aa-SW2 at N = {N} '
          f'(N_true={N_true} of {N_true_run} SIR particles, {NUM_SIM} run(s)):')
    print(f'    {"sampling floor":16s}: {sw2_floor[i_N]:.4f}')
    for i_m, (label, key, _) in enumerate(METHODS):
        sw2_mean[i_m,   i_N] = SW2_time[key].mean()
        sw2_std[i_m,    i_N] = SW2_time[key].std()
        runtime[i_m,    i_N] = runtimes[key]
        spinup[i_m,     i_N] = np.nanmean(steps[key][:, 0])
        online[i_m,     i_N] = np.nanmean(steps[key][:, 1:])
        online_std[i_m, i_N] = np.nanstd(steps[key][:, 1:])
        print(f'    {label:16s}: {sw2_mean[i_m, i_N]:.4f} +/- {sw2_std[i_m, i_N]:.4f}'
              f'   (spin-up {spinup[i_m, i_N]:7.3f} s,'
              f' online {online[i_m, i_N]:8.4f} s/step)')

    # ------------------------------------------------------------------
    # Per-N archive
    # ------------------------------------------------------------------
    archive = dict(
        randint=randint, t=t, Noise=Noise, tau=tau, L=L, dy=dy, N=N, fi=FI,
        sigma=sigma, gamma=gamma, sigma0=sigma0,
        SIGMA_L63=SIGMA_L63, RHO_L63=RHO_L63, BETA_L63=BETA_L63,
        DT_SUB=DT_SUB, N_SUB=N_SUB, integrator='rk4',
        Final_Number_ITERATION=np.array(
            [params[m]['Final_Number_ITERATION'] for m in
             ('otf', 'fmf', 'sbf', 'krf',
              'sif_ode', 'sif_sde')], dtype=np.int64),
        Final_Number_ITERATION_keys=np.array(
            ['otf', 'fmf', 'sbf', 'krf',
             'sif_ode', 'sif_sde']),
        # X0, X_ref and the per-method ensembles are deliberately NOT archived:
        # they are the bulk of the file and nothing in the error/time figures
        # reads them. N_true and N_true_run are kept so the archive still says
        # what the scores were measured against.
        Y_true=Y_True, X_true=X_True, NUM_SIM=NUM_SIM, T=T,
        N_true=N_true, N_true_run=N_true_run, sw2_floor=sw2_floor[i_N],
        **{f'time_{key}':  runtimes[key] for _, key, _ in METHODS},
        **{f'steps_{key}': steps[key]    for _, key, _ in METHODS},
        **{f'sw2_{key}':   SW2[key]      for _, key, _ in METHODS})
    if SAVE_PARTICLES:
        archive.update({f'X_{key}': results[key] for _, key, _ in METHODS})
    np.savez_compressed(os.path.join(
        DATA_DIR, f'DATA_file_L63_vs_particles_fi{FI}_N_{N}.npz'), **archive)

    # Refresh the summary now that this size is archived, so the run is
    # recoverable from here even if the next size never completes.
    write_summary(i_N + 1)

#%%
# ----------------------------------------------------------------------
# Sweep summary
# ----------------------------------------------------------------------
# Already written by the last in-loop call; repeated so the complete summary is
# still produced by this block if the loop bounds ever change.
write_summary(len(N_LIST))

#%%
# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
# Estimation error against ensemble size, with the sampling floor. Unlike the
# dimension sweep the floor genuinely DOES depend on the swept variable here --
# more particles resolve the posterior better -- so it is drawn as a curve.
plt.figure(figsize=(10, 6))
for i_m, (label, key, color) in enumerate(METHODS):
    plt.errorbar(N_LIST, sw2_mean[i_m], yerr=sw2_std[i_m] / np.sqrt(NUM_SIM),
                 color=color, lw=2.5, marker='o', capsize=3, label=label)
plt.plot(N_LIST, sw2_floor, color='k', linestyle='--', lw=2.0,
         label='sampling floor')
if labeling: plt.xlabel('ensemble size N', fontsize=fontsize)
if labeling: plt.ylabel('aa-SW2',          fontsize=fontsize)
plt.xscale('log')
plt.yscale('log')
plt.xticks(N_LIST, [str(N) for N in N_LIST])
plt.legend(fontsize=fontsize - 4, ncol=2)
plt.tight_layout()
os.makedirs('figs', exist_ok=True)
plt.savefig(f'figs/L63_sw2_vs_particles_fi{FI}.pdf', bbox_inches='tight')
plt.show()

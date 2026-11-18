"""
@author: Mohammad Al-Jarrah

Dimension sweep of the Linear-Quadratic benchmark: STEP 5 of the study. The
experiment of Quadratic.py repeated at several state dimensions, with the tuned
parameters HELD FIXED -- no method is retuned at any dimension.

For each n in N_LIST the system has L = 2n states and dy = n observations,
built from n decoupled 2D rotating blocks. Every filter is run NUM_SIM times,
each run against its OWN independently drawn true trajectory, scored by the
axis-aligned sliced 2-Wasserstein distance (aa-SW2) to that trajectory's
reference posterior, and the score is averaged over time so it can be plotted
against the dimension.

One FI level (set by FI at the top of the file) applies to the whole sweep, the
same for every method, so the only thing varying along the curve is n.

⚠️ Requires a warm-start checkpoint at EVERY dimension in N_LIST. The network's
input layer is sized from L and dy, so a checkpoint fitted at one dimension does
not load at another: step 2 must be repeated per dimension before this runs.

Outputs
-------
DATA_file_Quadratic_vs_dim_fi{FI}_n_{n}.npz : per-dimension archive — particles,
    reference posterior, per-time-step aa-SW2, runtimes, and the raw
    per-analysis-step timings steps_<KEY> of shape (NUM_SIM x T-1) (same
    layout as the archive written by Quadratic.py), plus that dimension's
    sampling floor. Particles are omitted when SAVE_PARTICLES is False.
DATA_file_Quadratic_vs_dim_fi{FI}.npz       : the sweep summary — sw2_mean, sw2_std,
    runtime, spinup, online and online_std as (len(METHODS) x len(N_LIST))
    matrices whose row order is given by method_keys, plus sw2_floor of shape
    (len(N_LIST),). spinup is the offline first analysis step, online the
    per-step cost of every step after it, and runtime the raw total the two are
    split out of.

The sampling floor is an N-particle draw from each reference SIR run, disjoint
from the N_true particles the methods are scored against, so it is an
independent sample of the same posterior: the aa-SW2 between the two is what a
perfect sampler attains and no filter can beat. It is stored as one value per
dimension because each dimension has its own reference run; measured, it stays
within a ~10% band across the whole sweep with no trend in n. Runs of this file
predating 2026-09-07 do not carry it; for those, dim_sampling_floor.py recovers
it by replaying the per-dimension seed.
figs/Quadratic_sw2_vs_dim_fi{FI}.pdf   : aa-SW2 against the state dimension.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from EnKF import EnKF
from SIR import SIR
from OTF import OTF
from FMF import FMF
from SIF import SIF
from SBF import SBF
from KRF import KRF

from param_multi_steps_fi import get_config

# ----------------------------------------------------------------------
# WHICH ONLINE BUDGET THIS SWEEP RUNS AT  —  set this before running
# ----------------------------------------------------------------------
# One FI level for the whole sweep, the same for every method: the point here is
# to vary the DIMENSION, so the budget is held fixed and the level chosen from
# step 4's error-vs-compute results.
#
# 1 | 4 | 16. The level resolves to a different absolute budget per method
# (fi x param_multi_steps_fi.FI_BASE[method]):
#
#     fi        fmf/sif      otf      krf/sbf
#      1            256       16             8
#      4           1024       64            32
#     16           4096      256           128
#
# It selects a whole tuned configuration, not just an iteration count: stage 2
# re-searched the optimiser block at every level, so changing FI here changes the
# learning rate, batch size, weight decay, clip and warmup too.
FI = 4

# Every .npz archive this script writes goes here, so the loaders have one
# place to look.
DATA_DIR = 'DATA'   # [was 'DATA'] nothing existing is overwritten
os.makedirs(DATA_DIR, exist_ok=True)

# fonttype 42 embeds TrueType fonts, which the journals require.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=16)

plt.close('all')

# --- Reproducibility ---

randint = np.random.randint(0, 1000)  # random seed drawn at runtime
print(randint)
np.random.seed(randint)
torch.manual_seed(randint)

# ----------------------------------------------------------------------
# Warm start — one checkpoint PER DIMENSION
# ----------------------------------------------------------------------
# 'require' rather than 'off': the settings this sweep runs were tuned against an
# already-trained network, so a random initialisation would test them in a regime
# they were never fitted for.
#
# ⚠️ Unlike the other two sweeps, this one needs a SEPARATE checkpoint at every n
# in N_LIST. The network's input layer is sized from L and dy, so the filename
# encodes them and a checkpoint fitted at one dimension cannot be loaded at
# another. Before running this file, step 2 must have been repeated at every
# dimension -- Quadratic_run_one_step.py with WARM_START = 'save' and the same
# tag, once per n, for all learned methods.
#
# 'require' is what makes a missing dimension a loud, named failure on the first
# filter call instead of a silent retrain that would quietly make one point of
# the curve incomparable with the rest.
WARM_START     = 'require'   # 'off'  no cache, train from scratch
                             # 'auto' load if a checkpoint exists, else train and save
                             # 'save' always train from scratch, then overwrite
                             # 'load' load if a checkpoint exists, never write one
WARM_START_TAG = 'quadratic'      # must match what stage 1 saved under
WARM_START_DIR = 'DATA/warm_start'

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


# The observation rule. Observing every other state (0, 2, ..., 2n-2) gives n
# observations for L = 2n states, one per decoupled 2D subsystem.
def h(x):
    return x[::2] * x[::2]


def A(x, t=0):
    # Reads the module-level F, which the sweep rebinds for every n.
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

#%%
# ----------------------------------------------------------------------
# Sweep configuration
# ----------------------------------------------------------------------
# State dimensions visited by the sweep. Each n gives L = 2n states and
# dy = n observations; the per-n cost grows with both, so the last entries
# dominate the total runtime.
N_LIST = [1, 3, 5, 8, 10, 15, 20]

tau = 1e-1                # time step
T   = int(2 / tau)         # number of time steps T = 5 s
t   = np.arange(0.0, tau * T, tau)

# Dynamical system — the 2x2 rotation block is the same at every dimension,
# only the number of copies in F = kron(I_n, F_block) changes.
alpha   = 0.9
a       = alpha * 0.99
b       = np.sqrt(1 - alpha ** 2)
F_block = np.array([[a, -b], [b, a]])

noise  = np.sqrt(1e-1)   # noise level std
sigma  = noise           # process noise
sigma0 = 1               # initial-state noise
gamma  = noise           # observation noise
Noise  = [sigma, gamma]

N       = int(1e4)   # ensemble particles
NUM_SIM = 20           # independent simulations. [was 10]

# A recorded runtime spans all NUM_SIM simulations and all T-1 steps of each. The
# archives keep the raw totals; this factor normalises them to one analysis step.
PER_STEP = 1.0 / (NUM_SIM * (T - 1))

# Reference posterior: SIR is run with N_true_run particles per 2D block but
# only the first N_true are kept — that is all the aa-SW2 comparison uses.
#
# N_true matches Quadratic_vs_final_iter.py's. aa_sw2 resolves both ensembles at
# min(N, N_true) quantiles, so N_true sets the resolution -- and the finite-sample
# bias -- of the metric itself: a sweep run against a 1e3-particle reference is
# not on the same scale as step 4's, which used 2e3.
N_true_run = int(1e5)
N_true     = int(2e3)

# The reference SIR is kept to N_true + N particles rather than truncated to
# N_true. The extra N are a slice DISJOINT from the reference, hence an
# independent draw from the same posterior, and the aa-SW2 between the two is
# the sampling floor: the error a perfect sampler attains at this N, which no
# filter can beat. Steps 4 and 6 have always reported it; this sweep did not,
# and without it the dimension figure has no way to show how much of each
# method's error is Monte-Carlo noise rather than filtering error.
#
# Measured, it is roughly FLAT in dimension -- 0.103, 0.110, 0.103, 0.098, 0.095,
# 0.092, 0.099 at n = 1..20 (seed 404, N = 1e4, N_true = 2e3), a ~10% band with
# no monotone trend. That is what should be expected rather than a surprise:
# aa_sw2 averages L one-dimensional distances, and each block is filtered by its
# own 2D SIR at the same particle count, so the per-coordinate sampling error
# does not grow with L and averaging more coordinates only steadies the mean. It
# is still computed per dimension rather than once, because each dimension has
# its own reference run and the value does move within that band.
N_keep = N_true + N

# Particles are the bulk of the archives — they grow with L, so roughly 8 MB per
# method at n = 10. Set False to keep only the aa-SW2 curves and runtimes.
SAVE_PARTICLES = False

# Method display label, archive key, and plot colour, in figure order.
METHODS = [
    ('EnKF',    'EnKF',        'C0'),
    ('SIR',     'SIR',         'C1'),
    ('OTF',     'OTF',         'C2'),
    ('SBF',     'SBF',         'C6'),
    ('KRF',     'KRF',         'C7'),
    ('FMF',     'FMF',         'C3'),
    ('SIF-ODE', 'SIF_ODE',     'C4'),
    ('SIF-SDE', 'SIF_SDE',     'C8'),
]

#%%
# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------
# Summary matrices, filled column by column: (len(METHODS) x len(N_LIST)).
# runtime is the raw total; spinup and online split it into the offline first
# analysis step and the per-step cost of every step after it. The split has to
# be reduced here rather than left to plotting time, because the summary
# carries no per-step arrays -- those stay in the per-dimension archives, so
# the matrices below remain re-derivable if the reduction ever changes.
sw2_mean   = np.zeros((len(METHODS), len(N_LIST)))
sw2_std    = np.zeros((len(METHODS), len(N_LIST)))
runtime    = np.zeros((len(METHODS), len(N_LIST)))
spinup     = np.zeros((len(METHODS), len(N_LIST)))
online     = np.zeros((len(METHODS), len(N_LIST)))
online_std = np.zeros((len(METHODS), len(N_LIST)))
sw2_floor  = np.zeros(len(N_LIST))
L_LIST     = [2*n for n in N_LIST]

# Row order of the matrices is method_keys; column order is n_list / L_list.
method_keys   = [key   for _, key, _ in METHODS]
method_labels = [label for label, _, _ in METHODS]


def write_summary(n_done):
    """
    Write the sweep summary covering the first n_done dimensions.

    Called after every point rather than once after the loop, so a sweep that is
    interrupted still leaves a loadable summary on disk holding the points that
    did finish. Only the completed columns are written -- the preallocated
    matrices are still zero-filled beyond n_done, and a zero column is
    indistinguishable from a measurement once it is in the archive.

    Parameters
    ----------
    n_done : int — number of sweep points completed so far
    """
    s = slice(None, n_done)
    # The filename carries FI: two runs of this file at different budgets are
    # different experiments and must not overwrite each other. The old
    # _const_time name was inherited from an earlier project and is not used.
    np.savez_compressed(os.path.join(DATA_DIR, f'DATA_file_Quadratic_vs_dim_fi{FI}.npz'),
        randint=randint, n_list=np.array(N_LIST[:n_done]), L_list=np.array(L_LIST[:n_done]),
        method_keys=np.array(method_keys), method_labels=np.array(method_labels),
        sw2_mean=sw2_mean[:, s], sw2_std=sw2_std[:, s], runtime=runtime[:, s],
        spinup=spinup[:, s], online=online[:, s], online_std=online_std[:, s],
        # One floor per dimension, not one for the sweep: it depends on L
        # through the metric, so unlike step 4's single constant this is a curve.
        sw2_floor=sw2_floor[s],
        fi=FI,
        N=N, NUM_SIM=NUM_SIM, T=T, tau=tau, N_true=N_true, N_true_run=N_true_run,
        complete=bool(n_done == len(N_LIST)))
    print(f'  [summary] {n_done} of {len(N_LIST)} points written'
          f'{" (complete)" if n_done == len(N_LIST) else ""}')

for i_dim, n in enumerate(N_LIST):

    # Re-seed per dimension so any single n can be reproduced on its own.
    np.random.seed(randint + n)
    torch.manual_seed(randint + n)

    L  = n*2   # number of states
    dy = n     # number of states observed

    # Rebind the module-level F that A() closes over.
    F = np.kron(np.eye(int(n)), F_block)

    print(f'\n{"="*70}\n n = {n}   (L = {L} states, dy = {dy} observations)\n{"="*70}')

    # Same FI level at every dimension -- the tuned configuration is held fixed
    # and only n varies. L and dy enter through INPUT_DIM alone; every other
    # value is dimension-independent, which is what makes reusing one tuning
    # across the whole sweep meaningful.
    parameters_otf     = get_config('otf',     L, dy, FI)
    parameters_fmf     = get_config('fmf',     L, dy, FI)
    parameters_sbf     = get_config('sbf',     L, dy, FI)
    parameters_krf     = get_config('krf',     L, dy, FI)
    parameters_sif_ode = get_config('sif_ode', L, dy, FI)  # eps>0, lambda=0, ODE inference
    parameters_sif_sde = get_config('sif_sde', L, dy, FI)  # eps>0, lambda=1, SDE inference

    # A mini-batch cannot exceed the ensemble it is drawn from. Harmless at
    # N = 1e4, where no tuned BATCH_SIZE comes close, but it keeps the guard in
    # place if N is ever lowered.
    for _p in (parameters_otf, parameters_fmf, parameters_sbf, parameters_krf,
               parameters_sif_ode, parameters_sif_sde):
        if _p['BATCH_SIZE'] > N:
            _p['BATCH_SIZE'] = N

    # ------------------------------------------------------------------
    # NUM_SIM independent true trajectories, one filter run each
    # ------------------------------------------------------------------
    # Every simulation draws its OWN state/observation trajectory, so the NUM_SIM
    # runs of each filter differ in the data they see as well as in their initial
    # ensemble X0 and their internal sampling noise. Each run therefore needs its
    # own reference posterior, and the error bars in the sweep figure combine
    # filter variability with trajectory variability.
    X_True = np.zeros((NUM_SIM, T, L,  1))
    Y_True = np.zeros((NUM_SIM, T, dy, 1))
    for k in range(NUM_SIM):
        X_True[k], Y_True[k] = Gen_True_Data(L, dy, T, sigma0, sigma, gamma, tau)

    # Independent initial ensembles, one per run.
    X0 = np.zeros((NUM_SIM, L, N))
    for k in range(NUM_SIM):
        X0[k,] = np.random.multivariate_normal(np.zeros(L), sigma0 * sigma0 * np.eye(L), N).T

    # ------------------------------------------------------------------
    # Reference posteriors — one per simulation
    # ------------------------------------------------------------------
    # n independent 2D SIR filters, one per decoupled block of F. Running SIR in
    # 2D rather than the full L = 2n space sidesteps the curse of dimensionality;
    # the per-block posteriors are exact because the dynamics, process/observation
    # noise, and initial prior all factorise across blocks, and block j is observed
    # only through y index j (h observes state index 2j). This is what keeps the
    # reference trustworthy at the large end of the sweep.
    #
    # All NUM_SIM trajectories go through one SIR call per block: SIR's leading
    # axis is the simulation axis, so run k is filtered against its own data.
    X_pool = np.zeros((NUM_SIM, T, L, N_keep))
    for j in range(n):
        # Block j occupies state indices [2j, 2j+1] and is observed by y index j.
        X0_blk = np.zeros((NUM_SIM, 2, N_true_run))
        for k in range(NUM_SIM):
            X0_blk[k] = np.random.multivariate_normal(
                np.zeros(2), sigma0 * sigma0 * np.eye(2), N_true_run).T
        Y_blk = Y_True[:, :, j:j+1, :]                      # (NUM_SIM, T, 1, 1)
        X_blk = SIR(Y_blk, X0_blk, A_block, h, t, Noise, device=devices[0])  # (NUM_SIM, T, 2, N_true_run)
        X_pool[:, :, 2*j:2*j+2, :] = X_blk[:, :, :, :N_keep]

    # The references every method is scored against, and beside them an
    # independent draw of N from the same run. Both slices come out of one SIR
    # call rather than two, so the floor costs no extra filtering -- only the
    # memory of carrying N_keep instead of N_true particles to this line.
    # Copied rather than sliced. A basic slice is a VIEW, so keeping X_ref as one
    # would pin the whole N_keep pool -- 1.5 GB at n = 20 -- alive for the rest of
    # the dimension, for the sake of the 256 MB actually needed. Copying costs one
    # transient duplicate and lets the pool go as soon as the floor is computed.
    X_ref            = X_pool[:, :, :, :N_true].copy()
    X_floor          = X_pool[:, :, :, N_true:N_true + N]
    sw2_floor[i_dim] = float(aa_sw2(X_floor, X_ref)[:, 1:].mean())
    del X_pool, X_floor
    print(f'\n Sampling floor at n = {n}, N = {N}: {sw2_floor[i_dim]:.4f}')

    # ------------------------------------------------------------------
    # Run the filters
    # ------------------------------------------------------------------
    # Per-method timing bucket for this dimension, filled in place by each filter
    # with a (NUM_SIM x T-1) array of per-analysis-step seconds. A defaultdict so
    # each dict exists at the moment its closure is defined, and so every job owns
    # a separate one.
    timings = defaultdict(dict)

    # Warm-start arguments go to every LEARNED filter; EnKF and SIR train
    # nothing and do not accept them. The checkpoint each one resolves to is
    # dimension-specific -- its filename carries this iteration's L and dy.
    _ws = dict(warm_start=WARM_START, warm_start_tag=WARM_START_TAG,
               warm_start_dir=WARM_START_DIR)

    jobs = [
        ('SBF',         lambda d: SBF(Y_True,  X0, A, h, t, Noise, parameters_sbf,     device=d, timing_out=timings['SBF'],         **_ws)),
        ('KRF',         lambda d: KRF(Y_True,  X0, A, h, t, Noise, parameters_krf,     device=d, timing_out=timings['KRF'],         **_ws)),
        ('OTF',         lambda d: OTF(Y_True,  X0, A, h, t, Noise, parameters_otf,     device=d, timing_out=timings['OTF'],         **_ws)),
        ('FMF',         lambda d: FMF(Y_True,  X0, A, h, t, Noise, parameters_fmf,     device=d, timing_out=timings['FMF'],         **_ws)),
        ('SIF_ODE',     lambda d: SIF(Y_True,  X0, A, h, t, Noise, parameters_sif_ode, device=d, timing_out=timings['SIF_ODE'],     **_ws)),
        ('SIF_SDE',     lambda d: SIF(Y_True,  X0, A, h, t, Noise, parameters_sif_sde, device=d, timing_out=timings['SIF_SDE'],     **_ws)),
        ('EnKF',        lambda d: EnKF(Y_True, X0, A, h, t, Noise, SIGMA=1e-6,         device=d, timing_out=timings['EnKF'])),
        ('SIR',         lambda d: SIR(Y_True,  X0, A, h, t, Noise,                     device=d, timing_out=timings['SIR'])),
    ]

    results, runtimes = _dispatch(jobs)

    # ------------------------------------------------------------------
    # Score against the reference posteriors
    # ------------------------------------------------------------------
    # One (NUM_SIM x T) array per method — each run against the reference of its
    # own trajectory — then averaged over time. t = 0 is
    # skipped: every method still holds the common prior ensemble there, before
    # any Bayesian update has been applied.
    SW2      = {name: aa_sw2(X, X_ref) for name, X in results.items()}
    SW2_time = {name: s[:, 1:].mean(axis=1) for name, s in SW2.items()}   # (NUM_SIM,)

    # Per-analysis-step wall-clock, (NUM_SIM x T-1) per method. Column 0 is the
    # spin-up step, which the learned filters can train offline -- no observation
    # has arrived yet -- while every later step refines an already-warm network.
    # PER_STEP smears the two together over a step count that does not change
    # with n, so the online column is what actually varies with the dimension.
    steps = {key: timings[key]['step_times'] for _, key, _ in METHODS}

    print(f'\n Time-averaged aa-SW2 at n = {n} '
          f'(N_true={N_true} of {N_true_run} SIR particles, {NUM_SIM} run(s)):')
    print(f'    {"sampling floor":16s}: {sw2_floor[i_dim]:.4f}')
    for i_m, (label, key, _) in enumerate(METHODS):
        sw2_mean[i_m,   i_dim] = SW2_time[key].mean()
        sw2_std[i_m,    i_dim] = SW2_time[key].std()
        runtime[i_m,    i_dim] = runtimes[key]      # raw total; PER_STEP normalises it
        spinup[i_m,     i_dim] = np.nanmean(steps[key][:, 0])
        online[i_m,     i_dim] = np.nanmean(steps[key][:, 1:])
        online_std[i_m, i_dim] = np.nanstd(steps[key][:, 1:])
        print(f'    {label:16s}: {sw2_mean[i_m, i_dim]:.4f} +/- {sw2_std[i_m, i_dim]:.4f}'
              f'   (spin-up {spinup[i_m, i_dim]:7.3f} s,'
              f' online {online[i_m, i_dim]:8.4f} s/step)')

    # ------------------------------------------------------------------
    # Per-dimension archive
    # ------------------------------------------------------------------
    # Particles, reference posteriors, per-time-step aa-SW2, and runtimes, so any
    # figure for this dimension can be redrawn without re-running the filters.
    archive = dict(
        randint=randint, t=t, Noise=Noise, tau=tau, n=n, L=L, dy=dy, N=N,
        # The budget the whole sweep was held at. Stored per point so an archive
        # is self-describing: two runs of this file at different FI produce
        # identically named files that are not comparable.
        fi=FI,
        Final_Number_ITERATION=np.array(
            [p['Final_Number_ITERATION'] for p in
             (parameters_otf, parameters_fmf, parameters_sbf, parameters_krf,
              parameters_sif_ode, parameters_sif_sde)],
            dtype=np.int64),
        # Stored per point as well as in the summary, so a single dimension's
        # archive can be read on its own without the sweep file beside it.
        sw2_floor=sw2_floor[i_dim],
        Final_Number_ITERATION_keys=np.array(
            ['otf', 'fmf', 'sbf', 'krf', 'sif_ode', 'sif_sde']),
        # The particle ensembles -- X0, X_ref and the per-method results -- are
        # deliberately NOT archived: they are the bulk of the file and nothing in
        # the error/time figures reads them. N_true and N_true_run are kept as
        # scalars so the archive still says what the scores were measured against.
        Y_true=Y_True, X_true=X_True,
        N_true=N_true, N_true_run=N_true_run,
        **{f'time_{key}':  runtimes[key] for _, key, _ in METHODS},
        **{f'steps_{key}': steps[key]    for _, key, _ in METHODS},
        **{f'sw2_{key}':   SW2[key]      for _, key, _ in METHODS})
    if SAVE_PARTICLES:
        archive.update({f'X_{key}': results[key] for _, key, _ in METHODS})
    np.savez_compressed(os.path.join(DATA_DIR,
        f'DATA_file_Quadratic_vs_dim_fi{FI}_n_{n}.npz'), **archive)

    # Refresh the summary now that this dimension is archived, so the run is
    # recoverable from here even if the next dimension never completes.
    write_summary(i_dim + 1)

#%%
# ----------------------------------------------------------------------
# Sweep summary
# ----------------------------------------------------------------------
# Already written by the last in-loop call above; repeated here so the complete
# summary is still produced by this block if the loop bounds ever change.
write_summary(len(N_LIST))

#%%
# Estimation error against the state dimension: aa-SW2 averaged over time and
# over the NUM_SIM independent trajectories, with the spread across runs as
# error bars.
labeling = True  # set False to hide all axis labels
fontsize = 16

plt.figure(figsize=(10, 6))
for i_m, (label, key, color) in enumerate(METHODS):
    plt.errorbar(L_LIST, sw2_mean[i_m], yerr=sw2_std[i_m],
                 color=color, lw=2.5, marker='o', capsize=3, label=label)
if labeling: plt.xlabel('state dimension L', fontsize=fontsize)
if labeling: plt.ylabel('aa-SW2',            fontsize=fontsize)
plt.yscale('log')
plt.xticks(L_LIST)
plt.legend(fontsize=fontsize - 4, ncol=2)
plt.tight_layout()
plt.savefig(f'figs/sim20_Quadratic_sw2_vs_dim_fi{FI}.pdf', bbox_inches='tight')
plt.show()

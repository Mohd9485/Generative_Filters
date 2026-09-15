"""
@author: Mohammad Al-Jarrah

Online-refinement sweep of the Linear-Quadratic benchmark: STEP 4 of the study.
The experiment of Quadratic.py repeated at each online budget the stage-2 sweep
tuned over.

Final_Number_ITERATION is the number of training iterations a learned filter
spends on each analysis step *after* the first. The first step trains from a
random initialisation and is the one step that can be done offline, since its
pairs come from the prior alone; every later step refines an already-warm
network, and Final_Number_ITERATION is how much refinement it gets. Sweeping it
therefore traces the accuracy the online budget buys, which is the cost a
deployed filter actually pays per observation.

Everything else is held fixed: n = 5 (L = 10 states, dy = 5 observations),
N = 10000 particles, and NUM_SIM = 10 simulations. The true trajectories, the
reference posteriors and the initial ensembles X0 are all generated ONCE,
outside the loop, and reused by every point, so the sweep isolates the effect of
the online budget alone.

Each point runs the configuration stage 2 tuned AT THAT LEVEL. The optimiser
block (learning rate, batch size, weight decay, clip, warmup) was re-searched at
every budget, so the points differ in far more than Final_Number_ITERATION --
they are separately tuned filters, not one filter given more iterations.
Every learned filter warm-starts from the stage-1 weights in DATA/warm_start/,
matching the regime stage 2 tuned in.

EnKF and SIR take no parameter dict and are mathematically invariant under this
sweep. They are run at every point regardless, as flat reference lines showing
what the learned filters are being measured against; their curves being flat is
the expected result, not a bug.

A sampling floor is reported alongside the filters: an independent N-particle
draw from the reference posteriors themselves, scored the same way. Because N is
fixed here the floor is a single constant, repeated across the sweep so the
summary keeps the same layout as the other _vs* archives.

Crash safety
------------
The summary is rewritten after every completed sweep point, holding only the
points finished so far, so it is always a loadable, self-consistent archive. If
the run dies at fi = 16 the summary already on disk covers fi = 1 and 4 and can
be plotted immediately -- no rebuild step and no hand editing. A point that raises
is recorded in failed_F and the sweep moves on to the next F rather than taking
the remaining points down with it.

Outputs
-------
DATA_file_Quadratic_vs_final_iter_fi_{fi}.npz : per-point archive — particles,
    per-time-step aa-SW2, runtimes, and the raw per-analysis-step timings
    steps_<KEY> of shape (NUM_SIM x T-1) (same layout as the archive written
    by Quadratic.py). Particles are omitted when SAVE_PARTICLES is False.
DATA_file_Quadratic_vs_final_iter.npz       : the sweep summary — sw2_mean,
    sw2_std, runtime, spinup, online and online_std as
    (len(METHODS) x points completed) matrices whose row order is given by
    method_keys, plus the sampling floor, failed_F and the complete flag.
figs/Quadratic_sw2_vs_compute.pdf           : aa-SW2 against MEASURED compute,
    the mean wall-clock seconds of one online analysis step. That is the axis
    worth reporting: one FI level buys a different number of iterations per
    method and those iterations cost different amounts, so an iteration axis
    compares unlike things while a time axis does not.

The archives store BOTH the level (fi) and the per-method resolved budget
(Final_Number_ITERATION, one entry per learned method, keyed by
Final_Number_ITERATION_keys) -- one level means a different amount of work for
each method, so the level alone does not identify what a point ran.
"""

import os
import sys
import time
import traceback
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

from param_multi_steps_fi import get_config, FI_BASE

# Every .npz archive this script writes goes here, so the loaders have one
# place to look.
DATA_DIR = 'DATA'   # [was 'DATA'] separate output dir, nothing existing overwritten
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
# Warm start — the stage-1 weights this sweep is built on
# ----------------------------------------------------------------------
# 'require' rather than 'off': the online settings this sweep tests were tuned
# by tuning_stage_two.py against an ALREADY-TRAINED network, so running
# them from a random initialisation would test them in a regime they were never
# fitted for. 'require' also fails loudly on a missing or mismatched checkpoint
# instead of silently retraining, which would make one point's spin-up column
# incomparable with the rest.
#
# Consequence for the timings: with a cache hit the spin-up step costs
# Final_Number_ITERATION rather than ITERATION, so the `spinup` column measures
# a warm refinement, NOT the cost of training a filter from scratch. The
# `online` column is unaffected and is the number the error-vs-compute figure
# is built from.
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
# Online refinement budgets visited by the sweep, as stage-2 FI LEVELS rather
# than absolute iteration counts. Each method resolves its own budget from the
# level: fi x param_multi_steps_fi.FI_BASE[method], which is 256 for fmf and the
# SIF rung, 16 for otf, 8 for krf and sbf. So one level is a very different
# amount of work per method --
#
#     fi        fmf/sif      otf      krf/sbf
#      1            256       16             8
#      4           1024       64            32
#     16           4096      256           128
#
# -- which is exactly why this sweep records computational time: the reportable
# figure is error against compute, not error against iteration count, and an
# iteration axis would compare unlike things across methods.
#
# A level is NOT just a budget. tuning_stage_two.py re-searched the whole
# optimiser block (lr, batch size, weight decay, clip, warmup) separately at
# every level, so each point below runs a different tuned configuration, not the
# same one with the iteration count swapped.
#
# Level 0 is the no-adaptation baseline, added to anchor the compute axis at its
# left end. It is not a tuned cell: param_multi_steps_fi serves stage 1's
# configuration there with Final_Number_ITERATION = 0, so with WARM_START =
# 'require' below every learned filter loads its stage-1 checkpoint and trains at
# NO analysis step -- the whole run is inference under the frozen spin-up network.
# It answers what the online budget is actually buying, which none of 1/4/16 can
# on its own. Note that it is not a zero-COST point: every method still
# integrates its ODE/SDE at each step, and SBF still samples two IPF trajectories
# per step, which is why the figure below is drawn against measured time.
F_LIST = [0, 1, 4, 16]

# Fixed dimension and ensemble size for the whole sweep.
n  = 5
L  = n * 2     # number of states
dy = n         # number of states observed
N  = int(1e4)  # ensemble particles

tau = 1e-1                # time step
# T = 50, the reporting horizon. T = 20 matches the horizon
# tuning_stage_two.py tuned at, and runs at that setting agree with the
# tuning far better (Spearman +0.68 to +0.89, against +0.11 to +0.54 at T = 50)
# -- the gap between the two is the objective mismatch, not a bad tuning.
# Both horizons are worth having: T = 50 is the only one that exposes
# long-horizon behaviour, notably SBF's divergence at fi = 16, which begins
# after t ~ 20 and is invisible at T = 20.
#
# Completed runs, all preserved:
#     DATA/step4_T50_Ntrue1000/  T=50, N_true=1000, seed 258
#     DATA/step4_T20_seed449/    T=20, N_true=2000, seed 449
#     DATA/step4_T20_seed875/    T=20, N_true=2000, seed 875
T = int(2 / tau)   # number of time steps T = 5 s
t = np.arange(0.0, tau * T, tau)

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

NUM_SIM = 20   # independent simulations. [was 10]

# A recorded runtime spans all NUM_SIM simulations and all T-1 steps of each. The
# archives keep the raw totals; this factor normalises them to one analysis step.
PER_STEP = 1.0 / (NUM_SIM * (T - 1))

# Reference posterior: SIR is run with N_true_run particles per 2D block; the
# first N_true are the reference, and a disjoint slice of N more supplies the
# independent sample used for the sampling floor.
# N_true = 2000 to match the tuner's n_sample, so the reference posterior is
# resolved at the same number of quantiles the tuning objective used.
N_true_run = int(1e5)
N_true     = int(2e3)
N_keep     = N_true + N

# Particles are the bulk of the archives — roughly 40 MB per method at N = 1e4.
# Set False to keep only the aa-SW2 curves and runtimes.
SAVE_PARTICLES = True

# Method display label, archive key, and plot colour, in figure order. EnKF
# and SIR ignore Final_Number_ITERATION entirely and are here as flat baselines.
METHODS = [
    ('EnKF',    'EnKF',        'C0'),
    ('SIR',     'SIR',         'C1'),
    ('OTF',     'OTF',         'C2'),
    ('FMF',     'FMF',         'C3'),
    ('SIF-ODE', 'SIF_ODE',     'C4'),
    ('SIF-SDE', 'SIF_SDE',     'C8'),
    ('SBF',     'SBF',         'C6'),
    ('KRF',     'KRF',         'C7'),
]

#%%
# ----------------------------------------------------------------------
# True trajectories, reference posteriors and initial ensembles — once
# ----------------------------------------------------------------------
# None of these depend on Final_Number_ITERATION, so building them outside the
# loop both saves the cost and guarantees that every F is scored against exactly
# the same targets from exactly the same starting ensemble. X0 is fixed here as
# well -- unlike the ensemble-size sweep, where it necessarily changes with N --
# so the only thing that varies across this sweep is the online budget itself.
X_True = np.zeros((NUM_SIM, T, L,  1))
Y_True = np.zeros((NUM_SIM, T, dy, 1))
for k in range(NUM_SIM):
    X_True[k], Y_True[k] = Gen_True_Data(L, dy, T, sigma0, sigma, gamma, tau)

# Initial ensembles — independent across the NUM_SIM runs, shared across F.
X0 = np.zeros((NUM_SIM, L, N))
for k in range(NUM_SIM):
    X0[k,] = np.random.multivariate_normal(np.zeros(L), sigma0 * sigma0 * np.eye(L), N).T

# n independent 2D SIR filters, one per decoupled block of F = kron(I_n, F_block).
# Running SIR in 2D rather than the full L = 2n space sidesteps the curse of
# dimensionality; the per-block posteriors are exact because the dynamics,
# process/observation noise, and initial prior all factorise across blocks, and
# block j is observed only through y index j (h observes state index 2j).
X_pool = np.zeros((NUM_SIM, T, L, N_keep))
for j in range(n):
    # Block j occupies state indices [2j, 2j+1] and is observed by y index j.
    X0_blk = np.zeros((NUM_SIM, 2, N_true_run))
    for k in range(NUM_SIM):
        X0_blk[k] = np.random.multivariate_normal(
            np.zeros(2), sigma0 * sigma0 * np.eye(2), N_true_run).T
    Y_blk = Y_True[:, :, j:j+1, :]                       # (NUM_SIM, T, 1, 1)
    X_blk = SIR(Y_blk, X0_blk, A_block, h, t, Noise, device=devices[0])  # (NUM_SIM, T, 2, N_true_run)
    X_pool[:, :, 2*j:2*j+2, :] = X_blk[:, :, :, :N_keep]

X_ref = X_pool[:, :, :, :N_true]  # the references every method is scored against

# Sampling floor: an N-particle slice of each reference SIR run that is disjoint
# from X_ref, so it is an independent draw from the same posterior. N is fixed
# across this sweep, so unlike the ensemble-size sweep this is one constant
# computed once rather than a curve.
X_floor       = X_pool[:, :, :, N_true:N_true + N]       # (NUM_SIM, T, L, N)
sw2_floor_val = float(aa_sw2(X_floor, X_ref)[:, 1:].mean())
print(f'\nSampling floor at N = {N}: {sw2_floor_val:.4f}')

#%%
# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------
# Results accumulate as lists rather than preallocated matrices, so the summary
# can be assembled from exactly the points that finished. A point that raises
# contributes nothing and leaves no zero-filled column behind to be mistaken for
# a measurement.
method_keys   = [key   for _, key, _ in METHODS]
method_labels = [label for label, _, _ in METHODS]

done_F   = []   # F values completed, in sweep order
done_row = []   # per-point dict: method key -> reduction scalars
failed_F = []   # F values whose sweep point raised


def write_summary(final=False):
    """
    Write the sweep summary covering every point completed so far.

    Called after each point rather than once at the end, so an interrupted run
    still leaves a loadable summary on disk holding the points that did finish.
    The matrices are (len(METHODS) x len(done_F)) -- they grow a column per
    point -- so the archive is always self-consistent and directly plottable,
    with F_list naming exactly the columns present.

    Parameters
    ----------
    final : bool — mark the summary complete; False while points remain
    """
    P    = len(done_F)
    mats = {name: np.zeros((len(METHODS), P)) for name in
            ('sw2_mean', 'sw2_std', 'runtime', 'spinup', 'online', 'online_std')}
    for i_p, row in enumerate(done_row):
        for i_m, key in enumerate(method_keys):
            for name in mats:
                mats[name][i_m, i_p] = row[key][name]

    np.savez_compressed(
        os.path.join(DATA_DIR, 'DATA_file_Quadratic_vs_final_iter.npz'),
        randint=randint, F_list=np.array(done_F), n=n, L=L, dy=dy, N=N,
        method_keys=np.array(method_keys), method_labels=np.array(method_labels),
        # The floor does not vary with F; repeated across the sweep so the
        # archive keeps the same (len(F_list),) layout as the other _vs* files.
        sw2_floor=np.full(P, sw2_floor_val),
        failed_F=np.array(failed_F, dtype=np.int64), complete=bool(final),
        NUM_SIM=NUM_SIM, T=T, tau=tau, N_true=N_true, N_true_run=N_true_run,
        **mats)
    print(f'  [summary] {P} of {len(F_LIST)} points written'
          f'{" (complete)" if final else ""}')


for i_F, F_ITER in enumerate(F_LIST):

    # Same seed at every point, not randint + F: the trajectories, references and
    # X0 are already fixed above, so seeding identically here starts each point
    # from the same RNG state and leaves Final_Number_ITERATION as the only thing
    # that differs. Each point is still reproducible on its own.
    np.random.seed(randint)
    torch.manual_seed(randint)

    print(f'\n{"="*70}\n FI level = {F_ITER}   '
          f'(n = {n}, L = {L}, dy = {dy}, N = {N})\n{"="*70}')

    try:
        # Every dict is the configuration stage 2 tuned AT THIS LEVEL -- the whole
        # optimiser block, not just the budget. Final_Number_ITERATION is already
        # resolved inside it (fi x the method's base), so nothing is overridden
        # here: writing F_ITER into it would replace 256/1024/4096 with 1/4/16 and
        # run every method at a near-zero budget.
        parameters_otf     = get_config('otf',     L, dy, F_ITER)
        parameters_fmf     = get_config('fmf',     L, dy, F_ITER)
        parameters_sbf     = get_config('sbf',     L, dy, F_ITER)
        parameters_krf     = get_config('krf',     L, dy, F_ITER)
        parameters_sif_ode = get_config('sif_ode', L, dy, F_ITER)  # eps>0, lambda=0, ODE inference
        parameters_sif_sde = get_config('sif_sde', L, dy, F_ITER)  # eps>0, lambda=1, SDE inference

        learned = (parameters_otf, parameters_fmf, parameters_sbf, parameters_krf,
                   parameters_sif_ode, parameters_sif_sde)

        # The resolved budgets differ per method, so record what each one actually
        # ran -- the level alone does not identify the work done.
        print('    resolved Final_Number_ITERATION: '
              + ', '.join(f'{k}={p["Final_Number_ITERATION"]}' for k, p in
                          zip(('otf', 'fmf', 'sbf', 'krf', 'sif_ode', 'sif_sde'), learned)))

        # A mini-batch cannot exceed the ensemble it is drawn from. Kept from the
        # ensemble-size sweep: harmless at N = 1e4, where no tuned BATCH_SIZE comes
        # close, but it keeps the guard in place if N is ever lowered.
        for p in learned:
            if p['BATCH_SIZE'] > N:
                p['BATCH_SIZE'] = N

        # ------------------------------------------------------------------
        # Run the filters
        # ------------------------------------------------------------------
        # Per-method timing bucket for this point, filled in place by each filter
        # with a (NUM_SIM x T-1) array of per-analysis-step seconds. A defaultdict
        # so each dict exists at the moment its closure is defined, and so every
        # job owns a separate one.
        timings = defaultdict(dict)

        # Warm-start arguments go to every LEARNED filter; EnKF and SIR train
        # nothing and do not accept them.
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
        # own trajectory — then averaged over time. t = 0 is skipped: every method
        # still holds its prior ensemble there, before any Bayesian update.
        SW2      = {name: aa_sw2(X, X_ref) for name, X in results.items()}
        SW2_time = {name: s[:, 1:].mean(axis=1) for name, s in SW2.items()}   # (NUM_SIM,)

        # Per-analysis-step wall-clock, (NUM_SIM x T-1) per method. Column 0 is the
        # spin-up step, which does not depend on Final_Number_ITERATION -- it runs
        # ITERATION iterations from a random initialisation regardless -- so it
        # should stay flat across this sweep while online rises with F.
        steps = {key: timings[key]['step_times'] for _, key, _ in METHODS}

        print(f'\n Time-averaged aa-SW2 at FI level = {F_ITER} '
              f'(N_true={N_true} of {N_true_run} SIR particles, {NUM_SIM} run(s)):')
        print(f'    {"sampling floor":16s}: {sw2_floor_val:.4f}')
        row = {}
        for i_m, (label, key, _) in enumerate(METHODS):
            row[key] = dict(
                sw2_mean   = float(SW2_time[key].mean()),
                sw2_std    = float(SW2_time[key].std()),
                runtime    = float(runtimes[key]),   # raw total; PER_STEP normalises it
                spinup     = float(np.nanmean(steps[key][:, 0])),
                online     = float(np.nanmean(steps[key][:, 1:])),
                online_std = float(np.nanstd(steps[key][:, 1:])),
            )
            print(f'    {label:16s}: {row[key]["sw2_mean"]:.4f} +/- {row[key]["sw2_std"]:.4f}'
                  f'   (spin-up {row[key]["spinup"]:7.3f} s,'
                  f' online {row[key]["online"]:8.4f} s/step)')

        # ------------------------------------------------------------------
        # Per-point archive
        # ------------------------------------------------------------------
        archive = dict(
            randint=randint, t=t, Noise=Noise, tau=tau, n=n, L=L, dy=dy, N=N,
            # fi is the LEVEL; the budget it resolved to differs per method, so
            # both are stored. Without the per-method row a reader cannot tell
            # what work a point actually did.
            fi=F_ITER,
            Final_Number_ITERATION=np.array(
                [p['Final_Number_ITERATION'] for p in learned], dtype=np.int64),
            Final_Number_ITERATION_keys=np.array(
                ['otf', 'fmf', 'sbf', 'krf', 'sif_ode', 'sif_sde']),
            X0=X0, Y_true=Y_True, X_true=X_True,
            X_ref=X_ref, N_true=N_true, N_true_run=N_true_run,
            sw2_floor=sw2_floor_val,
            **{f'time_{key}':  runtimes[key] for _, key, _ in METHODS},
            **{f'steps_{key}': steps[key]    for _, key, _ in METHODS},
            **{f'sw2_{key}':   SW2[key]      for _, key, _ in METHODS})
        if SAVE_PARTICLES:
            archive.update({f'X_{key}': results[key] for _, key, _ in METHODS})
        np.savez_compressed(os.path.join(DATA_DIR,
            f'DATA_file_Quadratic_vs_final_iter_fi_{F_ITER}.npz'), **archive)

        done_F.append(int(F_ITER))
        done_row.append(row)

    except Exception:
        # One point failing should not cost the points after it: record it, keep
        # the summary of what has finished, and carry on. The traceback goes to
        # the log so the failure is diagnosable rather than merely counted.
        failed_F.append(int(F_ITER))
        print(f'\n[FAIL] FI level = {F_ITER} raised; '
              f'continuing with the remaining points.')
        traceback.print_exc(file=sys.stdout)

    # Rewritten after every point, completed or failed, so the summary on disk
    # always reflects the run so far.
    write_summary(final=(i_F == len(F_LIST) - 1))

#%%
# ----------------------------------------------------------------------
# Console summary — time-averaged aa-SW2, method x FI level
# ----------------------------------------------------------------------
# The per-point block inside the loop prints one level at a time, which is what
# a running sweep can show but not what the comparison needs: the question is
# how each method moves ACROSS budgets, and that only reads off a table with the
# levels side by side. Built from done_row rather than re-loaded from disk, so it
# covers exactly the points this process completed and nothing left over from an
# earlier run.
if done_row:
    _w   = 11
    _rule = '=' * (20 + _w * len(done_F))

    print(f'\n{_rule}')
    print(f' Time-averaged aa-SW2   (NUM_SIM={NUM_SIM}, T={T}, N={N}, '
          f'N_true={N_true}, seed {randint})')
    print(_rule)
    print(f' {"method":18s}' + ''.join(f'{"fi=" + str(f):>{_w}}' for f in done_F))
    for label, key, _ in METHODS:
        print(f' {label:18s}'
              + ''.join(f'{row[key]["sw2_mean"]:>{_w}.4f}' for row in done_row))
    print(f' {"sampling floor":18s}'
          + ''.join(f'{sw2_floor_val:>{_w}.4f}' for _ in done_F))

    # One FI level is a different number of iterations for every method
    # (fi x FI_BASE), so the table above cannot be read without this one beside
    # it: its columns are levels, not equal amounts of work. EnKF and SIR train
    # nothing at any level and are shown as '-'.
    print(f'\n Resolved Final_Number_ITERATION  (fi x FI_BASE):')
    print(f' {"method":18s}' + ''.join(f'{"fi=" + str(f):>{_w}}' for f in done_F))
    for label, key, _ in METHODS:
        base = FI_BASE.get(key.lower())
        print(f' {label:18s}'
              + ''.join(f'{("-" if base is None else int(f) * base):>{_w}}'
                        for f in done_F))

    # The price of each cell above. fi = 0 trains at no step and is still not
    # free -- every method integrates its ODE/SDE at every step regardless --
    # which is why the figure is drawn against this and not against the level.
    print(f'\n Online cost, seconds per analysis step:')
    print(f' {"method":18s}' + ''.join(f'{"fi=" + str(f):>{_w}}' for f in done_F))
    for label, key, _ in METHODS:
        print(f' {label:18s}'
              + ''.join(f'{row[key]["online"]:>{_w}.4f}' for row in done_row))

    if failed_F:
        print(f'\n Levels that raised and are absent from the table: {failed_F}')
    print(_rule)

#%%
# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
# Estimation error against MEASURED COMPUTE: aa-SW2 averaged over time and over
# the NUM_SIM independent trajectories, with the spread across runs as error
# bars, plotted against the wall-clock cost of one online analysis step. Each
# learned method contributes one curve through its budgets, so the figure
# reads as "what a second of online training buys, per method" rather than as a
# comparison at equal iteration counts — which the per-method FI_BASE makes
# meaningless. The dashed floor is an independent N-particle draw from the
# reference posteriors — no sampler can do better at this N, so the gap above it
# is the actual filter error.
if done_F:
    summary  = dict(np.load(os.path.join(DATA_DIR,
                    'DATA_file_Quadratic_vs_final_iter.npz'), allow_pickle=True))
    labeling = True  # set False to hide all axis labels
    fontsize = 16

    # x is MEASURED cost, not the FI level: one level buys a different number of
    # iterations per method (fi x FI_BASE) and those iterations cost different
    # amounts, so the level axis compares unlike things across methods while
    # seconds do not. 'online' is the mean wall-clock of ONE online analysis
    # step -- the recurring cost of running the filter -- with the spin-up step
    # excluded, since it is charged once per simulation and not per assimilation.
    # It also admits fi = 0, which a log axis over the level itself cannot show.
    #
    # EnKF and SIR train nothing, so their sweep points are repeats of a single
    # run rather than a curve. Drawn as one marker at their mean cost and error,
    # which states the baseline without implying a budget dependence they do not
    # have.
    BASELINES = ('EnKF', 'SIR')

    # Column of each level in the summary matrices, so the fi = 0 end of a curve
    # can be marked whatever order the points completed in.
    F_col = {int(f): i for i, f in enumerate(summary['F_list'])}

    plt.figure(figsize=(10, 6))
    for i_m, (label, key, color) in enumerate(METHODS):
        x = summary['online'][i_m]
        y = summary['sw2_mean'][i_m]
        e = summary['sw2_std'][i_m]

        if key in BASELINES:
            plt.errorbar(x.mean(), y.mean(), yerr=e.mean(), color=color,
                         marker='D', markersize=9, capsize=3, linestyle='none',
                         label=label)
            continue

        # Sorted by cost, not by level: the curve is read left to right as
        # "what more compute buys", and nothing guarantees the two orders agree.
        order = np.argsort(x)
        plt.errorbar(x[order], y[order], yerr=e[order], color=color, lw=2.5,
                     marker='o', capsize=3, label=label)

        # The zero-budget end, ringed so it is not read as a tuned point: no
        # online training happened there, so it is the frozen warm-started
        # network and the baseline the rest of the curve is measured against.
        if 0 in F_col:
            j = F_col[0]
            plt.plot(x[j], y[j], color=color, marker='s', markersize=12,
                     markerfacecolor='none', markeredgewidth=2.0, linestyle='none')

    # Proxy handle, so the ring is explained once instead of nine times.
    if 0 in F_col:
        plt.plot([], [], color='k', marker='s', markersize=12,
                 markerfacecolor='none', markeredgewidth=2.0, linestyle='none',
                 label='fi = 0 (no online training)')

    # Flat in compute as well as in level -- the floor is a property of N and of
    # the reference posterior, so it is a line across the panel, not a curve.
    plt.axhline(summary['sw2_floor'][0], color='k', linestyle='--', lw=2.0,
                label='sampling floor')
    if labeling: plt.xlabel('wall-clock seconds per analysis step', fontsize=fontsize)
    if labeling: plt.ylabel('aa-SW2',                               fontsize=fontsize)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=fontsize - 4, ncol=2)
    plt.tight_layout()
    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/sim20_Quadratic_sw2_vs_compute.pdf', bbox_inches='tight')
    plt.show()

if failed_F:
    print(f'\n{"="*70}\n Sweep finished with failures at '
          f'FI levels: {failed_F}\n{"="*70}')

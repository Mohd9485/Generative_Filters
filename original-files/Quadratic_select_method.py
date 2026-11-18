"""
@author: Mohammad Al-Jarrah

Regenerate ONE learned method's stage-1 warm-start checkpoints for the
Linear-Quadratic benchmark, at every (L, dy) this repo's Quadratic scripts
need.

Why this file exists
---------------------
Quadratic.py's own Phase 1 trains a checkpoint at whatever single (L, dy) it
is set to (L = 10, dy = 5 -- also what Quadratic_vs_particles.py and
Quadratic_vs_final_iter.py run at). Quadratic_vs_dim.py instead sweeps L, and
needs a SEPARATE checkpoint at every dimension it visits; there is no single
Phase 1 run that produces all of them.

Some methods' checkpoints are also too large to keep in a public git
repository. KRF's triangular map builds one sub-network per state
coordinate (see KRF.py's module docstring), so its checkpoint size grows
with L rather than staying fixed like every other method's -- 28 MB at
L = 2, 551 MB at L = 40. Rather than shipping those files, this script
regenerates them locally: set `method` below and run it once before running
any of the four Quadratic scripts, and the checkpoint each of them needs
under warm_start='require' will already be on disk.

What it does
------------
The same Phase 1 as Quadratic.py's own spin-up -- param_one_step.get_config,
T_SPINUP steps, warm_start='auto' so an existing checkpoint is left alone --
just looped over every (L, dy) pair the four Quadratic scripts actually read
a checkpoint at, for ONE method rather than all six:

    N_LIST = [1, 3, 5, 8, 10, 15, 20]     L = 2n, dy = n

(L, dy) = (10, 5) is what Quadratic.py, Quadratic_vs_particles.py and
Quadratic_vs_final_iter.py all run at; the full list is what
Quadratic_vs_dim.py's sweep needs. Running this file once per method
reproduces exactly the checkpoints those four scripts expect to find, so it
is also the file to point at (rather than committing them) to keep any
oversized checkpoint out of the repository.

Writes DATA/warm_start/<method>_quadratic_L<L>_dy<dy>_<arch>_<digest>.pt --
nothing else. No filter comparison, no archive, no figure.
"""

import os
import sys
import time
import importlib
import numpy as np
import torch

# ----------------------------------------------------------------------
# Importing the filters from THIS directory, and proving that it happened
# ----------------------------------------------------------------------
# If another copy of this repository sits AHEAD of this one on sys.path, the
# filters resolve there instead -- the module names match, so nothing
# complains, and a run is silently built from two different versions of the
# same code. Pinning this directory at the front of sys.path for each import,
# then checking where the module actually came from, turns that mistake into
# an error at line one.
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
        sys.path[:] = saved

    where = os.path.dirname(os.path.abspath(module.__file__))
    if where != _HERE:
        raise ImportError(
            f'{module_name} resolved to {module.__file__!r}, not to {_HERE!r}. '
            f'Another copy of this repository is ahead on sys.path; the filters '
            f'there are a different version and must not be mixed with these.')
    return getattr(module, func_name)


OTF = _import_local('OTF', 'OTF')
FMF = _import_local('FMF', 'FMF')
SIF = _import_local('SIF', 'SIF')
SBF = _import_local('SBF', 'SBF')
KRF = _import_local('KRF', 'KRF')

get_config_one_step = _import_local('param_one_step', 'get_config')

#%%
# ----------------------------------------------------------------------
# WHICH METHOD TO REGENERATE  —  set this before running
# ----------------------------------------------------------------------
method = 'krf'   # 'fmf' | 'otf' | 'sif_ode' | 'sif_sde' | 'sbf' | 'krf'

FILTER = {'fmf': FMF, 'sif_sde': SIF, 'sif_ode': SIF,
          'otf': OTF, 'krf': KRF, 'sbf': SBF}[method]

# ----------------------------------------------------------------------
# Every (L, dy) a Quadratic script reads a checkpoint at
# ----------------------------------------------------------------------
# Matches Quadratic_vs_dim.py's own N_LIST exactly -- L = 2n, dy = n. n = 5
# (L = 10, dy = 5) is also what Quadratic.py, Quadratic_vs_particles.py and
# Quadratic_vs_final_iter.py run at, so it does not need listing twice.
N_LIST = [1, 3, 5, 8, 10, 15, 20]
DIMS   = [(2 * n, n) for n in N_LIST]

# ----------------------------------------------------------------------
# Warm start
# ----------------------------------------------------------------------
# 'auto': train only the dimensions missing from DATA/warm_start/, load and
# leave alone anything already there -- so re-running this file after it was
# interrupted, or after a fresh git clone that only has SOME checkpoints
# committed, does not retrain what is already on disk.
WARM_START     = 'auto'
WARM_START_TAG = 'quadratic'      # must match what the Quadratic scripts use
WARM_START_DIR = 'DATA/warm_start'

# Retrain and overwrite EVERY checkpoint regardless of what is on disk. Set
# this after editing param_one_step.py, or to replace weights of unknown
# provenance.
FORCE_REGEN = False

# Every filter writes its checkpoint at (k == 0, i == 0) and never again, so
# two time steps -- one analysis step -- is all Phase 1 needs.
T_SPINUP = 2

randint = 390
np.random.seed(randint)
torch.manual_seed(randint)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

os.makedirs(WARM_START_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# The Linear-Quadratic system — identical to Quadratic.py / Quadratic_vs_dim.py
# ----------------------------------------------------------------------
# n decoupled 2x2 rotation blocks: F = kron(I_n, F_block). Rebuilt per
# dimension below since F's size depends on n; A() closes over the module-
# level name and picks up whichever F is current when it is called.
tau    = 1e-1
noise  = np.sqrt(1e-1)   # noise level std
sigma  = noise           # process noise
sigma0 = 1               # initial-state noise
gamma  = noise           # observation noise
Noise  = [sigma, gamma]

alpha   = 0.9
a       = alpha * 0.99
b       = np.sqrt(1 - alpha ** 2)
F_block = np.array([[a, -b], [b, a]])


def h(x):
    return x[::2] * x[::2]   # observe every other state (0, 2, ...); n obs for L = 2n states


def A(x, t=0):
    try:
        return F @ x
    except Exception:
        return torch.from_numpy(F).to(dtype=torch.float32, device=x.device) @ x


def Gen_True_Data(L, dy, T, sigma0, sigma, gamma, tau):
    """Same generator Quadratic.py uses; only T differs (T_SPINUP here)."""
    x = np.zeros((T, L, 1))
    y = np.zeros((T, dy, 1))
    x[0,] = np.random.multivariate_normal(np.zeros(L), sigma0 * sigma0 * np.eye(L), 1).T
    for i in range(T - 1):
        x[i + 1, :] = A(x[i, :]) + np.random.multivariate_normal(
            np.zeros(L), sigma * sigma * np.eye(L), 1).T
        y[i + 1, :] = h(x[i + 1, :]) + np.random.multivariate_normal(
            np.zeros(dy), gamma * gamma * np.eye(dy), 1).T
    return x, y


# Ensemble size for the spin-up. Not part of the checkpoint's digest (see
# KRF.py's/OTF.py's own _warm_start_path docstring -- the network has the
# same shape at any N), but it does feed the BATCH_SIZE clamp below, so it is
# matched to what Quadratic.py itself trains Phase 1 with rather than chosen
# freely.
N = int(1e4)

_ws = dict(warm_start=('save' if FORCE_REGEN else WARM_START),
           warm_start_tag=WARM_START_TAG, warm_start_dir=WARM_START_DIR)
if FORCE_REGEN:
    print('FORCE_REGEN: retraining and overwriting every checkpoint.')


def _tag_files():
    """Basename -> mtime for every checkpoint of this problem currently on disk."""
    return {f: os.path.getmtime(os.path.join(WARM_START_DIR, f))
            for f in os.listdir(WARM_START_DIR)
            if f'_{WARM_START_TAG}_' in f}


print(f'\n=== Regenerating {method!r} at {len(DIMS)} dimension(s), '
      f'{T_SPINUP} steps, WARM_START={_ws["warm_start"]!r} ===')

for L, dy in DIMS:
    n = dy
    # Re-seeded per dimension, matching Quadratic_vs_dim.py, so any single
    # dimension's spin-up can be reproduced on its own.
    np.random.seed(randint + n)
    torch.manual_seed(randint + n)

    F = np.kron(np.eye(n), F_block)   # rebinds the name A() closes over

    p = get_config_one_step(method, L, dy)
    # A mini-batch cannot exceed the ensemble it is drawn from. Harmless at
    # N = 1e4, where no tuned BATCH_SIZE comes close, but it keeps the guard
    # in place if N is ever lowered.
    if p['BATCH_SIZE'] > N:
        p['BATCH_SIZE'] = N

    X0 = np.random.multivariate_normal(
        np.zeros(L), sigma0 * sigma0 * np.eye(L), N).T[None]   # (1, L, N)

    _, Y_spin = Gen_True_Data(L, dy, T_SPINUP, sigma0, sigma, gamma, tau)
    Y_spin = Y_spin[None]                                       # (1, T_SPINUP, dy, 1)
    t_spin = np.arange(0.0, tau * T_SPINUP, tau)

    tag    = f'L={L}, dy={dy}'
    before = _tag_files()
    t0     = time.time()
    FILTER(Y_spin, X0, A, h, t_spin, Noise, p, device=device, **_ws)
    written = sorted(f for f, mt in _tag_files().items() if before.get(f) != mt)

    if written:
        print(f'  ({tag}) trained in {time.time() - t0:6.1f} s -> {written[0]}')
    else:
        print(f'  ({tag}) already present, loaded (nothing trained).')

print('\nDone. DATA/warm_start/ now has an %r checkpoint at every dimension '
      'the Quadratic scripts need.' % method)

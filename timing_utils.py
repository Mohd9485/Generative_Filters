"""
@author: Mohammad Al-Jarrah

Shared timing helpers for the filtering methods.

The filters report two computational costs rather than one total: the spin-up
cost of the first analysis step, which trains a network from a random
initialisation and can be done offline, and the online cost of every later
step, which refines the already-warm network. Splitting them needs a wall clock
that is valid for GPU work, which is what sync_clock provides.
"""

import time
import torch


def sync_clock(device):
    """
    Wall-clock reading that is valid for GPU work.

    CUDA kernels are launched asynchronously, so time.time() on its own
    measures launch, not execution. Drain the device queue first, then read the
    clock. A whole-run timing gets away without this because the closing
    .cpu() transfer forces the synchronisation anyway; a per-step timing has no
    such barrier and would otherwise report launch overhead as the step cost.

    Parameters
    ----------
    device : torch.device or str or None — device the work is queued on; a
             non-CUDA device (or None) skips straight to the clock read

    Returns
    -------
    float — seconds since the epoch, read after the device has caught up
    """
    if device is not None and torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    return time.time()

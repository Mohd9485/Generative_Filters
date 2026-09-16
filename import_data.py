"""
@author: Mohammad Al-Jarrah

Reproduce the figures of the Linear-Quadratic and Lorenz-63 experiments from
their saved archives, without re-running any filter.

The SIF family has two variants:

    SIF-ODE     epsilon > 0, lambda = 0, deterministic ODE inference  (sif_ode)
    SIF-SDE     epsilon > 0, lambda = 1, stochastic SDE inference     (sif_sde)

A third variant -- lambda = 1 read out through the ODE, rather than the SDE --
existed early in this study as the middle step of a three-rung ablation
ladder, separating the two ingredients (epsilon > 0 and lambda > 0) one at a
time. Once that ablation had been made, the pair above is the comparison that
matters: the deterministic map against the stochastic one. It is no longer
computed by any driver script or carried by any archive.

Method selection is driven entirely by the METHODS list below: a key absent
from it is not read from an archive or drawn, so adding or removing a method
is a one-line edit here.

Archives consumed: one particle-trajectory archive each from Quadratic.py and
L63.py, an ensemble-size sweep from L63_vs_particles.py, and the three
Quadratic sweeps (Quadratic_vs_dim.py, Quadratic_vs_particles.py,
Quadratic_vs_final_iter.py). Each `load(...)` call below names the exact
archive and, where the choice is not obvious (which FI, which seed), why. The
three Quadratic sweep summaries are small -- they hold
only the aa-SW2 and timing matrices plus the swept values, not the particles -- so
those figures load instantly; the per-sweep particle archives are not needed
here.

Every sweep goes through the same helpers. The swept axis is a parameter rather
than something baked in -- 'L_list', 'N_list' or 'F_list' -- so the three
sections below differ only in which archive they load and what they call the x
axis. Any archive that is absent is skipped with a note and the rest still plot,
so a sweep that has not been run yet costs nothing but a [skip] line.

What to expect in the refinement sweep
--------------------------------------
Final_Number_ITERATION sets how many training iterations each analysis step
after the first is given, so `online` should rise with it, roughly linearly.
`spinup` should NOT: the first step trains from a random initialisation for
ITERATION iterations regardless, so a flat offline curve is the expected result
and a sloped one means the sweep is moving something it should not be. EnKF and
SIR take no parameter dict at all and are flat in every panel by construction.

Timing is reported in two parts rather than one. The first analysis step of a
learned filter trains a network from a random initialisation, and it is the one
step that can be done offline: its training pairs are drawn from the prior
alone, before any observation has arrived. Every step after it refines an
already-warm network, and that is the cost a deployed filter actually pays.
Each sweep therefore gets two timing figures rather than one — `online` in
seconds per analysis step, and the offline `spinup` in seconds paid once — and
they are drawn separately because they are not the same quantity and do not
share units. A single total divided by the step count blends the two and
reports a number no step ever took.

Failures are tabulated before the figures are drawn. A filter that diverges
leaves no mark on a log-scale plot: an aborted point is NaN, which matplotlib
skips silently, and a blown-up point is pulled off the top of the axis. Both
therefore read as "curve missing here" rather than as a failure. The table
names them explicitly, for each sweep, and is printed only when there is
something to report.

Archives written before that split carry only the totals. They are still
readable here: the timing figures fall back to runtime/step count and say so,
rather than failing on a missing key.

Caveat on the timing figures: the drivers dispatch NUM_GPU filters concurrently,
so every recorded runtime includes contention with the two other methods in its
dispatch batch. The curves are therefore comparable in shape and scaling but are
not isolated per-method benchmarks. The spin-up column also absorbs one-off CUDA
warm-up (context, allocator, cuBLAS handles) on whichever step touches a device
first — visible as a non-zero spin-up for EnKF and SIR, which train nothing.
"""

import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Configure matplotlib to embed fonts in PDF/PS outputs and set default font sizes.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['savefig.transparent'] = False   # True would also blank the axes patch
matplotlib.rcParams['savefig.facecolor']   = 'none'  # transparent outside the axes
matplotlib.rcParams['savefig.edgecolor']   = 'none'
matplotlib.rcParams['axes.facecolor']      = 'white' # white inside the axes
matplotlib.rcParams['figure.facecolor']    = 'none'
plt.rc('font', size=18)   # ticks and any text not sized explicitly
plt.close('all')

fontsize  = 17   # legends, the info box, point annotations, tick labels
labelsize = fontsize + 5   # axis labels and panel titles
labeling = True  # set False to hide all axis labels

# Legends across the whole figure set.
#   True  -- every sweep figure carries its own legend, so each one is readable
#            on its own.
#   False -- only the error-vs-dimension figure keeps a legend, and every other
#            figure is drawn without one. For a paper or a slide where the
#            figures appear together, one legend identifies the eight curves
#            for all of them, and repeating it eight times wastes the space
#            the curves could be using.
# The trajectory and density panels are unaffected either way: they name each
# method in its panel title rather than in a legend. Set via WITH_LEGENDS=1 in
# the environment to switch it without editing this file.
WITH_LEGENDS = os.environ.get('WITH_LEGENDS', '1') == '1'
ALL_LEGENDS  = WITH_LEGENDS*0

# Method labels, the archive keys holding their particles/runtimes, and plot
# colours. Six learned methods plus the two untrained baselines. C5 is simply
# unused, left free rather than reassigned so a colour always identifies the
# same method across every figure and every version of this file.
#
# This list is the ONLY place the method selection is expressed: both the
# density figures (which look up `X_<key>`) and the sweep figures (which
# filter the archive's method_keys through STYLE) take their membership from
# here. An archive carrying a column for a method not listed here -- from a
# version of a driver that computed one more or one fewer -- has that column
# skipped when it is read, rather than raising.
METHODS = [
    ('EnKF',    'EnKF',        'C0'),
    ('SIR',     'SIR',         'C1'),
    ('OTF',     'OTF',         'C2'),
    ('FMF',     'FMF',         'C3'),
    ('SIF-ODE', 'SIF_ODE',     'C4'),   # eps>0, lambda=0, ODE inference
    ('SIF-SDE', 'SIF_SDE',     'C8'),   # eps>0, lambda=1, SDE inference
    ('SBF',     'SBF',         'C6'),
    ('KRF',     'KRF',         'C7'),
]

# key -> (label, colour), so a sweep summary can be plotted in whatever order it
# stored its rows while still picking up the labels and colours used above.
STYLE = {key: (label, color) for label, key, color in METHODS}


def _rows_in_style(summary, what=''):
    """
    Row indices of the archive's methods that this variant draws, and their keys.

    A sweep archive's method_keys and METHODS above usually name the same set,
    but need not: an older or newer archive can carry a column this file
    doesn't draw. Rather than special-casing that at each call site, every
    function that reads `method_keys` passes it through here, so the
    selection lives in one list and a method can be added or removed by
    editing that list alone.

    Returns
    -------
    (list of int, list of str) — positions into the archive's matrices, and the
    keys at those positions, in the archive's own row order.
    """
    keys = [str(k) for k in summary['method_keys']]
    rows = [i for i, k in enumerate(keys) if k in STYLE]
    dropped = [k for k in keys if k not in STYLE]
    if dropped and what:
        print(f'[note] {what}: not drawn by this variant: {dropped}')
    return rows, [keys[i] for i in rows]


# Archives live here — the drivers all write into this directory.
DATA_DIR = 'DATA'

# All figures land here.
FIG_DIR = 'figs'
os.makedirs(FIG_DIR, exist_ok=True)
print(f'figures -> {FIG_DIR}/  (legends {"on" if WITH_LEGENDS else "off except where pinned"})')

# All archives -- the Quadratic sweeps, Quadratic.py's and L63.py's own runs,
# and L63_vs_particles.py's sweep -- are the 20-simulation re-run of steps 4, 5
# and 6 (2026-09-06/07). Two things worth knowing about that set:
#   * step 4 gained an fi = 0 level -- stage 1's configuration with a zero online
#     budget, so the filters train at NO analysis step and run purely on their
#     warm-started weights. It is the no-adaptation baseline the rest of the
#     budget axis is measured against, and it is why the axis now starts at zero.
#   * KRF's timing row in step 4 is an ISOLATED measurement (one GPU, nothing
#     else running), re-measured after the original contended numbers came out
#     non-monotone in the budget. The other seven methods' timings are still
#     contended, measured at a time when nine filters shared three GPUs —
#     this study's current eight plus the since-dropped third SIF rung, which
#     was still being computed then — so KRF's compute is not strictly
#     comparable with theirs. The errors are unaffected: the re-measurement
#     reproduced them bit-for-bit.

# Methods that read no training budget at all. Their sweep points are repeats of
# one experiment rather than a curve, so on the compute axis they are drawn as a
# single marker at their mean rather than as a line whose spread would imply a
# budget dependence they do not have -- the spread across their points is timing
# noise, and EnKF's four measurements differing by 0.0003 s is exactly that.
UNTRAINED = ('EnKF', 'SIR')

# A sweep point counts as blown up when it exceeds the method's own median
# across the sweep by this factor.
BLOWUP_FACTOR = 10.0


def load(path):
    """
    Load an .npz archive into a dict, or return None with a note if it is absent.

    A bare filename is resolved inside DATA_DIR, so call sites can keep naming
    archives the way the drivers name them. A path that already carries a
    directory component is used exactly as given.
    """
    if not os.path.dirname(path):
        path = os.path.join(DATA_DIR, path)
    if not os.path.exists(path):
        print(f'[skip] {path} not found — run the script that writes it first.')
        return None
    return dict(np.load(path, allow_pickle=True))


def crop_sweep(summary, x_key, xmax):
    """
    A copy of a sweep summary keeping only the points with x <= xmax.

    Every per-point array is sliced together -- the (methods x points) matrices
    on their last axis, the swept values and the sampling floor on their first --
    so the result is a self-consistent summary that plots exactly like an
    uncropped one. The original dict is not modified.

    Used rather than an axis limit because a cropped AXIS still lets the dropped
    column set the autoscale, and still lets report_failures name a point the
    figure does not show. Cropping the data means the panel and its reports agree
    on which points exist.

    The keys are listed explicitly rather than inferred from array shapes: with
    eight methods and six sweep points nothing collides today, but a sweep that
    happened to have as many points as methods would make a shape-based rule
    slice method_keys by mistake.

    Parameters
    ----------
    summary : dict — sweep summary archive contents.
    x_key   : str  — key holding the swept values ('N_list', 'L_list', 'F_list').
    xmax    : float — keep points with summary[x_key] <= xmax.

    Returns
    -------
    dict — the cropped copy.
    """
    x    = np.asarray(summary[x_key])
    keep = x <= xmax
    if keep.all():
        return summary
    out = dict(summary)
    out[x_key] = x[keep]
    for k in ('sw2_mean', 'sw2_std', 'runtime', 'spinup', 'online', 'online_std'):
        if k in out:
            out[k] = np.asarray(out[k])[:, keep]
    for k in ('sw2_floor',):
        if k in out and np.asarray(out[k]).ndim == 1:
            out[k] = np.asarray(out[k])[keep]
    print(f'[note] {x_key}: cropped to <= {xmax:g}; dropped {list(x[~keep])}')
    return out


def per_step_factor(data):
    """
    Conversion from a filter's total runtime to its cost per analysis step.

    Every recorded runtime covers one whole call: all NUM_SIM simulations, and
    all T-1 analysis steps within each. Dividing by their product gives the
    per-step figure. NUM_SIM and T are read from the archive when present (the
    sweep summaries store them) and otherwise recovered from X_true's shape.
    """
    num_sim = int(data['NUM_SIM']) if 'NUM_SIM' in data else data['X_true'].shape[0]
    T       = int(data['T'])       if 'T'       in data else data['X_true'].shape[1]
    return 1.0 / (num_sim * (T - 1))


def split_steps(steps):
    """
    Reduce a raw per-analysis-step timing array to (spin-up, online, online std).

    Parameters
    ----------
    steps : ndarray (NUM_SIM x T-1) — per-analysis-step seconds. Column 0 is the
            spin-up step; the rest are the online steps. Steps an aborted run
            never reached are NaN, and a method carried over from an archive
            with no per-step timing is NaN throughout — nanmean warns on such a
            row but the NaN it returns is the intended answer, so the warning is
            silenced rather than the value.

    Returns
    -------
    tuple of float — (spin-up seconds, online seconds/step, online std)
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return (float(np.nanmean(steps[:, 0])),
                float(np.nanmean(steps[:, 1:])),
                float(np.nanstd(steps[:, 1:])))


def report_runtimes(data):
    """
    Print the per-method computational time recorded in the archive, split into
    the offline spin-up and the online per-step cost.

    steps_<KEY> holds the raw (NUM_SIM x T-1) per-analysis-step timing. Column 0
    is the spin-up: the learned filters train from a random initialisation
    there, which is the one step that can be done offline, since its training
    pairs come from the prior alone and no observation has arrived yet. Every
    later step refines an already-warm network. The ratio is the number the
    comparison turns on — how many online steps the offline stage is worth.

    Archives predating the split carry only the totals, so those fall back to
    the old whole-run figure, which averages over both regimes.
    """
    if not any(f'steps_{key}' in data for _, key, _ in METHODS):
        per_step = per_step_factor(data)
        print('Per-method computational time (seconds per analysis step, spin-up and '
              'online blended — this archive predates the split):')
        for label, key, _ in METHODS:
            time_key = f'time_{key}'
            if time_key in data:
                total = float(data[time_key])
                print(f'    {label:16s}: {total * per_step:9.4f}   ({total:8.2f} s total)')
        return

    print('Per-method computational time (spin-up is the offline first analysis step; '
          'online is every step after it):')
    for label, key, _ in METHODS:
        if f'steps_{key}' not in data:
            continue
        spin, online, online_std = split_steps(data[f'steps_{key}'])
        total = float(data[f'time_{key}']) if f'time_{key}' in data else np.nan
        print(f'    {label:16s}: spin-up {spin:8.3f} s  |  '
              f'online {online:7.4f} +/- {online_std:.4f} s/step  |  '
              f'ratio {spin / online:6.1f}x  |  total {total:8.2f} s')


def report_offline(summary, x_key, xlabel):
    """
    Print the offline spin-up cost carried by a sweep summary, per method.

    The spin-up is paid once, so it does not belong on the online per-step
    curves the sweep figures draw — but it is still the price of deploying each
    method, and it varies across the sweep. The mean over the swept values is
    printed with the range beside it, because a mean alone would hide a spin-up
    that grows with the dimension or the ensemble size.

    Parameters
    ----------
    summary : dict — sweep summary archive contents.
    x_key   : str  — archive key holding the swept values ('L_list' or 'N_list').
    xlabel  : str  — name of the swept quantity, for the heading.
    """
    if 'spinup' not in summary:
        print(f'[skip] offline spin-up: this summary predates the timing split — '
              f'rerun the sweep to record it.')
        return

    x      = summary[x_key]
    rows, keys = _rows_in_style(summary)
    spinup     = summary['spinup'][rows]

    print(f'\nOffline spin-up (first analysis step), averaged over {xlabel} '
          f'= {", ".join(f"{v:g}" for v in x)}:')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for i_m, key in enumerate(keys):
            label, _ = STYLE.get(key, (key, None))
            row      = spinup[i_m]
            print(f'    {label:16s}: {np.nanmean(row):8.3f} s   '
                  f'(range {np.nanmin(row):7.3f} - {np.nanmax(row):7.3f} s)')


def report_completeness(summary, x_key, xlabel):
    """
    State which sweep points a summary covers, and whether the run finished.

    The sweep scripts rewrite their summary after every point, so a summary on
    disk may legitimately cover only part of the sweep -- and a partial one is
    indistinguishable from a complete one once loaded, since the axis key names
    only the columns present. The complete flag settles it, and failed_F names
    any point that raised and was skipped rather than never reached.

    Summaries written before those keys existed carry neither; the sweep points
    are still listed, and nothing is claimed about completeness.

    Parameters
    ----------
    summary : dict — sweep summary archive contents.
    x_key   : str  — archive key holding the swept values.
    xlabel  : str  — name of the swept quantity, for the heading.
    """
    x = summary[x_key]
    print(f'\nSweep points present: {xlabel} = {", ".join(f"{v:g}" for v in x)}')

    if 'complete' in summary:
        if bool(summary['complete']):
            print('    the sweep ran to completion.')
        else:
            print('    [warning] this summary is from a sweep that did not finish; '
                  'the points above are what completed.')

    failed = summary['failed_F'] if 'failed_F' in summary else np.array([])
    if np.size(failed):
        print(f'    [warning] these points raised and were skipped: '
              f'{", ".join(f"{v:g}" for v in np.atleast_1d(failed))}')


def timing_plot_kwargs(summary, name, which):
    """
    Pick the timing quantity a sweep figure should draw.

    The two regimes are plotted separately rather than as one curve, because
    they are different quantities: `online` is a rate, in seconds per analysis
    step, paid on every step a deployed filter takes, while `spinup` is a
    one-off cost in seconds. Drawing them on one axis would invite reading a
    ratio between numbers that do not share units.

    'online' prefers the recorded per-step cost and falls back to the stored
    total divided by the step count for summaries written before the split —
    that blends the one-off spin-up into every step, so it is labelled
    differently and announced rather than substituted silently. 'offline' has no
    such fallback: a total carries no record of what its first step cost, so the
    panel is skipped rather than guessed at.

    Parameters
    ----------
    summary : dict — sweep summary archive contents.
    name    : str  — archive name, used only in the notices.
    which   : str  — 'online' or 'offline'.

    Returns
    -------
    dict or None — keyword arguments for plot_vs_sweep, or None when the
                   quantity cannot be recovered from this summary.
    """
    if which == 'online':
        # online_std is carried by the summary but not drawn: on a log axis the
        # near-zero EnKF/SIR steps have std exceeding mean, so the bars truncate
        # against the axis floor and read as data rather than as clipping.
        if 'online' in summary:
            return dict(y_key='online', ylabel='online time per step (s)', scale=1.0)
        print(f'[note] {name}: no per-step timing in this summary; falling back to the '
              f'total runtime over the step count, which still carries the offline '
              f'first step in every point.')
        return dict(y_key='runtime', ylabel='wall-clock time per step (s)',
                    scale=per_step_factor(summary))

    if which == 'offline':
        if 'spinup' in summary:
            return dict(y_key='spinup', ylabel='offline spin-up time (s)', scale=1.0)
        print(f'[skip] {name}: offline spin-up figure — this summary predates the '
              f'timing split, and a total carries no record of its first step.')
        return None

    raise ValueError(f"which must be 'online' or 'offline', got {which!r}")


def report_failures(summary, x_key, xlabel, name):
    """
    Tabulate the sweep points where a method aborted or blew up.

    Neither failure mode is visible in the figures. An aborted point is NaN, and
    both matplotlib and a log axis drop it without comment; a blown-up point is
    finite but orders of magnitude above its neighbours, so it leaves the top of
    the axis. Either way the curve simply appears to be missing, which is
    indistinguishable from a sweep point that was never run.

    Some summaries carry per-run diagnostics (n_nan, n_runs_blown, sw2_run_max),
    which separate the two modes exactly. Summaries written by the sweep
    scripts themselves carry only sw2_mean, so the check degrades to what that
    alone can show: a non-finite entry is an abort of unknown extent, and an
    entry far above the method's own median across the sweep is a probable
    blow-up. The degraded mode is announced, so a thin report is never mistaken
    for a clean one.

    Parameters
    ----------
    summary : dict — sweep summary archive contents.
    x_key   : str  — archive key holding the swept values ('L_list' or 'N_list').
    xlabel  : str  — name of the swept quantity, for the heading.
    name    : str  — archive name, used in the heading.

    Returns
    -------
    int — number of failed (method, sweep point) pairs found.
    """
    x        = summary[x_key]
    rows, keys = _rows_in_style(summary)
    sw2        = summary['sw2_mean'][rows]
    detailed = 'n_nan' in summary

    rows = []
    for i_m, key in enumerate(keys):
        label, _ = STYLE.get(key, (key, None))

        # A method's own median across the sweep is the reference for "far
        # above": it is unchanged by one bad point, whereas a mean is not.
        finite = sw2[i_m][np.isfinite(sw2[i_m])]
        median = np.median(finite) if finite.size else np.nan

        for i_x, xv in enumerate(x):
            if detailed and summary['n_nan'][i_m, i_x] > 0:
                n_nan, n_tot = summary['n_nan'][i_m, i_x], summary['n_total'][i_m, i_x]
                rows.append((label, xv, 'ABORT',
                             f'{n_nan}/{n_tot} entries NaN ({n_nan / n_tot:.0%})'))
            elif detailed and summary['n_runs_blown'][i_m, i_x] > 0:
                rows.append((label, xv, 'blow-up',
                             f'{summary["n_runs_blown"][i_m, i_x]} run(s) diverged, '
                             f'worst {summary["sw2_run_max"][i_m, i_x]:.4g}'))
            elif not detailed and not np.isfinite(sw2[i_m, i_x]):
                rows.append((label, xv, 'ABORT', 'sw2_mean is NaN'))
            elif (not detailed and np.isfinite(median) and median > 0
                  and sw2[i_m, i_x] > BLOWUP_FACTOR * median):
                rows.append((label, xv, 'blow-up',
                             f'sw2_mean {sw2[i_m, i_x]:.4g} is '
                             f'{sw2[i_m, i_x] / median:.0f}x its sweep median'))

    if not rows:
        print(f'\nFailure report ({name}): no aborted or blown-up sweep points.')
        return 0

    print(f'\nFailure report ({name}) — points that vanish from the figures:')
    if not detailed:
        print('    [note] this summary carries no per-run diagnostics, so the '
              'counts above are inferred from sw2_mean, not exact.')
    print(f'    {"method":16s} {xlabel:>18s}  {"mode":<9s} detail')
    print(f'    {"-"*72}')
    for label, xv, mode, detail in rows:
        print(f'    {label:16s} {xv:>18g}  {mode:<9s} {detail}')
    return len(rows)


def plot_trajectories(data, fontsize=fontsize, labelsize=labelsize,
                      savename=None):
    """
    Particle trajectories for every method against the true state: one column
    per method, one row per state coordinate.

    Parameters
    ----------
    data     : dict — archive contents (must contain t, X_true and the X_* keys).
    savename : str  — optional path stub; if given, saves <stub>.pdf.
    """
    k             = 0
    t             = data['t']
    X_true        = data['X_true']
    L             = X_true.shape[2]
    plot_particle = 500

    present = [(label, key, color) for label, key, color in METHODS if f'X_{key}' in data]
    methods = len(present)

    plt.figure(figsize=(26, 16))
    for col, (label, key, color) in enumerate(present):
        X = data[f'X_{key}']          # shape (NUM_SIM x T x L x N).
        for l in range(L):
            plt.subplot(L, methods, methods * l + col + 1)
            plt.plot(t, X[k, :, l, :plot_particle], color=color, alpha=0.1, rasterized=True)
            plt.plot(t, X_true[k, :, l], color='k', linestyle='--', label='True state')
            if labeling: plt.xlabel('time', fontsize=labelsize)
            if l == 0:
                plt.title(label)
            if col == 0 and labeling:
                plt.ylabel(f'X({l + 1})', fontsize=labelsize)
            if l <= L-2:
                plt.gca().get_xaxis().set_visible(False)
            if col > 0:
                plt.gca().get_yaxis().set_visible(False)
            plt.ylim([-5,5])

    plt.tight_layout()
    if savename is not None:
        plt.savefig(f'{savename}.pdf', bbox_inches='tight')


def plot_single_state_density(data, coord=1, bins=120, smooth=1.5, vmax_pct=90,
                              fontsize=fontsize, labelsize=labelsize,
                              savename=None):
    """
    One state coordinate against time as a density, one panel per method.

    At each analysis step the particles are histogrammed along the state axis on
    a grid common to every panel, and the resulting (state x time) array is drawn
    with pcolormesh. Nothing connects consecutive time steps, so a bimodal
    posterior appears as two continuous bands with genuinely empty space between
    them, rather than as trajectories that cross from one mode to the other.

    The leading panel is the reference posterior X_ref, drawn the same way and on
    the same colour scale, so each filter can be compared against the target it
    approximates rather than only against the other filters.

    This is the reason a percentile band is not used: an interval between two
    quantiles necessarily spans both modes and fills the gap that carries no
    particles at all.

    Each column is normalised as a density (it integrates to one over the grid),
    so the shading reads as the conditional law at that time; the colour scale is
    shared across panels so that the nine remain comparable.

    Parameters
    ----------
    data     : dict  — archive contents (must contain t, X_true and the X_* keys).
    coord    : int   — 0-based state index to draw.
    bins     : int   — number of bins along the state axis.
    smooth   : float — Gaussian smoothing in bins along the state axis; 0 disables
                       it and leaves the raw histogram.
    vmax_pct : float — percentile of the density used as the top of the colour
                       scale, the counterpart of alpha in the scatter version.
                       Lowering it saturates more of the map and so strengthens
                       the colour; raising it towards 100 keeps more gradation
                       inside the peaks. Everything above it is clipped.
    savename : str   — optional path stub; if given, saves <stub>.pdf.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter1d

    k      = 0
    t      = data['t']
    X_true = data['X_true']

    # Panels: the reference posterior first, so every method can be read against
    # the distribution it is trying to match, then one panel per method. Each
    # entry carries its own (T x N) ensemble, so the indexing that produces it is
    # allowed to differ per source. The reference is drawn in greyscale to set it
    # apart from the coloured filters.
    #
    # X_ref is (NUM_SIM x T x L x N_true) — one reference per simulation, matching
    # simulation k's own trajectory. Archives written before per-simulation
    # trajectories stored a single shared reference, (T x L x N_true); both are
    # accepted so old figures can still be redrawn.
    panels = []
    if 'X_ref' in data:
        X_ref = data['X_ref']
        X_ref = X_ref[:, coord, :] if X_ref.ndim == 3 else X_ref[k, :, coord, :]
        panels.append(('Reference (SIR)', X_ref, 'k'))
    panels += [(label, data[f'X_{key}'][k, :, coord, :], color)
               for label, key, color in METHODS if f'X_{key}' in data]
    npanel = len(panels)

    # Common state grid: 0.5-99.5 percentile over all panels, widened by 10%.
    shown  = np.concatenate([Xp.ravel() for _, Xp, _ in panels])
    lo, hi = np.percentile(shown[np.isfinite(shown)], [0.5, 99.5])
    lo, hi = min(lo, X_true[k, :, coord].min()), max(hi, X_true[k, :, coord].max())
    pad    = 0.1 * (hi - lo)
    edges   = np.linspace(lo - pad, hi + pad, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Densities first, so the colour scale can be shared across the panels.
    dens = []
    for _, Xp, _ in panels:                                      # Xp: (T x N)
        D = np.stack([np.histogram(Xp[i], bins=edges, density=True)[0]
                      for i in range(Xp.shape[0])], axis=1)      # (bins x T)
        dens.append(gaussian_filter1d(D, smooth, axis=0) if smooth else D)
    vmax = np.percentile(np.concatenate([D.ravel() for D in dens]), vmax_pct)

    plt.figure(figsize=(2.9 * npanel, 4))
    for col, ((label, _, color), D) in enumerate(zip(panels, dens)):
        plt.subplot(1, npanel, col + 1)
        cmap = LinearSegmentedColormap.from_list('', ['white', color])
        plt.pcolormesh(t, centers, D, cmap=cmap, vmin=0, vmax=vmax,
                       shading='gouraud', rasterized=True)
        plt.plot(t, X_true[k, :, coord], color='k', linestyle='--', label='True state')
        plt.ylim(-5, 5)
        plt.title(label)
        if labeling: plt.xlabel('time', fontsize=labelsize)
        if col == 0 and labeling:
            plt.ylabel(f'X({coord + 1})', fontsize=labelsize)
        else:
            plt.gca().get_yaxis().set_visible(False)

    plt.tight_layout()
    if savename is not None:
        plt.savefig(f'{savename}.pdf', bbox_inches='tight')


def _panel_ensembles(data, coord, k=0):
    """
    (label, (T x N) ensemble, colour) for the reference and every method present.

    Shared by the density figures so they cannot drift apart in which panels they
    show or what order they show them in. X_ref is (NUM_SIM x T x L x N_true) --
    one reference per simulation, matching simulation k's own trajectory --
    though archives written before per-simulation trajectories stored a single
    shared (T x L x N_true), and both are accepted.
    """
    panels = []
    if 'X_ref' in data:
        X_ref = data['X_ref']
        X_ref = X_ref[:, coord, :] if X_ref.ndim == 3 else X_ref[k, :, coord, :]
        panels.append(('Reference (SIR)', X_ref, 'k'))
    panels += [(label, data[f'X_{key}'][k, :, coord, :], color)
               for label, key, color in METHODS if f'X_{key}' in data]
    return panels


def _state_grid(panels, X_true_coord, bins):
    """Common state grid for a set of panels: 0.5-99.5 percentile, widened 10%."""
    shown  = np.concatenate([Xp.ravel() for _, Xp, _ in panels])
    lo, hi = np.percentile(shown[np.isfinite(shown)], [0.5, 99.5])
    lo, hi = min(lo, X_true_coord.min()), max(hi, X_true_coord.max())
    pad    = 0.1 * (hi - lo)
    edges  = np.linspace(lo - pad, hi + pad, bins + 1)
    return edges, 0.5 * (edges[:-1] + edges[1:])


def _time_densities(panels, edges, smooth):
    """(bins x T) density per panel, optionally smoothed along the state axis."""
    from scipy.ndimage import gaussian_filter1d
    dens = []
    for _, Xp, _ in panels:                                      # Xp: (T x N)
        D = np.stack([np.histogram(Xp[i], bins=edges, density=True)[0]
                      for i in range(Xp.shape[0])], axis=1)      # (bins x T)
        dens.append(gaussian_filter1d(D, smooth, axis=0) if smooth else D)
    return dens


def plot_density_slice_rows(data, coord_top=1, coord_bot=5, t_index=27, bins=120,
                            smooth=1.5, slice_smooth=3.0, vmax_pct=90,
                            ylim=(-5, 5), collapse_factor=1.5, wspace=0.12,
                            top_gap=0.18,
                            fontsize=fontsize, labelsize=labelsize,
                            savename=None):
    """
    Two state coordinates as densities over time, plus the marked slice opened up.

    Three rows, one column per method with the reference posterior leading:

        row 1   density of X(coord_top + 1) against time
        row 2   density of X(coord_bot + 1) against time, with a dashed vertical
                line at t[t_index]
        row 3   the particles of X(coord_bot + 1) AT t[t_index], as a histogram

    Row 3 is drawn rotated -- the state value on the y axis, density on the x --
    so it shares row 2's vertical scale. Read straight down the dashed line and
    the distribution is in the same coordinates it was just plotted in; a
    conventional histogram would put that axis horizontally and rotate the
    correspondence by ninety degrees for no gain.

    Each row-3 panel carries the reference posterior behind it as a black outline,
    so whether a filter holds both modes or has collapsed onto one is visible in
    the panel itself rather than by looking back to the leading column.

    Why a density and not a percentile band: an interval between two quantiles
    necessarily spans both modes of a bimodal posterior and fills the gap that
    carries no particles at all. Nothing connects consecutive time steps either,
    so two modes appear as two bands with genuinely empty space between them
    rather than as trajectories crossing between them.

    Colour scales are shared within a row but NOT between rows: the two
    coordinates have their own spreads, and forcing one scale on both would
    flatten whichever is more concentrated.

    Parameters
    ----------
    data      : dict  — archive contents (needs t, X_true and the X_* keys).
    coord_top : int   — 0-based state index for row 1 (default 1 = X(2)).
    coord_bot : int   — 0-based state index for rows 2 and 3 (default 5 = X(6)).
    t_index   : int   — time index the slice is taken at; row 2's dashed line and
                        row 3's histogram both sit here. Default 27, i.e.
                        t = 2.7 s at tau = 0.1.
    bins      : int   — bins along the state axis, shared by the densities and
                        the histograms so the two read on one grid.
    smooth    : float — Gaussian smoothing in bins along the state axis for the
                        density rows; 0 leaves the raw histogram.
    slice_smooth : float — the same for row 3. Larger than `smooth` by default:
                        row 3 estimates a density from ONE time step, where the
                        density rows average over the whole record, so it carries
                        more sampling noise per bin. 0 leaves the raw histogram.
    vmax_pct  : float — percentile of the density used as the top of the colour
                        scale, per row.
    ylim      : tuple — state-axis limits, shared by all three rows.
    wspace   : float — horizontal gap between columns, as a fraction of a
                       panel's width. The same value the other two grid figures
                       use, so all three match; left to tight_layout they do not.
    top_gap  : float or None — vertical gap between rows 1 and 2, as a fraction
                       of a row's height. The two inter-row gaps are NOT alike:
                       row 1's x axis is hidden, so the space above row 2 is
                       empty, while the space above row 3 holds row 2's tick
                       labels and its 'time' label. A uniform layout gives them
                       the same gap and the top one then reads as slack. This
                       tightens it alone, by sliding row 1 down; what opens up
                       above is cropped by bbox_inches at save time. None leaves
                       the layout as tight_layout set it.
    collapse_factor : float — a row-3 panel whose peak density exceeds this
                        multiple of the reference's is treated as a collapse
                        spike and left out of the shared density limit, so one
                        degenerate ensemble cannot compress every other panel.
                        Such panels are clipped, and named on stdout. Raise it to
                        let taller peaks set the axis; a very large value
                        restores "fit the tallest panel".
    savename  : str   — optional path stub; if given, saves <stub>.pdf.
    """
    k      = 0
    t      = np.asarray(data['t'])
    X_true = data['X_true']

    if not (0 <= t_index < len(t)):
        print(f'[skip] plot_density_slice_rows: t_index={t_index} outside 0..{len(t)-1} '
              f'(this archive has T={len(t)}, t up to {t[-1]:.2f})')
        return
    t_slice = t[t_index]

    rows = []
    for coord in (coord_top, coord_bot):
        panels        = _panel_ensembles(data, coord, k)
        edges, cent   = _state_grid(panels, X_true[k, :, coord], bins)
        dens          = _time_densities(panels, edges, smooth)
        vmax          = np.percentile(np.concatenate([D.ravel() for D in dens]), vmax_pct)
        rows.append(dict(coord=coord, panels=panels, edges=edges, centers=cent,
                         dens=dens, vmax=vmax))
    npanel = len(rows[0]['panels'])

    # Row 3 reuses row 2's grid, so a bar sits at the same height as the shading
    # it was sliced out of. Normalised as a density, which is what makes
    # ensembles of different size comparable.
    #
    # Smoothed with the same Gaussian-on-a-grid estimator the density rows use,
    # so the slice and the shading it came from are the same kind of object
    # rather than a smooth field beside a ragged bar chart. One time step holds
    # far fewer particles than a whole row does, so the raw counts are noticeably
    # noisier and want more smoothing than `smooth` gives -- SIR especially,
    # where resampling leaves few distinct values. Set slice_smooth=0 to see the
    # bare histogram.
    from scipy.ndimage import gaussian_filter1d
    bot        = rows[1]
    slice_hist = [np.histogram(Xp[t_index], bins=bot['edges'], density=True)[0]
                  for _, Xp, _ in bot['panels']]
    if slice_smooth:
        slice_hist = [gaussian_filter1d(h, slice_smooth) for h in slice_hist]

    # The shared density axis is NOT the maximum over all panels. A filter that
    # has collapsed piles its whole mass into one narrow spike, so its peak
    # density sits far above every other panel's -- SIR reaches 0.69 here
    # against a 0.40 reference, because resampling leaves few distinct particles
    # -- and letting that one number set the axis compresses every well-behaved
    # panel into the left half of its box, which is the opposite of what the row
    # is for.
    #
    # So peaks more than `collapse_factor` times the reference's are treated as
    # collapse spikes and excluded from the limit; the axis is set by the tallest
    # panel that remains. Excluded panels are drawn clipped rather than rescaled,
    # and named below, so the clipping is a stated fact rather than a silent one.
    peaks    = np.array([h.max() for h in slice_hist])
    ref_peak = peaks[0] if bot['panels'][0][0].startswith('Reference') else np.median(peaks)
    keep     = peaks <= collapse_factor * ref_peak
    hmax     = (peaks[keep].max() if keep.any() else peaks.max()) * 1.15
    for (label, _, _), pk in zip(bot['panels'], peaks):
        if pk > hmax:
            print(f'[note] {label}: slice peak density {pk:.3f} exceeds the shared '
                  f'axis limit {hmax:.3f} and is clipped — its mass is concentrated '
                  f'in a narrow mode, {pk / ref_peak:.1f}x the reference peak.')
    ref_hist   = slice_hist[0] if bot['panels'][0][0].startswith('Reference') else None

    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap

    fig, axes = plt.subplots(3, npanel, figsize=(2.9 * npanel, 10.5),
                             gridspec_kw=dict(height_ratios=[1, 1, 1]))
    axes = np.atleast_2d(axes)

    # --- rows 1 and 2: density against time -------------------------------
    for i_row, row in enumerate(rows):
        for col, ((label, _, color), D) in enumerate(zip(row['panels'], row['dens'])):
            ax   = axes[i_row, col]
            cmap = LinearSegmentedColormap.from_list('', ['white', color])
            ax.pcolormesh(t, row['centers'], D, cmap=cmap, vmin=0, vmax=row['vmax'],
                          shading='gouraud', rasterized=True)
            ax.plot(t, X_true[k, :, row['coord']], color='k', linestyle='--', lw=1.2)
            # The slice marker goes on row 2 alone: it points at the row whose
            # distribution is opened up underneath, and putting it on both rows
            # would suggest row 3 shows both coordinates.
            #
            # Dash-dot and black. Black dashed is the true trajectory, here and
            # in row 3, so the marker must not be dashed; dash-dot is the nearest
            # style still free. Black rather than the panel colour because the
            # marker has to stay legible over shading of every hue, and on the
            # darker panels a same-coloured line sinks into the density it is
            # meant to be marking.
            #
            # The white halo keeps it readable where the shading is heaviest.
            if i_row == 1:
                ax.axvline(t_slice, color='k', linestyle='-.', lw=2.2, alpha=0.95,
                           path_effects=[pe.withStroke(linewidth=4.5, foreground='white')])
            ax.set_ylim(*ylim)
            if i_row == 0:
                ax.set_title(label, fontsize=labelsize)
            # Row 1 hides time: row 2 sits directly below it on the same range.
            # Row 2 keeps it, because a slice marker nobody can locate on an axis
            # is just a line -- and row 3 cannot carry it, its x axis is density.
            if i_row == 0:
                ax.get_xaxis().set_visible(False)
            elif labeling:
                ax.set_xlabel('time', fontsize=labelsize)
            if col == 0 and labeling:
                ax.set_ylabel(f'X({row["coord"] + 1})', fontsize=labelsize)
            else:
                ax.get_yaxis().set_visible(False)

    # --- row 3: the marked slice, rotated onto row 2's vertical scale ------
    for col, ((label, _, color), h) in enumerate(zip(bot['panels'], slice_hist)):
        ax = axes[2, col]
        # The reference behind every panel, so mode capture is judged in place.
        if ref_hist is not None and col > 0:
            ax.step(ref_hist, bot['centers'], where='mid', color='k', lw=1.4,
                    label='Reference (SIR)')
        ax.fill_betweenx(bot['centers'], 0, h, color=color, alpha=0.55, step='mid',
                         rasterized=True)
        ax.step(h, bot['centers'], where='mid', color=color, lw=1.6)
        ax.axhline(X_true[k, t_index, bot['coord']], color='k', linestyle='--', lw=1.2)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, hmax)
        # Labelled on every panel: each is a standalone distribution, and the
        # row is read across rather than as one axis shared left to right.
        if labeling:
            ax.set_xlabel('density', fontsize=labelsize)
        if col == 0 and labeling:
            ax.set_ylabel(f'X({bot["coord"] + 1})', fontsize=labelsize)
        else:
            ax.get_yaxis().set_visible(False)
        ax.set_xticks([])

    fig.tight_layout()
    # Same pinned column spacing as the other two grid figures. tight_layout
    # derives it from each figure's own tick-label extents, so grids built the
    # same way drift apart -- this one sat at 20% of a panel width against their
    # 12%. Pinning the three to one value is what makes them read as a set.
    fig.subplots_adjust(wspace=wspace)

    # Row 1 slides down to close the empty gap above row 2. Done after
    # subplots_adjust, which would otherwise reset the positions set here.
    if top_gap is not None:
        p0, p1 = axes[0][0].get_position(), axes[1][0].get_position()
        shift  = (p0.y0 - p1.y1) - top_gap * p0.height
        if shift > 0:
            for ax_ in axes[0]:
                q = ax_.get_position()
                ax_.set_position([q.x0, q.y0 - shift, q.width, q.height])

    if savename is not None:
        fig.savefig(f'{savename}.pdf', bbox_inches='tight')

def n_label(N):
    """
    Row label for an ensemble size: 'N=$10^3$' for an exact power of ten,
    'N=250' otherwise.

    The powers of ten are the sizes these figures are usually built from, and
    they are long enough written out that they crowd a y axis shared with the
    state name. Anything that is not an exact power is left as a plain integer
    rather than rounded into a nearby exponent.
    """
    k = int(round(np.log10(N)))
    return rf'N=$10^{k}$' if 10 ** k == N else f'N={N}'


def plot_density_rows_vs_N(datasets, coord=0, bins=120, smooth=1.5, vmax_pct=90,
                           ylim=None, wspace=0.12,
                           fontsize=fontsize, labelsize=labelsize,
                           savename=None):
    """
    One state coordinate as a density over time, one ROW per ensemble size.

    Each row is built exactly as `plot_single_state_density` builds its single
    row -- reference posterior first, then one panel per method, each column of a
    panel a histogram of that analysis step's particles along the state axis --
    stacked so the effect of N reads down a column instead of across two figures.

    The state grid and the colour scale are computed ONCE over every row. That is
    the whole reason this is not `plot_single_state_density` called twice: drawn
    separately, each row would get its own axes and its own `vmax`, and a band
    that looks tighter in the bottom row would be telling you nothing at all. On
    a shared grid and a shared scale, tighter means tighter.

    Columns are the methods common to every row, in METHODS order, so a column
    means the same thing top to bottom. A method present in one archive and not
    another is dropped rather than left as a hole.

    Note on the reference column: the reference posterior does not depend on N --
    it is the same `N_true`-particle SIR in every archive of a given run -- so
    that column should look near-identical down the figure. It is the control:
    if it does not, the rows are not comparable and something else differs
    between the archives.

    Parameters
    ----------
    datasets : list of (row_label, data) — one archive per row, top to bottom.
        The label closes the y axis of that row's first panel, after the state
        name: 'X(1), N=$10^2$'. n_label() builds the usual form.
    coord    : int   — 0-based state index to draw.
    bins     : int   — number of bins along the state axis.
    smooth   : float — Gaussian smoothing in bins along the state axis; 0 leaves
                       the raw histogram.
    vmax_pct : float — percentile of the density used as the top of the colour
                       scale; everything above it is clipped. Lowering it
                       saturates more of the map, raising it towards 100 keeps
                       more gradation inside the peaks.
    ylim     : (lo, hi) or None — None uses the full common state grid, which is
                       what keeps the rows comparable. A tuple crops every row
                       identically. Unlike the quadratic figures this is not
                       pinned to (-5, 5): L63's first state runs to roughly
                       +/- 20, and that limit would show almost nothing.
    wspace   : float — horizontal gap between columns, as a fraction of a panel's
                       width. Shared with plot_density_traj_rows; see there.
    savename : str   — optional path stub; if given, saves <stub>.pdf.
    """
    from matplotlib.colors import LinearSegmentedColormap

    k    = 0
    rows = [(label, _panel_ensembles(data, coord, k), data['t'],
             data['X_true'][k, :, coord])
            for label, data in datasets]

    # Column order from the first row, kept only where every row has it.
    order  = [lbl for lbl, _, _ in rows[0][1]]
    common = set.intersection(*[{lbl for lbl, _, _ in panels}
                                for _, panels, _, _ in rows])
    order  = [lbl for lbl in order if lbl in common]
    dropped = [lbl for lbl, _, _ in rows[0][1] if lbl not in common]
    if dropped:
        print(f'[density rows] not in every archive, dropped: {dropped}')
    npanel, nrow = len(order), len(rows)

    # One grid and one colour scale over every row — see the docstring.
    all_panels = [pan for _, panels, _, _ in rows
                  for pan in panels if pan[0] in order]
    all_true   = np.concatenate([X_true_c for _, _, _, X_true_c in rows])
    edges, centers = _state_grid(all_panels, all_true, bins)

    dens = []
    for _, panels, _, _ in rows:
        by_label = {lbl: (lbl, Xp, color) for lbl, Xp, color in panels}
        dens.append(_time_densities([by_label[lbl] for lbl in order], edges, smooth))
    vmax = np.percentile(
        np.concatenate([D.ravel() for row in dens for D in row]), vmax_pct)

    colors = {lbl: color for lbl, _, color in rows[0][1]}

    fig, axes = plt.subplots(nrow, npanel, squeeze=False,
                             figsize=(2.9 * npanel, 3.4 * nrow))
    for r, ((row_label, _, t, X_true_c), row_dens) in enumerate(zip(rows, dens)):
        for c, (lbl, D) in enumerate(zip(order, row_dens)):
            ax   = axes[r][c]
            cmap = LinearSegmentedColormap.from_list('', ['white', colors[lbl]])
            ax.pcolormesh(t, centers, D, cmap=cmap, vmin=0, vmax=vmax,
                          shading='gouraud', rasterized=True)
            ax.plot(t, X_true_c, color='k', linestyle='--', label='True state')
            ax.set_ylim(*(ylim if ylim is not None else (edges[0], edges[-1])))

            # Titles on the top row and x labels on the bottom row only: every
            # panel shares both axes, so repeating them is noise.
            if r == 0:
                ax.set_title(lbl, fontsize=labelsize)
            if r == nrow - 1 and labeling:
                ax.set_xlabel('time', fontsize=labelsize)
            else:
                ax.get_xaxis().set_visible(False)

            # The first column carries both, on one line: the state every panel
            # draws, then the ensemble size that distinguishes this row.
            if c == 0 and labeling:
                ax.set_ylabel(f'X({coord + 1}), {row_label}', fontsize=labelsize)
            else:
                ax.get_yaxis().set_visible(False)

    fig.tight_layout()
    # tight_layout derives the column spacing from each figure's own tick-label
    # extents, so two grids built the same way drift apart -- this one came out
    # at 20% of a panel width against the density/trajectory figure's 12%. Pin it
    # instead, to the same value in both, so the two read as one set.
    fig.subplots_adjust(wspace=wspace)
    if savename is not None:
        fig.savefig(f'{savename}.pdf', bbox_inches='tight')



def plot_density_traj_rows(top, bottom, coord=0, bins=120, smooth=1.5,
                           vmax_pct=90, plot_particle=500, alpha=0.08,
                           density_cols=('Reference (SIR)',),
                           ylim=None, wspace=0.12,
                           fontsize=fontsize, labelsize=labelsize,
                           savename=None):
    """
    Two rows over the same columns: a density row above a trajectory row.

    A companion to plot_density_rows_vs_N, which stacks two DENSITY rows to show
    what changes with the ensemble size. This one changes the representation
    instead: the top row is the same time-resolved density, the bottom row draws
    the individual particle paths of a second archive. They answer different
    questions about the same object and are easy to misread for one another --
    a density says where the mass is, a trajectory bundle says whether that mass
    is carried by particles that stay together or by particles that cross -- so
    putting them one above the other on a shared axis is the point of the figure.

    Both rows share ONE state grid and ONE y range, computed over both archives
    together. Drawn separately each row would set its own limits and a bundle
    that looks tighter below would be telling you nothing; on a shared range,
    tighter means tighter. The colour scale is taken from the density row alone,
    since the trajectory row does not use one.

    Columns are the methods present in BOTH archives, in METHODS order, so a
    column means the same thing top to bottom. The reference posterior leads,
    as it does in the density figures.

    Parameters
    ----------
    top      : (row_label, data) — archive drawn as a density.
    bottom   : (row_label, data) — archive drawn as particle trajectories.
    coord    : int   — 0-based state index to draw.
    bins     : int   — bins along the state axis for the density row.
    smooth   : float — Gaussian smoothing in bins along the state axis; 0 leaves
                       the raw histogram.
    vmax_pct : float — percentile of the density used as the top of the colour
                       scale; everything above is clipped.
    plot_particle : int — how many particle paths to draw in the bottom row.
                       Every path is one more line to rasterise and one more
                       layer of ink; 500 is enough to read the bundle's width
                       without turning it into a solid block.
    alpha    : float — per-path opacity in the bottom row.
    density_cols : tuple — columns drawn as a DENSITY in the bottom row too,
                       rather than as trajectories. The reference is there by
                       default and for a substantive reason: SIR resamples, so
                       particle index i at one step and at the next need not be
                       the same particle, and a line joining them traces a path
                       through ancestors that no particle took. Its time
                       marginals are exactly what the reference is for; its
                       "trajectories" are an artefact of the bookkeeping. Keeping
                       it a density in both rows also leaves the control panel
                       unchanged down the figure, which is what says the two rows
                       are comparable.
    ylim     : (lo, hi) or None — None uses the shared grid, which is what keeps
                       the rows comparable. A tuple crops both rows identically.
    wspace   : float — horizontal gap between columns, as a fraction of a panel's
                       width. Shared with plot_density_rows_vs_N so the two grids
                       match; left to tight_layout they do not.
    savename : str   — optional path stub; if given, saves <stub>.pdf.
    """
    from matplotlib.colors import LinearSegmentedColormap

    k = 0
    (top_label, top_data), (bot_label, bot_data) = top, bottom
    panels_top = _panel_ensembles(top_data, coord, k)
    panels_bot = _panel_ensembles(bot_data, coord, k)

    # Column order from the top row, kept only where the bottom row has it too.
    have_bot = {lbl for lbl, _, _ in panels_bot}
    order    = [lbl for lbl, _, _ in panels_top if lbl in have_bot]
    dropped  = [lbl for lbl, _, _ in panels_top if lbl not in have_bot]
    if dropped:
        print(f'[density/traj rows] not in both archives, dropped: {dropped}')
    npanel = len(order)

    by_top = {lbl: (lbl, Xp, c) for lbl, Xp, c in panels_top}
    by_bot = {lbl: (lbl, Xp, c) for lbl, Xp, c in panels_bot}
    colors = {lbl: c for lbl, _, c in panels_top}

    t_top = top_data['t']
    t_bot = bot_data['t']
    true_top = top_data['X_true'][k, :, coord]
    true_bot = bot_data['X_true'][k, :, coord]

    # One grid over BOTH rows — see the docstring.
    edges, centers = _state_grid(
        [by_top[l] for l in order] + [by_bot[l] for l in order],
        np.concatenate([true_top, true_bot]), bins)

    dens = _time_densities([by_top[l] for l in order], edges, smooth)

    # The columns the bottom row also draws as a density need their own, from
    # the bottom archive. One colour scale spans both rows, so the reference
    # panel above and below can be compared rather than merely looked at.
    bot_dens_cols = [l for l in order if l in density_cols]
    dens_bot = dict(zip(bot_dens_cols,
                        _time_densities([by_bot[l] for l in bot_dens_cols],
                                        edges, smooth))) if bot_dens_cols else {}
    vmax = np.percentile(
        np.concatenate([D.ravel() for D in dens]
                       + [D.ravel() for D in dens_bot.values()]), vmax_pct)

    # One x range for every panel. pcolormesh sets its limits tight to the data
    # while plot adds a 5% margin either side, so the density row came out at
    # (0, 4.9) and the trajectory row at (-0.245, 5.145) -- a 10% difference that
    # inset the paths relative to the densities above them and left the reference
    # column, itself a density, disagreeing with its own row.
    xlo = min(float(t_top[0]),  float(t_bot[0]))
    xhi = max(float(t_top[-1]), float(t_bot[-1]))
    if (float(t_top[0]), float(t_top[-1])) != (float(t_bot[0]), float(t_bot[-1])):
        print(f'[density/traj rows] the two archives span different times, '
              f'{t_top[0]}..{t_top[-1]} and {t_bot[0]}..{t_bot[-1]}; both rows are '
              f'drawn on the union {xlo}..{xhi}, so a column is NOT the same '
              f'interval top and bottom.')

    fig, axes = plt.subplots(2, npanel, squeeze=False,
                             figsize=(2.9 * npanel, 3.4 * 2))
    for c, lbl in enumerate(order):
        # --- top: density ---
        ax   = axes[0][c]
        cmap = LinearSegmentedColormap.from_list('', ['white', colors[lbl]])
        ax.pcolormesh(t_top, centers, dens[c], cmap=cmap, vmin=0, vmax=vmax,
                      shading='gouraud', rasterized=True)
        ax.plot(t_top, true_top, color='k', linestyle='--', label='True state')

        # --- bottom: trajectories, or a density for the reference column ---
        ax2 = axes[1][c]
        if lbl in dens_bot:
            ax2.pcolormesh(t_bot, centers, dens_bot[lbl], cmap=cmap,
                           vmin=0, vmax=vmax, shading='gouraud', rasterized=True)
        else:
            Xp = by_bot[lbl][1]                   # (T x N)
            ax2.plot(t_bot, Xp[:, :plot_particle], color=colors[lbl],
                     alpha=alpha, rasterized=True)
        ax2.plot(t_bot, true_bot, color='k', linestyle='--', label='True state')

        for r, ax_ in enumerate((ax, ax2)):
            ax_.set_ylim(*(ylim if ylim is not None else (edges[0], edges[-1])))
            ax_.set_xlim(xlo, xhi)
            # Titles on the top row and x labels on the bottom row only: every
            # panel shares both axes, so repeating them is noise.
            if r == 0:
                ax_.set_title(lbl, fontsize=labelsize)
                ax_.get_xaxis().set_visible(False)
            elif labeling:
                ax_.set_xlabel('time', fontsize=labelsize)
            # The first column carries the state and what distinguishes the row.
            if c == 0 and labeling:
                ax_.set_ylabel(f'X({coord + 1}), {top_label if r == 0 else bot_label}',
                               fontsize=labelsize)
            else:
                ax_.get_yaxis().set_visible(False)

    fig.tight_layout()
    # tight_layout derives the column spacing from each figure's own tick-label
    # extents, so two grids built the same way drift apart -- this one came out
    # at 20% of a panel width against the density/trajectory figure's 12%. Pin it
    # instead, to the same value in both, so the two read as one set.
    fig.subplots_adjust(wspace=wspace)
    if savename is not None:
        fig.savefig(f'{savename}.pdf', bbox_inches='tight')


def _resolve_yerr(summary, Yerr, yerr_kind, y_key):
    """
    Turn the archived spread into the bar the figure should actually draw.

    The archives store sw2_std, the standard deviation of the per-simulation
    time-averaged errors -- how much ONE run varies from trajectory to
    trajectory. That is a real quantity, but it is not the uncertainty of the
    thing these figures plot: the curve is a MEAN over NUM_SIM simulations, so
    the bar belonging to it is the uncertainty of that mean, std / sqrt(NUM_SIM).

    The mismatch is not cosmetic. On the dimension sweep at n = 1 the spread
    across trajectories exceeds the mean for three methods, so mean - std goes
    NEGATIVE -- unplottable on a log axis, clipped to 1% of the mean, and drawn
    as a bar plunging far below the sampling floor. Nothing there is claiming an
    error below the floor; it is a std bar attached to a mean. Against the
    standard error, all 63 cells of that sweep sit above the floor and none is
    negative.

    'std' is kept as the default so timing panels, whose online_std is a spread
    over (simulation, step) pairs rather than over simulations, are not silently
    rescaled by the wrong n.

    Parameters
    ----------
    summary   : dict — needs NUM_SIM for the 'se' conversion.
    Yerr      : ndarray or None — the archived spread, already scaled.
    yerr_kind : 'std' — draw it as stored; 'se' — divide by sqrt(NUM_SIM).
    y_key     : str — named in the warning if NUM_SIM is missing.

    Returns
    -------
    (ndarray or None, str) — the bar to draw, and a word describing it.
    """
    if Yerr is None or yerr_kind == 'std':
        return Yerr, 'std'
    if yerr_kind != 'se':
        raise ValueError(f"yerr_kind must be 'std' or 'se'; got {yerr_kind!r}")
    if 'NUM_SIM' not in summary:
        print(f'[warn] {y_key}: yerr_kind="se" needs NUM_SIM, which this archive '
              f'does not carry — drawing the stored std instead.')
        return Yerr, 'std'
    n_sim = int(summary['NUM_SIM'])
    print(f'[note] {y_key}: error bars are the standard error, '
          f'std/sqrt({n_sim}) = std/{np.sqrt(n_sim):.2f}.')
    return Yerr / np.sqrt(n_sim), 'se'


def plot_vs_sweep(summary, x_key, xlabel, y_key, ylabel=None, yerr_key=None, show_floor=False,
                  logx=False, scale=1.0, ylim=None, legend=True, xdiv=1,
                  yerr_kind='std', floor_kind='curve', legend_kw=None,
                  exclude=(), fontsize=fontsize, labelsize=labelsize,
                  savename=None):
    """
    One per-method quantity from a sweep summary, plotted against the swept value.

    Both summary archives store their per-method quantities — sw2_mean, sw2_std,
    runtime, and (once the timing split was recorded) spinup, online and
    online_std — as (methods x sweep) matrices sharing the row order given by
    method_keys, so any of them can be plotted through this one function and the
    rows are matched by key rather than by position.

    Parameters
    ----------
    summary    : dict — sweep summary archive contents.
    x_key      : str  — archive key holding the swept values ('L_list' or 'N_list').
    xlabel     : str  — axis label for the swept quantity.
    y_key      : str  — archive key holding the plotted matrix ('sw2_mean',
                        'online', 'spinup', 'runtime').
    ylabel     : str or None — axis label for the plotted quantity. None draws
                         no y label at all, for a panel that sits in a row
                         sharing one, or where the caption carries the units.
                         The ticks are unaffected, so the panel is still
                         readable; only the word disappears.

                         Note that timing_plot_kwargs supplies its own `ylabel`
                         in the dict it returns, so a timing panel cannot simply
                         be passed ylabel=None alongside **kw -- that is a
                         duplicate keyword. Pop it instead:
                         kw.pop('ylabel', None).
    yerr_key   : str   — optional archive key for error bars ('sw2_std',
                         'online_std').
    yerr_kind  : str   — 'std' draws the stored spread; 'se' divides it by
                         sqrt(NUM_SIM) to give the uncertainty of the plotted
                         mean. See _resolve_yerr for why that is the right bar
                         for an error curve and the wrong one for a timing curve.
    show_floor : bool  — overlay sw2_floor when the archive carries one.
    floor_kind : str   — 'curve' plots the archived floor point by point;
                         'mean' draws it as one horizontal line at the average
                         with a band over its range.

                         Which is right depends on the sweep, and getting it
                         wrong misleads in opposite directions. Against N the
                         floor genuinely falls (0.1288 at N=1e3 to 0.0906 at
                         N=5e4) -- a real, monotone dependence that a flat line
                         would erase. Against DIMENSION it does not depend on n
                         at all: every 2D block of F = kron(I_n, F_block) has the
                         same posterior law, so aa_sw2 is averaging more draws
                         from one distribution and the expectation is unchanged.
                         Its measured wobble there (0.0924-0.1098, 5.3%) is the
                         estimator's own Monte-Carlo scatter -- each dimension
                         reseeds with randint+n and draws its own trajectories
                         and reference -- and plotting it as a curve invites
                         reading a trend into noise. Three independent runs at
                         n=5 put the same quantity at 0.1125, 0.1030 and 0.0956,
                         a 6.7% spread with no dimension involved at all.
    logx       : bool  — log-scale the swept axis (ensemble sizes span two decades).
    scale      : float — multiplies the plotted matrix; used to turn the stored
                         total runtimes into per-step times via per_step_factor.
    ylim       : tuple — optional (low, high) bound. Autoscale keeps every point
                         visible; set this to crop an outlier that would
                         otherwise squash the rest of the curves.
    legend     : bool  — draw the legend. Set False for panels that sit beside
                         another sharing the same curves, so the legend is not
                         repeated across a multi-panel figure.
    exclude    : tuple — method keys to leave out of this panel. Filters the rows
                         before anything is drawn or scaled, so the axis is fitted
                         to what remains rather than to curves that are not there.
    legend_kw  : dict  — overrides merged into the plt.legend call, e.g.
                         dict(loc='lower right', fontsize=13, ncol=2). The
                         default box is two columns at fontsize-4 wherever
                         matplotlib finds room; a panel whose curves leave one
                         specific gap free is better served by naming it.
    xdiv       : float — divides the tick *labels* only, leaving the tick
                         positions at their true values so the spacing is
                         unchanged; use it to write large ensemble sizes in
                         thousands, with the factor stated in xlabel.
    savename   : str   — optional path stub; if given, saves <stub>.pdf.
    """
    x          = summary[x_key]
    rows, keys = _rows_in_style(summary, y_key)
    Y          = summary[y_key][rows] * scale
    Yerr       = summary[yerr_key][rows] * scale if yerr_key is not None else None

    if exclude:
        keep    = [i for i, k in enumerate(keys) if k not in exclude]
        dropped = [k for k in keys if k in exclude]
        if dropped:
            print(f'[note] {y_key}: excluded from this panel: {dropped}')
        keys = [keys[i] for i in keep]
        Y    = Y[keep]
        if Yerr is not None:
            Yerr = Yerr[keep]
    Yerr, _ = _resolve_yerr(summary, Yerr, yerr_kind, y_key)

    # A log axis silently drops non-positive and non-finite values, which reads
    # as a point missing from the figure rather than as an error. Name them.
    bad = ~(np.isfinite(Y) & (Y > 0))
    for i_m, i_x in zip(*np.nonzero(bad)):
        print(f'[warn] {y_key} for {keys[i_m]} at {x_key}={x[i_x]} is {Y[i_m, i_x]!r} '
              f'— not plottable on a log axis, the point will be missing.')

    # An error bar reaching y - err <= 0 has no lower end a log axis can place,
    # so matplotlib runs it off the bottom of the figure. Clip the lower arm to
    # LOW_FRAC of the mean: the bar then terminates inside the axes, with its
    # lower end marking a truncation rather than the true y - err.
    LOW_FRAC = 1e-2
    if Yerr is not None:
        lower = np.minimum(Yerr, Y * (1.0 - LOW_FRAC))
        for i_m, i_x in zip(*np.nonzero(lower < Yerr)):
            print(f'[warn] {y_key} for {keys[i_m]} at {x_key}={x[i_x]}: '
                  f'std {Yerr[i_m, i_x]:.3g} exceeds mean {Y[i_m, i_x]:.3g}; the lower '
                  f'error bar is truncated at {Y[i_m, i_x] * LOW_FRAC:.3g} to stay on the log axis.')

    plt.figure(figsize=(8, 6))
    for i_m, key in enumerate(keys):
        label, color = STYLE.get(key, (key, None))
        err = None if Yerr is None else np.vstack([lower[i_m], Yerr[i_m]])
        plt.errorbar(x, Y[i_m], yerr=err,
                     color=color, lw=2.5, marker='o', capsize=3, label=label)

    # An independent draw from the reference posterior — no sampler does better.
    if show_floor and 'sw2_floor' in summary:
        floor = np.asarray(summary['sw2_floor'], dtype=float)
        if floor_kind == 'mean':
            plt.axhline(floor.mean(), color='k', linestyle='--', lw=2.0,
                        label='sampling floor')
            # Line only, no band. The spread is reported below rather than drawn:
            # it is the estimator's own scatter, not a property of the sweep, and
            # a shaded band reads as a quantity that varies with the x axis --
            # which is exactly the impression drawing it as a constant is meant
            # to avoid.
            print(f'[note] sampling floor drawn as a constant at {floor.mean():.4f} '
                  f'(range {floor.min():.4f}-{floor.max():.4f}, '
                  f'{100 * floor.std() / floor.mean():.1f}% scatter).')
        elif floor_kind == 'curve':
            plt.plot(x, floor, color='k', linestyle='--', lw=2.0,
                     label='sampling floor')
        else:
            raise ValueError(f"floor_kind must be 'curve' or 'mean'; "
                             f"got {floor_kind!r}")

    if labeling: plt.xlabel(xlabel, fontsize=labelsize)
    if labeling and ylabel is not None: plt.ylabel(ylabel, fontsize=labelsize)
    if logx:
        # F_list starts at 0 since the no-adaptation level was added, and a log
        # axis silently drops non-positive values -- the point would simply be
        # absent, indistinguishable from one that was never run. symlog keeps the
        # decade spacing the positive budgets need and admits 0 on a linear
        # segment below the smallest of them.
        xa = np.asarray(x, dtype=float)
        if np.any(xa <= 0):
            plt.xscale('symlog', linthresh=float(xa[xa > 0].min()))
            print(f'[note] {x_key} contains {int((xa <= 0).sum())} non-positive '
                  f'value(s); using a symlog x axis so they stay on the figure.')
        else:
            plt.xscale('log')
    plt.yscale('log')
    if ylim is not None:
        plt.ylim(*ylim)
    # Positions stay at the true x values; only the labels are rescaled.
    plt.xticks(x, [f'{v / xdiv:g}' for v in x])
    # The sampling floor leads the legend. It is the reference every curve is
    # read against -- the error an independent draw from the reference posterior
    # attains, which no sampler beats -- so it belongs at the top of the key
    # rather than trailing eight methods. It is drawn last so the curves sit over
    # it, hence the reorder here rather than a reordered plot call.
    handles, labels = plt.gca().get_legend_handles_labels()
    order = ([i for i, l in enumerate(labels) if l == 'sampling floor']
             + [i for i, l in enumerate(labels) if l != 'sampling floor'])
    handles = [handles[i] for i in order]
    labels  = [labels[i]  for i in order]
    if legend:
        kw = dict(fontsize=fontsize, ncol=2)
        kw.update(legend_kw or {})
        plt.legend(handles, labels, **kw)
    plt.tight_layout()
    if savename is not None:
        # A panel with its y label suppressed is a DIFFERENT deliverable from the
        # labelled one, not a replacement, so it gets its own name rather than
        # overwriting. Without this the two variants collide on one filename and
        # whichever ran last silently wins.
        if ylabel is None:
            savename = f'{savename}_nolabel'
        plt.savefig(f'{savename}.pdf', bbox_inches='tight')


def plot_error_vs_time(summary, x_key, point_label, xlabel='online time per step (s)',
                       y_key='sw2_mean', ylabel='aa-SW2', time_key='online',
                       yerr_key='sw2_std', yerr_kind='std', annotate=True, show_floor=False,
                       mark_zero=True, xlim=None, ylim=None, legend=True,
                       legend_out=False, legend_refs_only=False, exclude=(),
                       info_box=False, yticks=None, ydiv=1.0,
                       fontsize=fontsize, labelsize=labelsize,
                       savename=None):
    """
    Error against compute cost, as one parametric curve per method.

    plot_vs_sweep draws every method against a single shared x vector -- the
    swept values, which are the same number for all of them. Compute is not like
    that: at a given budget each method pays its own price, so the x coordinate
    is per method as well as per sweep point. This function therefore walks each
    method's row of the timing matrix against its row of the error matrix,
    giving eight curves whose points are the sweep values rather than eight
    curves sharing an axis.

    That is the figure the study is actually after. The budgets are not
    compute-matched across methods -- an OTF iteration and an FMF iteration cost
    different amounts -- so a shared iteration axis compares unlike things,
    while a time axis puts every method on the one currency a deployed filter
    spends. Reading it: down is more accurate, left is cheaper, so the lower-left
    frontier is what a practitioner would pick from.

    Nothing is divided here. `online` is stored as nanmean(steps[:, 1:]), the
    mean cost of ONE analysis step after the first, so it is already a per-step
    rate. Pass time_key='runtime' with a per_step_factor scale if the blended
    total-over-steps figure is wanted instead -- that is a different quantity,
    since it folds the one-off spin-up into the per-step number.

    A curve is free to double back on itself. The drivers dispatch NUM_GPU
    filters at once, so a recorded time includes contention with whichever two
    methods shared its batch, and a point can come out cheaper than a smaller
    budget. That is measurement noise in x, not a non-monotonicity in the
    method, and it is left visible rather than sorted away. Two such inversions
    were chased down: KRF's proved to be contention and its row was re-measured
    in isolation, while SBF's fi=4 and fi=16 cells do equal gradient work
    (32x768 and 128x192 sample-iterations) and need no explaining.

    Two method groups are drawn differently, because they mean different things:

      * The methods in UNTRAINED read no budget, so their sweep points are
        repeats of one experiment. They get ONE marker at their mean, not a
        curve -- a line through four repeats would draw a budget dependence that
        does not exist.
      * The zero-budget point, where the sweep has one, is ringed. It is not a
        smaller budget of the same kind: nothing trains there at any analysis
        step, so it is the no-adaptation baseline the rest of each curve is
        measured against, and it is the left end of every curve.

    Parameters
    ----------
    summary     : dict — sweep summary archive contents.
    x_key       : str  — archive key holding the swept values, used only to
                         label the markers ('F_list', 'N_list', 'L_list').
    point_label : str  — what one swept value is called, for the annotations
                         and the legend note (e.g. 'FI').
    xlabel      : str  — axis label for the time axis.
    y_key       : str  — archive key holding the error matrix.
    ylabel      : str or None — axis label for the error. None draws no y label
                         at all, for a panel that sits in a row sharing one, or
                         where the caption carries the units; the ticks are
                         unaffected. Unlike plot_vs_sweep this KEEPS a default,
                         so an existing call that never mentioned it is not
                         silently stripped of its label. Suppressing it adds
                         '_nolabel' to the saved name, since the two variants are
                         different deliverables and would otherwise collide.
    time_key    : str  — archive key holding the timing matrix ('online',
                         'spinup', 'runtime').
    yerr_key    : str  — optional archive key for vertical error bars.
    yerr_kind   : str  — 'std' or 'se'; see _resolve_yerr.
    annotate    : bool — write the swept value beside each marker, so the
                         direction of travel along a curve is readable. Skipped
                         for UNTRAINED methods, whose single marker has no one
                         swept value to name.
    mark_zero   : bool — ring the zero-budget point and name it in the legend.
                         Silently inert on sweeps whose axis has no zero, so
                         the dimension and ensemble-size sweeps are unaffected.
    show_floor  : bool — draw the sampling floor as a horizontal band. It has no
                         compute cost, so it cannot be a curve here: it is drawn
                         at its mean level across the sweep, with the spread as
                         the band.
    xlim        : tuple — optional (low, high) bound on the time axis. EnKF and
                         SIR train nothing and sit three decades left of every
                         learned filter, so on a shared axis they compress the
                         region where the trade-off actually happens; cropping
                         to the learned range is what makes that region legible.
    ylim        : tuple — optional (low, high) bound on the error axis.
    legend      : bool — draw the legend.
    legend_out  : bool — place the legend outside the axes, to the right. Nine
                         methods make a bulky box, and once the axes are cropped
                         to the learned filters there is no empty corner left for
                         it to sit in without covering a curve.
    yticks      : sequence — y positions to label, in DATA units. A log axis
                         spanning under a decade otherwise labels its minor ticks
                         too, which is where '3 x 10^-1, 4 x 10^-1, 6 x 10^-1'
                         comes from; naming the three worth having is shorter and
                         no less precise.
    ydiv        : float — divides the tick LABELS only, leaving the positions at
                         their true values so nothing moves. Use it to lift a
                         common factor out of the ticks and into the axis label,
                         as xdiv does for plot_vs_sweep's horizontal axis: pass
                         ydiv=0.1 and label the axis '(x 10^-1)'.
    info_box    : bool  — write the numbers this panel does NOT draw into a box
                         in the lower-left corner: the sampling floor when it is
                         suppressed, and the mean error and cost of every
                         excluded method. Cropping a curve for legibility is only
                         honest if what was cropped stays legible somewhere, and
                         a reader comparing the learned filters still needs to
                         know where the untrained baselines sit.
    exclude     : tuple — method keys to leave out of the panel entirely. Unlike
                         xlim, which crops a curve out of VIEW while it still
                         sets the autoscale, this drops the rows before anything
                         is drawn or fitted -- so a panel meant for the learned
                         filters can be scaled to them rather than to EnKF and
                         SIR sitting three decades away.
    legend_refs_only : bool — keep only the two reference entries, the sampling
                         floor and the zero-budget ring, and drop the eight method
                         entries. The method colours are keyed by the legend on
                         the dimension figure, which ALL_LEGENDS=False makes the
                         one panel carrying it for the whole set, so repeating
                         them here buys nothing and costs the corner of the axes
                         a ten-entry box occupies. What is NOT redundant is the
                         meaning of the two reference marks, which appear on no
                         other panel.
    savename    : str  — optional path stub; if given, saves <stub>.pdf.
    """
    if time_key not in summary:
        print(f'[skip] plot_error_vs_time: archive has no {time_key!r}')
        return

    sweep      = summary[x_key]
    rows, keys = _rows_in_style(summary, y_key)
    X          = summary[time_key][rows]
    Y          = summary[y_key][rows]
    Yerr       = (summary[yerr_key][rows]
                  if yerr_key is not None and yerr_key in summary else None)

    # Recorded before the rows go, so info_box can still report them. Averaged
    # over the sweep because these methods read no budget: their points are
    # repeats of one experiment, not a curve.
    excluded_stats = [(STYLE.get(k, (k, None))[0],
                       float(np.nanmean(Y[i])), float(np.nanmean(X[i])))
                      for i, k in enumerate(keys) if k in exclude]

    if exclude:
        # `keep`, not `rows`: rows already indexes the archive, and reusing the
        # name here would silently make the second filter index the first one's
        # output with the first one's positions.
        keep    = [i for i, k in enumerate(keys) if k not in exclude]
        dropped = [k for k in keys if k in exclude]
        print(f'[note] {y_key}: excluded from this panel: {dropped}')
        keys = [keys[i] for i in keep]
        X, Y = X[keep], Y[keep]
        if Yerr is not None:
            Yerr = Yerr[keep]

    Yerr, _ = _resolve_yerr(summary, Yerr, yerr_kind, y_key)

    # Both axes are logarithmic, which silently drops non-positive and
    # non-finite values -- indistinguishable from a point that was never there.
    bad = ~(np.isfinite(X) & (X > 0) & np.isfinite(Y) & (Y > 0))
    for i_m, i_x in zip(*np.nonzero(bad)):
        print(f'[warn] {keys[i_m]} at {point_label}={sweep[i_x]}: '
              f'{time_key}={X[i_m, i_x]!r}, {y_key}={Y[i_m, i_x]!r} '
              f'— not plottable on log axes, the point will be missing.')

    # Same truncation rule plot_vs_sweep uses: an error bar reaching y - err <= 0
    # has no lower end a log axis can place, so clip it to LOW_FRAC of the mean.
    LOW_FRAC = 1e-2
    lower = None if Yerr is None else np.minimum(Yerr, Y * (1.0 - LOW_FRAC))

    # Column of the zero-budget level, if this sweep has one.
    zero = np.nonzero(np.asarray(sweep, dtype=float) == 0)[0]
    zero = int(zero[0]) if (mark_zero and zero.size) else None

    # Same 8 x 6 as the plot_vs_sweep panels, so the figures in a set share a
    # size. This was widened to 10 inches when the legend carried ten entries in
    # two columns and could not fit beside the data; with legend_refs_only that
    # box is a single line and the extra width bought nothing but an odd figure
    # out in the set.
    plt.figure(figsize=(8, 6))
    for i_m, key in enumerate(keys):
        label, color = STYLE.get(key, (key, None))
        err = None if Yerr is None else np.vstack([lower[i_m], Yerr[i_m]])

        if key in UNTRAINED:
            # One marker at the mean, with the mean error bar. A diamond so it
            # reads as a fixed reference rather than a point on someone's curve.
            e = None if Yerr is None else np.array([[np.nanmean(lower[i_m])],
                                                    [np.nanmean(Yerr[i_m])]])
            plt.errorbar(np.nanmean(X[i_m]), np.nanmean(Y[i_m]), yerr=e,
                         color=color, marker='D', markersize=9, capsize=3,
                         linestyle='none', label=label)
            continue

        plt.errorbar(X[i_m], Y[i_m], yerr=err,
                     color=color, lw=2.5, marker='o', capsize=3, label=label)
        if zero is not None and np.isfinite(X[i_m, zero]) and np.isfinite(Y[i_m, zero]):
            plt.plot(X[i_m, zero], Y[i_m, zero], color=color, marker='s',
                     markersize=12, markerfacecolor='none', markeredgewidth=2.0,
                     linestyle='none')
        if annotate:
            for i_x, v in enumerate(sweep):
                if np.isfinite(X[i_m, i_x]) and np.isfinite(Y[i_m, i_x]):
                    plt.annotate(f'{v:g}', (X[i_m, i_x], Y[i_m, i_x]),
                                 textcoords='offset points', xytext=(4, 4),
                                 fontsize=fontsize, color=color, alpha=0.8)

    # Proxy handle, so the ring is explained once instead of once per method.
    if zero is not None:
        plt.plot([], [], color='k', marker='s', markersize=12,
                 markerfacecolor='none', markeredgewidth=2.0, linestyle='none',
                 label='no online training')

    # The floor is an independent draw from the reference posterior: it is what a
    # perfect sampler attains, at no training cost, so it spans the whole axis
    # rather than sitting at one x.
    if 'sw2_floor' in summary:
        floor = np.asarray(summary['sw2_floor'], dtype=float)
        if show_floor:
            plt.axhline(floor.mean(), color='k', linestyle='--', lw=2.0,
                        label='sampling floor')
            if floor.size > 1:
                plt.axhspan(floor.min(), floor.max(), color='k', alpha=0.08, lw=0)
        else:
            # Not drawn, but not silently dropped either. On this panel the floor
            # sits far below every method -- the whole span is only ~1.1 decades
            # and the floor would spend 40% of the height on empty gap -- so the
            # axis is cropped to the methods and the number is reported instead.
            # It still belongs in the caption: it is what the errors are read
            # against, and the ratio below is the honest summary of the gap.
            # np.ptp(a), not a.ptp(): the method was removed in NumPy 2.0.
            rng = (f' (range {floor.min():.4f}-{floor.max():.4f})'
                   if floor.size > 1 and np.ptp(floor) > 0 else '')
            best = float(np.nanmin(Y))
            print(f'[note] sampling floor {floor.mean():.4f}{rng} — NOT drawn on '
                  f'this panel; the best method sits at {best:.4f}, '
                  f'{best / floor.mean():.1f}x above it.')

    # What the panel leaves out, stated in the corner rather than lost.
    if info_box:
        lines = []
        if 'sw2_floor' in summary:
            lines.append(f'sampling floor   {float(np.asarray(summary["sw2_floor"]).mean()):.4f}')
        for name, y_mean, x_mean in excluded_stats:
            lines.append(f'{name:<7} {y_mean:>7.4f}  @ {x_mean * 1e3:.2f} ms/step')
        if lines:
            plt.gca().text(0.02, 0.02, '\n'.join(lines),
                           transform=plt.gca().transAxes,
                           ha='left', va='bottom',
                           # One size for every element on the figure, so the
                           # box and the key read as the same typographic level.
                           fontsize=fontsize, family='monospace',
                           bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                                     edgecolor='0.6', alpha=0.95))

    if labeling: plt.xlabel(xlabel, fontsize=labelsize)
    if labeling and ylabel is not None: plt.ylabel(ylabel, fontsize=labelsize)
    plt.xscale('log')
    plt.yscale('log')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    else:
        # Fitted to what is actually drawn rather than left to autoscale, which
        # pads a log axis to round decades and here wasted most of the panel on
        # empty space above SIR and below the floor. The span is only ~1.1
        # decades, so a tight fit is what makes the crowded band between 0.26 and
        # 0.5 -- where six of the eight methods live -- legible.
        #
        # The bottom takes the floor into account: it is the reference the whole
        # figure is read against, so cropping it off would be worse than a little
        # empty space. The top leaves room for the legend when one sits inside.
        # Fitted to the points the panel actually SHOWS. With an xlim in force a
        # curve can leave the view while still dragging the y range to cover it,
        # which is how a zoom ends up mostly empty; restricting the fit to the
        # visible x window is what makes the crop do what it looks like it does.
        vis = np.ones_like(Y, dtype=bool)
        if xlim is not None:
            vis &= (X >= xlim[0]) & (X <= xlim[1])
        if not vis.any():
            vis[:] = True
        span_lo = np.nanmin(np.where(vis, Y - (0 if Yerr is None else lower), np.inf))
        span_hi = np.nanmax(np.where(vis, Y + (0 if Yerr is None else Yerr), -np.inf))
        if show_floor and 'sw2_floor' in summary:
            span_lo = min(span_lo, float(np.asarray(summary['sw2_floor']).min()))
        head = 1.10 if (legend and not legend_out and legend_refs_only) else \
               (2.0 if (legend and not legend_out) else 1.05)
        plt.ylim(span_lo * 0.92, span_hi * head)

    # After the scale and the limits, not before: plt.yscale('log') installs its
    # own locator and formatter and would undo an earlier plt.yticks call. The
    # minor formatter is silenced separately, or matplotlib keeps labelling the
    # decade's minor ticks underneath the three named here.
    if yticks is not None:
        from matplotlib.ticker import NullFormatter
        plt.yticks(list(yticks), [f'{v / ydiv:g}' for v in yticks])
        plt.gca().yaxis.set_minor_formatter(NullFormatter())

    # The two reference entries lead the legend, floor first, then the methods.
    # Both say what a curve is read against -- the floor is the error an
    # independent draw from the reference posterior attains, the ring marks
    # where no training happened at all -- so they belong at the top of the key
    # rather than trailing eight methods. Both are drawn after the curves so the
    # curves sit over them, hence the reorder here rather than reordered plot
    # calls.
    # legend_refs_only keeps just the ring. The floor needs no key: it is the only
    # dashed black horizontal on the panel and the axis makes its level readable,
    # whereas the ring marks specific points and would be unexplained without one.
    lead = ['no online training'] if legend_refs_only else \
           ['sampling floor', 'no online training']
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [i for l0 in lead for i, l in enumerate(labels) if l == l0]
    if not legend_refs_only:
        order += [i for i, l in enumerate(labels) if l not in lead]
    handles = [handles[i] for i in order]
    labels  = [labels[i]  for i in order]
    if legend:
        kw = (dict(loc='upper left', bbox_to_anchor=(1.02, 1.0), ncol=1)
              if legend_out else dict(loc='upper right',
                                      ncol=1 if legend_refs_only else 2))
        # The title names what a marker along a curve stands for, which is only
        # true when the points are labelled. With annotations off it is a stale
        # instruction and it costs the legend a whole row it does not need.
        title = dict(title=f'marker = {point_label}',
                     title_fontsize=fontsize) if annotate else {}
        plt.legend(handles, labels, fontsize=fontsize, **title, **kw)
    plt.tight_layout()
    if savename is not None:
        # Same rule as plot_vs_sweep: a label-suppressed panel gets its own name
        # rather than overwriting the labelled one.
        if ylabel is None:
            savename = f'{savename}_nolabel'
        plt.savefig(f'{savename}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    n = 5   # dimension of the Quadratic.py run to load

    # Dynamic benchmark: trajectories and runtimes.
    # Written by Quadratic.py at FI = 4, warm-started from the tuned checkpoints
    # (T = 20, N_true = 2000 — the protocol steps 4, 5 and 6 all ran on).
    dyn = load('DATA_file_Quadratic_FI_4_n_5_NUM_SIM_1_randint_390_N_10000.npz')
    if dyn is not None:
        report_runtimes(dyn)
        plot_trajectories(dyn, savename=f'{FIG_DIR}/Quadratic_n_{n}_from_data')
        # Three rows: X(2) and X(6) as densities over time, then the X(6)
        # particles at t = 2.7 opened up as a histogram, with the marker
        # on row 2 naming the slice row 3 came from. X(2) and X(6) are both
        # unobserved -- h sees the odd-numbered states -- which is where the
        # posterior is bimodal and the methods actually differ.
        plot_density_slice_rows(dyn, coord_top=1, coord_bot=5, t_index=27,
                          savename=f'{FIG_DIR}/Quadratic_n_{n}_density_slice_from_data')

    # ------------------------------------------------------------------
    # Step 7 — the transfer test on Lorenz-63
    # ------------------------------------------------------------------
    # The same density panels as the quadratic figure above, at two ensemble
    # sizes, for the first state only. Both archives come from one L63.py
    # configuration (FI = 4, seed 390) with nothing changed but N, so reading a
    # column top to bottom isolates the effect of the ensemble size.
    #
    # Coordinate 0 rather than an unobserved one: L63 observes x3 alone, so x1
    # is unobserved and is where the chaotic spread is widest.
    L63_N = (100, 1000)
    l63   = [(n_label(N_),
              load(f'DATA_file_L63_FI_4_NUM_SIM_1_randint_390_N_{N_}.npz'))
             for N_ in L63_N]
    l63   = [(label, d) for label, d in l63 if d is not None]
    if l63:
        plot_density_rows_vs_N(l63, coord=0,
                               savename=f'{FIG_DIR}/L63_density_rows_vs_N_from_data')

        # The same top row as the figure above, with the bottom row switched from
        # a density to the particle paths themselves at the larger ensemble. The
        # density says where the mass is; the trajectories say whether it is
        # carried by particles that stay together or by particles that cross.
        plot_density_traj_rows(l63[0], l63[-1], coord=0,
                               savename=f'{FIG_DIR}/L63_density_traj_rows_from_data')


    # ------------------------------------------------------------------
    # Step 7b — Lorenz-63, error and cost against ensemble size
    # ------------------------------------------------------------------
    # The L63 counterpart of the quadratic ensemble-size sweep above, from
    # L63_vs_particles.py. Same helpers, same axes; the only differences are the
    # archive it reads and the tick divisor, since this sweep starts two decades
    # lower (N = 100 rather than 1000) and reads better in hundreds.
    l63p = load('DATA_file_L63_vs_particles_fi4.npz')
    if l63p is not None:
        # Cropped at 5000. KRF diverged at N = 10000 -- its training loss went
        # non-finite at simulation 6, step 26, and the filter returned NaN from
        # there on -- so that column has no KRF point to draw. Rather than show
        # a panel with one method silently absent at its right-hand end, the
        # sweep stops where every method has a value. The abort is still in the
        # archive and in logs/step8_L63_vs_particles.log.
        l63p = crop_sweep(l63p, 'N_list', 5000)
        report_completeness(l63p, 'N_list', 'ensemble size N')
        report_offline(l63p, 'N_list', 'ensemble size N')
        report_failures(l63p, 'N_list', 'ensemble size N', 'L63_vs_particles')
        # The floor is drawn as a CURVE here, not the constant the dimension
        # figure uses: against N it genuinely falls, and steeply -- 1.52 at
        # N = 100 to 0.39 at N = 10000 -- because more particles resolve the
        # posterior better. Averaging that away would erase a real dependence.
        # No legend_kw: the default box -- two columns at fontsize-4, placed by
        # matplotlib -- is what the quadratic ensemble-size panel uses, and these
        # two are read as a pair, so they carry the same key at the same size in
        # the same 8 x 6 frame. The legend follows WITH_LEGENDS like every other
        # panel.
        plot_vs_sweep(l63p, 'N_list', 'ensemble size N (×100)',
                      'sw2_mean', ylabel='aa-SW2', yerr_key='sw2_std', yerr_kind='se',
                      show_floor=True, logx=True, xdiv=100, legend=ALL_LEGENDS,
                      savename=f'{FIG_DIR}/L63_sw2_vs_particles_from_data')
        plot_vs_sweep(l63p, 'N_list', 'ensemble size N (×100)',
                        'sw2_mean', yerr_key='sw2_std', yerr_kind='se',
                        show_floor=True, logx=True, xdiv=100, legend=ALL_LEGENDS,
                        savename=f'{FIG_DIR}/L63_sw2_vs_particles_from_data')
        kw = timing_plot_kwargs(l63p, 'L63_vs_particles', 'online')
        if kw is not None:
            plot_vs_sweep(l63p, 'N_list', 'ensemble size N (×100)',
                          logx=True, legend=ALL_LEGENDS, xdiv=100,
                          savename=f'{FIG_DIR}/L63_online_time_vs_particles_from_data', **kw)
        kw = timing_plot_kwargs(l63p, 'L63_vs_particles', 'offline')
        if kw is not None:
            plot_vs_sweep(l63p, 'N_list', 'ensemble size N (×100)',
                          logx=True, legend=ALL_LEGENDS, xdiv=100,
                          savename=f'{FIG_DIR}/L63_offline_time_vs_particles_from_data', **kw)

    # Dimension sweep: aa-SW2 and online time against the state dimension. The
    # offline spin-up is printed rather than plotted -- it is paid once, so it
    # does not belong on a per-step curve, but it is still part of the cost.
    dim = load('DATA_file_Quadratic_vs_dim_fi4.npz')
    if dim is not None:
        report_completeness(dim, 'L_list', 'state dimension n')
        report_offline(dim, 'L_list', 'state dimension n')
        report_failures(dim, 'L_list', 'state dimension n', 'vs_dim')
        # Legend drawn unconditionally: when ALL_LEGENDS is False this is the one
        # figure that carries it for the whole set.
        # The floor sits near 0.10 and the lowest method curve near 0.22, so the
        # lower-right of the panel is empty across the whole sweep -- the one
        # place a nine-entry box fits without covering a curve or the floor.
        # Smaller than the default to stay inside that band.
        # floor_kind='mean': the floor does not depend on the dimension, so it is
        # drawn as one line at the average with a band over the measured spread,
        # rather than as a curve whose wobble is only the estimator's own noise.
        plot_vs_sweep(dim, 'L_list', 'state dimension n',
                        'sw2_mean', yerr_key='sw2_std', yerr_kind='se', show_floor=True,
                        floor_kind='mean', legend=True,
                        legend_kw=dict(loc='lower right', ncol=2,
                                        fontsize=fontsize-2, labelspacing=0.35,
                                        columnspacing=1.1, handlelength=1.8,
                                        borderpad=0.45, framealpha=1.0,
                                        # Lifted well off the axis floor: the band
                                        # between the sampling floor near 0.10 and
                                        # the lowest method curve near 0.22 is
                                        # empty across the whole sweep, and this
                                        # pad centres the box in it rather than
                                        # letting the floor line graze its edge.
                                        borderaxespad=1.5),
                        savename=f'{FIG_DIR}/Quadratic_sw2_vs_dim_from_data')
        plot_vs_sweep(dim, 'L_list', 'state dimension n',
                        'sw2_mean', ylabel='aa-SW2', yerr_key='sw2_std', yerr_kind='se', show_floor=True,
                        floor_kind='mean', legend=True,
                        legend_kw=dict(loc='lower right', ncol=2,
                                        fontsize=fontsize-2, labelspacing=0.35,
                                        columnspacing=1.1, handlelength=1.8,
                                        borderpad=0.45, framealpha=1.0,
                                        # Lifted well off the axis floor: the band
                                        # between the sampling floor near 0.10 and
                                        # the lowest method curve near 0.22 is
                                        # empty across the whole sweep, and this
                                        # pad centres the box in it rather than
                                        # letting the floor line graze its edge.
                                        borderaxespad=1.5),
                        savename=f'{FIG_DIR}/Quadratic_sw2_vs_dim_from_data')
        # Online and offline are drawn on their own axes: one is seconds per
        # step, the other seconds paid once, so a shared axis would be a
        # category error rather than a compact figure.
        kw = timing_plot_kwargs(dim, 'vs_dim', 'online')
        if kw is not None:
            plot_vs_sweep(dim, 'L_list', 'state dimension n', legend=ALL_LEGENDS,
                          savename=f'{FIG_DIR}/Quadratic_online_time_vs_dim_from_data', **kw)
        kw = timing_plot_kwargs(dim, 'vs_dim', 'offline')
        if kw is not None:
            plot_vs_sweep(dim, 'L_list', 'state dimension n', legend=ALL_LEGENDS,
                          savename=f'{FIG_DIR}/Quadratic_offline_time_vs_dim_from_data', **kw)

    # Ensemble-size sweep: aa-SW2 and online time against N.
    par = load('DATA_file_Quadratic_vs_particles_fi4.npz')
    if par is not None:
        report_completeness(par, 'N_list', 'ensemble size N')
        report_offline(par, 'N_list', 'ensemble size N')
        report_failures(par, 'N_list', 'ensemble size N', 'vs_particles')
        plot_vs_sweep(par, 'N_list', 'ensemble size N (×1000)',
                      'sw2_mean', ylabel='aa-SW2', yerr_key='sw2_std', yerr_kind='se', show_floor=True,
                      logx=True, xdiv=1000, legend=ALL_LEGENDS,
                      savename=f'{FIG_DIR}/Quadratic_sw2_vs_particles_from_data')
        plot_vs_sweep(par, 'N_list', 'ensemble size N (×1000)',
                        'sw2_mean', yerr_key='sw2_std', yerr_kind='se', show_floor=True,
                        logx=True, xdiv=1000, legend=ALL_LEGENDS,
                        savename=f'{FIG_DIR}/Quadratic_sw2_vs_particles_from_data')
        # The cropped companion to the panel above is no longer produced. It
        # existed for a run whose low-N points diverged far enough to force the
        # axis out to 1e6, flattening every curve into the bottom decade; this
        # 20-simulation set has no such points -- report_failures reports none --
        # so the full-range panel is already readable and the crop added nothing.
        kw = timing_plot_kwargs(par, 'vs_particles', 'online')
        if kw is not None:
            plot_vs_sweep(par, 'N_list', 'ensemble size N (×1000)',
                          logx=True, legend=ALL_LEGENDS, xdiv=1000,
                          savename=f'{FIG_DIR}/Quadratic_online_time_vs_particles_from_data', **kw)
        kw = timing_plot_kwargs(par, 'vs_particles', 'offline')
        if kw is not None:
            plot_vs_sweep(par, 'N_list', 'ensemble size N (×1000)',
                          logx=True, legend=ALL_LEGENDS, xdiv=1000,
                          savename=f'{FIG_DIR}/Quadratic_offline_time_vs_particles_from_data', **kw)

    # Refinement sweep: aa-SW2 and online time against Final_Number_ITERATION,
    # the training budget every analysis step after the first is given. Same
    # helpers as the two sweeps above -- only the axis key differs.
    # The same 20-simulation set at T = 20 as steps 5 and 6 above, so the three
    # sweeps are mutually comparable, and a horizon change would move the error.
    #
    # Its budget axis now starts at fi = 0, the no-adaptation level: stage 1's
    # configuration with Final_Number_ITERATION zeroed, so every filter runs on
    # its warm-started weights and trains at no step. It anchors the left end of
    # both the budget and the compute axes, and is the point the rest of each
    # curve is worth measuring against. It is also why plot_vs_sweep falls back
    # to a symlog x axis here: a log axis would drop it.
    fin = load('DATA_file_Quadratic_vs_final_iter.npz')
    if fin is not None:
        report_completeness(fin, 'F_list', 'Final_Number_ITERATION')
        report_offline(fin, 'F_list', 'Final_Number_ITERATION')
        report_failures(fin, 'F_list', 'Final_Number_ITERATION', 'vs_final_iter')
        plot_vs_sweep(fin, 'F_list', 'Final_Number_ITERATION',
                      'sw2_mean', ylabel='aa-SW2', yerr_key='sw2_std', yerr_kind='se', show_floor=True,
                      logx=True, legend=ALL_LEGENDS,
                      savename=f'{FIG_DIR}/Quadratic_sw2_vs_final_iter_from_data')
        # Same curves cropped to the band the methods actually separate in, so a
        # single diverged point cannot flatten every other curve into the bottom
        # decade. report_failures above names whatever leaves the axis.
        plot_vs_sweep(fin, 'F_list', 'Final_Number_ITERATION',
                      'sw2_mean', ylabel='aa-SW2', yerr_key='sw2_std', yerr_kind='se', show_floor=True,
                      logx=True, ylim=(1e-1, 1e2), legend=ALL_LEGENDS,
                      savename=f'{FIG_DIR}/Quadratic_sw2_vs_final_iter_from_data_zoom')
        kw = timing_plot_kwargs(fin, 'vs_final_iter', 'online')
        if kw is not None:
            plot_vs_sweep(fin, 'F_list', 'Final_Number_ITERATION',
                          logx=True, legend=ALL_LEGENDS,
                          savename=f'{FIG_DIR}/Quadratic_online_time_vs_final_iter_from_data', **kw)
        kw = timing_plot_kwargs(fin, 'vs_final_iter', 'offline')
        if kw is not None:
            plot_vs_sweep(fin, 'F_list', 'Final_Number_ITERATION',
                          logx=True, legend=ALL_LEGENDS,
                          savename=f'{FIG_DIR}/Quadratic_offline_time_vs_final_iter_from_data', **kw)
        # The same budgets as the figure above, re-plotted against what they
        # COST rather than against the iteration count.
        #
        # KRF's x coordinates come from an isolated re-measurement -- one GPU,
        # nothing else running -- while every other method's are contended,
        # recorded at a time when nine filters shared three GPUs -- this
        # study's current eight plus the since-dropped third SIF rung, which
        # was still being computed then. KRF was re-measured because its
        # contended timings were not monotone in the budget (fi = 4
        # came out cheaper than fi = 1 on four times the iterations); in
        # isolation they are. The two conditions are not interchangeable and
        # isolation made KRF SLOWER, not faster, so its position on this axis is
        # conservative rather than flattering. The errors are untouched. The budgets are not
        # compute-matched across methods, so the iteration axis compares unlike
        # things; this one puts every method in seconds per analysis step. Down
        # is more accurate, left is cheaper, so the lower-left frontier is the
        # set of methods worth choosing between.
        # Legend drawn unconditionally, and annotations off. The two reference
        # entries it carries -- the zero-budget ring and the sampling floor --
        # are what make the panel readable without a caption, and the per-point
        # budget labels cannot be placed legibly in the region right of 1e-1
        # where six methods overlap. Set annotate=True to get them back.
        # show_floor=False: see the note in plot_error_vs_time. The floor is
        # printed rather than plotted here, which lets the axis crop to the band
        # the methods actually occupy instead of spending most of its height on
        # the gap down to it.
        plot_error_vs_time(fin, 'F_list', 'FI', show_floor=False, legend=True,
                           annotate=False, legend_refs_only=True, yerr_kind='se',
                           ylabel='aa-SW2',
                           savename=f'{FIG_DIR}/Quadratic_sw2_vs_online_time_from_data')
        # The same panel with no y label. NOT **kw: at this point in the driver
        # `kw` still holds the last timing_plot_kwargs result (y_key='spinup',
        # ylabel=..., scale=1.0), which would plot the spin-up time instead of
        # the error -- and plot_error_vs_time has no `scale`, so it raises first.
        # ylabel=None is the whole change; the '_nolabel' suffix is added to the
        # save name automatically, so the two panels do not collide.
        plot_error_vs_time(fin, 'F_list', 'FI', show_floor=False, legend=True,
                           annotate=False, legend_refs_only=True, yerr_kind='se',
                           ylabel=None,
                           savename=f'{FIG_DIR}/Quadratic_sw2_vs_online_time_from_data')
        # Cropped to the learned filters. EnKF and SIR train nothing, so they sit
        # near 1e-3 s/step and push every method that does train into the right
        # quarter of the full panel above; dropping those three empty decades is
        # what makes the accuracy/compute frontier readable. Their error is still
        # on the full panel, and they are not part of a training-budget trade-off
        # in any case.
        # EnKF and SIR are dropped rather than cropped. They train nothing, so
        # they sit three decades left of every learned filter and are not part of
        # a training-budget trade-off at all; excluding the rows lets both axes
        # scale to the methods that are, instead of being pinned by hand. No ylim
        # and no floor line for the same reason -- both would stretch the axis
        # back over ground nothing occupies. The floor is printed instead.
        # Neither axis is pinned. Once EnKF and SIR are excluded the only thing a
        # crop would remove is OTF's zero-budget point at ~3 ms, which is a real
        # measurement and the left end of its curve -- so both axes are left to
        # fit the learned methods as they are.
        # The ring is drawn on this panel too, so it needs its key here as well --
        # inside the axes rather than legend_out, since a single entry parked in
        # its own column outside the frame would cost more width than it uses.
        plot_error_vs_time(fin, 'F_list', 'FI', show_floor=False, legend=True,
                           legend_refs_only=True, annotate=False, yerr_kind='se',
                           exclude=UNTRAINED, info_box=True,
                           # Three named ticks with the common factor lifted into
                           # the label. The panel spans well under a decade, so
                           # the default formatter writes every one of them as
                           # 'n x 10^-1' and the exponent is repeated three times
                           # for no information.
                           yticks=(0.3, 0.4, 0.6), ydiv=0.1,
                           ylabel='aa-SW2 (×$10^{-1}$)',
                           savename=f'{FIG_DIR}/Quadratic_sw2_vs_online_time_from_data_zoom',
                           labelsize=labelsize, fontsize=fontsize)
        # The same panel with the ticks written out in full instead of factored:
        # ydiv is left at its default of 1, so the three positions are labelled
        # 0.3, 0.4 and 0.6 and the axis label carries no multiplier. Same figure,
        # two tick conventions -- hence a name of its own, or this would
        # overwrite the panel above rather than sit beside it.
        plot_error_vs_time(fin, 'F_list', 'FI', show_floor=False, legend=True,
                           legend_refs_only=True, annotate=False, yerr_kind='se',
                           exclude=UNTRAINED, info_box=True,
                           yticks=(0.3, 0.4, 0.6),
                           ylabel=None,
                           savename=f'{FIG_DIR}/Quadratic_sw2_vs_online_time_from_data_zoom',
                           labelsize=labelsize, fontsize=fontsize)

    plt.show()

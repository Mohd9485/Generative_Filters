"""
@author: Mohammad Al-Jarrah

Tuned parameters for every filter, in one self-contained file.

    from param_one_step import get_config
    params = get_config('sif_ode', L, dy)

Every value is written out in the units the filters consume, so this file
imports nothing. Each method must be run with the matching implementation named
in FILTERS; get_filter() does the import for you.

Provenance: the SMAC one-step sweep of 2026-08-26 (tuning_stage_one.py),
100 trials per method at NUM_SIM = 2, T = 2, zero failures. Incumbents are in
logs/best_one_step_<method>_sim2.json.

    method         file        aa-SW2
    sif_ode        SIF.py      0.042639
    sif_sde        SIF.py      0.043387
    fmf            FMF.py      0.044220
    krf            KRF.py      0.062629
    otf            OTF.py      0.077176
    sbf            SBF.py      0.087079

Final_Number_ITERATION was pinned as a Constant in every search space, so those
values are carried over rather than searched.
"""


# ----------------------------------------------------------------------
# Which implementation each method must run with.
# ----------------------------------------------------------------------
FILTERS = {
    'fmf':         ('FMF',  'FMF'),
    'sif_sde':     ('SIF',  'SIF'),
    'sif_ode':     ('SIF',  'SIF'),
    'otf':         ('OTF',  'OTF'),
    'krf':         ('KRF',  'KRF'),
    'sbf':         ('SBF',  'SBF'),
}

METHODS = tuple(FILTERS)


# ----------------------------------------------------------------------
# Flow Matching Filter
# ----------------------------------------------------------------------
_FMF = {
    'normalization': 'None',

    # the block-triangular flow itself
    'SOLVER': 'heun',
    'ODE_STEPS': 200,

    # network
    'NUM_NEURON': 320,
    'num_resblocks': 1,
    'COND_MODE': 'concat',
    'TIME_EMBED_DIM': 20,
    'TIME_SCALE': 100.0,
    'TIME_MAX_PERIOD': 50.0,

    # optimisation
    'LearningRate': 0.000748376559,
    'BATCH_SIZE': 576,
    'SCHEDULE': 'cosine',
    'WARMUP_FRAC': 0.02,

    # data / budget
    'RESAMPLE_DATA': True,
    'COLD_START': False,
    'ITERATION': 65536,
    'Final_Number_ITERATION': 256*4,

    # stabiliser block
    'ZERO_INIT_OUT': True,
    'USE_LAYERNORM': True,
    'OPTIMIZER': 'adamw',
    'WEIGHT_DECAY': 0.0,
    'GRAD_CLIP': 0.5,
}

# ----------------------------------------------------------------------
# Stochastic Interpolant Filter, two rungs of the ablation ladder.
# NOISE_LEVEL / DENOISER_WEIGHT / INFERENCE_MODE define the variant; each rung
# was tuned in its own right, so every searched key is listed per rung.
# ----------------------------------------------------------------------
_SIF_SDE = {
    'normalization': 'None',
    'NUM_NEURON': 576,
    'BATCH_SIZE': 704,
    'LearningRate': 0.000730013106,
    'ITERATION': 32768,
    'Final_Number_ITERATION': 256*4,
    'num_resblocks': 1,
    'ODE_STEPS': 100,
    'NOISE_LEVEL': 0.1814779544991,    # epsilon
    'INFERENCE_MODE': 'sde',
    'DENOISER_WEIGHT': 1.0,            # lambda
    'TIME_EMBED_DIM': 40,
    'TIME_SCALE': 50.0,
    'TIME_MAX_PERIOD': 50.0,
    'SCHEDULE': 'cosine',
    'WARMUP_FRAC': 0.0,
    'RESAMPLE_DATA': True,
    'COLD_START': False,
    'SOLVER': 'heun',
    'ZERO_INIT_OUT': True,
    'USE_LAYERNORM': True,
    'OPTIMIZER': 'adamw',
    'WEIGHT_DECAY': 0.01,
    'GRAD_CLIP': 1.0,
}

# epsilon > 0, lambda = 0: velocity head alone, so no denoiser loss regularises
# the trunk.
_SIF_ODE = dict(
    _SIF_SDE,
    INFERENCE_MODE='ode',
    DENOISER_WEIGHT=0.0,
    NUM_NEURON=768,
    BATCH_SIZE=576,
    NOISE_LEVEL=0.1433125539948,
    LearningRate=0.0008260079241,
    ITERATION=8192,
    Final_Number_ITERATION=256*4,
    ODE_STEPS=200,
    TIME_EMBED_DIM=30,
    TIME_SCALE=150.0,
    WEIGHT_DECAY=0.01,
    GRAD_CLIP=2.0,
    WARMUP_FRAC=0.05,
)

# ----------------------------------------------------------------------
# Optimal Transport Filter
# ----------------------------------------------------------------------
_OTF = {
    'normalization': 'None',
    'NUM_NEURON': [256, 512],          # critic, map
    'BATCH_SIZE': 576,
    'LearningRate': [0.0002421871224, 0.00034122139],   # critic, map
    'ITERATION': 2048,
    'Final_Number_ITERATION': 16*4,
    'K_in': 15,
    'num_resblocks': [1, 1],           # critic, map

    'T_RESIDUAL': True,
    'ZERO_INIT_OUT': True,

    # 'SCHEDULE' is deliberately absent, so OTF.py's own default
    # ('warm_restarts') applies, unlike FMF/SIF which pin 'cosine'.
    'OPTIMIZER': 'adamw',
    'WEIGHT_DECAY': 0.001,
    'GRAD_CLIP': 0.25,
    'WARMUP_FRAC': 0.05,
}

# ----------------------------------------------------------------------
# Knothe-Rosenblatt Filter
# ----------------------------------------------------------------------
_KRF = {
    'normalization': 'None',
    'NUM_NEURON': 768,
    'BATCH_SIZE': 768,
    'LearningRate': 3.2608338e-06,
    'ITERATION': 4096,
    'Final_Number_ITERATION': 8*4,
    'depth': 4,
    'quad_points': 4,

    'COLD_START': False,
    'SCHEDULE': 'cosine',
    'WARMUP_FRAC': 0.1,
    'GRAD_CLIP': 1.0,

    'OPTIMIZER': 'adamw',
    'WEIGHT_DECAY': 0.1,
}

# ----------------------------------------------------------------------
# Schrodinger Bridge Filter
# ----------------------------------------------------------------------
_SBF = {
    'normalization': 'None',
    'NUM_NEURON': 640,
    'TIME_EMBED_DIM': 50,
    'num_resblocks': 1,
    'SDE_STEPS': 80,
    'G_DIFFUSION': 1.0,
    'ITERATION': 1024,
    'Final_Number_ITERATION': 8*4,
    'BATCH_SIZE': 320,
    'LR': 0.0002392084557,
    'NUM_STAGES': 3,
    'EMA_DECAY': 0.99,

    # Requires the patched SBF.py, which hardcoded Adam and never read
    # WEIGHT_DECAY before; pre-patch file kept at SBF.py.bak.
    'OPTIMIZER': 'adamw',
    'WEIGHT_DECAY': 0.001,
    'GRAD_CLIP': 0.25,
}

_CONFIGS = {
    'fmf':         _FMF,
    'sif_sde':     _SIF_SDE,
    'sif_ode':     _SIF_ODE,        # epsilon>0, lambda=0, ODE inference
    'otf':         _OTF,
    'krf':         _KRF,
    'sbf':         _SBF,
}


def get_config(tune, L, dy):
    """
    Return the tuned parameter dict for one method.

    Parameters
    ----------
    tune : one of METHODS —
        'fmf' | 'sif_sde' | 'sif_ode' | 'otf' | 'krf' | 'sbf'
    L, dy : int — state and observation dimensions. These enter only through
        INPUT_DIM; every other value is dimension-independent.

    Returns
    -------
    dict — parameters for the implementation named in FILTERS[tune].
    """
    if tune not in _CONFIGS:
        raise ValueError(f'unknown method {tune!r}; available: {sorted(_CONFIGS)}')
    params = dict(_CONFIGS[tune])
    params['INPUT_DIM'] = [L, dy]
    return params


def get_filter(tune):
    """
    Import and return the filter function that goes with get_config(tune, ...).

    Requires the repository root on sys.path (the modules are imported by name,
    e.g. FMF).
    """
    import importlib
    module_name, func_name = FILTERS[tune]
    return getattr(importlib.import_module(module_name), func_name)

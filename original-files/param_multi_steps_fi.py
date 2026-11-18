"""
@author: Mohammad Al-Jarrah

Stage-2 tuned parameters, one set per (method, FINAL_ITER) pair.

    from param_multi_steps_fi import get_config
    params = get_config('sif_sde', L, dy, fi=4)

Companion to param_multi_steps.py, which serves a single configuration per
method. This file serves THREE TUNED configurations -- one for each online
budget the stage-2 sweep tuned over -- because the sweep tuned the optimiser
separately at every budget. The three differ in far more than
Final_Number_ITERATION: learning rate, batch size, weight decay, clip and
warmup were each re-searched at each budget, so taking one budget's optimiser
settings and only swapping the iteration count would not reproduce any
measured result.

A FOURTH cell, fi = 0, is served alongside them and is NOT a tuned
configuration. It is the stage-1 configuration of param_one_step.py copied
verbatim with Final_Number_ITERATION set to 0, so that a run at fi = 0 trains
nothing at any analysis step: with warm_start='require' every filter loads its
stage-1 checkpoint and is charged an online budget of zero from step 0 onward,
making the whole run pure inference under the frozen spin-up network. It is the
no-adaptation baseline of the error-against-compute curve, not a measured
optimum, and no aa-SW2 is quoted for it below because none was searched.

Because nothing trains at fi = 0, the optimiser block carried in that cell --
learning rate, batch size, weight decay, clip, warmup -- is never read: those
keys are consumed only inside the training loop that never executes. They are
written out anyway so the cell states exactly which configuration it came from.
For the same reason, stage-1 values rather than any stage-2 cell's values is a
choice about provenance, not about behaviour: at zero iterations the two are
byte-identical runs, since stage 1 and stage 2 agree on every key read outside
the training loop.

Self-contained: every value is written out in the units the filters consume,
so this file imports nothing.

Provenance of the fi = 1, 4 and 16 cells: the SMAC multi-step sweep of 2026-08-27/28
(tuning_stage_two.py) at NUM_SIM = 2, T = 20, warm-started from the
stage-1 weights in DATA/warm_start/. 100 trials per cell, 2100 trials in
total, zero failures. All 21 cells are filled.
Incumbents: logs/best_multi_steps_<method>_fi<FI>_sim2.json

fi is the RAW search value; the resolved online budget is fi x a per-method
base, which is why the same fi means very different amounts of work:

    method          base    fi=0    fi=1    fi=4   fi=16
    fmf            x256        0     256    1024    4096
    sif_ode        x256        0     256    1024    4096
    sif_sde        x256        0     256    1024    4096
    otf            x16         0      16      64     256
    krf            x8          0       8      32     128
    sbf            x8          0       8      32     128

fi = 0 resolves to a zero budget for every method, which is the one level at
which the per-method base does not matter -- and the one level at which the
methods are therefore directly comparable in iterations. They are still not
comparable in TIME: a zero-iteration step is not a zero-cost step, since every
method still integrates its ODE/SDE at each analysis step and SBF additionally
samples two full IPF trajectories per step.

Best aa-SW2 at each TUNED cell (fi = 0 is absent: it was never searched):

    method          fi=1       fi=4       fi=16
    fmf           0.358548   0.332700   0.319637
    sif_ode       0.331189   0.318155   0.271044
    sif_sde       0.340216   0.275000   0.266910
    otf           0.408612   0.349522   0.332348
    krf           0.328851   0.352541   0.353544
    sbf           0.459991   0.443238   0.411021

Every method improves monotonically with budget EXCEPT krf, whose best is at
fi = 1 and which degrades at both larger budgets -- the one method for which
more online training makes things worse.
"""


# ----------------------------------------------------------------------
# Which implementation each method must run with.
# ----------------------------------------------------------------------
FILTERS = {
    'fmf':         ('FMF', 'FMF'),
    'sif_ode':     ('SIF', 'SIF'),   # eps>0, lambda=0, ODE inference
    'sif_sde':     ('SIF', 'SIF'),
    'otf':         ('OTF', 'OTF'),
    'krf':         ('KRF', 'KRF'),
    'sbf':         ('SBF', 'SBF'),
}

METHODS     = tuple(FILTERS)
FINAL_ITERS = (0, 1, 4, 16)

# Resolved online budget = fi x this base. Recorded so a caller can report the
# absolute iteration count without re-deriving it from the tuner.
FI_BASE = {
    'fmf':         256,
    'sif_ode':     256,
    'sif_sde':     256,
    'otf':         16,
    'krf':         8,
    'sbf':         8,
}

# ----------------------------------------------------------------------
# Flow Matching Filter
# ----------------------------------------------------------------------
_FMF = {
    0  : {   # stage-1 config, online budget zeroed  |  Final_Number_ITERATION 0
        'normalization':          'None',
        'NUM_NEURON':             320,
        'num_resblocks':          1,
        'COND_MODE':              'concat',
        'TIME_EMBED_DIM':         20,
        'TIME_SCALE':             100.0,
        'TIME_MAX_PERIOD':        50.0,
        'ODE_STEPS':              200,
        'SOLVER':                 'heun',
        'LearningRate':           0.000748376559,
        'BATCH_SIZE':             576,
        'SCHEDULE':               'cosine',
        'WARMUP_FRAC':            0.02,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.0,
        'GRAD_CLIP':              0.5,
        'ITERATION':              65536,
        'Final_Number_ITERATION': 0,
        'RESAMPLE_DATA':          True,
        'COLD_START':             False,
        'ZERO_INIT_OUT':          True,
        'USE_LAYERNORM':          True,
    },

    1  : {   # aa-SW2 0.358548   |   Final_Number_ITERATION 256
        'normalization':          'None',
        'NUM_NEURON':             320,
        'num_resblocks':          1,
        'COND_MODE':              'concat',
        'TIME_EMBED_DIM':         20,
        'TIME_SCALE':             100.0,
        'TIME_MAX_PERIOD':        50.0,
        'ODE_STEPS':              200,
        'SOLVER':                 'heun',
        'LearningRate':           0.0001247337992,
        'BATCH_SIZE':             640,
        'SCHEDULE':               'cosine',
        'WARMUP_FRAC':            0.1,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.01,
        'GRAD_CLIP':              1.0,
        'ITERATION':              65536,
        'Final_Number_ITERATION': 256,
        'RESAMPLE_DATA':          True,
        'COLD_START':             False,
        'ZERO_INIT_OUT':          True,
        'USE_LAYERNORM':          True,
    },

    4  : {   # aa-SW2 0.332700   |   Final_Number_ITERATION 1024
        'normalization':          'None',
        'NUM_NEURON':             320,
        'num_resblocks':          1,
        'COND_MODE':              'concat',
        'TIME_EMBED_DIM':         20,
        'TIME_SCALE':             100.0,
        'TIME_MAX_PERIOD':        50.0,
        'ODE_STEPS':              200,
        'SOLVER':                 'heun',
        'LearningRate':           3.07514419e-05,
        'BATCH_SIZE':             576,
        'SCHEDULE':               'cosine',
        'WARMUP_FRAC':            0.05,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.01,
        'GRAD_CLIP':              0.5,
        'ITERATION':              65536,
        'Final_Number_ITERATION': 1024,
        'RESAMPLE_DATA':          True,
        'COLD_START':             False,
        'ZERO_INIT_OUT':          True,
        'USE_LAYERNORM':          True,
    },

    16 : {   # aa-SW2 0.319637   |   Final_Number_ITERATION 4096
        'normalization':          'None',
        'NUM_NEURON':             320,
        'num_resblocks':          1,
        'COND_MODE':              'concat',
        'TIME_EMBED_DIM':         20,
        'TIME_SCALE':             100.0,
        'TIME_MAX_PERIOD':        50.0,
        'ODE_STEPS':              200,
        'SOLVER':                 'heun',
        'LearningRate':           0.0007645567945,
        'BATCH_SIZE':             512,
        'SCHEDULE':               'cosine',
        'WARMUP_FRAC':            0.05,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.01,
        'GRAD_CLIP':              2.0,
        'ITERATION':              65536,
        'Final_Number_ITERATION': 4096,
        'RESAMPLE_DATA':          True,
        'COLD_START':             False,
        'ZERO_INIT_OUT':          True,
        'USE_LAYERNORM':          True,
    },

}

# ----------------------------------------------------------------------
# Stochastic Interpolant Filter -- ODE
# ----------------------------------------------------------------------
_SIF_ODE = {
    0  : {   # stage-1 config, online budget zeroed  |  Final_Number_ITERATION 0
        'normalization':          'None',
        'NUM_NEURON':             768,
        'num_resblocks':          1,
        'TIME_EMBED_DIM':         30,
        'TIME_SCALE':             150.0,
        'TIME_MAX_PERIOD':        50.0,
        'ODE_STEPS':              200,
        'SOLVER':                 'heun',
        'NOISE_LEVEL':            0.1433125539948,
        'INFERENCE_MODE':         'ode',
        'DENOISER_WEIGHT':        0.0,
        'LearningRate':           0.0008260079241,
        'BATCH_SIZE':             576,
        'SCHEDULE':               'cosine',
        'WARMUP_FRAC':            0.05,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.01,
        'GRAD_CLIP':              2.0,
        'ITERATION':              8192,
        'Final_Number_ITERATION': 0,
        'RESAMPLE_DATA':          True,
        'COLD_START':             False,
        'ZERO_INIT_OUT':          True,
        'USE_LAYERNORM':          True,
    },

    1  : {   # aa-SW2 0.331189   |   Final_Number_ITERATION 256
        'normalization':          'None',
        'NUM_NEURON':             768,
        'num_resblocks':          1,
        'TIME_EMBED_DIM':         30,
        'TIME_SCALE':             150.0,
        'TIME_MAX_PERIOD':        50.0,
        'ODE_STEPS':              200,
        'SOLVER':                 'heun',
        'NOISE_LEVEL':            0.1433125539948,
        'INFERENCE_MODE':         'ode',
        'DENOISER_WEIGHT':        0.0,
        'LearningRate':           3.7869521e-05,
        'BATCH_SIZE':             576,
        'SCHEDULE':               'cosine',
        'WARMUP_FRAC':            0.0,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.01,
        'GRAD_CLIP':              0.5,
        'ITERATION':              8192,
        'Final_Number_ITERATION': 256,
        'RESAMPLE_DATA':          True,
        'COLD_START':             False,
        'ZERO_INIT_OUT':          True,
        'USE_LAYERNORM':          True,
    },

    4  : {   # aa-SW2 0.318155   |   Final_Number_ITERATION 1024
        'normalization':          'None',
        'NUM_NEURON':             768,
        'num_resblocks':          1,
        'TIME_EMBED_DIM':         30,
        'TIME_SCALE':             150.0,
        'TIME_MAX_PERIOD':        50.0,
        'ODE_STEPS':              200,
        'SOLVER':                 'heun',
        'NOISE_LEVEL':            0.1433125539948,
        'INFERENCE_MODE':         'ode',
        'DENOISER_WEIGHT':        0.0,
        'LearningRate':           1.43796471e-05,
        'BATCH_SIZE':             512,
        'SCHEDULE':               'cosine',
        'WARMUP_FRAC':            0.02,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.001,
        'GRAD_CLIP':              2.0,
        'ITERATION':              8192,
        'Final_Number_ITERATION': 1024,
        'RESAMPLE_DATA':          True,
        'COLD_START':             False,
        'ZERO_INIT_OUT':          True,
        'USE_LAYERNORM':          True,
    },

    16 : {   # aa-SW2 0.271044   |   Final_Number_ITERATION 4096
        'normalization':          'None',
        'NUM_NEURON':             768,
        'num_resblocks':          1,
        'TIME_EMBED_DIM':         30,
        'TIME_SCALE':             150.0,
        'TIME_MAX_PERIOD':        50.0,
        'ODE_STEPS':              200,
        'SOLVER':                 'heun',
        'NOISE_LEVEL':            0.1433125539948,
        'INFERENCE_MODE':         'ode',
        'DENOISER_WEIGHT':        0.0,
        'LearningRate':           0.0001455891043,
        'BATCH_SIZE':             384,
        'SCHEDULE':               'cosine',
        'WARMUP_FRAC':            0.02,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.0,
        'GRAD_CLIP':              2.0,
        'ITERATION':              8192,
        'Final_Number_ITERATION': 4096,
        'RESAMPLE_DATA':          True,
        'COLD_START':             False,
        'ZERO_INIT_OUT':          True,
        'USE_LAYERNORM':          True,
    },

}

# ----------------------------------------------------------------------
# Stochastic Interpolant Filter -- SDE
# ----------------------------------------------------------------------
_SIF_SDE = {
    0  : {   # stage-1 config, online budget zeroed  |  Final_Number_ITERATION 0
        'normalization':          'None',
        'NUM_NEURON':             576,
        'num_resblocks':          1,
        'TIME_EMBED_DIM':         40,
        'TIME_SCALE':             50.0,
        'TIME_MAX_PERIOD':        50.0,
        'ODE_STEPS':              100,
        'SOLVER':                 'heun',
        'NOISE_LEVEL':            0.1814779544991,
        'INFERENCE_MODE':         'sde',
        'DENOISER_WEIGHT':        1.0,
        'LearningRate':           0.000730013106,
        'BATCH_SIZE':             704,
        'SCHEDULE':               'cosine',
        'WARMUP_FRAC':            0.0,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.01,
        'GRAD_CLIP':              1.0,
        'ITERATION':              32768,
        'Final_Number_ITERATION': 0,
        'RESAMPLE_DATA':          True,
        'COLD_START':             False,
        'ZERO_INIT_OUT':          True,
        'USE_LAYERNORM':          True,
    },

    1  : {   # aa-SW2 0.340216   |   Final_Number_ITERATION 256
        'normalization':          'None',
        'NUM_NEURON':             576,
        'num_resblocks':          1,
        'TIME_EMBED_DIM':         40,
        'TIME_SCALE':             50.0,
        'TIME_MAX_PERIOD':        50.0,
        'ODE_STEPS':              100,
        'SOLVER':                 'heun',
        'NOISE_LEVEL':            0.1814779544991,
        'INFERENCE_MODE':         'sde',
        'DENOISER_WEIGHT':        1.0,
        'LearningRate':           0.0001369101213,
        'BATCH_SIZE':             384,
        'SCHEDULE':               'cosine',
        'WARMUP_FRAC':            0.02,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.01,
        'GRAD_CLIP':              0.25,
        'ITERATION':              32768,
        'Final_Number_ITERATION': 256,
        'RESAMPLE_DATA':          True,
        'COLD_START':             False,
        'ZERO_INIT_OUT':          True,
        'USE_LAYERNORM':          True,
    },

    4  : {   # aa-SW2 0.275000   |   Final_Number_ITERATION 1024
        'normalization':          'None',
        'NUM_NEURON':             576,
        'num_resblocks':          1,
        'TIME_EMBED_DIM':         40,
        'TIME_SCALE':             50.0,
        'TIME_MAX_PERIOD':        50.0,
        'ODE_STEPS':              100,
        'SOLVER':                 'heun',
        'NOISE_LEVEL':            0.1814779544991,
        'INFERENCE_MODE':         'sde',
        'DENOISER_WEIGHT':        1.0,
        'LearningRate':           0.0008075814685,
        'BATCH_SIZE':             640,
        'SCHEDULE':               'cosine',
        'WARMUP_FRAC':            0.0,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.001,
        'GRAD_CLIP':              2.0,
        'ITERATION':              32768,
        'Final_Number_ITERATION': 1024,
        'RESAMPLE_DATA':          True,
        'COLD_START':             False,
        'ZERO_INIT_OUT':          True,
        'USE_LAYERNORM':          True,
    },

    16 : {   # aa-SW2 0.266910   |   Final_Number_ITERATION 4096
        'normalization':          'None',
        'NUM_NEURON':             576,
        'num_resblocks':          1,
        'TIME_EMBED_DIM':         40,
        'TIME_SCALE':             50.0,
        'TIME_MAX_PERIOD':        50.0,
        'ODE_STEPS':              100,
        'SOLVER':                 'heun',
        'NOISE_LEVEL':            0.1814779544991,
        'INFERENCE_MODE':         'sde',
        'DENOISER_WEIGHT':        1.0,
        'LearningRate':           0.0002356428186,
        'BATCH_SIZE':             640,
        'SCHEDULE':               'cosine',
        'WARMUP_FRAC':            0.02,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.1,
        'GRAD_CLIP':              0.25,
        'ITERATION':              32768,
        'Final_Number_ITERATION': 4096,
        'RESAMPLE_DATA':          True,
        'COLD_START':             False,
        'ZERO_INIT_OUT':          True,
        'USE_LAYERNORM':          True,
    },

}

# ----------------------------------------------------------------------
# Optimal Transport Filter
# ----------------------------------------------------------------------
_OTF = {
    # 'SCHEDULE' is absent from every OTF cell, so OTF.py's own default
    # ('warm_restarts') applies -- unlike FMF/SIF, which pin 'cosine'.
    0  : {   # stage-1 config, online budget zeroed  |  Final_Number_ITERATION 0
        'normalization':          'None',
        'NUM_NEURON':             [256, 512],
        'num_resblocks':          [1, 1],
        'K_in':                   15,
        'LearningRate':           [0.0002421871224, 0.00034122139],
        'BATCH_SIZE':             576,
        'WARMUP_FRAC':            0.05,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.001,
        'GRAD_CLIP':              0.25,
        'ITERATION':              2048,
        'Final_Number_ITERATION': 0,
        'ZERO_INIT_OUT':          True,
        'T_RESIDUAL':             True,
    },

    1  : {   # aa-SW2 0.408612   |   Final_Number_ITERATION 16
        'normalization':          'None',
        'NUM_NEURON':             [256, 512],
        'num_resblocks':          [1, 1],
        'K_in':                   15,
        'LearningRate':           [6.91457374e-05, 1.32078658e-05],
        'BATCH_SIZE':             128,
        'WARMUP_FRAC':            0.1,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.01,
        'GRAD_CLIP':              1.0,
        'ITERATION':              2048,
        'Final_Number_ITERATION': 16,
        'ZERO_INIT_OUT':          True,
        'T_RESIDUAL':             True,
    },

    4  : {   # aa-SW2 0.349522   |   Final_Number_ITERATION 64
        'normalization':          'None',
        'NUM_NEURON':             [256, 512],
        'num_resblocks':          [1, 1],
        'K_in':                   15,
        'LearningRate':           [4.41565955e-05, 0.0001503236843],
        'BATCH_SIZE':             576,
        'WARMUP_FRAC':            0.02,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.0,
        'GRAD_CLIP':              0.25,
        'ITERATION':              2048,
        'Final_Number_ITERATION': 64,
        'ZERO_INIT_OUT':          True,
        'T_RESIDUAL':             True,
    },

    16 : {   # aa-SW2 0.332348   |   Final_Number_ITERATION 256
        'normalization':          'None',
        'NUM_NEURON':             [256, 512],
        'num_resblocks':          [1, 1],
        'K_in':                   15,
        'LearningRate':           [1.0649665e-05, 1.26150552e-05],
        'BATCH_SIZE':             448,
        'WARMUP_FRAC':            0.1,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.01,
        'GRAD_CLIP':              2.0,
        'ITERATION':              2048,
        'Final_Number_ITERATION': 256,
        'ZERO_INIT_OUT':          True,
        'T_RESIDUAL':             True,
    },

}

# ----------------------------------------------------------------------
# Knothe-Rosenblatt Filter
# ----------------------------------------------------------------------
_KRF = {
    0  : {   # stage-1 config, online budget zeroed  |  Final_Number_ITERATION 0
        'normalization':          'None',
        'NUM_NEURON':             768,
        'depth':                  4,
        'quad_points':            4,
        'LearningRate':           3.2608338e-06,
        'BATCH_SIZE':             768,
        'SCHEDULE':               'cosine',
        'WARMUP_FRAC':            0.1,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.1,
        'GRAD_CLIP':              1.0,
        'ITERATION':              4096,
        'Final_Number_ITERATION': 0,
        'COLD_START':             False,
    },

    1  : {   # aa-SW2 0.328851   |   Final_Number_ITERATION 8
        'normalization':          'None',
        'NUM_NEURON':             768,
        'depth':                  4,
        'quad_points':            4,
        'LearningRate':           2.1014722e-06,
        'BATCH_SIZE':             256,
        'SCHEDULE':               'cosine',
        'WARMUP_FRAC':            0.0,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.1,
        'GRAD_CLIP':              1.0,
        'ITERATION':              4096,
        'Final_Number_ITERATION': 8,
        'COLD_START':             False,
    },

    4  : {   # aa-SW2 0.352541   |   Final_Number_ITERATION 32
        'normalization':          'None',
        'NUM_NEURON':             768,
        'depth':                  4,
        'quad_points':            4,
        'LearningRate':           1.0961332e-06,
        'BATCH_SIZE':             320,
        'SCHEDULE':               'cosine',
        'WARMUP_FRAC':            0.05,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.01,
        'GRAD_CLIP':              1.0,
        'ITERATION':              4096,
        'Final_Number_ITERATION': 32,
        'COLD_START':             False,
    },

    16 : {   # aa-SW2 0.353544   |   Final_Number_ITERATION 128
        'normalization':          'None',
        'NUM_NEURON':             768,
        'depth':                  4,
        'quad_points':            4,
        'LearningRate':           1.0710193e-06,
        'BATCH_SIZE':             128,
        'SCHEDULE':               'cosine',
        'WARMUP_FRAC':            0.0,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.0,
        'GRAD_CLIP':              1.0,
        'ITERATION':              4096,
        'Final_Number_ITERATION': 128,
        'COLD_START':             False,
    },

}

# ----------------------------------------------------------------------
# Schrodinger Bridge Filter
# ----------------------------------------------------------------------
_SBF = {
    0  : {   # stage-1 config, online budget zeroed  |  Final_Number_ITERATION 0
        'normalization':          'None',
        'NUM_NEURON':             640,
        'num_resblocks':          1,
        'TIME_EMBED_DIM':         50,
        'SDE_STEPS':              80,
        'G_DIFFUSION':            1.0,
        'LR':                     0.0002392084557,
        'BATCH_SIZE':             320,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.001,
        'GRAD_CLIP':              0.25,
        'EMA_DECAY':              0.99,
        'ITERATION':              1024,
        'Final_Number_ITERATION': 0,
        'NUM_STAGES':             3,
    },

    1  : {   # aa-SW2 0.459991   |   Final_Number_ITERATION 8
        'normalization':          'None',
        'NUM_NEURON':             640,
        'num_resblocks':          1,
        'TIME_EMBED_DIM':         50,
        'SDE_STEPS':              80,
        'G_DIFFUSION':            1.0,
        'LR':                     0.0003314109255,
        'BATCH_SIZE':             768,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.001,
        'GRAD_CLIP':              2.0,
        'EMA_DECAY':              0.99,
        'ITERATION':              1024,
        'Final_Number_ITERATION': 8,
        'NUM_STAGES':             3,
    },

    4  : {   # aa-SW2 0.443238   |   Final_Number_ITERATION 32
        'normalization':          'None',
        'NUM_NEURON':             640,
        'num_resblocks':          1,
        'TIME_EMBED_DIM':         50,
        'SDE_STEPS':              80,
        'G_DIFFUSION':            1.0,
        'LR':                     0.0004509968836,
        'BATCH_SIZE':             768,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.1,
        'GRAD_CLIP':              0.5,
        'EMA_DECAY':              0.99,
        'ITERATION':              1024,
        'Final_Number_ITERATION': 32,
        'NUM_STAGES':             3,
    },

    16 : {   # aa-SW2 0.411021   |   Final_Number_ITERATION 128
        'normalization':          'None',
        'NUM_NEURON':             640,
        'num_resblocks':          1,
        'TIME_EMBED_DIM':         50,
        'SDE_STEPS':              80,
        'G_DIFFUSION':            1.0,
        'LR':                     0.000588581244,
        'BATCH_SIZE':             192,
        'OPTIMIZER':              'adamw',
        'WEIGHT_DECAY':           0.001,
        'GRAD_CLIP':              1.0,
        'EMA_DECAY':              0.99,
        'ITERATION':              1024,
        'Final_Number_ITERATION': 128,
        'NUM_STAGES':             3,
    },

}

_CONFIGS = {
    'fmf':         _FMF,
    'sif_ode':     _SIF_ODE,
    'sif_sde':     _SIF_SDE,
    'otf':         _OTF,
    'krf':         _KRF,
    'sbf':         _SBF,
}


def get_config(tune, L, dy, fi):
    """
    Return the parameter dict for one method at one online budget.

    Parameters
    ----------
    tune : one of METHODS —
        'fmf' | 'sif_ode' | 'sif_sde' | 'otf' | 'krf' | 'sbf'
    L, dy : int — state and observation dimensions. These enter only through
        INPUT_DIM; every other value is dimension-independent.
    fi : one of FINAL_ITERS — 0, 1, 4 or 16. The RAW search value, not the
        iteration count. The resolved budget is fi * FI_BASE[tune], and is
        already baked into the returned 'Final_Number_ITERATION'.

        1, 4 and 16 are the tuned cells. 0 is the no-adaptation baseline: the
        stage-1 configuration with a zero online budget, so the run is pure
        inference under the warm-started network and trains at no step. It is
        served here rather than from param_one_step.py so that one call site,
        one fi knob, covers the whole error-against-compute curve.

    Returns
    -------
    dict — parameters for the implementation named in FILTERS[tune].

    Raises
    ------
    ValueError — unknown method or unknown fi. Never returns a neighbouring
        budget's settings as a substitute: the optimiser was tuned per budget,
        so a silent fallback would report a configuration never measured.
    """
    if tune not in _CONFIGS:
        raise ValueError(f'unknown method {tune!r}; available: {sorted(_CONFIGS)}')
    if fi not in FINAL_ITERS:
        raise ValueError(f'unknown fi {fi!r}; available: {FINAL_ITERS}')
    params = dict(_CONFIGS[tune][fi])
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

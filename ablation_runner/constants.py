"""Shared constants for the ablation_runner toolchain.

W_WARP_VALUE is the source of truth for the post-tune w_threshold_warp
across run_ablation.py / run_profiling.py / kernel_breakdown.py. Only
tuning_w.py sweeps W; everywhere else this constant is the answer.
"""
from pathlib import Path

# ============================================================
# Post-tune winning W threshold. Source of truth — update if a new
# sweep on tuning_w.py finds a different winner for the workload.
# ============================================================
W_WARP_VALUE = 4


# Variants under test.
ALL_VARIANTS  = ['node_grouped', 'full_walk', 'node_grouped_global_only']
NG_VARIANTS   = ['node_grouped', 'node_grouped_global_only']

# Datasets used by every script in this directory.
DATASETS      = ('delicious', 'coin', 'flight')


# Per-dataset (wpn, num_batches, num_windows, max_walk_len). A40-sized.
PRESETS = {
    'delicious': ( 8, 30, 15, 20),
    'coin':      (20,  4,  1, 80),
    'flight':    (20,  4,  1, 80),
}

# ============================================================
# Output base + binary path.
# ============================================================
DEFAULT_OUTPUT_BASE = 'ablation_results'
DEFAULT_BLOCK_DIM   = 256

# Scripts live in <co>/tempest-benchmarks/ablation_runner/. The C++ binary
# lives in the sibling temporal-random-walk repo at
# <co>/temporal-random-walk/build/bin/ablation_streaming — i.e.
# ../../temporal-random-walk/build relative to this directory.
ABLATION_DIR = Path(__file__).resolve().parent
TRW_BUILD    = ABLATION_DIR.parent.parent / 'temporal-random-walk' / 'build'
DEFAULT_BIN  = str(TRW_BUILD / 'bin' / 'ablation_streaming')
DEFAULT_NSYS = 'nsys'

# ============================================================
# Fixed per-call CLI args for ablation_streaming.
# ============================================================
USE_GPU     = '1'
PICKER      = 'exponential_weight'
IS_DIRECTED = '0'
TIMESCALE   = '-1'

# ============================================================
# Outlier rejection / phase-2 sample sizes.
# Used by all scripts that aggregate measurements.
# ============================================================
# Iterative median-relative threshold: drop the value whose deviation
# from the running median exceeds OUTLIER_THRESHOLD * median, until the
# worst remaining is within threshold OR len(kept) == MIN_KEEP.
OUTLIER_THRESHOLD = 0.15
MIN_KEEP          = 3

# Aggregate-W tie-break band for tuning. Tightened to 0.5% — at 1% the
# band swallowed almost every W on the laptop traces.
NOISE_BAND        = 0.005

FINAL_RUNS_THR    = 5  # final throughput runs per (dataset, variant)

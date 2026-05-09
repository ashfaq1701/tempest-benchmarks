"""Server-scale dataset overlay for the second batch of datasets:
wiki-talk-temporal, ml_tgbl-comment, sx-stackoverflow.

Re-exports everything from constants.py except DATASETS and PRESETS,
which are overridden here. Scale matches the coin/flight cell of
constants.py — designed for an A40-class GPU. Lower wpn or raise nb/nw
if the server you're targeting OOMs on sx-stackoverflow (2.6M nodes,
the largest of the three).

The run_ablation_new.py / run_profiling_new.py scripts import from
this module instead of constants.py so they can share helpers but
target the new dataset set.
"""
# pylint: disable=wildcard-import,unused-wildcard-import
from constants import (  # noqa: F401  (re-exported for sibling scripts)
    W_WARP_VALUE,
    ALL_VARIANTS, NG_VARIANTS,
    DEFAULT_OUTPUT_BASE, DEFAULT_BLOCK_DIM,
    ABLATION_DIR, TRW_BUILD, DEFAULT_BIN, DEFAULT_NSYS,
    USE_GPU, PICKER, IS_DIRECTED, TIMESCALE,
    OUTLIER_THRESHOLD, MIN_KEEP, NOISE_BAND,
    FINAL_RUNS_THR,
)


# Datasets in this batch.
DATASETS = ('wiki-talk', 'tgbl-comment', 'sx-stackoverflow')


# Per-dataset (wpn, num_batches, num_windows, max_walk_len). A40-sized,
# matching the coin/flight cell of constants.py:
#   - wiki-talk-temporal:  7.8M edges, 1.14M nodes — same shape as coin.
#   - ml_tgbl-comment   : 44.3M edges,  995K nodes — between coin and flight.
#   - sx-stackoverflow  : 63.5M edges, 2.6M nodes — same edge count as
#       flight but ~2.6× the nodes; nw=2 halves active-per-window so the
#       walk-set fits without dropping wpn.
PRESETS = {
    'wiki-talk':        (20, 4, 1, 80),
    'tgbl-comment':     (20, 4, 1, 80),
    'sx-stackoverflow': (20, 4, 2, 80),
}

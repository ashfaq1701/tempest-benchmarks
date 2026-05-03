#!/usr/bin/env python3
"""W-threshold tuning sweep — extracted from the original Phase 1 of
run_ablation.py.

Sweeps `--w-threshold-warp ∈ TUNE_W_VALUES` for `node_grouped` only on
each (dataset, variant). FW ignores the parameter entirely.
node_grouped_global_only shares the scheduler with node_grouped, so it
inherits the winning W rather than being tuned separately. Picks ONE
universal W across datasets via mean-of-per-row-normalized steps/sec;
ties (within NOISE_BAND) break to the smallest W (most conservative).

This script is diagnostic only. It does NOT update constants.W_WARP_VALUE.
The rest of the toolchain (run_ablation / run_profiling / kernel_breakdown)
reads W_WARP_VALUE from constants.py directly. If this sweep produces a
different winner from the current constant, update constants.py manually.

Usage:
  cd ablation_runner
  python3 tuning_w.py coin.csv flight.csv delicious.csv \\
      --output ablation_results --block-dim 256

Default --binary is ../../temporal-random-walk/build/bin/ablation_streaming.
"""
import argparse
import statistics
import sys
from pathlib import Path

from constants import (
    DATASETS, NG_VARIANTS, PRESETS,
    DEFAULT_BIN, DEFAULT_BLOCK_DIM, DEFAULT_OUTPUT_BASE,
    NOISE_BAND,
)
from common import invoke, mean_std, reject_outliers, fmt_drop, write_csv


# Phase 1 tunes node_grouped only; node_grouped_global_only inherits.
TUNE_VARIANTS   = ['node_grouped']
TUNE_W_VALUES   = [1, 2, 4, 8, 16, 32, 64]
TUNE_RUNS_PER_W = 5


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('coin_csv',      help='Path to ml_tgbl-coin.csv')
    ap.add_argument('flight_csv',    help='Path to ml_tgbl-flight.csv')
    ap.add_argument('delicious_csv', help='Path to delicious(_clip).csv')
    ap.add_argument('--binary',     default=DEFAULT_BIN,
                    help=f'Path to ablation_streaming (default: {DEFAULT_BIN}).')
    ap.add_argument('--output',     default=DEFAULT_OUTPUT_BASE,
                    help=f'CSV output base; writes <base>_tuning.csv '
                         f'(default: {DEFAULT_OUTPUT_BASE}).')
    ap.add_argument('--block-dim',  type=int, default=DEFAULT_BLOCK_DIM,
                    help=f'block_dim passed to ablation_streaming '
                         f'(default: {DEFAULT_BLOCK_DIM}).')
    args = ap.parse_args()

    if not Path(args.binary).is_file():
        ap.error(f'binary not found: {args.binary}')
    paths = {'coin': args.coin_csv, 'flight': args.flight_csv,
             'delicious': args.delicious_csv}
    for name, p in paths.items():
        if not Path(p).is_file():
            ap.error(f'{name} CSV not found: {p}')

    out_base = Path(args.output)
    out_tune = out_base.parent / f'{out_base.name}_tuning.csv'

    print(f'# binary           : {args.binary}')
    print(f'# block_dim        : {args.block_dim}')
    print(f'# tune W values    : {TUNE_W_VALUES}')
    print(f'# tune variants    : {TUNE_VARIANTS}  '
          f'(NG_global_only inherits; FW skipped)')
    print(f'# tune runs per W  : {TUNE_RUNS_PER_W}')
    print(f'# tie-break band   : {NOISE_BAND:.1%}  '
          f'(smallest W wins within this gap)')
    print(f'# tuning CSV       : {out_tune}')
    print()

    tuning_rows = []
    tune_means: dict = {}

    print('=' * 70)
    print('Phase 1: W-threshold tuning sweep (node_grouped only)')
    print('=' * 70)
    print()

    for ds in DATASETS:
        wpn, nb, nw, mwl = PRESETS[ds]
        for variant in TUNE_VARIANTS:
            tune_means[(ds, variant)] = {}
            for w in TUNE_W_VALUES:
                tune_sps = []
                for r in range(TUNE_RUNS_PER_W):
                    tag = (f'  tune  {ds:>9} / {variant:<26} / '
                           f'W={w:<3} run {r+1}/{TUNE_RUNS_PER_W}')
                    print(f'{tag} ...', end=' ', flush=True)
                    try:
                        t, s, a = invoke(args.binary, paths[ds], variant,
                                         wpn, nb, nw, mwl, args.block_dim, w)
                    except RuntimeError as e:
                        print(f'FAIL ({e})', file=sys.stderr)
                        continue
                    print(f'thr={t/1e6:6.3f}M w/s  steps={s/1e6:7.3f}M s/s  '
                          f'avg_len={a:5.2f}')
                    tune_sps.append(s)
                    tuning_rows.append({
                        'phase':            'tune',
                        'dataset':          ds,
                        'variant':          variant,
                        'w_threshold_warp': w,
                        'run':              r + 1,
                        'wpn':              wpn,
                        'num_batches':      nb,
                        'num_windows':      nw,
                        'max_walk_len':     mwl,
                        'block_dim':        args.block_dim,
                        'walks_per_sec':    t,
                        'steps_per_sec':    s,
                        'avg_walk_length':  a,
                    })
                if tune_sps:
                    kept, dropped = reject_outliers(tune_sps)
                    tune_means[(ds, variant)][w] = statistics.mean(kept)
                    msg = fmt_drop('steps/sec', dropped, len(kept), len(tune_sps),
                                   scale=1e6, unit='M s/s')
                    if msg: print(msg)

    # Tuning matrix table — mean steps/sec per (dataset, variant) × W.
    print()
    print('=== Tuning matrix (mean steps/sec, M; * = per-row best) ===')
    print()
    header_w = '  '.join(f'{w:>7}' for w in TUNE_W_VALUES)
    print(f'{"dataset":<9}  {"variant":<26}  W=  {header_w}')
    print('-' * (9 + 2 + 26 + 2 + 4 + len(header_w) + 2))
    for ds in DATASETS:
        for variant in TUNE_VARIANTS:
            ws_means = tune_means.get((ds, variant), {})
            if not ws_means:
                continue
            best_w_row = max(ws_means, key=ws_means.get)
            cells = []
            for w in TUNE_W_VALUES:
                v = ws_means.get(w)
                if v is None:
                    cells.append(f'{"--":>7}')
                else:
                    mark = '*' if w == best_w_row else ' '
                    cells.append(f'{mark}{v/1e6:6.3f}')
            print(f'{ds:<9}  {variant:<26}      {"  ".join(cells)}')
    print()

    # Aggregate to ONE universal W (mean of per-row-normalized steps/sec).
    norm_table: dict = {}
    for key, ws_means in tune_means.items():
        if not ws_means:
            continue
        row_max = max(ws_means.values())
        norm_table[key] = ({w: v / row_max for w, v in ws_means.items()}
                           if row_max > 0
                           else {w: 0.0 for w in ws_means})
    agg_score: dict = {}
    for w in TUNE_W_VALUES:
        fracs = [norm_table[k][w] for k in norm_table if w in norm_table[k]]
        if fracs:
            agg_score[w] = statistics.mean(fracs)

    abs_best_score = max(agg_score.values()) if agg_score else 0.0
    eligible = [w for w, s in agg_score.items() if s >= abs_best_score - NOISE_BAND]
    winning_w = min(eligible) if eligible else 1
    abs_best_w = max(agg_score, key=agg_score.get) if agg_score else 1
    tied = (winning_w != abs_best_w)

    print('=== Aggregate score across datasets (mean normalized steps/sec) ===')
    print()
    print(f'{"W":>4}  {"mean frac":>10}  {"  pick":<10}')
    print('-' * 30)
    for w in TUNE_W_VALUES:
        s = agg_score.get(w)
        if s is None:
            print(f'{w:>4}  {"--":>10}'); continue
        marker = ''
        if w == winning_w:    marker = '<-- chosen'
        elif w == abs_best_w: marker = '(abs max)'
        print(f'{w:>4}  {s:>10.4f}  {marker}')
    print()
    if tied:
        print(f'Note: W={winning_w} chosen over W={abs_best_w} '
              f'(score gap < {NOISE_BAND:.0%}; smallest W in tie band wins).')
    elif (abs_best_score - min(agg_score.values())) < NOISE_BAND:
        print(f'Note: aggregate scores spread less than {NOISE_BAND:.0%} '
              f'across all W — likely no real signal.')
    print()
    print(f'=== Sweep winner: W={winning_w} ===')
    print()
    print('Note: this script is diagnostic. constants.W_WARP_VALUE is the '
          'source of truth for run_ablation / run_profiling / '
          'kernel_breakdown. If this sweep produced a different winner, '
          'update constants.py manually.')

    if tuning_rows:
        write_csv(out_tune, tuning_rows)
        print(f'\nWrote {len(tuning_rows)} tuning rows to {out_tune}')
        return 0
    print('No successful tuning runs.', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())

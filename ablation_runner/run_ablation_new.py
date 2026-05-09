#!/usr/bin/env python3
"""
Throughput-finals runner for the second dataset batch (wiki-talk-temporal,
ml_tgbl-comment, sx-stackoverflow). Mirrors run_ablation.py's contract
exactly — it just imports DATASETS / PRESETS from constants_new.py.

W is sourced from constants.W_WARP_VALUE (post-tune winner). FW ignores
w_threshold_warp entirely, so its W is hardcoded to 1.

Optimization target is steps/sec, not walks/sec — see run_ablation.py
for the rationale.

Output:
  <output>_final.csv — one row per (dataset, variant) over FINAL_RUNS_THR
  runs. run_profiling_new.py reads this back to merge nsys columns.

Usage (from ablation_runner/):
  python3 run_ablation_new.py wiki-talk.csv tgbl-comment.csv \\
      sx-stackoverflow.csv --output ablation_results_new --block-dim 256

Server-only: PRESETS in constants_new.py are A40-sized.
"""
import argparse
import sys
from pathlib import Path

from constants_new import (
    ALL_VARIANTS, NG_VARIANTS, DATASETS, PRESETS,
    DEFAULT_BIN, DEFAULT_BLOCK_DIM, DEFAULT_OUTPUT_BASE,
    FINAL_RUNS_THR,
    W_WARP_VALUE,
)
from common import invoke, mean_std, reject_outliers, fmt_drop, write_csv


# FW ignores w_threshold_warp entirely (only NG uses it). Pin to 1 in
# the CLI calls; value is irrelevant for FW correctness or perf.
W_FW_HARDCODED = 1


def cell_w(variant: str) -> int:
    return W_WARP_VALUE if variant in NG_VARIANTS else W_FW_HARDCODED


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('wiki_talk_csv',        help='Path to wiki-talk-temporal.csv')
    ap.add_argument('tgbl_comment_csv',     help='Path to ml_tgbl-comment.csv')
    ap.add_argument('sx_stackoverflow_csv', help='Path to sx-stackoverflow.csv')
    ap.add_argument('--binary',     default=DEFAULT_BIN,
                    help=f'Path to ablation_streaming (default: {DEFAULT_BIN}).')
    ap.add_argument('--output',     default=DEFAULT_OUTPUT_BASE,
                    help=f'CSV output base; writes <base>_final.csv '
                         f'(default: {DEFAULT_OUTPUT_BASE}).')
    ap.add_argument('--block-dim',  type=int, default=DEFAULT_BLOCK_DIM,
                    help=f'block_dim passed to ablation_streaming '
                         f'(default: {DEFAULT_BLOCK_DIM}).')
    args = ap.parse_args()

    if not Path(args.binary).is_file():
        ap.error(f'binary not found: {args.binary}')
    paths = {'wiki-talk':        args.wiki_talk_csv,
             'tgbl-comment':     args.tgbl_comment_csv,
             'sx-stackoverflow': args.sx_stackoverflow_csv}
    for name, p in paths.items():
        if not Path(p).is_file():
            ap.error(f'{name} CSV not found: {p}')

    out_base  = Path(args.output)
    out_final = out_base.parent / f'{out_base.name}_final.csv'

    print(f'# binary           : {args.binary}')
    print(f'# block_dim        : {args.block_dim}')
    print(f'# W (NG variants)  : {W_WARP_VALUE}  (constants.W_WARP_VALUE)')
    print(f'# W (FW)           : {W_FW_HARDCODED}  (hardcoded; FW ignores W)')
    print(f'# final thr runs   : {FINAL_RUNS_THR}  (per dataset × variant)')
    print(f'# final CSV        : {out_final}')
    print()

    # ===========================================================
    # Throughput finals — FINAL_RUNS_THR per (dataset, variant).
    # ===========================================================
    print('=' * 70)
    print(f'Throughput — {FINAL_RUNS_THR} runs per (dataset, variant)')
    print('=' * 70)
    print()

    thr_stats: dict = {}
    for ds in DATASETS:
        wpn, nb, nw, mwl = PRESETS[ds]
        for variant in ALL_VARIANTS:
            w = cell_w(variant)
            thrs, sps, lens = [], [], []
            for r in range(FINAL_RUNS_THR):
                tag = (f'  thr   {ds:>17} / {variant:<26} / W={w:<3} '
                       f'run {r+1}/{FINAL_RUNS_THR}')
                print(f'{tag} ...', end=' ', flush=True)
                try:
                    t, s, a = invoke(args.binary, paths[ds], variant,
                                     wpn, nb, nw, mwl, args.block_dim, w)
                except RuntimeError as e:
                    print(f'FAIL ({e})', file=sys.stderr); continue
                print(f'thr={t/1e6:6.3f}M w/s  steps={s/1e6:7.3f}M s/s  '
                      f'avg_len={a:5.2f}')
                thrs.append(t); sps.append(s); lens.append(a)
            kept_sps,  dropped_sps  = reject_outliers(sps)
            kept_thrs, dropped_thrs = reject_outliers(thrs)
            kept_lens, _            = reject_outliers(lens)
            for msg in [
                fmt_drop('steps/sec', dropped_sps, len(kept_sps), len(sps),
                         scale=1e6, unit='M s/s'),
                (None if dropped_thrs == dropped_sps else
                 fmt_drop('walks/sec', dropped_thrs, len(kept_thrs), len(thrs),
                          scale=1e6, unit='M w/s')),
            ]:
                if msg: print(msg)
            tm, td = mean_std(kept_thrs)
            sm, sd = mean_std(kept_sps)
            lm, _  = mean_std(kept_lens)
            thr_stats[(ds, variant)] = {
                'w': w, 'n_thr_raw': len(thrs),
                'walks_per_sec_mean': tm, 'walks_per_sec_std': td,
                'walks_per_sec_n_kept': len(kept_thrs),
                'steps_per_sec_mean': sm, 'steps_per_sec_std': sd,
                'steps_per_sec_n_kept': len(kept_sps),
                'avg_walk_length_mean': lm,
            }

    # Build final rows.
    final_rows = []
    for ds in DATASETS:
        for variant in ALL_VARIANTS:
            ts = thr_stats.get((ds, variant), {})
            if not ts:
                continue
            row = {'dataset': ds, 'variant': variant,
                   'w_threshold_warp': ts.get('w', cell_w(variant))}
            row.update(ts)
            final_rows.append(row)

    if not final_rows:
        print('No successful throughput runs.', file=sys.stderr); return 1

    write_csv(out_final, final_rows)

    print()
    print(f'Wrote {len(final_rows)} final rows to {out_final}')
    print()

    # ===========================================================
    # Final markdown report.
    # ===========================================================
    print('=' * 70)
    print('Final report — throughput')
    print('=' * 70)
    print()
    print(f'| {"dataset":<17} | {"variant":<26} | {"W":>3} | '
          f'{"steps/s mean (M)":>16} | {"std":>7} | '
          f'{"walks/s mean (M)":>16} | {"std":>7} | '
          f'{"avg_len":>7} |')
    print('|' + '|'.join('-' * w for w in
          [19, 28, 5, 18, 9, 18, 9, 9]) + '|')
    for r in final_rows:
        print(f'| {r["dataset"]:<17} | {r["variant"]:<26} | '
              f'{r["w_threshold_warp"]:>3} | '
              f'{r["steps_per_sec_mean"]/1e6:>16.3f} | '
              f'{r["steps_per_sec_std"]/1e6:>7.3f} | '
              f'{r["walks_per_sec_mean"]/1e6:>16.3f} | '
              f'{r["walks_per_sec_std"]/1e6:>7.3f} | '
              f'{r["avg_walk_length_mean"]:>7.2f} |')
    return 0


if __name__ == '__main__':
    sys.exit(main())

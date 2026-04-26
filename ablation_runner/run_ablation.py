#!/usr/bin/env python3
"""
Server-side ablation runner: drives ablation_streaming across
(dataset × kernel_launch_type × run) and writes per-run metrics to CSV.

Optimization target: **steps/sec** (work actually done across all walks).
Not walks/sec — a length-1 walk that did nothing still counts as one walk,
so walks/sec is fooled by bugs that drop walks. steps/sec captures the
total computational work and is monotone in real throughput.

Two-phase design:
  Phase 1 (tune)       — sweep --w-threshold-warp ∈ {1,2,4,8,16,32,64} for the
                         two NODE_GROUPED variants (FW ignores the parameter,
                         so it is not tuned). TUNE_RUNS_PER_W runs per
                         (dataset, variant, W). Picks ONE universal W across
                         datasets by averaging the per-row-normalized
                         **steps/sec**; ties (within NOISE_BAND) break to the
                         smallest W (most conservative).
  Phase 2 (throughput) — FINAL_RUNS_THR runs per (dataset, variant) at the
                         universal W (NG variants) and at W=1 for FW. Reject
                         outliers per metric; store mean ± std of walks/sec,
                         steps/sec, avg_walk_length.

For nsys-derived kernel/NVTX metrics, run `run_phase3.py` AFTER this script
finishes. It reuses the constants/helpers defined here, hardcodes the
winning W, profiles each (dataset, variant), and merges the nsys columns
into <base>_final.csv plus writes <base>_ingest.csv.

Datasets:  coin, flight, delicious — each with its own (wpn, nb, nw, mwl)
           preset sized for an A40 (48 GB) class GPU.
Variants:  full_walk, node_grouped, node_grouped_global_only.

Outputs (CSV; --output BASE produces BASE_tuning.csv and BASE_final.csv):
  - <base>_tuning.csv   : one row per Phase-1 invocation
                          (phase, dataset, variant, W, run, metrics, config)
  - <base>_final.csv    : one row per (dataset, variant) summarising
                          Phase-2 throughput stats. run_phase3.py later
                          extends this CSV with nsys-derived columns.

Usage (from outside the build directory):
  python3 run_ablation.py coin.csv flight.csv delicious.csv \\
      --output ablation_results --block-dim 256

Default --binary is ./build/bin/ablation_streaming.
"""
import argparse
import csv
import re
import statistics
import subprocess
import sys
from pathlib import Path

ALL_VARIANTS = ['full_walk', 'node_grouped', 'node_grouped_global_only']
NG_VARIANTS  = ['node_grouped', 'node_grouped_global_only']
DATASETS     = ('coin', 'flight', 'delicious')

DEFAULT_BIN          = './build/bin/ablation_streaming'
DEFAULT_OUTPUT_BASE  = 'ablation_results'
DEFAULT_BLOCK_DIM    = 256

TUNE_W_VALUES        = [1, 2, 4, 8, 16, 32, 64]
TUNE_RUNS_PER_W      = 5
FINAL_RUNS_THR       = 5

# Outlier rejection: from N samples, drop the value(s) whose deviation from
# the (current) median exceeds OUTLIER_THRESHOLD * median, iteratively.
# Stops when worst remaining is within threshold OR len(kept) == MIN_KEEP.
# 0.15 catches the obvious 50%+ flukes (e.g. 4.8 vs cluster ~12), the
# moderate ~25% dips, AND the borderline ~15-20% server-contention dips
# (e.g. 47 vs cluster ~55) — while still preserving normal 5% run-to-run
# noise (5% << 15%).
OUTLIER_THRESHOLD    = 0.15
MIN_KEEP             = 3
# Aggregate-W tie-break band. If multiple Ws are within NOISE_BAND of the
# absolute best aggregate score, the smallest W wins (most conservative).
NOISE_BAND           = 0.01

# Per-dataset (wpn, num_batches, num_windows, max_walk_len). A40-sized.
# - coin / flight : full spec from the paper plan (wpn=20, mwl=80, ~33% window).
# - delicious     : finer streaming cadence — 50 batches × 5% window
#                   (window = 2.5 × batch). wpn=1, mwl=20 keep walk-output VRAM
#                   in check on 33.8 M nodes (13.5 GB); active graph at 5%
#                   window adds ~1 GB.
# Invariant: num_windows <= num_batches — otherwise window < batch and the
# older half of every batch is ingested then immediately expired before
# the walk call (wasted ingest, no walk samples on those edges).
PRESETS = {
    'coin':      (20,  5,  3, 80),
    'flight':    (20,  5,  3, 80),
    'delicious': ( 1, 50, 20, 20),
}

USE_GPU     = '1'
PICKER      = 'exponential_weight'
IS_DIRECTED = '0'
TIMESCALE   = '-1'

THROUGHPUT_RE = re.compile(r'^Throughput:\s+([\d.eE+-]+)\s+walks/sec', re.MULTILINE)
STEPS_RE      = re.compile(r'^Steps/sec:\s+([\d.eE+-]+)\s+steps/sec',  re.MULTILINE)
AVGLEN_RE     = re.compile(r'^Final avg walk length:\s*([\d.eE+-]+)',  re.MULTILINE)


def build_run_argv(binary, data, klt, wpn, nb, nw, mwl, block_dim, w_threshold_warp):
    return [binary, data, USE_GPU, PICKER, klt, IS_DIRECTED,
            str(wpn), str(nb), str(nw), str(mwl), TIMESCALE,
            str(block_dim), str(w_threshold_warp)]


def parse_throughput(stdout):
    m_t = THROUGHPUT_RE.search(stdout)
    m_s = STEPS_RE.search(stdout)
    m_a = AVGLEN_RE.search(stdout)
    if not (m_t and m_s and m_a):
        raise RuntimeError(
            f'missing one of Throughput/Steps/AvgLen in stdout tail:\n{stdout[-500:]}')
    return float(m_t.group(1)), float(m_s.group(1)), float(m_a.group(1))


def invoke(binary, data, klt, wpn, nb, nw, mwl, block_dim, w_threshold_warp):
    cmd = build_run_argv(binary, data, klt, wpn, nb, nw, mwl, block_dim, w_threshold_warp)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f'exit {proc.returncode}. stderr tail:\n{proc.stderr[-500:]}')
    return parse_throughput(proc.stdout)


def mean_std(xs):
    if not xs:
        return 0.0, 0.0
    mu = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return mu, sd


def reject_outliers(xs, threshold_frac=OUTLIER_THRESHOLD, min_keep=MIN_KEEP):
    """Iterative median-relative outlier rejection. See module docstring."""
    kept = list(xs)
    dropped: list = []
    while len(kept) > min_keep:
        med = statistics.median(kept)
        if med == 0:
            break
        devs = [abs(x - med) / med for x in kept]
        max_idx = max(range(len(kept)), key=lambda i: devs[i])
        if devs[max_idx] > threshold_frac:
            dropped.append(kept.pop(max_idx))
        else:
            break
    return kept, dropped


def fmt_drop(label, dropped, kept_n, raw_n, scale=1.0, unit=''):
    if not dropped:
        return None
    s = ', '.join(f'{d/scale:.3f}{unit}' for d in dropped)
    return f'    [{len(dropped)} outlier(s) rejected ({label}): {s}; kept {kept_n}/{raw_n}]'


def write_csv(path, rows):
    """Write a list-of-dicts to CSV. Field order = union of keys across
    rows in insertion order. Idempotent — overwrites if file exists."""
    if not rows:
        return
    fieldnames, seen = [], set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k); seen.add(k)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)


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
                    help=f'CSV output base; writes <base>_tuning.csv and '
                         f'<base>_final.csv (default: {DEFAULT_OUTPUT_BASE}).')
    ap.add_argument('--block-dim',  type=int, default=DEFAULT_BLOCK_DIM,
                    help=f'block_dim passed to ablation_streaming (default: {DEFAULT_BLOCK_DIM}).')
    args = ap.parse_args()

    if not Path(args.binary).is_file():
        ap.error(f'binary not found: {args.binary}')
    paths = {'coin': args.coin_csv, 'flight': args.flight_csv, 'delicious': args.delicious_csv}
    for name, p in paths.items():
        if not Path(p).is_file():
            ap.error(f'{name} CSV not found: {p}')

    out_base = Path(args.output)
    out_tune  = out_base.parent / f'{out_base.name}_tuning.csv'
    out_final = out_base.parent / f'{out_base.name}_final.csv'

    print(f'# binary           : {args.binary}')
    print(f'# block_dim        : {args.block_dim}')
    print(f'# tune W values    : {TUNE_W_VALUES}')
    print(f'# tune runs per W  : {TUNE_RUNS_PER_W}  (NG variants only; FW skipped)')
    print(f'# final thr runs   : {FINAL_RUNS_THR}  (per dataset × variant at winning W)')
    print(f'# outlier threshold: {OUTLIER_THRESHOLD:.0%} (min_keep={MIN_KEEP})')
    print(f'# tuning CSV       : {out_tune}    (one row per Phase-1 invocation)')
    print(f'# final CSV        : {out_final}     (one row per dataset × variant)')
    print()

    tuning_rows = []  # raw Phase-1 rows

    # ===========================================================
    # Phase 1: W-threshold tuning sweep (NG variants only).
    # Optimize on steps/sec (work done), not walks/sec (walk count).
    # ===========================================================
    print('=' * 70)
    print('Phase 1: W-threshold tuning sweep')
    print('=' * 70)
    print()

    tune_means: dict = {}
    for ds in DATASETS:
        wpn, nb, nw, mwl = PRESETS[ds]
        for variant in NG_VARIANTS:
            tune_means[(ds, variant)] = {}
            for w in TUNE_W_VALUES:
                tune_sps = []
                for r in range(TUNE_RUNS_PER_W):
                    tag = f'  tune  {ds:>9} / {variant:<26} / W={w:<3} run {r+1}/{TUNE_RUNS_PER_W}'
                    print(f'{tag} ...', end=' ', flush=True)
                    try:
                        t, s, a = invoke(args.binary, paths[ds], variant,
                                         wpn, nb, nw, mwl, args.block_dim, w)
                    except RuntimeError as e:
                        print(f'FAIL ({e})', file=sys.stderr); continue
                    print(f'thr={t/1e6:6.3f}M w/s  steps={s/1e6:7.3f}M s/s  avg_len={a:5.2f}')
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
        for variant in NG_VARIANTS:
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
        norm_table[key] = {w: v / row_max for w, v in ws_means.items()} \
                          if row_max > 0 else {w: 0.0 for w in ws_means}
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
    print(f'=== Universal W chosen: {winning_w} ===')
    print()

    cell_w = {(ds, v): (winning_w if v in NG_VARIANTS else 1)
              for ds in DATASETS for v in ALL_VARIANTS}

    # Persist tuning CSV NOW. Phase 1 typically takes the longest; if Phase 2
    # explodes, we must not lose this data.
    write_csv(out_tune, tuning_rows)
    print(f'(checkpoint) wrote {len(tuning_rows)} tuning rows to {out_tune}')
    print()

    # ===========================================================
    # Phase 2: throughput finals — FINAL_RUNS_THR per (dataset, variant).
    # ===========================================================
    print('=' * 70)
    print(f'Phase 2: throughput — {FINAL_RUNS_THR} runs per (dataset, variant)')
    print('=' * 70)
    print()

    thr_stats: dict = {}
    for ds in DATASETS:
        wpn, nb, nw, mwl = PRESETS[ds]
        for variant in ALL_VARIANTS:
            w = cell_w[(ds, variant)]
            thrs, sps, lens = [], [], []
            for r in range(FINAL_RUNS_THR):
                tag = f'  thr   {ds:>9} / {variant:<26} / W={w:<3} run {r+1}/{FINAL_RUNS_THR}'
                print(f'{tag} ...', end=' ', flush=True)
                try:
                    t, s, a = invoke(args.binary, paths[ds], variant,
                                     wpn, nb, nw, mwl, args.block_dim, w)
                except RuntimeError as e:
                    print(f'FAIL ({e})', file=sys.stderr); continue
                print(f'thr={t/1e6:6.3f}M w/s  steps={s/1e6:7.3f}M s/s  avg_len={a:5.2f}')
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

    # Build final rows from Phase-2 stats.
    final_rows = []
    for ds in DATASETS:
        for variant in ALL_VARIANTS:
            ts = thr_stats.get((ds, variant), {})
            if not ts:
                continue
            row = {'dataset': ds, 'variant': variant,
                   'w_threshold_warp': ts.get('w', 1)}
            row.update(ts)
            final_rows.append(row)

    if not tuning_rows:
        print('No successful tuning runs.', file=sys.stderr); return 1
    if not final_rows:
        print('No successful throughput runs.', file=sys.stderr); return 1

    write_csv(out_final, final_rows)

    print()
    print(f'Wrote {len(final_rows)} final rows to {out_final}')
    print()

    # ===========================================================
    # Final markdown report. steps/sec leads (optimization metric).
    # ===========================================================
    print('=' * 70)
    print('Final report — Phase 2 throughput')
    print('=' * 70)
    print()
    print(f'| {"dataset":<9} | {"variant":<26} | {"W":>3} | '
          f'{"steps/s mean (M)":>16} | {"std":>7} | '
          f'{"walks/s mean (M)":>16} | {"std":>7} | '
          f'{"avg_len":>7} |')
    print('|' + '|'.join('-' * w for w in
          [11, 28, 5, 18, 9, 18, 9, 9]) + '|')
    for r in final_rows:
        print(f'| {r["dataset"]:<9} | {r["variant"]:<26} | '
              f'{r["w_threshold_warp"]:>3} | '
              f'{r["steps_per_sec_mean"]/1e6:>16.3f} | '
              f'{r["steps_per_sec_std"]/1e6:>7.3f} | '
              f'{r["walks_per_sec_mean"]/1e6:>16.3f} | '
              f'{r["walks_per_sec_std"]/1e6:>7.3f} | '
              f'{r["avg_walk_length_mean"]:>7.2f} |')
    return 0


if __name__ == '__main__':
    sys.exit(main())

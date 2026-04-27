#!/usr/bin/env python3
"""
TEA+/Tempest picker × CPU/GPU comparison runner (bulk-only).

For each (dataset × edge_picker × mode) cell, drives
walk_sampling_speed_test N times and aggregates ingest- and walk-sampling
timings into mean ± std (after outlier rejection). Output is one flat
row per cell.

Purpose: produce the comparison row that backs Section
"Detailed Comparison with TEA+/TEA". CPU mode acts as the TEA+ proxy
(no GPU advantage); GPU mode is the production Tempest path. Same
binary, same data, same picker — only `use_gpu` flips.

Pickers swept (edge picker; start picker fixed to Uniform):
    ExponentialIndex, ExponentialWeight, Linear, TemporalNode2Vec

Datasets: growth, delicious. Both passed as positional args.

Modes: CPU and GPU. The kernel-launch-type is intrinsic to the
picker, not the mode:
  - ExponentialIndex / ExponentialWeight / Linear → NODE_GROUPED (always).
  - TemporalNode2Vec                              → FULL_WALK    (always).
The CPU codepath ignores kernel_launch_type (the std walker handles
all pickers), so the same KLT value is passed in both modes and the
CSV records the picker's intrinsic kernel choice. On GPU, NG's
dispatcher already bypasses to per_walk_step_kernel for Node2Vec, so
declaring FULL_WALK explicitly avoids the scheduler setup/teardown
overhead that NG would still pay around the per-walk kernel.

Walks: 1 walk per node, max walk length 80, undirected — matching the
TEA+ paper's experimental config (Table tab:comparison_with_tea).

Outlier rejection mirrors run_ablation.py: median-relative threshold
0.15, MIN_KEEP=3.

Usage:
    python3 run_picker_comparison.py <growth_csv> <delicious_csv> \\
        [--binary BUILD/BIN/walk_sampling_speed_test] \\
        [--output picker_comparison.csv] \\
        [--runs 5] \\
        [--timeout 1800]
"""
import argparse
import csv
import re
import statistics
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PICKERS = [
    'ExponentialIndex',
    'ExponentialWeight',
    'Linear',
    'TemporalNode2Vec',
]
START_PICKER = 'Uniform'
DATASETS     = ['growth', 'delicious']
MODES        = ['GPU', 'CPU']

# Picker -> kernel_launch_type. Mode-independent: the CPU codepath
# ignores klt (the std walker dispatches all pickers internally), so the
# CSV column reflects the picker's intrinsic kernel choice in both rows.
# Node2Vec routes to FULL_WALK because its prev_node-dependent CDF
# defeats the coop scheduler's panel sharing.
def kernel_launch_type(picker: str) -> str:
    if picker == 'TemporalNode2Vec':
        return 'FULL_WALK'
    return 'NODE_GROUPED'


# Match the TEA+ paper's experimental config.
NUM_WALKS_PER_NODE = 1
MAX_WALK_LEN       = 80
IS_DIRECTED        = 0      # undirected
NUM_TOTAL_WALKS    = 0      # ignored when num_walks_per_node != -1

DEFAULT_BIN     = './build/bin/walk_sampling_speed_test'
DEFAULT_OUTPUT  = 'picker_comparison.csv'
DEFAULT_RUNS    = 5
DEFAULT_TIMEOUT = 1800       # 30 min per run; CPU + Node2Vec can be slow.

# Outlier rejection (same algorithm as run_ablation.py).
OUTLIER_THRESHOLD = 0.15
MIN_KEEP          = 3

# ---------------------------------------------------------------------------
# Output parsers — keyed off walk_sampling_speed_test's parseable lines.
# ---------------------------------------------------------------------------
INGEST_RE = re.compile(r'^Ingest time:\s+([\d.eE+-]+)\s+seconds', re.MULTILINE)
WALK_RE   = re.compile(r'^Walk time:\s+([\d.eE+-]+)\s+seconds',   re.MULTILINE)
WPS_RE    = re.compile(r'^\s*Walks/sec:\s+([\d.eE+-]+)',          re.MULTILINE)
SPS_RE    = re.compile(r'^\s*Steps/sec:\s+([\d.eE+-]+)',          re.MULTILINE)
AVL_RE    = re.compile(r'^Average walk length:\s+([\d.eE+-]+)',   re.MULTILINE)


def reject_outliers(xs, threshold_frac=OUTLIER_THRESHOLD, min_keep=MIN_KEEP):
    """Iterative median-relative outlier rejection. See run_ablation.py."""
    kept = list(xs)
    while len(kept) > min_keep:
        med = statistics.median(kept)
        if med == 0:
            break
        devs = [abs(x - med) / med for x in kept]
        i = max(range(len(kept)), key=lambda j: devs[j])
        if devs[i] > threshold_frac:
            kept.pop(i)
        else:
            break
    return kept


def mean_std(xs):
    if not xs:
        return 0.0, 0.0
    mu = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return mu, sd


def invoke(binary, csv_path, use_gpu, picker, klt, timeout_sec):
    """Run walk_sampling_speed_test once; return parsed metrics dict."""
    cmd = [
        binary, csv_path,
        str(int(use_gpu)),
        str(IS_DIRECTED),
        str(NUM_TOTAL_WALKS),
        str(NUM_WALKS_PER_NODE),
        str(MAX_WALK_LEN),
        picker,
        START_PICKER,
        klt,
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        check=False, timeout=timeout_sec,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f'exit {proc.returncode}\n'
            f'stderr tail:\n{proc.stderr[-500:]}\n'
            f'stdout tail:\n{proc.stdout[-500:]}'
        )

    out = proc.stdout

    def grab(rx, name):
        m = rx.search(out)
        if not m:
            raise RuntimeError(
                f'missing "{name}" in stdout. Last 500 chars:\n{out[-500:]}'
            )
        return float(m.group(1))

    return {
        'ingest_s':         grab(INGEST_RE, 'Ingest time'),
        'walk_s':           grab(WALK_RE,   'Walk time'),
        'walks_per_sec':    grab(WPS_RE,    'Walks/sec'),
        'steps_per_sec':    grab(SPS_RE,    'Steps/sec'),
        'avg_walk_length':  grab(AVL_RE,    'Average walk length'),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('growth_csv',     help='Path to growth.csv')
    ap.add_argument('delicious_csv',  help='Path to delicious.csv')
    ap.add_argument('--binary',  default=DEFAULT_BIN,
                    help=f'Path to walk_sampling_speed_test '
                         f'(default: {DEFAULT_BIN})')
    ap.add_argument('--output',  default=DEFAULT_OUTPUT,
                    help=f'CSV output path (default: {DEFAULT_OUTPUT})')
    ap.add_argument('--runs',    type=int, default=DEFAULT_RUNS,
                    help=f'Runs per (dataset × picker × mode) cell '
                         f'(default: {DEFAULT_RUNS})')
    ap.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT,
                    help=f'Per-run timeout in seconds. CPU + Node2Vec on '
                         f'large graphs can be slow. (default: '
                         f'{DEFAULT_TIMEOUT})')
    args = ap.parse_args()

    if not Path(args.binary).is_file():
        ap.error(f'binary not found: {args.binary}')
    paths = {'growth': args.growth_csv, 'delicious': args.delicious_csv}
    for name, p in paths.items():
        if not Path(p).is_file():
            ap.error(f'{name} CSV not found: {p}')

    print(f'# binary  : {args.binary}')
    print(f'# runs    : {args.runs} per (dataset × picker × mode) cell')
    print(f'# timeout : {args.timeout}s per run')
    print(f'# output  : {args.output}')
    print(f'# config  : wpn={NUM_WALKS_PER_NODE}, mwl={MAX_WALK_LEN}, '
          f'undirected, start_picker={START_PICKER}')
    print()

    rows = []
    for ds in DATASETS:
        for picker in PICKERS:
            for mode in MODES:
                klt     = kernel_launch_type(picker)
                use_gpu = (mode == 'GPU')

                cell_runs = []
                tag = f'  {ds:>9} / {picker:<18} / {mode:<3} / {klt:<26}'
                for i in range(args.runs):
                    print(f'{tag} run {i+1}/{args.runs} ...',
                          end=' ', flush=True)
                    try:
                        r = invoke(args.binary, paths[ds],
                                   use_gpu, picker, klt, args.timeout)
                    except (RuntimeError, subprocess.TimeoutExpired) as e:
                        print(f'FAIL ({type(e).__name__}: {e})',
                              file=sys.stderr)
                        continue
                    print(
                        f'ingest={r["ingest_s"]:7.3f}s  '
                        f'walk={r["walk_s"]:7.3f}s  '
                        f'wps={r["walks_per_sec"]/1e3:8.1f}k  '
                        f'avg_len={r["avg_walk_length"]:5.2f}'
                    )
                    cell_runs.append(r)

                if not cell_runs:
                    print(f'  -> all {args.runs} runs failed; skipping cell',
                          file=sys.stderr)
                    continue

                ingest = [r['ingest_s']        for r in cell_runs]
                walk   = [r['walk_s']          for r in cell_runs]
                wps    = [r['walks_per_sec']   for r in cell_runs]
                sps    = [r['steps_per_sec']   for r in cell_runs]
                lens   = [r['avg_walk_length'] for r in cell_runs]

                ingest_kept = reject_outliers(ingest)
                walk_kept   = reject_outliers(walk)
                wps_kept    = reject_outliers(wps)
                sps_kept    = reject_outliers(sps)
                lens_kept   = reject_outliers(lens)

                im,  isd  = mean_std(ingest_kept)
                wm,  wsd  = mean_std(walk_kept)
                wpm, wpsd = mean_std(wps_kept)
                spm, spsd = mean_std(sps_kept)
                lm,  _    = mean_std(lens_kept)

                rows.append({
                    'dataset':              ds,
                    'picker':               picker,
                    'mode':                 mode,
                    'kernel_launch_type':   klt,
                    'start_picker':         START_PICKER,
                    'num_walks_per_node':   NUM_WALKS_PER_NODE,
                    'max_walk_length':      MAX_WALK_LEN,
                    'is_directed':          IS_DIRECTED,
                    'n_runs_raw':           len(cell_runs),
                    'n_runs_kept':          len(walk_kept),
                    'ingest_seconds_mean':  im,
                    'ingest_seconds_std':   isd,
                    'walk_seconds_mean':    wm,
                    'walk_seconds_std':     wsd,
                    'walks_per_sec_mean':   wpm,
                    'walks_per_sec_std':    wpsd,
                    'steps_per_sec_mean':   spm,
                    'steps_per_sec_std':    spsd,
                    'avg_walk_length_mean': lm,
                })

    if not rows:
        print('No successful cells; nothing to write.', file=sys.stderr)
        return 1

    out_path = Path(args.output)
    fields = list(rows[0].keys())
    with out_path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print('=' * 90)
    print('Summary (mean ± std across kept runs)')
    print('=' * 90)
    print(f'| {"dataset":<9} | {"picker":<18} | {"mode":<3} | '
          f'{"ingest s":>13} | {"walk s":>13} | {"M w/s":>8} | '
          f'{"avg_len":>7} |')
    print('|' + '|'.join('-' * w for w in
          [11, 20, 5, 15, 15, 10, 9]) + '|')
    for r in rows:
        ingest = f'{r["ingest_seconds_mean"]:6.3f}±{r["ingest_seconds_std"]:5.3f}'
        walk   = f'{r["walk_seconds_mean"]:6.3f}±{r["walk_seconds_std"]:5.3f}'
        print(f'| {r["dataset"]:<9} | {r["picker"]:<18} | {r["mode"]:<3} | '
              f'{ingest:>13} | {walk:>13} | '
              f'{r["walks_per_sec_mean"]/1e6:>8.3f} | '
              f'{r["avg_walk_length_mean"]:>7.2f} |')
    print()
    print(f'Wrote {len(rows)} rows to {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

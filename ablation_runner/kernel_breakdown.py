#!/usr/bin/env python3
"""
Kernel-firing breakdown across all 3 datasets for one NG variant.

Profiles ONE nsys run per dataset (`nsys profile --stats=true`), then
queries each resulting sqlite to count kernel launches inside
`walk_sampling_batch` NVTX ranges. Prints a per-dataset table and a
side-by-side summary.

Buckets reported:
  solo          — node_grouped_solo_kernel
  warp_smem     — node_grouped_warp_smem_kernel
  warp_global   — node_grouped_warp_global_kernel
  block_smem    — node_grouped_block_smem_kernel
  block_global  — node_grouped_block_global_kernel
  multi_block   — expand_block_tasks_kernel  (scheduler kernel that
                  runs once per step with block-tier tasks; mega-hub
                  splitting fires when W > W_THRESHOLD_MULTI_BLOCK
                  = 8192)
  start_edges   — pick_start_edges_kernel  (one launch per walk call)
  filter_alive, gather, partition_w, partition_g — scheduler stages

"other" gathers thrust/CUB internal kernels (sort, RLE, scan, etc.).

Usage:
  python3 kernel_breakdown.py coin.csv flight.csv delicious.csv \\
      --variant node_grouped --w 8 \\
      --nsys /its/home/ms2420/cuda-12.6/bin/nsys
"""
import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

from run_ablation import DATASETS, PRESETS, build_run_argv, parse_throughput


# Bucket name -> demangled-name substring. Order matters for substring
# matches: more specific buckets first.
BUCKETS = [
    ('solo',         'node_grouped_solo_kernel'),
    ('warp_smem',    'node_grouped_warp_smem_kernel'),
    ('warp_global',  'node_grouped_warp_global_kernel'),
    ('block_smem',   'node_grouped_block_smem_kernel'),
    ('block_global', 'node_grouped_block_global_kernel'),
    ('multi_block',  'expand_block_tasks_kernel'),
    ('start_edges',  'pick_start_edges_kernel'),
    ('filter_alive', 'walk_alive_flags_kernel'),
    ('gather',       'gather_last_nodes_kernel'),
    ('partition_w',  'partition_by_w_kernel'),
    ('partition_g',  'partition_by_g_kernel'),
]

DEFAULT_BIN       = './build/bin/ablation_streaming'
DEFAULT_NSYS      = 'nsys'
DEFAULT_VARIANT   = 'node_grouped'
DEFAULT_W         = 8
DEFAULT_BLOCK_DIM = 256


def profile_and_count(nsys_bin, binary, dataset_csv, variant, preset, w,
                      block_dim, rep_path):
    """Profile once, return (per-bucket dict, n NVTX ranges, throughput)."""
    wpn, nb, nw, mwl = preset
    rep_path = Path(rep_path)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [nsys_bin, 'profile', '--trace=cuda,nvtx', '--stats=true',
           '--force-overwrite=true',
           '--output', str(rep_path.with_suffix(''))] + \
          build_run_argv(binary, dataset_csv, variant,
                         wpn, nb, nw, mwl, block_dim, w)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f'nsys profile exit {proc.returncode}.\n'
            f'stderr tail:\n{proc.stderr[-500:]}')

    try:
        thr_t, sps_t, avg_len = parse_throughput(proc.stdout)
    except RuntimeError:
        thr_t = sps_t = avg_len = None

    sql_path = rep_path.with_suffix('.sqlite')
    if not sql_path.exists():
        raise RuntimeError(f'sqlite not produced at {sql_path}')

    db = sqlite3.connect(sql_path)
    try:
        ranges = db.execute(
            "SELECT start, end FROM NVTX_EVENTS "
            "WHERE text = 'walk_sampling_batch' ORDER BY start"
        ).fetchall()
        if not ranges:
            raise RuntimeError(f'no walk_sampling_batch NVTX in {sql_path}')

        where = ' OR '.join(f'(k.start>={s} AND k.end<={e})' for s, e in ranges)
        rows = db.execute(f"""
            SELECT COALESCE(d.value, m.value, sn.value, '?') AS name,
                   COUNT(*) AS n,
                   COALESCE(SUM(k.end - k.start), 0) AS total_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds d  ON d.id  = k.demangledName
            LEFT JOIN StringIds m  ON m.id  = k.mangledName
            LEFT JOIN StringIds sn ON sn.id = k.shortName
            WHERE {where}
            GROUP BY name
        """).fetchall()
    finally:
        db.close()

    counts = {b: {'n': 0, 'total_ns': 0} for b, _ in BUCKETS}
    counts['other'] = {'n': 0, 'total_ns': 0}
    for name, n, total_ns in rows:
        bucket = 'other'
        for b_name, b_pattern in BUCKETS:
            if b_pattern in name:
                bucket = b_name
                break
        counts[bucket]['n'] += n
        counts[bucket]['total_ns'] += total_ns

    return counts, len(ranges), {'thr': thr_t, 'sps': sps_t, 'avg_len': avg_len}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('coin_csv',      help='Path to ml_tgbl-coin.csv')
    ap.add_argument('flight_csv',    help='Path to ml_tgbl-flight.csv')
    ap.add_argument('delicious_csv', help='Path to delicious(_clip).csv')
    ap.add_argument('--variant', default=DEFAULT_VARIANT,
                    choices=['node_grouped', 'node_grouped_global_only',
                             'full_walk'])
    ap.add_argument('--w', type=int, default=DEFAULT_W,
                    help=f'w_threshold_warp (default: {DEFAULT_W})')
    ap.add_argument('--block-dim', type=int, default=DEFAULT_BLOCK_DIM)
    ap.add_argument('--binary',     default=DEFAULT_BIN)
    ap.add_argument('--nsys',       default=DEFAULT_NSYS)
    ap.add_argument('--rep-dir',    default='kernel_breakdown_nsys',
                    help='Directory for per-dataset .nsys-rep / .sqlite')
    args = ap.parse_args()

    paths = {'coin': args.coin_csv, 'flight': args.flight_csv,
             'delicious': args.delicious_csv}
    for name, p in paths.items():
        if not Path(p).is_file():
            ap.error(f'{name} CSV not found: {p}')
    if not Path(args.binary).is_file():
        ap.error(f'binary not found: {args.binary}')

    print(f'# variant   : {args.variant}')
    print(f'# W         : {args.w}')
    print(f'# block_dim : {args.block_dim}')
    print(f'# nsys      : {args.nsys}')
    print(f'# rep dir   : {args.rep_dir}')
    print()

    results = {}
    for ds in DATASETS:
        wpn, nb, nw, mwl = PRESETS[ds]
        print(f'=== {ds} (wpn={wpn} nb={nb} nw={nw} mwl={mwl}) ===')
        rep = Path(args.rep_dir) / f'{ds}_{args.variant}.nsys-rep'
        try:
            counts, n_ranges, thr = profile_and_count(
                args.nsys, args.binary, paths[ds], args.variant,
                PRESETS[ds], args.w, args.block_dim, rep)
        except RuntimeError as e:
            print(f'  FAIL: {e}', file=sys.stderr)
            continue
        results[ds] = (counts, n_ranges, thr)
        print(f'  walk_sampling_batch ranges (= timed batches): {n_ranges}')
        if thr['thr'] is not None:
            print(f'  throughput: {thr["thr"]/1e6:.2f} M w/s, '
                  f'{thr["sps"]/1e6:.2f} M s/s, avg_len={thr["avg_len"]:.2f}')
        print()

    if not results:
        print('No successful runs.', file=sys.stderr)
        return 1

    # Combined matrix: rows = buckets, columns = datasets.
    bucket_names = [b for b, _ in BUCKETS] + ['other']
    print('=' * 78)
    print('Kernel launches per bucket (inside walk_sampling_batch NVTX ranges)')
    print('=' * 78)
    print()

    # Header.
    head = f'{"bucket":<14}'
    for ds in DATASETS:
        if ds in results:
            head += f' | {ds:^21}'
    print(head)

    subhead = f'{"":<14}'
    for ds in DATASETS:
        if ds in results:
            subhead += f' | {"launches":>10} {"total ms":>10}'
    print(subhead)
    print('-' * len(head))

    for b in bucket_names:
        line = f'{b:<14}'
        for ds in DATASETS:
            if ds not in results:
                continue
            cell = results[ds][0][b]
            if cell['n'] == 0:
                line += f' | {0:>10} {"—":>10}'
            else:
                line += f' | {cell["n"]:>10} {cell["total_ns"]/1e6:>10.2f}'
        print(line)

    print()
    print('Notes:')
    print('  - "multi_block" = expand_block_tasks_kernel from the scheduler;')
    print('    runs once per step that had block-tier tasks. Mega-hub splitting')
    print('    fires when W > W_THRESHOLD_MULTI_BLOCK = 8192.')
    print('    This counts STEPS with block-tier work, not splits per se.')
    print('  - "other" = thrust/CUB internal kernels (sort, RLE, scan,')
    print('    partition, reduce, for_each). Not part of the pick/scheduler')
    print('    tier nomenclature.')
    print('  - "launches" is the kernel-launch count inside the timed walk')
    print('    NVTX range (excludes ingest, warmup, and stats-summary work).')
    return 0


if __name__ == '__main__':
    sys.exit(main())

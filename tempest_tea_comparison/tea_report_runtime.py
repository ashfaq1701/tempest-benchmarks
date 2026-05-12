#!/usr/bin/env python3
"""
TEA paper Table 4 reproduction — TEA-reimpl on growth + delicious for
all three biases.  Backs the "we reproduce the original TEA paper's
published numbers on the same datasets" line of the Tempest writeup.

Drives our CPU tea-reimpl binary (../../tea-reimpl/build/tea_walk).
Prints median wall time side-by-side with the paper's Table 4 column,
and the multiplier between them.  Every tunable parameter is a CLI
flag — pass --help for the full surface.  Paper-strict exp is
timescale_bound=-1 (raw exp(t_i) with cancellation, no rescale).
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (resolved relative to this script's location — no hardcoded /home).
# ---------------------------------------------------------------------------
HERE         = Path(__file__).resolve().parent
TEA_REIMPL   = HERE.parent.parent / 'tea-reimpl'
TEA_BIN      = TEA_REIMPL / 'build' / 'tea_walk'
ENV_DEFAULT  = HERE / '.env'

# TEA paper Table 4 — TEA in-memory column, wall time in seconds.
# Three-bias × two-dataset constants from the published paper.
PAPER_TABLE4 = {
    'growth':    {'linear': 0.56,  'exponential': 2.93,  'temporal_node2vec': 3.52},
    'delicious': {'linear': 7.98,  'exponential': 38.84, 'temporal_node2vec': 59.82},
}

# ---------------------------------------------------------------------------
# stdout parsers (match tea_walk's print_run_stats format)
# ---------------------------------------------------------------------------
WALK_TIME_RE = re.compile(r'^Walks done:\s+\d+\s+\(([\d.eE+-]+)\s+s\)', re.MULTILINE)
WPS_RE       = re.compile(r'^Throughput:\s+([\d.eE+-]+)\s+walks/sec',   re.MULTILINE)
SPS_RE       = re.compile(r'^Steps/sec:\s+([\d.eE+-]+)\s+steps/sec',    re.MULTILINE)
AVL_RE       = re.compile(r'^Final avg walk length:\s+([\d.eE+-]+)',    re.MULTILINE)


def load_env(path: Path) -> dict:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        k, _, v = line.partition('=')
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def grab(rx: re.Pattern, tag: str, text: str) -> float:
    m = rx.search(text)
    if not m:
        raise RuntimeError(
            f'missing "{tag}" in tea_walk stdout. Last 500 chars:\n{text[-500:]}')
    return float(m.group(1))


def run_tea(args, data_path: str, bias: str, variant: str) -> dict:
    """One tea_walk invocation; returns wall time + throughput metrics."""
    cmd = [
        str(TEA_BIN), data_path, bias, variant,
        str(int(args.is_directed)),
        str(args.walks_per_node),
        str(args.max_walk_len),
        str(args.timescale_bound),
    ]
    env = {**os.environ, 'OMP_NUM_THREADS': str(args.omp_threads)}
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f'tea_walk exit {proc.returncode}\nstderr tail:\n{proc.stderr[-500:]}')
    out = proc.stdout
    return {
        'walk_s':         grab(WALK_TIME_RE, 'Walks done',          out),
        'walks_per_sec':  grab(WPS_RE,       'Throughput',          out),
        'steps_per_sec':  grab(SPS_RE,       'Steps/sec',           out),
        'avg_walk_len':   grab(AVL_RE,       'Final avg walk length', out),
    }


def parse_variant_map(s: str) -> dict:
    """Parse a string like 'growth=tea_hpat,delicious=tea_pat'."""
    out = {}
    for kv in s.split(','):
        kv = kv.strip()
        if not kv:
            continue
        k, _, v = kv.partition('=')
        out[k.strip()] = v.strip()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--env', type=Path, default=ENV_DEFAULT,
                   help=f'.env file with dataset paths (default: {ENV_DEFAULT})')
    p.add_argument('--tea-binary', type=Path, default=TEA_BIN,
                   help=f'Path to tea_walk binary (default: {TEA_BIN})')
    p.add_argument('--walks-per-node', type=int, default=10,
                   help='Walks started from each active vertex (default: 10, '
                        'matching DeepWalk/CTDNE convention used in TEA paper §5.1)')
    p.add_argument('--max-walk-len', type=int, default=80,
                   help='Maximum walk length (default: 80, matching paper §5.1)')
    p.add_argument('--is-directed', type=int, default=1, choices=(0, 1),
                   help='Treat the graph as directed (default: 1)')
    p.add_argument('--timescale-bound', type=float, default=-1.0,
                   help='Exponential-bias timescale rescale. -1 = strict TEA '
                        '(raw exp(t_i) with cancellation, no rescale, paper §2.3 II). '
                        '>0 = rescale per-vertex time span to this magnitude. '
                        '(default: -1)')
    p.add_argument('--runs', type=int, default=3,
                   help='Number of runs per (dataset, bias) cell (default: 3)')
    p.add_argument('--omp-threads', type=int, default=os.cpu_count() or 16,
                   help='OMP_NUM_THREADS for tea_walk (default: nproc)')
    p.add_argument('--variants', type=parse_variant_map,
                   default='growth=tea_hpat,delicious=tea_pat',
                   help='Per-dataset TEA variant, comma-separated key=value pairs. '
                        'tea_pat for memory-bound graphs (delicious overflows HPAT '
                        'on a 24 GB laptop; paper §3.2 describes PAT as the fallback). '
                        "(default: 'growth=tea_hpat,delicious=tea_pat')")
    p.add_argument('--datasets', type=lambda s: tuple(d.strip() for d in s.split(',')),
                   default=('growth', 'delicious'),
                   help='Comma-separated datasets to run (default: growth,delicious — '
                        'paper Table 4 in-memory rows we have data for)')
    p.add_argument('--biases', type=lambda s: tuple(b.strip() for b in s.split(',')),
                   default=('linear', 'exponential', 'temporal_node2vec'),
                   help='Comma-separated biases to run '
                        '(default: linear,exponential,temporal_node2vec)')
    p.add_argument('--env-key-map',
                   type=parse_variant_map,
                   default='growth=GROWTH_PATH,delicious=DELICIOUS_PATH,'
                           'tgbl-comment=TGBL_COMMENT_PATH,tgbl-flight=TGBL_FLIGHT_PATH,'
                           'hub-synthetic=HUB_SYNTHETIC_PATH',
                   help='Map of dataset label → env-var key (default covers '
                        'all five known datasets)')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not Path(args.tea_binary).is_file():
        sys.stderr.write(f'ERROR: TEA binary not found at {args.tea_binary}\n'
                         f'  Build with: cd {TEA_REIMPL} && '
                         f'cmake --build build -j --target tea_walk\n')
        return 1
    if not args.env.is_file():
        sys.stderr.write(f'ERROR: env file not found at {args.env}\n')
        return 1
    env = load_env(args.env)

    # Resolve dataset paths from the env using the env-key-map.
    paths = {}
    for ds in args.datasets:
        env_key = args.env_key_map.get(ds)
        if env_key is None:
            sys.stderr.write(f'ERROR: no env-key mapped for dataset {ds!r}; '
                             f'add it via --env-key-map.\n')
            return 1
        p = env.get(env_key)
        if not p or not Path(p).is_file():
            sys.stderr.write(f'ERROR: {ds} CSV not found ({env_key}={p!r})\n')
            return 1
        paths[ds] = p

    print('=== TEA paper Table 4 reproduction ===')
    print(f'Binary    : {args.tea_binary}')
    print(f'Params    : wpn={args.walks_per_node}, max_walk_len={args.max_walk_len}, '
          f'timescale_bound={args.timescale_bound}, '
          f'OMP_NUM_THREADS={args.omp_threads}')
    print(f'Runs/cell : {args.runs}')
    print(f'Variants  : {args.variants}')
    print()

    results: dict = {ds: {} for ds in args.datasets}
    for ds in args.datasets:
        variant = args.variants.get(ds)
        if variant is None:
            sys.stderr.write(f'ERROR: no variant mapped for dataset {ds!r}; '
                             f'add it via --variants.\n')
            return 1
        for bias in args.biases:
            walk_times = []
            for run in range(1, args.runs + 1):
                print(f'  {ds:9s} / {bias:18s} / {variant} / run {run}/{args.runs} ...',
                      end=' ', flush=True)
                try:
                    r = run_tea(args, paths[ds], bias, variant)
                except RuntimeError as e:
                    print(f'FAIL ({e})')
                    continue
                walk_times.append(r['walk_s'])
                print(f"walk={r['walk_s']:6.2f}s  "
                      f"sps={r['steps_per_sec']/1e6:5.2f}M  "
                      f"avg_len={r['avg_walk_len']:.2f}")
            results[ds][bias] = walk_times

    # ----- Side-by-side report (only for datasets that have paper rows) -----
    print()
    print('=' * 78)
    print('Paper Table 4 (TEA, in-memory) vs ours — wall time in seconds')
    print('=' * 78)
    print(f'| {"dataset":<10} | {"bias":<18} | '
          f'{"paper":>7} | {"r1":>7} | {"r2":>7} | {"r3":>7} | '
          f'{"median":>7} | {"ratio":>7} |')
    print('|' + '|'.join('-' * w for w in [12, 20, 9, 9, 9, 9, 9, 9]) + '|')
    for ds in args.datasets:
        for bias in args.biases:
            paper = PAPER_TABLE4.get(ds, {}).get(bias)
            runs  = results[ds].get(bias, [])
            r_strs = [f'{x:7.2f}' for x in runs] + ['    n/a'] * (args.runs - len(runs))
            r_strs = r_strs[:3]  # only first 3 columns in the report
            median  = statistics.median(runs) if runs else float('nan')
            paper_s = f'{paper:7.2f}' if paper is not None else '    n/a'
            if runs and paper:
                ratio   = median / paper
                ratio_s = f'{ratio:6.2f}×'
            else:
                ratio_s = '    n/a'
            print(f'| {ds:<10} | {bias:<18} | '
                  f'{paper_s:>7} | {r_strs[0]:>7} | {r_strs[1]:>7} | {r_strs[2]:>7} | '
                  f'{median:>7.2f} | {ratio_s:>7} |')
    print()
    print('  paper  = TEA paper Table 4 (their hardware: 2× Xeon E5-2640 v2,')
    print('           16 cores total, 94 GB DRAM).')
    print('  ratio  = ours_median / paper.  >1 means ours is slower; <1 faster.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

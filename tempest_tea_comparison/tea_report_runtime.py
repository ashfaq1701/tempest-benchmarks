#!/usr/bin/env python3
"""
TEA paper Table 4 reproduction — TEA-reimpl on growth + delicious for all
three biases.

This is a head-to-head comparison of *runtime* against the original TEA
paper's published numbers.  Since the original TEA source isn't
publicly available, the only way to make the comparison defensible is
to run our reimplementation under the paper's stated experimental
config — exactly.  So the paper-defining parameters are hardcoded:

    walks_per_node  = 10        (DeepWalk/CTDNE default, paper §5.1)
    max_walk_len    = 80 for growth, 20 for delicious
                                (paper §5.1 default is 80, but delicious
                                 has 33.8M vertices: a 33.8M × 10 × 80 ×
                                 16 B output buffer is ≈ 433 GB, which
                                 OOMs any single-node box.  Walks on
                                 delicious die at avg length ≈ 2.16
                                 under exp-bias and ≈ 2.16 under linear
                                 anyway — mwl=20 is more than enough
                                 to never truncate a real walk, and the
                                 output-buffer footprint drops to ~108 GB
                                 which fits a server.)
    timescale_bound = -1        (paper §2.3.II Eq 3: pure exp(t_i) with
                                 the t_cur cancellation, no rescale;
                                 overridable via --timescale-bound for
                                 sensitivity studies — note that any
                                 positive value rescales the per-vertex
                                 time range and shifts the exp/node2vec
                                 distribution off the paper's spec, so
                                 the ratio column for those two biases
                                 stops being a strict reproduction.
                                 Linear is unaffected by this knob.)
    is_directed     = 1         (paper §2.1 walk model is directed)
    omp_threads     = 16        (paper §5.1: 2× Xeon E5-2640 v2 = 16 cores;
                                 overridable via --omp-threads for runs on
                                 boxes with a different core count — note
                                 that doing so moves the comparison off the
                                 paper's hardware and the "ratio" column
                                 becomes a cross-hardware figure rather
                                 than a like-for-like reproduction.)

The remaining paper-config params have no CLI override — overriding
them would invalidate the comparison.  CLI surfaces: --runs (measurement
methodology), --env (env location), --omp-threads (hardware match),
--timescale-bound (exp/node2vec distribution).

Datasets, variants and biases are fixed: growth + delicious, the two
in-memory rows of paper Table 4 that the laptop can run end-to-end.
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE        = Path(__file__).resolve().parent
TEA_REIMPL  = HERE.parent.parent / 'tea-reimpl'
TEA_BIN     = TEA_REIMPL / 'build' / 'tea_walk'
ENV_DEFAULT = HERE / '.env'

# ---------------------------------------------------------------------------
# TEA paper §5.1 experimental config — hardcoded by design (see module docstring).
# OMP thread count and timescale_bound are overridable via CLI; defaults
# match the paper (16 cores, no rescale).
# ---------------------------------------------------------------------------
TIMESCALE_BOUND_DEFAULT = -1.0
IS_DIRECTED             = 1
OMP_THREADS_DEFAULT     = 16

# Datasets: the two in-memory rows of paper Table 4 the laptop can run.
# Each entry is (label, env-key, tea_variant, walks_per_node, max_walk_len).
#
# tea_pat is the memory-bound fallback for delicious (paper §3.2 describes
# PAT as the fallback when HPAT's footprint overflows DRAM).
#
# growth: paper-default wpn=10, mwl=80.  Its avg walk ≈ 4.3 under linear
# and 3.1 under exp, so 80 is plenty of headroom; wpn=10 is the
# DeepWalk/CTDNE default that the TEA paper §5.1 reports.
#
# delicious: wpn=10, mwl=20.  wpn=10 stays at the paper default; mwl=20
# is the only deviation, dictated by the output-buffer footprint.
# Walks die at avg length ≈ 2.16 under every bias, so mwl=20 never
# truncates a real walk.  walks_out buffer scales as num_walks ×
# max_walk_len × sizeof(NodeStep); 33.78M vertices × wpn=10 × mwl=20 ×
# 16 B ≈ 108 GB, fits a server-class box (the paper Xeon had 94 GB
# DRAM — needs swap or a beefier server).
DATASETS = (
    # (label, env-key, tea_variant, walks_per_node, max_walk_len)
    ('growth',    'GROWTH_PATH',    'tea_hpat', 10, 80),
    ('delicious', 'DELICIOUS_PATH', 'tea_pat',  10, 20),
)

# Bias name → tea_walk picker string.
BIASES = ('linear', 'exponential', 'temporal_node2vec')

# Paper Table 4, TEA in-memory column — wall time in seconds.
PAPER_TABLE4 = {
    'growth':    {'linear': 0.56, 'exponential': 2.93,  'temporal_node2vec': 3.52},
    'delicious': {'linear': 7.98, 'exponential': 38.84, 'temporal_node2vec': 59.82},
}

# ---------------------------------------------------------------------------
# stdout parsers (match tea_walk's print_run_stats format)
# ---------------------------------------------------------------------------
WALK_TIME_RE = re.compile(r'^Walks done:\s+\d+\s+\(([\d.eE+-]+)\s+s\)',  re.MULTILINE)
SPS_RE       = re.compile(r'^Steps/sec:\s+([\d.eE+-]+)\s+steps/sec',     re.MULTILINE)
AVL_RE       = re.compile(r'^Final avg walk length:\s+([\d.eE+-]+)',     re.MULTILINE)


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
        raise RuntimeError(f'missing "{tag}" in tea_walk stdout. Last 500 chars:\n{text[-500:]}')
    return float(m.group(1))


def run_tea(data_path: str, bias: str, variant: str,
            walks_per_node: int, max_walk_len: int,
            timescale_bound: float, omp_threads: int) -> dict:
    cmd = [
        str(TEA_BIN), data_path, bias, variant,
        str(IS_DIRECTED),
        str(walks_per_node),
        str(max_walk_len),
        str(timescale_bound),
    ]
    env = {**os.environ, 'OMP_NUM_THREADS': str(omp_threads)}
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f'tea_walk exit {proc.returncode}\nstderr tail:\n{proc.stderr[-500:]}')
    return {
        'walk_s':        grab(WALK_TIME_RE, 'Walks done',            proc.stdout),
        'steps_per_sec': grab(SPS_RE,       'Steps/sec',             proc.stdout),
        'avg_walk_len':  grab(AVL_RE,       'Final avg walk length', proc.stdout),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--env', type=Path, default=ENV_DEFAULT,
                    help=f'.env with dataset paths (default: {ENV_DEFAULT})')
    ap.add_argument('--runs', type=int, default=3,
                    help='runs per (dataset, bias) cell — stability measurement, '
                         'not a paper-config knob (default: 3)')
    ap.add_argument('--omp-threads', type=int, default=OMP_THREADS_DEFAULT,
                    help=f'OMP_NUM_THREADS for tea_walk. Default '
                         f'{OMP_THREADS_DEFAULT} matches the TEA paper §5.1 '
                         f'hardware (2× Xeon E5-2640 v2 = 16 cores). Override '
                         f'when the run host has a different core count; '
                         f'doing so makes the "ratio" column a cross-hardware '
                         f'figure rather than a like-for-like reproduction.')
    ap.add_argument('--timescale-bound', type=float,
                    default=TIMESCALE_BOUND_DEFAULT,
                    help=f'exp-bias rescale passed to tea_walk. Default '
                         f'{TIMESCALE_BOUND_DEFAULT} = paper §2.3.II Eq 3 '
                         f'(pure exp(t_i), no rescale). Any positive value '
                         f'rescales the per-vertex time range and shifts the '
                         f'exp/node2vec distribution off paper spec — the '
                         f'ratio column for those biases stops being a strict '
                         f'reproduction (linear is unaffected).')
    args = ap.parse_args()
    if args.omp_threads <= 0:
        sys.stderr.write(f'ERROR: --omp-threads must be > 0 '
                         f'(got {args.omp_threads})\n')
        return 1

    if not TEA_BIN.is_file():
        sys.stderr.write(f'ERROR: TEA binary not found at {TEA_BIN}\n')
        return 1
    if not args.env.is_file():
        sys.stderr.write(f'ERROR: env file not found at {args.env}\n')
        return 1
    env = load_env(args.env)

    omp_threads     = args.omp_threads
    timescale_bound = args.timescale_bound
    omp_note = '' if omp_threads == OMP_THREADS_DEFAULT \
                  else f' (overridden; paper default = {OMP_THREADS_DEFAULT})'
    ts_note  = '' if timescale_bound == TIMESCALE_BOUND_DEFAULT \
                  else f' (overridden; paper default = {TIMESCALE_BOUND_DEFAULT})'

    print('=== TEA paper Table 4 reproduction ===')
    print(f'Binary    : {TEA_BIN}')
    print(f'Fixed     : timescale_bound={timescale_bound}{ts_note}, '
          f'is_directed={IS_DIRECTED}, '
          f'OMP_NUM_THREADS={omp_threads}{omp_note}')
    print(f'Per-ds    : ' + ', '.join(
        f'{ds}(wpn={wpn},mwl={mwl})' for ds, _, _, wpn, mwl in DATASETS))
    print(f'Runs/cell : {args.runs}')
    print()

    results: dict = {ds: {} for ds, _, _, _, _ in DATASETS}
    for ds, env_key, variant, wpn, mwl in DATASETS:
        data_path = env.get(env_key)
        if not data_path or not Path(data_path).is_file():
            sys.stderr.write(f'ERROR: {ds} CSV not found ({env_key}={data_path!r})\n')
            return 1
        for bias in BIASES:
            walk_times = []
            for run in range(1, args.runs + 1):
                print(f'  {ds:9s} / {bias:18s} / {variant} / '
                      f'wpn={wpn:<2} mwl={mwl:<3} / '
                      f'run {run}/{args.runs} ...',
                      end=' ', flush=True)
                try:
                    r = run_tea(data_path, bias, variant, wpn, mwl,
                                timescale_bound, omp_threads)
                except RuntimeError as e:
                    print(f'FAIL ({e})')
                    continue
                walk_times.append(r['walk_s'])
                print(f"walk={r['walk_s']:6.2f}s  "
                      f"sps={r['steps_per_sec']/1e6:5.2f}M  "
                      f"avg_len={r['avg_walk_len']:.2f}")
            results[ds][bias] = walk_times

    # ----- Side-by-side report -----
    print()
    print('=' * 78)
    print('Paper Table 4 (TEA, in-memory) vs ours — wall time in seconds')
    print('=' * 78)
    print(f'| {"dataset":<10} | {"bias":<18} | '
          f'{"paper":>7} | {"r1":>7} | {"r2":>7} | {"r3":>7} | '
          f'{"median":>7} | {"ratio":>7} |')
    print('|' + '|'.join('-' * w for w in [12, 20, 9, 9, 9, 9, 9, 9]) + '|')
    for ds, _, _, _, _ in DATASETS:
        for bias in BIASES:
            paper  = PAPER_TABLE4[ds][bias]
            runs   = results[ds].get(bias, [])
            r_strs = [f'{x:7.2f}' for x in runs] + ['    n/a'] * 3
            median = statistics.median(runs) if runs else float('nan')
            ratio  = (median / paper) if runs else float('nan')
            ratio_s = f'{ratio:6.2f}×' if runs else '    n/a'
            print(f'| {ds:<10} | {bias:<18} | '
                  f'{paper:>7.2f} | {r_strs[0]:>7} | {r_strs[1]:>7} | {r_strs[2]:>7} | '
                  f'{median:>7.2f} | {ratio_s:>7} |')
    print()
    print('  paper  = TEA paper Table 4 (their hardware: 2× Xeon E5-2640 v2,')
    print('           16 cores total, 94 GB DRAM).')
    print('  ratio  = ours_median / paper.  >1 means ours is slower; <1 faster.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

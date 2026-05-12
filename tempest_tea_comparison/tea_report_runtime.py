#!/usr/bin/env python3
"""
TEA paper Table 4 reproduction — growth + delicious, 3 biases × 3 runs.

Backs the "we reproduce the original TEA paper's published numbers on
the same datasets" line of the Tempest writeup.  Drives our CPU
tea-reimpl binary (../../tea-reimpl/build/tea_walk) with the paper's
default walk parameters (wpn=10, max_walk_len=80, strict-paper exp
with `timescale_bound=-1` — no rescale), prints the median wall time
side-by-side with the paper's Table 4 column, and the multiplier
between them.
"""
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
ENV_PATH    = HERE / '.env'

# ---------------------------------------------------------------------------
# Configuration (paper §5.1: walk_length=80, DeepWalk default wpn=10)
# ---------------------------------------------------------------------------
WALKS_PER_NODE   = 10
MAX_WALK_LEN     = 80
IS_DIRECTED      = 1
TIMESCALE_BOUND  = -1            # strict paper TEA: exp(t_i − t_max_u), no rescale
RUNS_PER_CELL    = 3
OMP_THREADS      = '16'

# TEA paper Table 4 — TEA in-memory column, wall time in seconds.
# (Linear, Exponential, TemporalNode2Vec)
PAPER_TABLE4 = {
    'growth':    {'linear': 0.56,  'exponential': 2.93,  'temporal_node2vec': 3.52},
    'delicious': {'linear': 7.98,  'exponential': 38.84, 'temporal_node2vec': 59.82},
}

# tea_hpat fits growth comfortably on a 24 GB-free laptop; delicious
# (33.8M nodes, max_D=4.4M) overflows the HPAT+aux memory budget and
# must fall back to tea_pat — the paper itself describes PAT as the
# memory-bound fallback (§3.2 last paragraph).
VARIANT = {
    'growth':    'tea_hpat',
    'delicious': 'tea_pat',
}

# ---------------------------------------------------------------------------
# stdout parsers (match tea_walk's print_run_stats format)
# ---------------------------------------------------------------------------
WALK_TIME_RE = re.compile(r'^Walks done:\s+\d+\s+\(([\d.eE+-]+)\s+s\)', re.MULTILINE)
WPS_RE       = re.compile(r'^Throughput:\s+([\d.eE+-]+)\s+walks/sec',   re.MULTILINE)
SPS_RE       = re.compile(r'^Steps/sec:\s+([\d.eE+-]+)\s+steps/sec',    re.MULTILINE)
AVL_RE       = re.compile(r'^Final avg walk length:\s+([\d.eE+-]+)',    re.MULTILINE)


def load_env(path: Path) -> dict:
    """Parse a simple KEY=VALUE .env file."""
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


def run_tea(data_path: str, bias: str, variant: str) -> dict:
    """One tea_walk invocation; returns wall time + throughput metrics."""
    cmd = [
        str(TEA_BIN), data_path, bias, variant,
        str(IS_DIRECTED), str(WALKS_PER_NODE),
        str(MAX_WALK_LEN), str(TIMESCALE_BOUND),
    ]
    env = {**os.environ, 'OMP_NUM_THREADS': OMP_THREADS}
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


def main() -> int:
    if not TEA_BIN.is_file():
        sys.stderr.write(f'ERROR: TEA binary not found at {TEA_BIN}\n'
                         f'  Build with: cd {TEA_REIMPL} && '
                         f'cmake --build build -j --target tea_walk\n')
        return 1
    if not ENV_PATH.is_file():
        sys.stderr.write(f'ERROR: env file not found at {ENV_PATH}\n')
        return 1

    env = load_env(ENV_PATH)
    paths = {
        'growth':    env.get('GROWTH_PATH'),
        'delicious': env.get('DELICIOUS_PATH'),
    }
    for ds, p in paths.items():
        if not p or not Path(p).is_file():
            sys.stderr.write(f'ERROR: {ds} CSV not found in env ({p!r})\n')
            return 1

    print('=== TEA paper Table 4 reproduction ===')
    print(f'Binary    : {TEA_BIN}')
    print(f'Params    : wpn={WALKS_PER_NODE}, max_walk_len={MAX_WALK_LEN}, '
          f'timescale_bound={TIMESCALE_BOUND}, OMP_NUM_THREADS={OMP_THREADS}')
    print(f'Runs/cell : {RUNS_PER_CELL}')
    print(f'Variant   : per-dataset → {VARIANT}')
    print()

    results = {ds: {} for ds in paths}
    for ds, csv_path in paths.items():
        variant = VARIANT[ds]
        for bias in ('linear', 'exponential', 'temporal_node2vec'):
            walk_times = []
            for run in range(1, RUNS_PER_CELL + 1):
                print(f'  {ds:9s} / {bias:18s} / {variant} / run {run}/{RUNS_PER_CELL} ...',
                      end=' ', flush=True)
                try:
                    r = run_tea(csv_path, bias, variant)
                except RuntimeError as e:
                    print(f'FAIL ({e})')
                    continue
                walk_times.append(r['walk_s'])
                print(f"walk={r['walk_s']:6.2f}s  "
                      f"sps={r['steps_per_sec']/1e6:5.2f}M  "
                      f"avg_len={r['avg_walk_len']:.2f}")
            results[ds][bias] = walk_times

    # ----- Final side-by-side report -----
    print()
    print('=' * 78)
    print('Paper Table 4 (TEA, in-memory) vs ours — wall time in seconds')
    print('=' * 78)
    header = (f'| {"dataset":<10} | {"bias":<18} | '
              f'{"paper":>7} | {"ours r1":>7} | {"r2":>7} | {"r3":>7} | '
              f'{"median":>7} | {"ratio":>7} |')
    print(header)
    print('|' + '|'.join('-' * w for w in [12, 20, 9, 9, 9, 9, 9, 9]) + '|')
    for ds in ('growth', 'delicious'):
        for bias in ('linear', 'exponential', 'temporal_node2vec'):
            paper = PAPER_TABLE4[ds][bias]
            runs  = results[ds].get(bias, [])
            r_strs = [f'{x:7.2f}' for x in runs] + ['    n/a'] * (3 - len(runs))
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

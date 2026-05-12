#!/usr/bin/env python3
"""
Tempest (GPU) vs TEA-reimpl (CPU) side-by-side comparison across four
datasets and three biases.

Runs both engines on the same data with matching walk parameters,
collects steps/sec across multiple repeats, and prints mean ± std per
(dataset, bias, engine) cell.  Tempest is invoked in bulk (single
ingest, single walk pass — no streaming batches/windows).

Tempest binary  : ../../temporal-random-walk/build/bin/walk_sampling_speed_test
TEA binary      : ../../tea-reimpl/build/tea_walk
Datasets        : growth, delicious, ml_tgbl-comment, ml_tgbl-flight (.env)
Biases          : linear, exponential, temporal_node2vec
Engines         : Tempest (GPU, NODE_GROUPED), TEA (CPU, tea_hpat/tea_pat)
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
TRW         = HERE.parent.parent / 'temporal-random-walk'
TEA_BIN     = TEA_REIMPL / 'build' / 'tea_walk'

# walk_sampling_speed_test landing point depends on which build dir
# was used.  Look at the standard build/ first; fall back to
# cmake-build-debug (CLion default, despite the name it builds Release).
TEMPEST_BIN_CANDIDATES = [
    TRW / 'build' / 'bin' / 'walk_sampling_speed_test',
    TRW / 'cmake-build-debug' / 'bin' / 'walk_sampling_speed_test',
]

ENV_PATH = HERE / '.env'

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Tempest's walk_sampling_speed_test hardcodes timescale_bound=34; match
# it on TEA so both engines sample from the same exp distribution.
WALKS_PER_NODE   = 1                 # wpn=1 fits an 8 GB laptop GPU
MAX_WALK_LEN     = 80
IS_DIRECTED      = 1
TIMESCALE_BOUND  = 34                # Tempest's hardcoded default
RUNS_PER_CELL    = 3
OMP_THREADS      = '16'

# Tempest bulk-mode knobs.  walk_sampling_speed_test does single-ingest /
# single-walk-pass by construction, so "batches=1, window=1" is the
# semantic — no streaming, no sliding window.
TEMPEST_START_PICKER = 'ExponentialWeight'
TEMPEST_KLT          = 'NODE_GROUPED'

# bias-name → (tempest picker, tea picker) mapping.
BIAS_PICKERS = [
    ('linear',            'Linear',            'linear'),
    ('exponential',       'ExponentialWeight', 'exponential'),
    ('temporal_node2vec', 'TemporalNode2Vec',  'temporal_node2vec'),
]

DATASETS = [
    # (label, env-key, tea_variant)
    # delicious's HPAT footprint overflows a 24 GB laptop; the paper
    # §3.2 itself describes PAT as the memory-bound fallback.
    ('growth',         'GROWTH_PATH',         'tea_hpat'),
    ('delicious',      'DELICIOUS_PATH',      'tea_pat'),
    ('tgbl-comment',   'TGBL_COMMENT_PATH',   'tea_hpat'),
    ('tgbl-flight',    'TGBL_FLIGHT_PATH',    'tea_hpat'),
    ('hub-synthetic',  'HUB_SYNTHETIC_PATH',  'tea_hpat'),
]

# ---------------------------------------------------------------------------
# stdout parsers
# ---------------------------------------------------------------------------
# tea_walk
TEA_SPS_RE   = re.compile(r'^Steps/sec:\s+([\d.eE+-]+)\s+steps/sec', re.MULTILINE)
TEA_WPS_RE   = re.compile(r'^Throughput:\s+([\d.eE+-]+)\s+walks/sec', re.MULTILINE)
TEA_AVL_RE   = re.compile(r'^Final avg walk length:\s+([\d.eE+-]+)', re.MULTILINE)
# walk_sampling_speed_test
TEMP_SPS_RE  = re.compile(r'^\s*Steps/sec:\s+([\d.eE+-]+)', re.MULTILINE)
TEMP_WPS_RE  = re.compile(r'^\s*Walks/sec:\s+([\d.eE+-]+)', re.MULTILINE)
TEMP_AVL_RE  = re.compile(r'^Average walk length:\s+([\d.eE+-]+)', re.MULTILINE)


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
            f'missing "{tag}" in stdout. Last 500 chars:\n{text[-500:]}')
    return float(m.group(1))


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def run_tempest(binary: Path, data_path: str, tempest_picker: str) -> dict:
    """One walk_sampling_speed_test invocation (GPU, bulk)."""
    cmd = [
        str(binary), data_path,
        '1',                          # use_gpu
        str(IS_DIRECTED),             # is_directed
        '-1',                         # num_total_walks (ignored when wpn != -1)
        str(WALKS_PER_NODE),
        str(MAX_WALK_LEN),
        tempest_picker,               # edge_picker
        TEMPEST_START_PICKER,
        TEMPEST_KLT,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f'tempest exit {proc.returncode}\nstderr tail:\n{proc.stderr[-500:]}')
    out = proc.stdout
    return {
        'steps_per_sec': grab(TEMP_SPS_RE, 'Steps/sec',  out),
        'walks_per_sec': grab(TEMP_WPS_RE, 'Walks/sec',  out),
        'avg_walk_len':  grab(TEMP_AVL_RE, 'Average walk length', out),
    }


def run_tea(data_path: str, bias: str, variant: str) -> dict:
    """One tea_walk invocation (CPU)."""
    cmd = [
        str(TEA_BIN), data_path, bias, variant,
        str(IS_DIRECTED), str(WALKS_PER_NODE),
        str(MAX_WALK_LEN), str(TIMESCALE_BOUND),
    ]
    env = {**os.environ, 'OMP_NUM_THREADS': OMP_THREADS}
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f'tea exit {proc.returncode}\nstderr tail:\n{proc.stderr[-500:]}')
    out = proc.stdout
    return {
        'steps_per_sec': grab(TEA_SPS_RE, 'Steps/sec',             out),
        'walks_per_sec': grab(TEA_WPS_RE, 'Throughput',            out),
        'avg_walk_len':  grab(TEA_AVL_RE, 'Final avg walk length', out),
    }


def repeat(label_prefix: str, runner, *args) -> list[float]:
    """Call `runner(*args)` RUNS_PER_CELL times; return list of steps/sec.
    Failures are logged and skipped (the cell still reports the kept runs)."""
    sps = []
    for i in range(1, RUNS_PER_CELL + 1):
        print(f'    {label_prefix} run {i}/{RUNS_PER_CELL} ...', end=' ', flush=True)
        try:
            r = runner(*args)
        except RuntimeError as e:
            short = str(e).splitlines()[0]
            print(f'FAIL ({short})')
            continue
        sps.append(r['steps_per_sec'])
        print(f"sps={r['steps_per_sec']/1e6:6.2f}M  "
              f"avg_len={r['avg_walk_len']:.2f}")
    return sps


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float('nan'), float('nan')
    mu = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return mu, sd


def main() -> int:
    tempest_bin = first_existing(TEMPEST_BIN_CANDIDATES)
    if tempest_bin is None:
        sys.stderr.write('ERROR: Tempest walk_sampling_speed_test binary not found.\n'
                         f'  Looked at:\n')
        for p in TEMPEST_BIN_CANDIDATES:
            sys.stderr.write(f'    {p}\n')
        return 1
    if not TEA_BIN.is_file():
        sys.stderr.write(f'ERROR: TEA binary not found at {TEA_BIN}\n')
        return 1
    if not ENV_PATH.is_file():
        sys.stderr.write(f'ERROR: env file not found at {ENV_PATH}\n')
        return 1
    env = load_env(ENV_PATH)

    print('=== Tempest (GPU) vs TEA-reimpl (CPU) — bulk steps/sec ===')
    print(f'Tempest bin   : {tempest_bin}')
    print(f'TEA bin       : {TEA_BIN}')
    print(f'Params        : wpn={WALKS_PER_NODE}, max_walk_len={MAX_WALK_LEN}, '
          f'timescale_bound={TIMESCALE_BOUND}, OMP_NUM_THREADS={OMP_THREADS}')
    print(f'Runs/cell     : {RUNS_PER_CELL}')
    print(f'Tempest mode  : GPU, {TEMPEST_KLT}, start_picker={TEMPEST_START_PICKER} '
          f'(bulk: batches=1, window=1)')
    print()

    # results[ds_label][bias_name] = {'tempest': [...], 'tea': [...]}
    results: dict = {}

    for ds_label, env_key, tea_variant in DATASETS:
        data_path = env.get(env_key)
        if not data_path or not Path(data_path).is_file():
            print(f'  [skip] {ds_label}: dataset path missing ({env_key}={data_path!r})')
            results[ds_label] = None
            continue
        print(f'--- {ds_label}  ({data_path})  TEA variant: {tea_variant} ---')
        results[ds_label] = {}

        for bias_name, tempest_picker, tea_picker in BIAS_PICKERS:
            print(f'  bias={bias_name}')
            tempest_sps = repeat('Tempest', run_tempest, tempest_bin,
                                 data_path, tempest_picker)
            tea_sps     = repeat('TEA    ', run_tea,
                                 data_path, tea_picker, tea_variant)
            results[ds_label][bias_name] = {'tempest': tempest_sps, 'tea': tea_sps}
        print()

    # ----- Side-by-side summary -----
    print('=' * 100)
    print('Steps/sec (×10⁶) — mean ± std across kept runs')
    print('=' * 100)
    print(f'| {"dataset":<13} | {"bias":<18} | '
          f'{"Tempest (GPU)":>16} | {"TEA (CPU)":>16} | {"speedup":>10} |')
    print('|' + '|'.join('-' * w for w in [15, 20, 18, 18, 12]) + '|')

    for ds_label, _, _ in DATASETS:
        cell = results.get(ds_label)
        if cell is None:
            print(f'| {ds_label:<13} | {"(dataset missing)":<18} | '
                  f'{"":>16} | {"":>16} | {"":>10} |')
            continue
        for bias_name, _, _ in BIAS_PICKERS:
            tempest_sps = cell[bias_name]['tempest']
            tea_sps     = cell[bias_name]['tea']
            t_mu, t_sd  = mean_std(tempest_sps)
            a_mu, a_sd  = mean_std(tea_sps)

            def fmt(mu: float, sd: float) -> str:
                if mu != mu:                            # NaN
                    return '   n/a'
                return f'{mu/1e6:6.2f} ± {sd/1e6:5.2f}'

            if t_mu == t_mu and a_mu == a_mu and t_mu > 0:
                ratio = a_mu / t_mu
                ratio_s = f'{ratio:5.2f}× TEA' if ratio >= 1.0 else f'{1.0/ratio:5.2f}× Tem'
            else:
                ratio_s = '       n/a'
            print(f'| {ds_label:<13} | {bias_name:<18} | '
                  f'{fmt(t_mu, t_sd):>16} | {fmt(a_mu, a_sd):>16} | {ratio_s:>10} |')
    print()
    print('  speedup column reads "x.xx× TEA" when TEA is faster,')
    print('                       "x.xx× Tem" when Tempest is faster.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Tempest (GPU) vs TEA-reimpl (CPU) side-by-side comparison across the
five datasets registered in .env (growth, delicious, tgbl-comment,
tgbl-flight, hub-synthetic), three biases each (linear, exponential,
temporal_node2vec), with per-dataset walk presets.

Both engines are driven in bulk (single ingest, single walk pass — no
streaming batches/windows; Tempest's ablation_runner nb/nw knobs are
streaming-only and don't apply here).  Every tunable parameter is a
CLI flag — pass --help to see them.

Tempest binary  : ../../temporal-random-walk/build/bin/walk_sampling_speed_test
                  (also looks at ../cmake-build-debug/bin/ as a fallback,
                  since CLion default builds there).
TEA binary      : ../../tea-reimpl/build/tea_walk
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
TRW         = HERE.parent.parent / 'temporal-random-walk'
TEA_BIN     = TEA_REIMPL / 'build' / 'tea_walk'

TEMPEST_BIN_CANDIDATES = [
    TRW / 'build' / 'bin' / 'walk_sampling_speed_test',
    TRW / 'cmake-build-debug' / 'bin' / 'walk_sampling_speed_test',
]
ENV_DEFAULT = HERE / '.env'

# ---------------------------------------------------------------------------
# Per-dataset walk presets — single source of truth, mirrors the
# ablation_runner constants.py 4-tuple shape (wpn, num_batches, num_windows,
# max_walk_len).  The bulk walk_sampling_speed_test ignores num_batches /
# num_windows (those are streaming-only knobs in ablation_streaming); we
# carry them for transparency / cross-script consistency.
#
# growth/delicious values follow the ablation_runner conventions for the
# corresponding TEA paper Table 4 rows; tgbl-* follow ablation_runner's
# 4-dataset preset row from constants.py; hub-synthetic uses the laptop
# config recommended in temporal-random-walk/synthetic_data_generator/
# README.txt (wpn=50 / mwl=80 for an 8 GB GPU; the README's A40 recipe is
# wpn=500 / mwl=100, which OOMs a laptop).
# ---------------------------------------------------------------------------
DATASET_PRESETS_DEFAULT = {
    # (wpn, num_batches, num_windows, max_walk_len)
    'growth':         (10,  1,  1, 80),
    'delicious':      ( 8, 30, 13, 10),
    'tgbl-comment':   (20,  4,  1, 80),
    'tgbl-flight':    (20,  4,  1, 80),
    'hub-synthetic':  (50,  1,  1, 80),
}

# label → env-var key.  Centralized so the script doesn't carry hardcoded
# paths anywhere — every dataset is resolved through .env.
ENV_KEY_MAP_DEFAULT = {
    'growth':         'GROWTH_PATH',
    'delicious':      'DELICIOUS_PATH',
    'tgbl-comment':   'TGBL_COMMENT_PATH',
    'tgbl-flight':    'TGBL_FLIGHT_PATH',
    'hub-synthetic':  'HUB_SYNTHETIC_PATH',
}

# TEA variant per dataset.  tea_pat is the memory-bound fallback the paper
# §3.2 describes for graphs whose HPAT+aux footprint overflows DRAM
# (delicious on a 24 GB laptop).
TEA_VARIANT_DEFAULT = {
    'growth':         'tea_hpat',
    'delicious':      'tea_pat',
    'tgbl-comment':   'tea_hpat',
    'tgbl-flight':    'tea_hpat',
    'hub-synthetic':  'tea_hpat',
}

# (display label, Tempest picker enum string, TEA picker string)
BIAS_PICKERS = [
    ('linear',            'Linear',            'linear'),
    ('exponential',       'ExponentialWeight', 'exponential'),
    ('temporal_node2vec', 'TemporalNode2Vec',  'temporal_node2vec'),
]

# ---------------------------------------------------------------------------
# stdout parsers
# ---------------------------------------------------------------------------
TEA_SPS_RE  = re.compile(r'^Steps/sec:\s+([\d.eE+-]+)\s+steps/sec',     re.MULTILINE)
TEA_WPS_RE  = re.compile(r'^Throughput:\s+([\d.eE+-]+)\s+walks/sec',    re.MULTILINE)
TEA_AVL_RE  = re.compile(r'^Final avg walk length:\s+([\d.eE+-]+)',     re.MULTILINE)
TEMP_SPS_RE = re.compile(r'^\s*Steps/sec:\s+([\d.eE+-]+)',              re.MULTILINE)
TEMP_WPS_RE = re.compile(r'^\s*Walks/sec:\s+([\d.eE+-]+)',              re.MULTILINE)
TEMP_AVL_RE = re.compile(r'^Average walk length:\s+([\d.eE+-]+)',       re.MULTILINE)


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


def first_existing(paths):
    for p in paths:
        if p.is_file():
            return p
    return None


def run_tempest(args, binary, data_path, tempest_picker, wpn, mwl):
    """One walk_sampling_speed_test invocation (GPU, bulk)."""
    cmd = [
        str(binary), data_path,
        '1',                                       # use_gpu
        str(int(args.is_directed)),
        '-1',                                      # num_total_walks (ignored when wpn != -1)
        str(wpn),
        str(mwl),
        tempest_picker,
        args.tempest_start_picker,
        args.tempest_klt,
        '',                                        # walk_dump_file (empty = skip)
        str(args.timescale_bound),
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


def run_tea(args, data_path, bias, variant, wpn, mwl):
    """One tea_walk invocation (CPU)."""
    cmd = [
        str(args.tea_binary), data_path, bias, variant,
        str(int(args.is_directed)),
        str(wpn),
        str(mwl),
        str(args.timescale_bound),
    ]
    env = {**os.environ, 'OMP_NUM_THREADS': str(args.omp_threads)}
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


def repeat(args, label_prefix, runner, *runner_args):
    sps = []
    for i in range(1, args.runs + 1):
        print(f'    {label_prefix} run {i}/{args.runs} ...', end=' ', flush=True)
        try:
            r = runner(*runner_args)
        except RuntimeError as e:
            short = str(e).splitlines()[0]
            print(f'FAIL ({short})')
            continue
        sps.append(r['steps_per_sec'])
        print(f"sps={r['steps_per_sec']/1e6:6.2f}M  "
              f"avg_len={r['avg_walk_len']:.2f}")
    return sps


def mean_std(xs):
    if not xs:
        return float('nan'), float('nan')
    mu = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return mu, sd


def parse_preset_override(s: str):
    """Parse 'growth=10,80 delicious=8,10' into {'growth': (10, 80), ...}.

    Only wpn,mwl pairs — nb/nw aren't tunable for the bulk binary.
    """
    out = {}
    for entry in re.split(r'[ ,;]+\s*(?=\w+=)|\s+', s.strip()):
        entry = entry.strip()
        if not entry:
            continue
        k, _, v = entry.partition('=')
        parts = [p.strip() for p in v.split(',')]
        if len(parts) < 2:
            raise argparse.ArgumentTypeError(
                f'preset override {entry!r} needs wpn,mwl (got {v!r})')
        out[k.strip()] = (int(parts[0]), int(parts[1]))
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--env', type=Path, default=ENV_DEFAULT,
                   help=f'.env file with dataset paths (default: {ENV_DEFAULT})')
    p.add_argument('--tea-binary', type=Path, default=TEA_BIN,
                   help=f'Path to tea_walk binary (default: {TEA_BIN})')
    p.add_argument('--tempest-binary', type=Path,
                   default=None,
                   help='Path to walk_sampling_speed_test (default: tries '
                        f'{TEMPEST_BIN_CANDIDATES[0]} then '
                        f'{TEMPEST_BIN_CANDIDATES[1]})')
    p.add_argument('--datasets', type=lambda s: tuple(d.strip() for d in s.split(',')),
                   default=tuple(DATASET_PRESETS_DEFAULT.keys()),
                   help='Comma-separated datasets to run (default: all five)')
    p.add_argument('--biases',
                   type=lambda s: tuple(b.strip() for b in s.split(',')),
                   default=tuple(name for name, _, _ in BIAS_PICKERS),
                   help='Comma-separated biases to run '
                        '(default: linear,exponential,temporal_node2vec)')
    p.add_argument('--runs', type=int, default=3,
                   help='Number of runs per (dataset, bias, engine) cell '
                        '(default: 3)')
    p.add_argument('--is-directed', type=int, default=1, choices=(0, 1),
                   help='Treat graphs as directed (default: 1)')
    p.add_argument('--timescale-bound', type=float, default=-1.0,
                   help='Exponential-bias timescale rescale, passed to both '
                        'engines so they sample from the same distribution. '
                        "-1 = strict-paper TEA (raw exp(t_i) with cancellation). "
                        '(default: -1)')
    p.add_argument('--omp-threads', type=int, default=os.cpu_count() or 16,
                   help='OMP_NUM_THREADS for tea_walk (default: nproc)')
    p.add_argument('--tempest-start-picker', default='ExponentialWeight',
                   help='Tempest first-hop picker (default: ExponentialWeight, '
                        "the walk_sampling_speed_test default)")
    p.add_argument('--tempest-klt', default='NODE_GROUPED',
                   choices=('NODE_GROUPED', 'FULL_WALK', 'NODE_GROUPED_GLOBAL_ONLY'),
                   help='Tempest kernel launch type (default: NODE_GROUPED)')
    p.add_argument('--preset-override', type=parse_preset_override, default={},
                   help='Override per-dataset (wpn,mwl) presets.  '
                        "Example: 'growth=10,80 delicious=8,10'.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not Path(args.tea_binary).is_file():
        sys.stderr.write(f'ERROR: TEA binary not found at {args.tea_binary}\n')
        return 1
    tempest_bin = (
        Path(args.tempest_binary)
        if args.tempest_binary
        else first_existing(TEMPEST_BIN_CANDIDATES)
    )
    if tempest_bin is None or not tempest_bin.is_file():
        sys.stderr.write('ERROR: Tempest walk_sampling_speed_test binary not found\n')
        return 1
    if not args.env.is_file():
        sys.stderr.write(f'ERROR: env file not found at {args.env}\n')
        return 1
    env = load_env(args.env)

    # Resolve presets (default + overrides) per dataset.
    def preset_for(ds: str):
        wpn0, nb, nw, mwl0 = DATASET_PRESETS_DEFAULT[ds]
        if ds in args.preset_override:
            wpn0, mwl0 = args.preset_override[ds]
        return wpn0, nb, nw, mwl0

    print('=== Tempest (GPU) vs TEA-reimpl (CPU) — bulk steps/sec ===')
    print(f'Tempest bin   : {tempest_bin}')
    print(f'TEA bin       : {args.tea_binary}')
    print(f'timescale_bound : {args.timescale_bound}   '
          f'OMP_NUM_THREADS: {args.omp_threads}   runs/cell: {args.runs}')
    print(f'Tempest mode  : GPU, {args.tempest_klt}, '
          f'start_picker={args.tempest_start_picker} (bulk: batches=1, window=1)')
    print('Per-dataset presets (wpn / mwl):')
    for ds in args.datasets:
        wpn, _, _, mwl = preset_for(ds)
        print(f'  {ds:<14} wpn={wpn:>3}  mwl={mwl:>3}  '
              f'variant={TEA_VARIANT_DEFAULT[ds]}')
    print()

    # results[ds][bias] = {'tempest': [...sps...], 'tea': [...sps...]}
    results: dict = {}

    for ds in args.datasets:
        env_key = ENV_KEY_MAP_DEFAULT.get(ds)
        if env_key is None:
            print(f'  [skip] {ds}: no env-key mapping')
            results[ds] = None
            continue
        data_path = env.get(env_key)
        if not data_path or not Path(data_path).is_file():
            print(f'  [skip] {ds}: dataset missing ({env_key}={data_path!r})')
            results[ds] = None
            continue
        wpn, _nb, _nw, mwl = preset_for(ds)
        variant = TEA_VARIANT_DEFAULT[ds]
        print(f'--- {ds}  ({data_path})  wpn={wpn} mwl={mwl} variant={variant} ---')
        results[ds] = {}

        for bias_name, tempest_picker, tea_picker in BIAS_PICKERS:
            if bias_name not in args.biases:
                continue
            print(f'  bias={bias_name}')
            tempest_sps = repeat(args, 'Tempest', run_tempest,
                                 args, tempest_bin, data_path, tempest_picker, wpn, mwl)
            tea_sps     = repeat(args, 'TEA    ', run_tea,
                                 args, data_path, tea_picker, variant, wpn, mwl)
            results[ds][bias_name] = {'tempest': tempest_sps, 'tea': tea_sps}
        print()

    # ----- Side-by-side summary -----
    print('=' * 110)
    print('Steps/sec (×10⁶) — mean ± std across kept runs')
    print('=' * 110)
    print(f'| {"dataset":<14} | {"bias":<18} | '
          f'{"Tempest (GPU)":>16} | {"TEA (CPU)":>16} | {"speedup":>14} |')
    print('|' + '|'.join('-' * w for w in [16, 20, 18, 18, 16]) + '|')

    for ds in args.datasets:
        cell = results.get(ds)
        if cell is None:
            print(f'| {ds:<14} | {"(skipped)":<18} |                  |                  |               |')
            continue
        for bias_name, _, _ in BIAS_PICKERS:
            if bias_name not in args.biases:
                continue
            entry = cell.get(bias_name)
            if entry is None:
                continue
            t_mu, t_sd = mean_std(entry['tempest'])
            a_mu, a_sd = mean_std(entry['tea'])

            def fmt(mu: float, sd: float) -> str:
                if mu != mu:  # NaN
                    return '   n/a'
                return f'{mu/1e6:6.2f} ± {sd/1e6:5.2f}'

            if t_mu == t_mu and a_mu == a_mu and t_mu > 0:
                ratio = a_mu / t_mu
                ratio_s = (f'{ratio:5.2f}× TEA' if ratio >= 1.0
                           else f'{1.0/ratio:5.2f}× Tem')
            else:
                ratio_s = '       n/a'
            print(f'| {ds:<14} | {bias_name:<18} | '
                  f'{fmt(t_mu, t_sd):>16} | {fmt(a_mu, a_sd):>16} | {ratio_s:>14} |')
    print()
    print('  speedup column reads "x.xx× TEA" when TEA is faster,')
    print('                       "x.xx× Tem" when Tempest is faster.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

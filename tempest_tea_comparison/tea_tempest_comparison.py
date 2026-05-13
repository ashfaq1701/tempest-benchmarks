#!/usr/bin/env python3
"""
Tempest (GPU) vs TEA-reimpl (CPU) side-by-side bulk comparison.

Both engines run FORWARD-IN-TIME walks so they sample from the same
distribution that the TEA paper defines in §2.1 (Γ_t(u) = {t_i > t_prev},
out-edges, t_i strictly increasing along the path).  TEA-reimpl is
forward-only by design; Tempest's walk_sampling_speed_test is invoked
with walk_direction=Forward_In_Time so the two engines walk in the same
temporal direction over the same candidate edge set.

Runs both engines on the five datasets registered in .env (growth,
delicious, tgbl-comment, tgbl-flight, hub-synthetic) across three
biases (linear, exponential, temporal_node2vec), with per-dataset
walks-per-node and max-walk-length presets, reports mean ± std of
steps/sec.

Tempest runs in bulk via walk_sampling_speed_test (single ingest,
single walk pass — the binary has no streaming concept; nb/nw=1 is
implicit).  TEA-reimpl runs via tea_walk with tea_hpat by default,
falling back to tea_pat for delicious whose HPAT footprint overflows
a 24 GB laptop (paper §3.2 describes PAT as the memory-bound fallback).

Tunables on the CLI:
    --env path                .env file with dataset paths
    --runs N                  runs per (dataset, bias, engine) cell
    --timescale-bound X       exp-bias rescale, passed to both engines
                              (default 100; keeps both engines' exp picker
                              well-conditioned on raw unix timestamps)
    --omp-threads N           OMP_NUM_THREADS for tea_walk

Hardcoded module constants (edit this file to change them):
    PRESETS                   per-dataset (walks_per_node, max_walk_len)
    ENV_KEY                   per-dataset env-var name in .env
    TEA_VARIANT               per-dataset tea_walk variant (hpat / pat)
    BIAS_PICKERS              (display, Tempest enum, TEA enum) per bias
    IS_DIRECTED               per-dataset {ds: 0|1}.  growth, delicious,
                              tgbl-* are directed temporal interactions
                              (user→item, sender→receiver).  hub-synthetic
                              is undirected by the generator's design
                              (synthetic_data_generator/README.txt).  Both
                              engines get the same per-dataset value, so
                              the comparison stays internally fair.
    TEMPEST_START_PICKER      walk_sampling_speed_test's default
    TEMPEST_KLTS              ['NODE_GROUPED'] — Tempest's headline
                              scheduler; the FULL_WALK per-walk dispatch
                              is no longer benchmarked here.
    TEMPEST_WALK_DIRECTION    Forward_In_Time — matches TEA-reimpl's
                              hardwired forward-walk convention (paper
                              §2.1: t_i strictly increasing along the
                              path) so the two engines sample the same
                              distribution.
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
# Per-dataset (walks_per_node, max_walk_len).  Mirrors the working-set
# sizes used by ablation_runner; delicious is a small-walk-many-batches
# config (its HPAT footprint dominates so we run shorter walks), the
# rest run paper-default mwl=80.  hub-synthetic uses the laptop config
# from temporal-random-walk/synthetic_data_generator/README.txt (the
# A40 recipe wpn=500/mwl=100 OOMs an 8 GB GPU).
# ---------------------------------------------------------------------------
PRESETS = {
    #                wpn  mwl
    'growth':         (20, 80),
    'delicious':      ( 4, 10),   # max that fits within working-set budget
    'tgbl-comment':   (20, 80),
    'tgbl-flight':    (20, 80),
    # hub-synthetic: README's A40-recommended config (the dataset is sized
    # for A40; laptop overflows at this wpn and would need (50, 80)).
    # synthetic_data_generator/README.txt §"Walk config".
    'hub-synthetic':  (500, 100),
}

# label → env-var key.
ENV_KEY = {
    'growth':         'GROWTH_PATH',
    'delicious':      'DELICIOUS_PATH',
    'tgbl-comment':   'TGBL_COMMENT_PATH',
    'tgbl-flight':    'TGBL_FLIGHT_PATH',
    'hub-synthetic':  'HUB_SYNTHETIC_PATH',
}

# TEA variant per dataset.  tea_pat is the memory-bound fallback.
TEA_VARIANT = {
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

# Fixed across all runs.
#
# IS_DIRECTED: the TEA paper §2.1 just defines G = (V, E, R) without
# specifying directionality.  Tempest's ablation_runner uses IS_DIRECTED=0
# by convention, but the datasets we run are all directed temporal
# interactions (growth/delicious: user→item actions; tgbl-*: directed
# temporal events).  Both engines get the same value either way, so the
# comparison is internally fair; we pick 1 here to match the actual
# data semantics and to stay consistent with tea_report_runtime.py
# (which also uses 1 so the laptop figures across both scripts come
# from the same edge interpretation).
# Per-dataset directionality.  growth, delicious, tgbl-* are directed
# temporal interactions (user→item, sender→receiver).  hub-synthetic is
# undirected by construction — the generator emits each hub-hub edge
# without an implied direction, and the README's bench_synthetic harness
# treats it that way.  Both engines get the same value per dataset so the
# comparison stays internally fair.
IS_DIRECTED = {
    'growth':         1,
    'delicious':      1,
    'tgbl-comment':   1,
    'tgbl-flight':    1,
    'hub-synthetic':  0,
}
TEMPEST_START_PICKER   = 'ExponentialWeight'     # walk_sampling_speed_test default
# Tempest is benchmarked under its headline NODE_GROUPED scheduler only.
# The simpler FULL_WALK per-walk dispatch is no longer reported here.
TEMPEST_KLTS           = ['NODE_GROUPED']
# Walks go FORWARD-IN-TIME in both engines.  TEA-reimpl is forward-only
# (paper §2.1: Γ_t(u) = {t_i > t_prev}, out-edges, monotone-increasing
# along the path).  Tempest is told to run forward via this CLI arg so
# both engines sample from the same candidate-set definition and the
# steps/sec comparison is apples-to-apples on the same algorithm.
TEMPEST_WALK_DIRECTION = 'Forward_In_Time'

# ---------------------------------------------------------------------------
# stdout parsers
# ---------------------------------------------------------------------------
TEA_SPS_RE  = re.compile(r'^Steps/sec:\s+([\d.eE+-]+)\s+steps/sec',     re.MULTILINE)
TEA_AVL_RE  = re.compile(r'^Final avg walk length:\s+([\d.eE+-]+)',     re.MULTILINE)
TEMP_SPS_RE = re.compile(r'^\s*Steps/sec:\s+([\d.eE+-]+)',              re.MULTILINE)
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
        raise RuntimeError(f'missing "{tag}" in stdout. Last 500 chars:\n{text[-500:]}')
    return float(m.group(1))


def first_existing(paths):
    return next((p for p in paths if p.is_file()), None)


def run_tempest(tempest_bin, data_path, picker, wpn, mwl, timescale, klt,
                is_directed):
    cmd = [
        str(tempest_bin), data_path,
        '1',                     # use_gpu
        str(is_directed),
        '-1',                    # num_total_walks (ignored when wpn != -1)
        str(wpn), str(mwl),
        picker, TEMPEST_START_PICKER, klt,
        '',                      # walk_dump_file (empty = skip)
        str(timescale),
        TEMPEST_WALK_DIRECTION,  # match TEA-reimpl's forward-only convention
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f'tempest exit {proc.returncode}\nstderr tail:\n{proc.stderr[-500:]}')
    return {
        'steps_per_sec': grab(TEMP_SPS_RE, 'Steps/sec',           proc.stdout),
        'avg_walk_len':  grab(TEMP_AVL_RE, 'Average walk length', proc.stdout),
    }


def run_tea(tea_bin, data_path, picker, variant, wpn, mwl, timescale,
            omp_threads, is_directed):
    cmd = [
        str(tea_bin), data_path, picker, variant,
        str(is_directed),
        str(wpn), str(mwl), str(timescale),
    ]
    env = {**os.environ, 'OMP_NUM_THREADS': str(omp_threads)}
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f'tea exit {proc.returncode}\nstderr tail:\n{proc.stderr[-500:]}')
    return {
        'steps_per_sec': grab(TEA_SPS_RE, 'Steps/sec',             proc.stdout),
        'avg_walk_len':  grab(TEA_AVL_RE, 'Final avg walk length', proc.stdout),
    }


def repeat(runs, label, runner, *runner_args):
    sps = []
    for i in range(1, runs + 1):
        print(f'    {label} run {i}/{runs} ...', end=' ', flush=True)
        try:
            r = runner(*runner_args)
        except RuntimeError as e:
            print(f'FAIL ({str(e).splitlines()[0]})')
            continue
        sps.append(r['steps_per_sec'])
        print(f"sps={r['steps_per_sec']/1e6:6.2f}M  avg_len={r['avg_walk_len']:.2f}")
    return sps


def mean_std(xs):
    if not xs:
        return float('nan'), float('nan')
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def fmt_cell(mu: float, sd: float, kept: int, total: int) -> str:
    """Format a summary cell as 'mean ± std (kept/total)'.  When kept < total
    the n-of-N suffix is shown so partial-run cells are visually flagged."""
    if mu != mu:  # NaN
        return f'    n/a ({kept}/{total})'
    base = f'{mu/1e6:6.2f} ± {sd/1e6:5.2f}'
    return base if kept == total else f'{base} ({kept}/{total})'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--env', type=Path, default=ENV_DEFAULT,
                    help=f'.env with dataset paths (default: {ENV_DEFAULT})')
    ap.add_argument('--runs', type=int, default=3,
                    help='runs per (dataset, bias, engine) cell (default: 3)')
    ap.add_argument('--timescale-bound', type=float, default=100.0,
                    help='exp-bias rescale passed to both engines '
                         '(default: 100).  Keeps the exp picker numerically '
                         'well-conditioned on raw unix timestamps: TEA '
                         'underflows old edges and Tempest overflows recent '
                         'edges at timescale_bound=-1, so the two engines '
                         'diverge.  Any sane positive value (~30-100) aligns '
                         'them; 100 leaves headroom for high-span datasets.')
    ap.add_argument('--omp-threads', type=int, default=os.cpu_count() or 16,
                    help='OMP_NUM_THREADS for tea_walk (default: nproc)')
    args = ap.parse_args()

    if not TEA_BIN.is_file():
        sys.stderr.write(f'ERROR: TEA binary not found at {TEA_BIN}\n')
        return 1
    tempest_bin = first_existing(TEMPEST_BIN_CANDIDATES)
    if tempest_bin is None:
        sys.stderr.write('ERROR: Tempest walk_sampling_speed_test not found\n')
        return 1
    if not args.env.is_file():
        sys.stderr.write(f'ERROR: env file not found at {args.env}\n')
        return 1
    env = load_env(args.env)

    print('=== Tempest (GPU) vs TEA-reimpl (CPU) — bulk steps/sec ===')
    print(f'Tempest bin   : {tempest_bin}')
    print(f'TEA bin       : {TEA_BIN}')
    print(f'timescale_bound : {args.timescale_bound}   '
          f'OMP_NUM_THREADS: {args.omp_threads}   runs/cell: {args.runs}')
    print(f'Tempest KLT   : {", ".join(TEMPEST_KLTS)}')
    print(f'Tempest start picker: {TEMPEST_START_PICKER}')
    print(f'walk direction: {TEMPEST_WALK_DIRECTION} (TEA-reimpl is forward-only)')
    print(f'is_directed   : per-dataset — '
          + ', '.join(f'{ds}={d}' for ds, d in IS_DIRECTED.items()))
    print()

    # results[ds][bias_name] = (tempest_sps_by_klt, tea_sps)
    #   tempest_sps_by_klt: {klt: [sps,...]}
    #   tea_sps          : [sps,...]
    results: dict = {}
    for ds, (wpn, mwl) in PRESETS.items():
        data_path = env.get(ENV_KEY[ds])
        if not data_path or not Path(data_path).is_file():
            print(f'  [skip] {ds}: {ENV_KEY[ds]} missing or path invalid ({data_path!r})')
            results[ds] = None
            continue
        variant     = TEA_VARIANT[ds]
        is_directed = IS_DIRECTED[ds]
        print(f'--- {ds}  ({data_path})  wpn={wpn} mwl={mwl} '
              f'variant={variant} is_directed={is_directed} ---')
        results[ds] = {}
        for bias_name, tempest_picker, tea_picker in BIAS_PICKERS:
            print(f'  bias={bias_name}')
            tempest_sps_by_klt = {}
            for klt in TEMPEST_KLTS:
                tempest_sps_by_klt[klt] = repeat(
                    args.runs, f'Tempest [{klt:<12}]', run_tempest,
                    tempest_bin, data_path, tempest_picker,
                    wpn, mwl, args.timescale_bound, klt, is_directed)
            a_sps = repeat(args.runs, 'TEA    ', run_tea,
                           TEA_BIN, data_path, tea_picker, variant,
                           wpn, mwl, args.timescale_bound, args.omp_threads,
                           is_directed)
            results[ds][bias_name] = (tempest_sps_by_klt, a_sps)
        print()

    # ----- One summary table per Tempest KLT -----
    def render_table(klt: str):
        print('=' * 110)
        print(f'Tempest [{klt}] vs TEA-reimpl — steps/sec (×10⁶), mean ± std  '
              '(n/N suffix when partial cell)')
        print('=' * 110)
        print(f'| {"dataset":<14} | {"bias":<18} | '
              f'{"Tempest (GPU)":>22} | {"TEA (CPU)":>22} | {"speedup":>14} |')
        print('|' + '|'.join('-' * w for w in [16, 20, 24, 24, 16]) + '|')
        for ds in PRESETS:
            cell = results.get(ds)
            if cell is None:
                print(f'| {ds:<14} | {"(skipped)":<18} |                        '
                      f'|                        |                |')
                continue
            for bias_name, _, _ in BIAS_PICKERS:
                tempest_sps_by_klt, a_sps = cell[bias_name]
                t_sps = tempest_sps_by_klt.get(klt, [])
                t_mu, t_sd = mean_std(t_sps)
                a_mu, a_sd = mean_std(a_sps)
                if t_mu == t_mu and a_mu == a_mu and t_mu > 0:
                    ratio = a_mu / t_mu
                    ratio_s = (f'{ratio:5.2f}× TEA' if ratio >= 1.0
                               else f'{1.0/ratio:5.2f}× Tem')
                else:
                    ratio_s = '         n/a'
                print(f'| {ds:<14} | {bias_name:<18} | '
                      f'{fmt_cell(t_mu, t_sd, len(t_sps), args.runs):>22} | '
                      f'{fmt_cell(a_mu, a_sd, len(a_sps), args.runs):>22} | '
                      f'{ratio_s:>14} |')
        print()
        print('  "x.xx× TEA" = TEA faster; "x.xx× Tem" = Tempest faster.')
        print()

    for klt in TEMPEST_KLTS:
        render_table(klt)
    return 0


if __name__ == '__main__':
    sys.exit(main())

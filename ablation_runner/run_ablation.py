#!/usr/bin/env python3
"""
Server-side ablation runner: drives ablation_streaming across
(dataset × kernel_launch_type × run) and writes per-run metrics to CSV.

Optimization target: **steps/sec** (work actually done across all walks).
Not walks/sec — a length-1 walk that did nothing still counts as one walk,
so walks/sec is fooled by bugs that drop walks. steps/sec captures the
total computational work and is monotone in real throughput.

Three-phase design:
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
  Phase 3 (nsys)       — NSYS_RUNS_PER_CELL runs per (dataset, variant) under
                         nsys profile (cuda+nvtx trace; no gpu-metrics — that
                         requires root). Each run's .nsys-rep is exported to
                         SQLite and we query for kernel/NVTX activity inside
                         the `walk_sampling_batch` ranges to extract:
                           - kern_time_per_call_ms     (sum kernel duration / N batches)
                           - kern_launches_per_call    (# kernel launches / N batches)
                           - per_kernel_us             (sum kern dur / sum kern count)
                           - gpu_active_frac           (sum kern dur / sum NVTX dur)
                         Reject outliers per metric (same rule as Phase 2).

Datasets:  coin, flight, delicious — each with its own (wpn, nb, nw, mwl)
           preset sized for an A40 (48 GB) class GPU.
Variants:  full_walk, node_grouped, node_grouped_global_only.

Outputs (CSV; --output BASE produces BASE_tuning.csv and BASE_final.csv):
  - <base>_tuning.csv   : one row per Phase-1 invocation
                          (phase, dataset, variant, W, run, metrics, config)
  - <base>_final.csv    : one row per (dataset, variant) summarising Phase 2
                          throughput stats and Phase 3 nsys-derived metrics

Usage (from outside the build directory):
  python3 run_ablation.py coin.csv flight.csv delicious.csv \\
      --output ablation_results --block-dim 256

Default --binary is ./build/bin/ablation_streaming, --nsys is `nsys`.
"""
import argparse
import csv
import re
import sqlite3
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
DEFAULT_NSYS         = 'nsys'

TUNE_W_VALUES        = [1, 2, 4, 8, 16, 32, 64]
TUNE_RUNS_PER_W      = 5
FINAL_RUNS_THR       = 5
NSYS_RUNS_PER_CELL   = 3

# Outlier rejection: from N samples, drop the value(s) whose deviation from
# the (current) median exceeds OUTLIER_THRESHOLD * median, iteratively.
# Stops when worst remaining is within threshold OR len(kept) == MIN_KEEP.
# 0.20 catches the obvious 50%+ flukes (e.g. 4.8 vs cluster ~12) and the
# moderate ~25% dips, while preserving normal 5% run-to-run noise.
OUTLIER_THRESHOLD    = 0.20
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


def invoke_nsys(nsys_bin, rep_path, binary, data, klt, wpn, nb, nw, mwl,
                block_dim, w_threshold_warp):
    """Run the binary under nsys profile; returns (rep_path, walks/s, steps/s, avg_len)."""
    cmd = [nsys_bin, 'profile', '--trace=cuda,nvtx',
           '--force-overwrite=true', '--output', str(rep_path)] + \
          build_run_argv(binary, data, klt, wpn, nb, nw, mwl,
                         block_dim, w_threshold_warp)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f'nsys exit {proc.returncode}. stderr tail:\n{proc.stderr[-500:]}')
    t, s, a = parse_throughput(proc.stdout)
    return rep_path, t, s, a


def export_to_sqlite(nsys_bin, rep_path):
    """nsys export -t sqlite ... → returns path to .sqlite next to .nsys-rep."""
    rep_path = Path(rep_path)
    sql_path = rep_path.with_suffix('.sqlite')
    if (not sql_path.exists()) or (sql_path.stat().st_mtime < rep_path.stat().st_mtime):
        proc = subprocess.run(
            [nsys_bin, 'export', '-t', 'sqlite',
             '--force-overwrite=true', str(rep_path)],
            capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f'nsys export exit {proc.returncode}. stderr tail:\n{proc.stderr[-500:]}')
    return sql_path


_NVTX_TABLE_CANDIDATES = (
    'NVTX_EVENTS',          # nsys <= 2024.x (and most current Linux installs)
    'NVTX_PUSHPOP_EVENTS',  # newer nsys schema variant
    'NSYS_EVENTS_NVTX_PUSHPOP',
)


def _find_nvtx_table(db):
    """Return whichever NVTX-events table the SQLite export contains.
    nsys versions differ in table naming. Raises if none is found."""
    rows = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    have = {r[0] for r in rows}
    for c in _NVTX_TABLE_CANDIDATES:
        if c in have:
            return c
    nvtx_like = sorted(t for t in have if 'NVTX' in t.upper())
    raise RuntimeError(
        'no NVTX events table in SQLite export — checked '
        f'{list(_NVTX_TABLE_CANDIDATES)}. Tables matching NVTX: '
        f'{nvtx_like or "(none)"}. Confirm nsys was invoked with '
        '`--trace=cuda,nvtx` and that the binary links libnvToolsExt.')


def _extract_walk(db):
    """Walk-side metrics from `walk_sampling_batch` NVTX ranges."""
    nvtx_tbl = _find_nvtx_table(db)
    ranges = db.execute(
        f"SELECT start, end FROM {nvtx_tbl} WHERE text = 'walk_sampling_batch' "
        "ORDER BY start"
    ).fetchall()
    if not ranges:
        return None
    per_call = []
    for s, e in ranges:
        row = db.execute(
            "SELECT COUNT(*), COALESCE(SUM(end - start), 0) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE start >= ? AND end <= ?", (s, e)
        ).fetchone()
        n_kern, t_kern_ns = row[0], row[1]
        per_call.append((n_kern, t_kern_ns / 1e6, (e - s) / 1e6))
    n_calls       = len(per_call)
    total_kern_ms = sum(r[1] for r in per_call)
    total_n_kern  = sum(r[0] for r in per_call)
    total_nvtx_ms = sum(r[2] for r in per_call)
    return {
        'kern_time_per_call_ms':  total_kern_ms / n_calls,
        'kern_launches_per_call': total_n_kern / n_calls,
        'per_kernel_us':          (total_kern_ms / max(1, total_n_kern)) * 1000.0,
        'gpu_active_frac':        total_kern_ms / total_nvtx_ms if total_nvtx_ms else 0.0,
    }


def _extract_ingest(db):
    """Ingest-side metrics from `ingestion_batch` NVTX ranges. Returns
    None if the binary did not emit ingestion NVTX (e.g., walk-only run).

    Bucket criteria (substring match on resolved kernel name):
      - sort+merge :   `RadixSort` or `merge_kernel`
      - weight     :   `compute_per_node_weights`
    H2D bytes/duration come from CUPTI_ACTIVITY_KIND_MEMCPY with
    copyKind = 1 (Host-to-Device).
    """
    nvtx_tbl = _find_nvtx_table(db)
    ranges = db.execute(
        f"SELECT start, end FROM {nvtx_tbl} WHERE text = 'ingestion_batch' "
        "ORDER BY start"
    ).fetchall()
    if not ranges:
        return None
    per_call = []
    for s, e in ranges:
        nvtx_ms = (e - s) / 1e6
        kerns = db.execute("""
            SELECT k.start, k.end,
                   COALESCE(d.value, m.value, sn.value, '') AS name
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds d  ON d.id  = k.demangledName
            LEFT JOIN StringIds m  ON m.id  = k.mangledName
            LEFT JOIN StringIds sn ON sn.id = k.shortName
            WHERE k.start >= ? AND k.end <= ?
        """, (s, e)).fetchall()
        n_launches    = len(kerns)
        sort_ms       = sum((k[1] - k[0]) for k in kerns
                            if 'RadixSort' in k[2] or 'merge_kernel' in k[2]) / 1e6
        weight_ms     = sum((k[1] - k[0]) for k in kerns
                            if 'compute_per_node_weights' in k[2]) / 1e6
        h2d_ms = (db.execute(
            "SELECT COALESCE(SUM(end - start), 0) "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE start >= ? AND end <= ? AND copyKind = 1", (s, e)
        ).fetchone()[0]) / 1e6
        per_call.append({
            'total_ms':  nvtx_ms,
            'sort_ms':   sort_ms,
            'weight_ms': weight_ms,
            'h2d_ms':    h2d_ms,
            'launches':  n_launches,
        })
    n = len(per_call)
    return {
        'ingest_total_ms':  sum(c['total_ms']  for c in per_call) / n,
        'ingest_sort_ms':   sum(c['sort_ms']   for c in per_call) / n,
        'ingest_weight_ms': sum(c['weight_ms'] for c in per_call) / n,
        'ingest_h2d_ms':    sum(c['h2d_ms']    for c in per_call) / n,
        'ingest_launches':  sum(c['launches']  for c in per_call) / n,
    }


def extract_nsys_metrics(nsys_bin, rep_path):
    """Returns (walk_dict, ingest_dict). Either may be None if the
    corresponding NVTX range is absent OR if the corresponding extractor
    raised (the failure is logged to stderr; the OTHER extractor still
    runs). Raises only if the SQLite export itself fails or both
    extractors failed."""
    sql_path = export_to_sqlite(nsys_bin, rep_path)
    db = sqlite3.connect(sql_path)
    try:
        try:
            walk = _extract_walk(db)
        except Exception as e:
            print(f'    [walk extraction failed for {rep_path}: '
                  f'{type(e).__name__}: {e}]', file=sys.stderr)
            walk = None
        try:
            ingest = _extract_ingest(db)
        except Exception as e:
            print(f'    [ingest extraction failed for {rep_path}: '
                  f'{type(e).__name__}: {e}]', file=sys.stderr)
            ingest = None
        if walk is None and ingest is None:
            raise RuntimeError(
                f'no walk_sampling_batch or ingestion_batch NVTX (or both '
                f'extractors failed) in {rep_path}')
        return walk, ingest
    finally:
        db.close()


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
    ap.add_argument('--nsys',       default=DEFAULT_NSYS,
                    help=f'Path to nsys binary (default: {DEFAULT_NSYS}).')
    ap.add_argument('--nsys-dir',   default=None,
                    help='Directory for .nsys-rep / .sqlite outputs '
                         '(default: <output_base>_nsys/ next to --output).')
    args = ap.parse_args()

    if not Path(args.binary).is_file():
        ap.error(f'binary not found: {args.binary}')
    paths = {'coin': args.coin_csv, 'flight': args.flight_csv, 'delicious': args.delicious_csv}
    for name, p in paths.items():
        if not Path(p).is_file():
            ap.error(f'{name} CSV not found: {p}')

    out_base = Path(args.output)
    out_tune   = out_base.parent / f'{out_base.name}_tuning.csv'
    out_final  = out_base.parent / f'{out_base.name}_final.csv'
    out_ingest = out_base.parent / f'{out_base.name}_ingest.csv'
    nsys_dir   = Path(args.nsys_dir) if args.nsys_dir else \
                 out_base.parent / f'{out_base.name}_nsys'
    nsys_dir.mkdir(parents=True, exist_ok=True)

    print(f'# binary           : {args.binary}')
    print(f'# nsys             : {args.nsys}')
    print(f'# block_dim        : {args.block_dim}')
    print(f'# tune W values    : {TUNE_W_VALUES}')
    print(f'# tune runs per W  : {TUNE_RUNS_PER_W}  (NG variants only; FW skipped)')
    print(f'# final thr runs   : {FINAL_RUNS_THR}  (per dataset × variant at winning W)')
    print(f'# nsys runs/cell   : {NSYS_RUNS_PER_CELL}')
    print(f'# outlier threshold: {OUTLIER_THRESHOLD:.0%} (min_keep={MIN_KEEP})')
    print(f'# tuning CSV       : {out_tune}    (one row per Phase-1 invocation)')
    print(f'# final CSV        : {out_final}     (one row per dataset × variant)')
    print(f'# ingest CSV       : {out_ingest}    (one row per dataset; variant-agnostic)')
    print(f'# nsys reports dir : {nsys_dir}')
    print()

    tuning_rows = []  # raw Phase-1 rows
    final_rows  = []  # one aggregated row per (dataset, variant)

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

    # Persist tuning CSV NOW. Phase 1 typically takes the longest; if any
    # later phase explodes, we must not lose this data.
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

    # Persist throughput-only finals NOW. Phase 3 may fail (nsys version
    # mismatch, sqlite schema differences, profiling overhead crashes);
    # don't lose Phase 2 work to a Phase 3 explosion.
    thr_only_rows = []
    for ds in DATASETS:
        for variant in ALL_VARIANTS:
            ts = thr_stats.get((ds, variant), {})
            if not ts:
                continue
            row = {'dataset': ds, 'variant': variant,
                   'w_threshold_warp': ts.get('w', 1)}
            row.update(ts)
            thr_only_rows.append(row)
    write_csv(out_final, thr_only_rows)
    print()
    print(f'(checkpoint) wrote {len(thr_only_rows)} throughput-only rows to '
          f'{out_final} (Phase 3 will overwrite with nsys columns added)')

    # ===========================================================
    # Phase 3: nsys-driven kernel/NVTX metrics.
    # ===========================================================
    print()
    print('=' * 70)
    print(f'Phase 3: nsys profiling — {NSYS_RUNS_PER_CELL} runs per (dataset, variant)')
    print('=' * 70)
    print()

    nsys_stats: dict = {}
    # Ingest is variant-agnostic (add_multiple_edges is the same path across
    # all three KernelLaunchTypes). Pool every nsys run for a given dataset.
    ingest_pool: dict = {ds: {'total': [], 'sort': [], 'weight': [],
                              'h2d':   [], 'launches': []} for ds in DATASETS}
    for ds in DATASETS:
        wpn, nb, nw, mwl = PRESETS[ds]
        for variant in ALL_VARIANTS:
            w = cell_w[(ds, variant)]
            kt, kn, pk, gf = [], [], [], []
            for r in range(NSYS_RUNS_PER_CELL):
                tag = f'  nsys  {ds:>9} / {variant:<26} / W={w:<3} run {r+1}/{NSYS_RUNS_PER_CELL}'
                print(f'{tag} ...', end=' ', flush=True)
                rep = nsys_dir / f'{ds}_{variant}_run{r+1}.nsys-rep'
                try:
                    invoke_nsys(args.nsys, rep, args.binary, paths[ds], variant,
                                wpn, nb, nw, mwl, args.block_dim, w)
                    walk, ingest = extract_nsys_metrics(args.nsys, rep)
                except Exception as e:
                    # Catch ALL — including sqlite3.OperationalError for
                    # nsys-version schema mismatches. One bad rep skips the
                    # rep, not the run; checkpointed CSVs are already on disk.
                    print(f'FAIL ({type(e).__name__}: {e})', file=sys.stderr)
                    continue
                if walk is not None:
                    kt.append(walk['kern_time_per_call_ms'])
                    kn.append(walk['kern_launches_per_call'])
                    pk.append(walk['per_kernel_us'])
                    gf.append(walk['gpu_active_frac'])
                if ingest is not None:
                    ingest_pool[ds]['total'   ].append(ingest['ingest_total_ms'])
                    ingest_pool[ds]['sort'    ].append(ingest['ingest_sort_ms'])
                    ingest_pool[ds]['weight'  ].append(ingest['ingest_weight_ms'])
                    ingest_pool[ds]['h2d'     ].append(ingest['ingest_h2d_ms'])
                    ingest_pool[ds]['launches'].append(ingest['ingest_launches'])
                line = ''
                if walk is not None:
                    line += (f'kern_t={walk["kern_time_per_call_ms"]:7.2f}ms  '
                             f'launches={walk["kern_launches_per_call"]:7.1f}  '
                             f'kern_us={walk["per_kernel_us"]:8.2f}  '
                             f'active={walk["gpu_active_frac"]:.3f}')
                if ingest is not None:
                    line += (f'  || ing_t={ingest["ingest_total_ms"]:6.1f}ms  '
                             f'sort={ingest["ingest_sort_ms"]:5.1f}ms  '
                             f'wgt={ingest["ingest_weight_ms"]:5.1f}ms')
                print(line)
            kept_kt, dr_kt = reject_outliers(kt)
            kept_kn, dr_kn = reject_outliers(kn)
            kept_pk, dr_pk = reject_outliers(pk)
            kept_gf, dr_gf = reject_outliers(gf)
            for label, dropped, kept_n, raw_n, scale, unit in [
                ('kern_time_ms',    dr_kt, len(kept_kt), len(kt), 1.0, 'ms'),
                ('kern_launches',   dr_kn, len(kept_kn), len(kn), 1.0, ''),
                ('per_kernel_us',   dr_pk, len(kept_pk), len(pk), 1.0, 'us'),
                ('gpu_active_frac', dr_gf, len(kept_gf), len(gf), 1.0, ''),
            ]:
                msg = fmt_drop(label, dropped, kept_n, raw_n, scale=scale, unit=unit)
                if msg: print(msg)
            kt_m, kt_s = mean_std(kept_kt)
            kn_m, kn_s = mean_std(kept_kn)
            pk_m, pk_s = mean_std(kept_pk)
            gf_m, gf_s = mean_std(kept_gf)
            nsys_stats[(ds, variant)] = {
                'n_nsys_raw': len(kt),
                'kern_time_per_call_ms_mean':  kt_m, 'kern_time_per_call_ms_std':  kt_s,
                'kern_launches_per_call_mean': kn_m, 'kern_launches_per_call_std': kn_s,
                'per_kernel_us_mean':          pk_m, 'per_kernel_us_std':          pk_s,
                'gpu_active_frac_mean':        gf_m, 'gpu_active_frac_std':        gf_s,
            }

    # Combine Phase 2 + Phase 3 into one row per (dataset, variant).
    for ds in DATASETS:
        for variant in ALL_VARIANTS:
            ts = thr_stats.get((ds, variant), {})
            ns = nsys_stats.get((ds, variant), {})
            if not ts and not ns:
                continue
            row = {'dataset': ds, 'variant': variant,
                   'w_threshold_warp': ts.get('w', 1)}
            row.update(ts); row.update(ns)
            final_rows.append(row)

    # Aggregate ingest pool per dataset (variant-agnostic — same code path).
    # Outlier rejection per metric independently. n_samples = up to
    # NSYS_RUNS_PER_CELL × len(ALL_VARIANTS).
    ingest_rows = []
    for ds in DATASETS:
        pool = ingest_pool.get(ds, {})
        if not pool or not pool.get('total'):
            continue
        kept = {}
        for k in ('total', 'sort', 'weight', 'h2d', 'launches'):
            kept_k, dropped_k = reject_outliers(pool[k])
            kept[k] = kept_k
            if dropped_k:
                msg = fmt_drop(f'ingest_{k}', dropped_k, len(kept_k),
                               len(pool[k]), scale=1.0,
                               unit='ms' if k != 'launches' else '')
                if msg: print(msg)
        tot_m, tot_s = mean_std(kept['total'])
        srt_m, srt_s = mean_std(kept['sort'])
        wgt_m, wgt_s = mean_std(kept['weight'])
        h2d_m, h2d_s = mean_std(kept['h2d'])
        lau_m, lau_s = mean_std(kept['launches'])
        ingest_rows.append({
            'dataset':                  ds,
            'n_samples_raw':            len(pool['total']),
            'n_samples_kept_total':     len(kept['total']),
            'ingest_total_ms_mean':     tot_m, 'ingest_total_ms_std':     tot_s,
            'ingest_sort_ms_mean':      srt_m, 'ingest_sort_ms_std':      srt_s,
            'ingest_weight_ms_mean':    wgt_m, 'ingest_weight_ms_std':    wgt_s,
            'ingest_h2d_ms_mean':       h2d_m, 'ingest_h2d_ms_std':       h2d_s,
            'ingest_launches_mean':     lau_m, 'ingest_launches_std':     lau_s,
            # Convenience derived columns (% of total ingest wall time):
            'ingest_sort_frac':         srt_m / tot_m if tot_m else 0.0,
            'ingest_weight_frac':       wgt_m / tot_m if tot_m else 0.0,
            'ingest_h2d_frac':          h2d_m / tot_m if tot_m else 0.0,
        })

    # ===========================================================
    # Final CSV writes. Tuning CSV was already written at the Phase-1
    # checkpoint and tuning_rows hasn't changed since — no rewrite.
    # Final CSV gets overwritten here with nsys columns merged in
    # (the Phase-2 checkpoint had throughput-only columns). Ingest CSV
    # is fresh.
    # ===========================================================
    if not final_rows:
        print('No successful final runs.', file=sys.stderr); return 1

    write_csv(out_final, final_rows)
    if ingest_rows:
        write_csv(out_ingest, ingest_rows)

    print()
    print(f'Wrote {len(final_rows)} final rows to {out_final}')
    if ingest_rows:
        print(f'Wrote {len(ingest_rows)} ingest rows to {out_ingest}')
    print()

    # ===========================================================
    # Final markdown report.
    # ===========================================================
    print('=' * 70)
    print('Final report — Phase 2 throughput + Phase 3 nsys metrics')
    print('=' * 70)
    print()
    print(f'| {"dataset":<9} | {"variant":<26} | {"W":>3} | '
          f'{"steps/s mean (M)":>16} | {"std":>7} | '
          f'{"walks/s mean (M)":>16} | {"std":>7} | '
          f'{"avg_len":>7} | '
          f'{"kern ms/call":>12} | {"std":>7} | '
          f'{"launches/call":>13} | {"std":>7} | '
          f'{"kern μs":>9} | {"std":>7} | '
          f'{"gpu_active":>10} | {"std":>7} |')
    print('|' + '|'.join('-' * w for w in
          [11, 28, 5, 18, 9, 18, 9, 9, 14, 9, 15, 9, 11, 9, 12, 9]) + '|')
    for r in final_rows:
        print(f'| {r["dataset"]:<9} | {r["variant"]:<26} | '
              f'{r["w_threshold_warp"]:>3} | '
              f'{r.get("steps_per_sec_mean", 0)/1e6:>16.3f} | '
              f'{r.get("steps_per_sec_std", 0)/1e6:>7.3f} | '
              f'{r.get("walks_per_sec_mean", 0)/1e6:>16.3f} | '
              f'{r.get("walks_per_sec_std", 0)/1e6:>7.3f} | '
              f'{r.get("avg_walk_length_mean", 0):>7.2f} | '
              f'{r.get("kern_time_per_call_ms_mean", 0):>12.2f} | '
              f'{r.get("kern_time_per_call_ms_std", 0):>7.2f} | '
              f'{r.get("kern_launches_per_call_mean", 0):>13.1f} | '
              f'{r.get("kern_launches_per_call_std", 0):>7.1f} | '
              f'{r.get("per_kernel_us_mean", 0):>9.2f} | '
              f'{r.get("per_kernel_us_std", 0):>7.2f} | '
              f'{r.get("gpu_active_frac_mean", 0):>10.3f} | '
              f'{r.get("gpu_active_frac_std", 0):>7.3f} |')

    # Ingest report — one row per dataset, pooling all variants × runs.
    if ingest_rows:
        print()
        print('=' * 70)
        print('Ingest report — variant-agnostic; pooled across all Phase 3 runs')
        print('=' * 70)
        print()
        print(f'| {"dataset":<9} | {"n":>3} | '
              f'{"total ms":>10} | {"std":>7} | '
              f'{"sort ms":>9} | {"std":>7} | {"sort %":>6} | '
              f'{"weight ms":>10} | {"std":>7} | {"weight %":>8} | '
              f'{"H2D ms":>8} | {"std":>7} | {"H2D %":>6} | '
              f'{"launches":>8} | {"std":>7} |')
        print('|' + '|'.join('-' * w for w in
              [11, 5, 12, 9, 11, 9, 8, 12, 9, 10, 10, 9, 8, 10, 9]) + '|')
        for r in ingest_rows:
            print(f'| {r["dataset"]:<9} | {r["n_samples_kept_total"]:>3} | '
                  f'{r["ingest_total_ms_mean"]:>10.2f} | {r["ingest_total_ms_std"]:>7.2f} | '
                  f'{r["ingest_sort_ms_mean"]:>9.2f} | {r["ingest_sort_ms_std"]:>7.2f} | '
                  f'{r["ingest_sort_frac"]*100:>5.1f}% | '
                  f'{r["ingest_weight_ms_mean"]:>10.2f} | {r["ingest_weight_ms_std"]:>7.2f} | '
                  f'{r["ingest_weight_frac"]*100:>7.1f}% | '
                  f'{r["ingest_h2d_ms_mean"]:>8.2f} | {r["ingest_h2d_ms_std"]:>7.2f} | '
                  f'{r["ingest_h2d_frac"]*100:>5.1f}% | '
                  f'{r["ingest_launches_mean"]:>8.1f} | {r["ingest_launches_std"]:>7.1f} |')
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Profiling runner: nsys-based kernel/NVTX metric extraction at a hardcoded
winning W. Use this AFTER run_ablation.py has produced:
  - <base>_tuning.csv  (Phase 1 — W sweep)
  - <base>_final.csv   (Phase 2 — throughput stats)

This script does the nsys-derived profiling step on top of those CSVs:

  1. Profiles each (dataset, variant) NSYS_RUNS_PER_CELL times, OR
     reuses existing <nsys_dir>/<dataset>_<variant>_run<r>.nsys-rep
     files if --reuse-existing is set.
  2. Forces a fresh nsys export to .sqlite every time (no mtime cache —
     nsys can return exit 0 with no actual sqlite write, and a stale
     empty file would silently break downstream queries).
  3. Validates the sqlite has the expected tables before running
     queries; any cell with a bad rep/sqlite is logged and skipped.
  4. Extracts walk-side and ingest-side metrics with the same outlier
     rejection rule as run_ablation.py.
  5. Merges nsys columns into <base>_final.csv (preserves Phase-2
     throughput data; just adds nsys columns).
  6. Writes <base>_ingest.csv (one row per dataset, variant-agnostic
     since add_multiple_edges is the same path across variants).

Usage:
  python3 run_profiling.py coin.csv flight.csv delicious.csv \\
      --base ablation_results_2 --block-dim 256 \\
      --nsys /its/home/ms2420/cuda-12.6/bin/nsys

  # Or, to retry extraction on existing .nsys-rep files without
  # re-profiling (fast; useful when the toolkit nsys is now correct
  # and the originals are good):
  python3 run_profiling.py ... --reuse-existing
"""
import argparse
import csv
import sqlite3
import statistics
import subprocess
import sys
from pathlib import Path

# Reuse the building blocks from run_ablation.py so logic stays in one
# place (outlier rejection threshold, throughput-line parser, etc.).
# nsys-specific helpers live here — Phase 1+2 don't need them.
from run_ablation import (
    ALL_VARIANTS, NG_VARIANTS, DATASETS, PRESETS,
    build_run_argv, parse_throughput,
    mean_std, reject_outliers, fmt_drop, write_csv,
)

NSYS_RUNS_PER_CELL = 3

# Candidate names for the NVTX-events table across nsys versions.
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
        n_launches = len(kerns)
        sort_ms    = sum((k[1] - k[0]) for k in kerns
                         if 'RadixSort' in k[2] or 'merge_kernel' in k[2]) / 1e6
        weight_ms  = sum((k[1] - k[0]) for k in kerns
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

# ===========================================================
# Hardcoded winning config (from Phase 1 of run on 2026-04-26).
# NG variants use this W; FW ignores w_threshold_warp so it's W=1.
# ===========================================================
WINNING_W_NG = 8
WINNING_W_FW = 1

DEFAULT_BIN          = './build/bin/ablation_streaming'
DEFAULT_BASE         = 'ablation_results_2'
DEFAULT_BLOCK_DIM    = 256
DEFAULT_NSYS         = 'nsys'


def cell_w(variant: str) -> int:
    return WINNING_W_NG if variant in NG_VARIANTS else WINNING_W_FW


def invoke_nsys(nsys_bin, rep_path, binary, data, klt, wpn, nb, nw, mwl,
                block_dim, w_threshold_warp):
    """Run binary under `nsys profile --stats=true`. With --stats=true,
    nsys writes the .sqlite NEXT TO the .nsys-rep as part of the profile
    run — no separate `nsys export` step needed. This sidesteps a bug in
    nsys 2024.4.2 (HPC toolkit at /its/home/ms2420/cuda-12.6/bin/) where
    `nsys export` on a --stats=false rep returns exit 0 but writes a
    sqlite with zero tables. Confirmed by side-by-side: a manually
    re-profiled run with --stats=true produces a populated sqlite, while
    the script's old profile-without-stats + separate-export produced
    an empty one for the SAME workload."""
    cmd = [nsys_bin, 'profile', '--trace=cuda,nvtx', '--stats=true',
           '--force-overwrite=true', '--output', str(rep_path)] + \
          build_run_argv(binary, data, klt, wpn, nb, nw, mwl,
                         block_dim, w_threshold_warp)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f'nsys profile exit {proc.returncode}. stderr tail:\n'
            f'{proc.stderr[-500:]}')
    return parse_throughput(proc.stdout)


def validate_sqlite(rep_path):
    """The .sqlite is produced as a side effect of `nsys profile
    --stats=true` (see invoke_nsys). This function just verifies the
    file exists and has the schema we depend on — no separate export
    step. Raises on bad sqlite (missing or schema-incomplete)."""
    rep_path = Path(rep_path).resolve()
    sql_path = rep_path.with_suffix('.sqlite')
    if not sql_path.exists():
        raise RuntimeError(
            f'sqlite not found at {sql_path}. The .nsys-rep was likely '
            f'profiled without --stats=true (e.g. via run_ablation.py '
            f'before this fix), so the sqlite was never produced. '
            f'Re-profile (drop --reuse-existing) to fix.')
    if sql_path.stat().st_size == 0:
        raise RuntimeError(f'sqlite exists but is empty: {sql_path}')
    db = sqlite3.connect(sql_path)
    try:
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r[0] for r in rows}
        missing = [t for t in ('CUPTI_ACTIVITY_KIND_KERNEL', 'NVTX_EVENTS')
                   if t not in names]
        if missing:
            raise RuntimeError(
                f'sqlite at {sql_path} missing required tables: {missing}. '
                f'Total tables: {len(names)}. The rep was likely profiled '
                f'without --stats=true; re-profile to fix.')
    finally:
        db.close()
    return sql_path
    # Verify the export actually wrote the schema we depend on.
    db = sqlite3.connect(sql_path)
    try:
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r[0] for r in rows}
        missing = [t for t in ('CUPTI_ACTIVITY_KIND_KERNEL', 'NVTX_EVENTS')
                   if t not in names]
        if missing:
            raise RuntimeError(
                f'nsys export produced sqlite missing required tables: '
                f'{missing}. Total tables in export: {len(names)}. '
                f'The .nsys-rep at {rep_path} is likely corrupt — '
                f're-profile it.')
    finally:
        db.close()
    return sql_path


def extract_metrics(rep_path):
    """Validate the .sqlite (created by `nsys profile --stats=true`)
    and run both extractors. Either may return None; raises if both
    fail or if the sqlite is missing/incomplete."""
    sql_path = validate_sqlite(rep_path)
    db = sqlite3.connect(sql_path)
    try:
        try:
            walk = _extract_walk(db)
        except Exception as e:
            print(f'    [walk extraction failed: {type(e).__name__}: {e}]',
                  file=sys.stderr)
            walk = None
        try:
            ingest = _extract_ingest(db)
        except Exception as e:
            print(f'    [ingest extraction failed: {type(e).__name__}: {e}]',
                  file=sys.stderr)
            ingest = None
        if walk is None and ingest is None:
            raise RuntimeError('both extractors failed (no NVTX ranges?)')
        return walk, ingest
    finally:
        db.close()


def read_existing_final_csv(path):
    """Read <base>_final.csv if it exists. Returns dict keyed by
    (dataset, variant) → row-dict. Empty dict if file missing."""
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            out[(r['dataset'], r['variant'])] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('coin_csv',      help='Path to ml_tgbl-coin.csv')
    ap.add_argument('flight_csv',    help='Path to ml_tgbl-flight.csv')
    ap.add_argument('delicious_csv', help='Path to delicious(_clip).csv')
    ap.add_argument('--base',       default=DEFAULT_BASE,
                    help=f'Output base prefix (default: {DEFAULT_BASE}).')
    ap.add_argument('--binary',     default=DEFAULT_BIN,
                    help=f'Path to ablation_streaming (default: {DEFAULT_BIN}).')
    ap.add_argument('--block-dim',  type=int, default=DEFAULT_BLOCK_DIM,
                    help=f'block_dim passed to ablation_streaming (default: {DEFAULT_BLOCK_DIM}).')
    ap.add_argument('--nsys',       default=DEFAULT_NSYS,
                    help=f'Path to nsys binary (default: {DEFAULT_NSYS}).')
    ap.add_argument('--nsys-dir',   default=None,
                    help='Directory for .nsys-rep / .sqlite outputs '
                         '(default: <base>_nsys/ next to --base).')
    ap.add_argument('--reuse-existing', action='store_true',
                    help='Skip nsys profile; assume <nsys_dir>/<ds>_<v>_run<r>'
                         '.nsys-rep already exists. Still re-exports + extracts.')
    args = ap.parse_args()

    if not Path(args.binary).is_file() and not args.reuse_existing:
        ap.error(f'binary not found: {args.binary}')
    paths = {'coin': args.coin_csv, 'flight': args.flight_csv,
             'delicious': args.delicious_csv}
    for name, p in paths.items():
        if not Path(p).is_file():
            ap.error(f'{name} CSV not found: {p}')

    base = Path(args.base)
    out_final  = base.parent / f'{base.name}_final.csv'
    out_ingest = base.parent / f'{base.name}_ingest.csv'
    nsys_dir   = Path(args.nsys_dir) if args.nsys_dir else \
                 base.parent / f'{base.name}_nsys'
    nsys_dir.mkdir(parents=True, exist_ok=True)

    print(f'# binary           : {args.binary}')
    print(f'# nsys             : {args.nsys}')
    print(f'# block_dim        : {args.block_dim}')
    print(f'# winning W (NG)   : {WINNING_W_NG}  (FW uses {WINNING_W_FW})')
    print(f'# nsys runs/cell   : {NSYS_RUNS_PER_CELL}')
    print(f'# reuse-existing   : {args.reuse_existing}')
    print(f'# nsys reports dir : {nsys_dir}')
    print(f'# final CSV (read+overwrite) : {out_final}')
    print(f'# ingest CSV (write)         : {out_ingest}')
    print()

    # ===========================================================
    # Phase 3: nsys profile + metric extraction.
    # ===========================================================
    print('=' * 70)
    print(f'Phase 3 only: nsys profiling — {NSYS_RUNS_PER_CELL} runs per cell')
    print('=' * 70)
    print()

    nsys_stats: dict = {}
    ingest_pool: dict = {ds: {'total': [], 'sort': [], 'weight': [],
                              'h2d':   [], 'launches': []} for ds in DATASETS}

    for ds in DATASETS:
        wpn, nb, nw, mwl = PRESETS[ds]
        for variant in ALL_VARIANTS:
            w = cell_w(variant)
            kt, kn, pk, gf = [], [], [], []
            for r in range(NSYS_RUNS_PER_CELL):
                rep = nsys_dir / f'{ds}_{variant}_run{r+1}.nsys-rep'
                tag = f'  {ds:>9} / {variant:<26} / W={w:<3} run {r+1}/{NSYS_RUNS_PER_CELL}'
                print(f'{tag} ...', end=' ', flush=True)
                try:
                    if not args.reuse_existing:
                        invoke_nsys(args.nsys, rep, args.binary, paths[ds],
                                    variant, wpn, nb, nw, mwl,
                                    args.block_dim, w)
                    walk, ingest = extract_metrics(rep)
                except Exception as e:
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
            for label, dropped, kept_n, raw_n, unit in [
                ('kern_time_ms',    dr_kt, len(kept_kt), len(kt), 'ms'),
                ('kern_launches',   dr_kn, len(kept_kn), len(kn), ''),
                ('per_kernel_us',   dr_pk, len(kept_pk), len(pk), 'us'),
                ('gpu_active_frac', dr_gf, len(kept_gf), len(gf), ''),
            ]:
                msg = fmt_drop(label, dropped, kept_n, raw_n, unit=unit)
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

    # ===========================================================
    # Aggregate ingest pool per dataset (variant-agnostic).
    # ===========================================================
    ingest_rows = []
    for ds in DATASETS:
        pool = ingest_pool[ds]
        if not pool['total']:
            continue
        kept = {}
        for k in ('total', 'sort', 'weight', 'h2d', 'launches'):
            kept_k, dropped_k = reject_outliers(pool[k])
            kept[k] = kept_k
            if dropped_k:
                msg = fmt_drop(f'ingest_{k}', dropped_k, len(kept_k),
                               len(pool[k]),
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
            'ingest_sort_frac':         srt_m / tot_m if tot_m else 0.0,
            'ingest_weight_frac':       wgt_m / tot_m if tot_m else 0.0,
            'ingest_h2d_frac':          h2d_m / tot_m if tot_m else 0.0,
        })

    # ===========================================================
    # Merge nsys columns into existing <base>_final.csv (preserving
    # Phase-2 throughput data) and write back.
    # ===========================================================
    existing = read_existing_final_csv(out_final)
    if not existing:
        print(f'WARNING: {out_final} not found — writing nsys-only rows.',
              file=sys.stderr)
    final_rows = []
    for ds in DATASETS:
        for variant in ALL_VARIANTS:
            row = dict(existing.get((ds, variant), {}))
            row.setdefault('dataset', ds)
            row.setdefault('variant', variant)
            row.setdefault('w_threshold_warp', cell_w(variant))
            ns = nsys_stats.get((ds, variant), {})
            row.update({k: v for k, v in ns.items()})
            final_rows.append(row)

    if final_rows:
        write_csv(out_final, final_rows)
    if ingest_rows:
        write_csv(out_ingest, ingest_rows)

    print()
    print(f'Wrote {len(final_rows)} merged final rows to {out_final}')
    if ingest_rows:
        print(f'Wrote {len(ingest_rows)} ingest rows to {out_ingest}')
    print()

    # ===========================================================
    # Final report.
    # ===========================================================
    print('=' * 70)
    print('Phase 3 nsys metrics report')
    print('=' * 70)
    print()
    print(f'| {"dataset":<9} | {"variant":<26} | '
          f'{"kern ms/call":>12} | {"std":>7} | '
          f'{"launches/call":>13} | {"std":>7} | '
          f'{"kern μs":>9} | {"std":>7} | '
          f'{"gpu_active":>10} | {"std":>7} |')
    print('|' + '|'.join('-' * w for w in
          [11, 28, 14, 9, 15, 9, 11, 9, 12, 9]) + '|')
    for r in final_rows:
        print(f'| {r["dataset"]:<9} | {r["variant"]:<26} | '
              f'{float(r.get("kern_time_per_call_ms_mean", 0) or 0):>12.2f} | '
              f'{float(r.get("kern_time_per_call_ms_std", 0) or 0):>7.2f} | '
              f'{float(r.get("kern_launches_per_call_mean", 0) or 0):>13.1f} | '
              f'{float(r.get("kern_launches_per_call_std", 0) or 0):>7.1f} | '
              f'{float(r.get("per_kernel_us_mean", 0) or 0):>9.2f} | '
              f'{float(r.get("per_kernel_us_std", 0) or 0):>7.2f} | '
              f'{float(r.get("gpu_active_frac_mean", 0) or 0):>10.3f} | '
              f'{float(r.get("gpu_active_frac_std", 0) or 0):>7.3f} |')

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

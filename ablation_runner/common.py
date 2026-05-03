"""Shared helpers for tuning_w / run_ablation / run_profiling /
kernel_breakdown. Everything that's identical across the four scripts
lives here: the ablation_streaming arg layout, the throughput stdout
parser, outlier rejection, CSV writer."""
import csv
import re
import statistics
import subprocess

from constants import OUTLIER_THRESHOLD, MIN_KEEP, USE_GPU, PICKER, IS_DIRECTED, TIMESCALE


THROUGHPUT_RE = re.compile(r'^Throughput:\s+([\d.eE+-]+)\s+walks/sec', re.MULTILINE)
STEPS_RE      = re.compile(r'^Steps/sec:\s+([\d.eE+-]+)\s+steps/sec',  re.MULTILINE)
AVGLEN_RE     = re.compile(r'^Final avg walk length:\s*([\d.eE+-]+)',  re.MULTILINE)


def build_run_argv(binary, data, klt, wpn, nb, nw, mwl, block_dim, w_threshold_warp):
    """Positional args for ablation_streaming. Values that are fixed
    across all scripts (use_gpu, picker, is_directed, timescale) are
    injected from constants.py — keep that list short and authoritative."""
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


def mean_std(xs):
    if not xs:
        return 0.0, 0.0
    mu = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return mu, sd


def reject_outliers(xs, threshold_frac=OUTLIER_THRESHOLD, min_keep=MIN_KEEP):
    """Iterative median-relative outlier rejection. See constants.py for
    OUTLIER_THRESHOLD / MIN_KEEP rationale."""
    kept = list(xs)
    dropped = []
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

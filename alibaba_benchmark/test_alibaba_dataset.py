import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tempest import Tempest

MAX_WALK_LEN = 100


def human_readable_count(n):
    if n >= 1_000_000_000:
        billions = n // 1_000_000_000
        millions = (n % 1_000_000_000) / 1_000_000
        return f"{billions} billion {millions:.1f} million"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f} million"
    elif n >= 1_000:
        return f"{n / 1_000:.1f} thousand"
    else:
        return str(n)


def main(base_dir, minutes_per_step, window_size, walk_count, walk_bias, use_gpu, kernel_mode):
    runtime_start = time.time()

    running_device = "GPU" if use_gpu else "CPU"
    print(f"---- Running on {running_device}. ----\n")

    t = Tempest(
        is_directed=True,
        use_gpu=use_gpu,
        max_time_capacity=window_size
    )

    edge_addition_times = []
    walk_times = []

    total_minutes_data_processed = 0

    total_edges_per_iteration = []
    active_edges_per_iteration = []

    total_edges_added = 0

    for i in range(0, 20160, minutes_per_step):
        dfs = [pd.read_parquet(os.path.join(base_dir, f'data_{i + j}.parquet')) for j in range(minutes_per_step)]
        merged_df = pd.concat(dfs, ignore_index=True)
        final_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

        total_edges_added += len(final_df)

        sources = final_df['u'].astype(np.int32).values
        targets = final_df['i'].astype(np.int32).values
        timestamps = final_df['ts'].astype(np.int64).values

        edge_addition_start_time = time.time()
        t.add_edges(sources, targets, timestamps)
        edge_addition_time = time.time() - edge_addition_start_time

        edge_addition_times.append(edge_addition_time)

        active_edge_count = t.edge_count()
        total_edges_per_iteration.append(total_edges_added)
        active_edges_per_iteration.append(active_edge_count)

        walk_start_time = time.time()
        t.get_walks_global(
            num_walks_total=walk_count,
            max_walk_len=MAX_WALK_LEN,
            walk_bias=walk_bias,
            start_bias="Uniform",
            direction="Forward",
            kernel_mode=kernel_mode
        )
        walk_sampling_time = time.time() - walk_start_time
        walk_times.append(walk_sampling_time)

        total_minutes_data_processed += minutes_per_step
        print(
            f"{total_minutes_data_processed} minutes data processed | "
            f"Edge addition time: {edge_addition_time:.3f}s | "
            f"Walk sampling time: {walk_sampling_time:.3f}s | "
            f"Total edges: {human_readable_count(total_edges_added)} | "
            f"Active edges: {human_readable_count(active_edge_count)}"
        )

    print('Completed processing all data')
    results = {
        'total_runtime': time.time() - runtime_start,
        'edge_addition_time': edge_addition_times,
        'walk_sampling_time': walk_times,
        'total_edges': total_edges_per_iteration,
        'active_edges': active_edges_per_iteration
    }

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'
    )
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'alibaba_streaming_result.pickle')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nTotal runtime: {results['total_runtime']:.2f} seconds")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Tempest Alibaba streaming benchmark")

    parser.add_argument(
        '--use_gpu', action='store_true',
        help='Enable GPU acceleration'
    )

    parser.add_argument(
        '--window_size', type=int, default=1_800_000,
        help='Sliding window size in milliseconds (default: 1_800_000 = 30 minutes)'
    )

    parser.add_argument(
        '--minutes_per_step', type=int, default=3,
        help='Increment size in minutes (default: 3)'
    )

    parser.add_argument(
        '--walk_count', type=int, default=1_000_000,
        help='Number of walks to generate (default: 1_000_000)'
    )

    parser.add_argument(
        '--walk_bias', type=str, default='ExponentialIndex',
        help='Walk bias type (default: ExponentialIndex)'
    )

    parser.add_argument(
        '--kernel_mode', type=str, default='NodeGrouped',
        help='Kernel mode (default: NodeGrouped)'
    )

    args = parser.parse_args()

    base_dir = os.environ.get('ALIBABA_DATASET_PATH')
    if not base_dir:
        raise RuntimeError('ALIBABA_DATASET_PATH environment variable is not set')

    print(f"Base dir: {base_dir}")
    print(f"Use GPU: {args.use_gpu}")
    print(f"Window size: {args.window_size} ms")
    print(f"Walk bias: {args.walk_bias}")
    print(f"Kernel mode: {args.kernel_mode}")

    main(
        base_dir,
        args.minutes_per_step,
        args.window_size,
        args.walk_count,
        args.walk_bias,
        args.use_gpu,
        args.kernel_model
    )

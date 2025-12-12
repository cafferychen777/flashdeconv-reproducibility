#!/usr/bin/env python
"""
FlashDeconv Scalability Benchmark.

This script reproduces the scalability benchmark results from the FlashDeconv paper,
demonstrating linear O(N) time and memory scaling.

Test scales: 1K, 5K, 10K, 20K, 50K, 100K spots
Output: Runtime, memory usage, and accuracy at each scale

Usage:
    python benchmark_scalability.py --output_dir ./results
    python benchmark_scalability.py --max_spots 50000  # Limit max scale

Requirements:
    - FlashDeconv (pip install flashdeconv)
    - psutil (pip install psutil)
"""

import gc
import os
import sys
import time
import argparse
import threading

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from flashdeconv import FlashDeconv

# Try to import psutil for memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Memory monitoring will be limited.")


def get_memory_mb():
    """Get current process memory in MB."""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    else:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == 'darwin':
            return rusage.ru_maxrss / 1e6
        else:
            return rusage.ru_maxrss / 1e3


class PeakMemoryMonitor:
    """Monitor peak memory usage in a background thread."""

    def __init__(self, interval=0.02):
        self.interval = interval
        self.peak_mb = 0
        self.running = False
        self.thread = None

    def _monitor(self):
        while self.running:
            current = get_memory_mb()
            self.peak_mb = max(self.peak_mb, current)
            time.sleep(self.interval)

    def start(self):
        self.peak_mb = get_memory_mb()
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        return self.peak_mb

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.peak_mb


def get_simulated_reference(n_types=10, n_genes=2000, seed=42):
    """Generate simulated scRNA-seq reference signatures."""
    np.random.seed(seed)

    # Base expression: log-normal with sparsity
    ref = np.random.lognormal(mean=1.0, sigma=1.5, size=(n_types, n_genes))

    # Sparsify
    ref[ref < 1.5] = 0

    # Add cell-type specific markers
    n_markers = 20
    for k in range(n_types):
        marker_idx = np.random.choice(n_genes, n_markers, replace=False)
        ref[k, marker_idx] *= np.random.uniform(5, 15, n_markers)

    # Normalize rows
    row_sums = ref.sum(axis=1, keepdims=True)
    ref = ref / row_sums * 10000

    return ref


def generate_spatial_data(ref, n_spots, seed=None):
    """Generate Spotless-style synthetic spatial data."""
    if seed is not None:
        np.random.seed(seed)

    n_types, n_genes = ref.shape

    # Generate true proportions (Dirichlet)
    beta = np.random.dirichlet(np.ones(n_types), size=n_spots)

    # Generate synthetic counts
    expected_counts = beta @ ref

    # Simulate variable sequencing depth
    depths = np.random.normal(20000, 5000, size=n_spots)
    depths = np.maximum(depths, 5000)

    # Scale and add Poisson noise
    Y = np.zeros((n_spots, n_genes))
    for i in range(n_spots):
        profile = expected_counts[i, :]
        if profile.sum() > 0:
            profile = profile / profile.sum() * depths[i]
        Y[i, :] = np.random.poisson(np.maximum(profile, 0))

    # Generate grid coordinates
    grid_size = int(np.ceil(np.sqrt(n_spots)))
    coords = np.array([[i % grid_size, i // grid_size] for i in range(n_spots)], dtype=float)

    return Y.astype(float), beta, coords


def run_single_benchmark(n_spots, n_genes=2000, n_types=10):
    """Run FlashDeconv benchmark at given scale."""

    gc.collect()
    baseline_mem = get_memory_mb()

    print(f"\n{'=' * 60}")
    print(f"Benchmark: {n_spots:,} spots x {n_genes:,} genes x {n_types} types")
    print(f"{'=' * 60}")

    # Generate data
    print("Generating synthetic data...")
    t0 = time.time()

    X = get_simulated_reference(n_types=n_types, n_genes=n_genes, seed=42)
    Y, beta_true, coords = generate_spatial_data(X, n_spots=n_spots, seed=42)

    data_time = time.time() - t0
    print(f"  Data generation: {data_time:.2f}s")
    print(f"  Y: {Y.shape}, X: {X.shape}")

    # Run FlashDeconv with memory monitoring
    print("\nRunning FlashDeconv...")
    monitor = PeakMemoryMonitor(interval=0.01)
    monitor.start()

    try:
        t0 = time.time()
        model = FlashDeconv(
            sketch_dim=256,
            lambda_spatial=100,
            preprocess="log_cpm",
            n_hvg=min(1000, n_genes),
            rho_sparsity=0.01,
            max_iter=100,
            verbose=False,
            random_state=42,
        )
        beta_pred = model.fit_transform(Y, X, coords)
        elapsed = time.time() - t0
        success = True
    except Exception as e:
        print(f"  ERROR: {e}")
        elapsed = None
        beta_pred = None
        success = False

    peak_mem = monitor.stop()

    # Compute accuracy
    if success and beta_pred is not None:
        corr, _ = pearsonr(beta_true.flatten(), beta_pred.flatten())
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Accuracy (Pearson r): {corr:.4f}")
        print(f"  Peak memory: {peak_mem:.0f} MB")
        print(f"  Memory delta: {peak_mem - baseline_mem:.0f} MB")
    else:
        corr = None

    # Cleanup
    del Y, X, beta_true, coords
    if beta_pred is not None:
        del beta_pred
    gc.collect()

    return {
        'n_spots': n_spots,
        'n_genes': n_genes,
        'n_types': n_types,
        'success': success,
        'time_sec': elapsed,
        'accuracy': corr,
        'peak_mem_mb': peak_mem,
        'mem_delta_mb': peak_mem - baseline_mem,
    }


def create_figure(df, output_path):
    """Create publication-ready 3-panel scalability figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping figure generation")
        return

    df = df[df['success'] == True].copy()
    if len(df) < 2:
        print("Not enough data points to create figure")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    spots_k = df['n_spots'].values / 1000
    times = df['time_sec'].values
    mems = df['mem_delta_mb'].values / 1024  # GB
    accs = df['accuracy'].values

    # Panel A: Runtime
    ax1 = axes[0]
    ax1.plot(spots_k, times, 'o-', color='#2E86AB', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of spots (thousands)', fontsize=11)
    ax1.set_ylabel('Runtime (seconds)', fontsize=11)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # O(N) reference line
    x_ref = np.array([spots_k.min(), spots_k.max()])
    y_ref = times[0] * (x_ref / x_ref[0])
    ax1.plot(x_ref, y_ref, '--', color='gray', alpha=0.5, label='O(N) reference')
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('A. Linear time scaling', fontweight='bold', fontsize=11)

    # Panel B: Memory
    ax2 = axes[1]
    ax2.plot(spots_k, mems, 's-', color='#E94F37', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of spots (thousands)', fontsize=11)
    ax2.set_ylabel('Peak memory (GB)', fontsize=11)
    ax2.set_xscale('log')
    ax2.axhline(y=16, color='gray', linestyle='--', alpha=0.5, label='16 GB limit')
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('B. Linear memory scaling', fontweight='bold', fontsize=11)

    # Panel C: Accuracy
    ax3 = axes[2]
    ax3.plot(spots_k, accs, 'D-', color='#28A745', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of spots (thousands)', fontsize=11)
    ax3.set_ylabel('Accuracy (Pearson r)', fontsize=11)
    ax3.set_xscale('log')
    y_min = max(0.80, min(accs) - 0.05)
    y_max = min(1.0, max(accs) + 0.02)
    ax3.set_ylim(y_min, y_max)
    ax3.axhline(y=0.90, color='gray', linestyle='--', alpha=0.5, label='r = 0.90')
    ax3.legend(frameon=False, fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_title('C. Accuracy preserved at scale', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nFigure saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='FlashDeconv Scalability Benchmark')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory for output results')
    parser.add_argument('--max_spots', type=int, default=100000,
                        help='Maximum number of spots to test')
    args = parser.parse_args()

    print("=" * 70)
    print("FlashDeconv Scalability Benchmark")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # Test scales
    all_scales = [1_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    spot_counts = [s for s in all_scales if s <= args.max_spots]

    n_genes = 2000
    n_types = 10

    results = []

    for n_spots in spot_counts:
        result = run_single_benchmark(n_spots, n_genes, n_types)
        results.append(result)

        if not result['success']:
            print(f"\nStopping: benchmark failed at {n_spots:,} spots")
            break

    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "scalability_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Spots':>10} {'Time (s)':>10} {'Memory (GB)':>12} {'Accuracy':>10}")
    print("-" * 50)
    for _, row in df[df['success'] == True].iterrows():
        print(f"{row['n_spots']:>10,} {row['time_sec']:>10.2f} "
              f"{row['mem_delta_mb']/1024:>12.2f} {row['accuracy']:>10.4f}")

    # Create figure
    if len(df[df['success'] == True]) >= 3:
        fig_path = os.path.join(args.output_dir, "scalability_figure.png")
        create_figure(df, fig_path)


if __name__ == "__main__":
    main()
